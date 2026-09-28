"""
attack_shaping.py

Detects a sound's attack (start + peak) and, if present, its decay stage
(the transition from the attack peak into a stable sustain level), and
provides the two crossfade curves used to blend an attack-only edit back
into the whole-signal edit without an audible seam.

Why this exists: of the 7 timbral models, hardness (Timbral_Hardness) is
the one whose formula gives real weight to a short window specifically
after the attack transient -- verified against the actual timbral_models
source, not assumed. Everything else in this tool edits with a single
per-band gain applied uniformly across the WHOLE signal, which works fine
for whole-signal/per-note-averaged parameters but is wasteful for
hardness: the same gain change that shapes the attack window also
reshapes the sustain equally, even when only the attack needed to move.
priority_optimizer.py uses this module to run a second, attack-scoped
edit and cross it into the normal whole-signal edit -- see
refine_attack() there.

Deliberately self-contained: this does its own simple, robust onset/peak/
decay detection rather than calling into timbral_models' internal
attack-detection functions directly. Those are tuned for hardness's own
feature extraction, carry a fair amount of tunable machinery, and aren't
really meant as a public API -- depending on them here would tie this
tool's editing behaviour to timbral_models' internal implementation
details AND its exact installed version. Everything here can be tested
on its own with plain numpy, with no audio-analysis dependency at all.

No GUI. No Max/OSC. All user-editable settings live in config.py -- edit
that file, then re-run priority_optimizer.py / main.py.
"""

import numpy as np

from config import (
    ATTACK_ENVELOPE_HOP_MS,
    ATTACK_ENVELOPE_WINDOW_MS,
    ATTACK_ONSET_THRESHOLD_FRAC,
    ATTACK_PEAK_SEARCH_MAX_MS,
    ATTACK_CENTROID_WINDOW_MS,
    DEFAULT_XFADE_DURATION_MS,
    DECAY_STABILIZATION_TOLERANCE_DB,
    DECAY_STABILIZATION_HOLD_MS,
    DECAY_SEARCH_MAX_MS,
)
from spectral_optimizer import apply_band_gains, db_to_linear


# ============================================================
# Core functions — shouldn't need to touch below here to test scenarios.
# All the values that WOULD normally need editing live in config.py.
# ============================================================

def compute_envelope(audio: np.ndarray, fs: int, hop_ms: float = ATTACK_ENVELOPE_HOP_MS,
                      window_ms: float = ATTACK_ENVELOPE_WINDOW_MS) -> tuple:
    """Overlapping RMS envelope: each estimate averages `window_ms` of
    audio (window_ms >= hop_ms, so estimates overlap), stepped every
    `hop_ms`. The overlap matters -- see ATTACK_ENVELOPE_WINDOW_MS's
    comment in config.py: a plain non-overlapping block barely longer
    than one cycle of a low-pitched fundamental produces a phase-dependent
    RMS wobble that a naive stabilisation check can mistake for real
    envelope movement. Returns (envelope_linear, envelope_db, hop_samples)
    -- linear for threshold-fraction comparisons (onset, peak), dB for the
    decay-stabilisation tolerance check, since that's naturally a
    perceptual (log) comparison."""
    hop_samples = max(1, int(round(hop_ms / 1000.0 * fs)))
    window_samples = max(hop_samples, int(round(window_ms / 1000.0 * fs)))
    n_hops = max(1, int(np.ceil(len(audio) / hop_samples)))

    envelope_linear = np.zeros(n_hops)
    for i in range(n_hops):
        start = i * hop_samples
        block = audio[start:start + window_samples]
        if len(block):
            envelope_linear[i] = np.sqrt(np.mean(block.astype(float) ** 2))

    envelope_db = 20.0 * np.log10(np.clip(envelope_linear, 1e-9, None))
    return envelope_linear, envelope_db, hop_samples


def detect_attack_and_decay(audio: np.ndarray, fs: int) -> dict:
    """Finds the attack (onset + peak) and, if one exists, the decay stage
    following it -- the stretch between the peak and the point where the
    envelope stabilises into a steady sustain level. Returns a dict:

      attack_start_idx, peak_idx   -- sample indices of attack start/peak
      decay_detected                -- bool
      decay_end_idx                 -- sample index where stabilisation
                                        was found (only when decay_detected)
      xfade_start_idx, xfade_duration_samples
                                     -- where/how long the crossfade should
                                        run. When decay_detected AND its
                                        natural length is at least
                                        DEFAULT_XFADE_DURATION_MS, this IS
                                        the decay stage (start=peak, length
                                        = measured decay length) -- the
                                        transition rides the sound's own
                                        decay. Otherwise (no decay found,
                                        or one shorter than the minimum
                                        usable crossfade) both fall back to
                                        ATTACK_CENTROID_WINDOW_MS after
                                        attack_start_idx / DEFAULT_XFADE_
                                        DURATION_MS -- a forced, too-short
                                        crossfade sounds abrupt, so the
                                        floor takes over rather than
                                        shrinking to fit.
      used_fallback                 -- bool, True whenever the branch above
                                        used the config defaults rather
                                        than a measured decay stage.
    """
    n = len(audio)
    envelope_linear, envelope_db, hop_samples = compute_envelope(audio, fs)
    hop_ms = hop_samples / fs * 1000.0

    peak_rms = float(np.max(envelope_linear)) if len(envelope_linear) else 0.0
    onset_threshold = ATTACK_ONSET_THRESHOLD_FRAC * peak_rms

    onset_hops = np.where(envelope_linear >= onset_threshold)[0]
    attack_start_hop = int(onset_hops[0]) if len(onset_hops) else 0
    attack_start_idx = attack_start_hop * hop_samples

    peak_search_hops = max(1, int(round(ATTACK_PEAK_SEARCH_MAX_MS / hop_ms)))
    search_end_hop = min(len(envelope_linear), attack_start_hop + peak_search_hops + 1)
    if search_end_hop > attack_start_hop:
        peak_hop = attack_start_hop + int(np.argmax(envelope_linear[attack_start_hop:search_end_hop]))
    else:
        peak_hop = attack_start_hop
    peak_idx = peak_hop * hop_samples

    # -- decay-stabilisation search --
    hold_hops = max(1, int(round(DECAY_STABILIZATION_HOLD_MS / hop_ms)))
    decay_search_end_hop = min(len(envelope_db), peak_hop + int(round(DECAY_SEARCH_MAX_MS / hop_ms)) + 1)

    decay_detected = False
    stabilised_hop = None
    for i in range(peak_hop, max(peak_hop, decay_search_end_hop - hold_hops)):
        window = envelope_db[i:i + hold_hops + 1]
        if len(window) < 2:
            continue
        # Range over the WHOLE hold window, not just adjacent-hop deltas --
        # adjacent-only deltas can be individually tiny even while a signal
        # keeps smoothly drifting (e.g. a slow swell), which would get
        # mistaken for a stabilised plateau one hop at a time. The window's
        # total range catches that cumulative drift correctly.
        if (np.max(window) - np.min(window)) <= DECAY_STABILIZATION_TOLERANCE_DB:
            stabilised_hop = i
            decay_detected = True
            break

    decay_end_idx = stabilised_hop * hop_samples if decay_detected else None
    decay_length_ms = (stabilised_hop - peak_hop) * hop_ms if decay_detected else 0.0

    if decay_detected and decay_length_ms >= DEFAULT_XFADE_DURATION_MS:
        xfade_start_idx = peak_idx
        xfade_duration_samples = int(round(decay_length_ms / 1000.0 * fs))
        used_fallback = False
    else:
        xfade_start_idx = attack_start_idx + int(round(ATTACK_CENTROID_WINDOW_MS / 1000.0 * fs))
        xfade_duration_samples = int(round(DEFAULT_XFADE_DURATION_MS / 1000.0 * fs))
        used_fallback = True

    # keep the window inside the signal -- a short test clip or a source
    # with almost no tail could otherwise push start+duration past the end.
    xfade_start_idx = int(np.clip(xfade_start_idx, 0, max(0, n - 1)))
    xfade_duration_samples = int(np.clip(xfade_duration_samples, 0, n - xfade_start_idx))

    return {
        "attack_start_idx": attack_start_idx,
        "peak_idx": peak_idx,
        "decay_detected": decay_detected,
        "decay_end_idx": decay_end_idx,
        "xfade_start_idx": xfade_start_idx,
        "xfade_duration_samples": xfade_duration_samples,
        "used_fallback": used_fallback,
    }


def crossfade_linear(a: np.ndarray, b: np.ndarray, start_idx: int, n_samples: int) -> np.ndarray:
    """a before the window, a linear (amplitude) ramp from a to b across
    the window, b after it. Correct choice for two signals that stay
    phase-correlated through the transition -- e.g. the same tonal source
    reshaped by two different band-gain vectors: at the window's midpoint
    each is at 0.5 gain, and since they're in phase, 0.5+0.5 sums back to
    the original amplitude (0dB) rather than boosting it."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    start = int(np.clip(start_idx, 0, n))
    dur = int(np.clip(n_samples, 0, n - start))

    result = np.array(b, dtype=float, copy=True)
    result[:start] = a[:start]
    if dur > 0:
        ramp = np.linspace(0.0, 1.0, dur, endpoint=True)
        result[start:start + dur] = a[start:start + dur] * (1.0 - ramp) + b[start:start + dur] * ramp
    return result


def crossfade_equal_power(a: np.ndarray, b: np.ndarray, start_idx: int, n_samples: int) -> np.ndarray:
    """Same shape as crossfade_linear, but with a constant-power (roughly
    cosine/sine) curve instead of a straight ramp. Correct choice for two
    signals that are NOT phase-correlated through the transition -- e.g.
    the noise component, where a linear fade of two independently-shaped
    noise edits can produce an audible dip in loudness at the midpoint,
    since their energy doesn't sum coherently the way in-phase tonal
    content does. At the window's midpoint each signal is at ~0.707 gain,
    so uncorrelated power sums back to the original (0.707^2+0.707^2=1)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    start = int(np.clip(start_idx, 0, n))
    dur = int(np.clip(n_samples, 0, n - start))

    result = np.array(b, dtype=float, copy=True)
    result[:start] = a[:start]
    if dur > 0:
        ramp = np.linspace(0.0, np.pi / 2.0, dur, endpoint=True)
        gain_a, gain_b = np.cos(ramp), np.sin(ramp)
        result[start:start + dur] = a[start:start + dur] * gain_a + b[start:start + dur] * gain_b
    return result


def crossfade_edit(tonal_audio: np.ndarray, fs: int,
                    tonal_attack_gains_db: np.ndarray, tonal_pass1_gains_db: np.ndarray,
                    noise_audio: np.ndarray, noise_attack_gains_db: np.ndarray, noise_pass1_gains_db: np.ndarray,
                    xfade_start_idx: int, xfade_duration_samples: int) -> np.ndarray:
    """Builds the composite that refine_attack() searches over and
    ultimately writes out: each channel gets its own attack-focused edit
    AND its own pass-1 whole-signal edit (both full-duration, both via
    the same apply_band_gains() used everywhere else in this tool), then
    each channel is crossfaded from its attack edit into its pass-1 edit
    independently. ONLY after each channel is independently crossfaded
    are the two summed -- crossfading the already-summed tonal+noise
    composite with one shared curve would be wrong for whichever channel
    it didn't match.

    Both channels use crossfade_linear here, not crossfade_equal_power --
    this is NOT the general "tonal vs noise" choice described in
    crossfade_linear/crossfade_equal_power's own docstrings, and is worth
    being explicit about why: apply_band_gains() edits by scaling STFT
    MAGNITUDE only and leaves phase untouched, for both channels alike.
    So noise_before and noise_after here aren't two independent,
    decorrelated noise bursts -- they're the exact same underlying
    noise_audio recording, phase-locked to each other by construction,
    just reshaped by two different gain vectors. That's the same
    "same source, reshaped two ways" situation the tonal channel is in,
    and crossfade_equal_power's +3dB midpoint bump (verified in
    test_attack_shaping.py) applies just as much to two phase-locked
    noise edits as to two phase-locked tonal edits -- including the
    degenerate case where the attack-pass search leaves a channel's gains
    identical to pass 1's, where equal-power would inject an audible
    bump into a spot where NOTHING was actually meant to change.
    crossfade_equal_power stays available in this module for a genuinely
    decorrelated crossfade (two independently-generated signals); it's
    just not what this specific composite needs for either channel.

    noise_audio=None skips the noise channel entirely, same convention as
    mix_channels()."""
    tonal_before = apply_band_gains(tonal_audio, fs, db_to_linear(tonal_attack_gains_db))
    tonal_after = apply_band_gains(tonal_audio, fs, db_to_linear(tonal_pass1_gains_db))
    tonal_out = crossfade_linear(tonal_before, tonal_after, xfade_start_idx, xfade_duration_samples)

    if noise_audio is None:
        return tonal_out

    noise_before = apply_band_gains(noise_audio, fs, db_to_linear(noise_attack_gains_db))
    noise_after = apply_band_gains(noise_audio, fs, db_to_linear(noise_pass1_gains_db))
    noise_out = crossfade_linear(noise_before, noise_after, xfade_start_idx, xfade_duration_samples)

    n = min(len(tonal_out), len(noise_out))
    return tonal_out[:n] + noise_out[:n]
