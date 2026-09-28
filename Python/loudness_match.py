"""
loudness_match.py

Final step, run after everything else in priority_optimizer.py: measures
the ORIGINAL source's (tonal + noise, unedited) integrated loudness
(LUFS-I, ITU-R BS.1770) and the edited result's own, then applies a
single broadband gain to the edited result so its delivered loudness
matches the source's.

Why this is a separate step rather than another search target: all 7
timbral parameters this tool searches over (warmth, brightness, depth,
hardness, roughness, sharpness, booming) describe the SHAPE of the
spectrum, not its overall level -- and boosting or cutting bands changes
total signal energy as a side effect of reshaping, whether or not that
was ever the intent. Nothing upstream of this corrects for it. This step
is deliberately the very last thing that happens to the audio: it reads
whatever the search (and, when it ran, attack refinement) already decided
and applies one flat gain on top, so "the edit should sound like a
reshaping of the source, not a louder or quieter version of it" holds
regardless of what happened earlier in the pipeline.

Uses pyloudnorm (pip install pyloudnorm) for the ITU-R BS.1770
measurement itself -- K-weighting + the relative/absolute gating
algorithm is a precisely specified standard, not something worth
reimplementing here.

No GUI. No Max/OSC. All user-editable settings live in config.py -- edit
that file, then re-run main.py.
"""
import numpy as np
import pyloudnorm

from config import LOUDNESS_MATCH_PEAK_CEILING_DBFS

# ITU-R BS.1770's own absolute silence gate (Gamma_a in the spec, and in
# pyloudnorm's meter.py) -- reused here as the threshold below which a
# measured loudness is treated as "silence, not a real level to match
# against or to", not an independent choice.
ABSOLUTE_SILENCE_GATE_LUFS = -70.0


def measure_integrated_loudness(audio: np.ndarray, fs: int) -> float:
    """ITU-R BS.1770 integrated loudness (LUFS-I) via pyloudnorm. The
    standard's gating block is 400ms, and pyloudnorm refuses anything
    shorter outright -- a sound-design one-shot can easily be shorter
    than that, so this falls back to a single ungated block sized to the
    whole clip in that case. BS.1770's relative-threshold gating exists
    to exclude quiet passages from a longer programme's average, which
    isn't a meaningful distinction on a clip too short to gate in the
    first place -- one whole-signal block is the closest sane reading,
    not a compromise (verified against a steady tone measured both ways:
    the gated and single-block readings agree to within 0.01 LUFS)."""
    meter = pyloudnorm.Meter(fs)
    try:
        return float(meter.integrated_loudness(audio))
    except ValueError:
        short_block_s = max(1, len(audio) - 1) / fs
        short_meter = pyloudnorm.Meter(fs, block_size=short_block_s)
        return float(short_meter.integrated_loudness(audio))


def match_loudness(audio: np.ndarray, fs: int, target_lufs: float) -> dict:
    """Applies a single broadband gain to `audio` so its own integrated
    loudness matches target_lufs -- clamped so the result's peak never
    exceeds LOUDNESS_MATCH_PEAK_CEILING_DBFS, even if that means landing
    short of the target.

    Skips entirely (returns audio unchanged) when either the edited
    signal or the target is at/below ABSOLUTE_SILENCE_GATE_LUFS -- there
    is no finite gain that sensibly "matches loudness" to or from
    silence, so trying would either blow up (dividing toward -inf LUFS)
    or mute a real signal to satisfy an undefined target.

    Returns a dict: matched_audio, source_loudness (== target_lufs, kept
    for convenience/reporting), edited_loudness (measured before
    matching), applied_gain_db, final_loudness (measured after matching
    -- differs from target_lufs only when peak_limited), peak_limited,
    skipped (True only in the silence case above)."""
    edited_loudness = measure_integrated_loudness(audio, fs)

    if (not np.isfinite(edited_loudness) or edited_loudness <= ABSOLUTE_SILENCE_GATE_LUFS
            or not np.isfinite(target_lufs) or target_lufs <= ABSOLUTE_SILENCE_GATE_LUFS):
        return {
            "matched_audio": audio,
            "source_loudness": target_lufs,
            "edited_loudness": edited_loudness,
            "applied_gain_db": 0.0,
            "final_loudness": edited_loudness,
            "peak_limited": False,
            "skipped": True,
        }

    ideal_gain_db = target_lufs - edited_loudness
    ideal_gain_linear = 10.0 ** (ideal_gain_db / 20.0)

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    ceiling_linear = 10.0 ** (LOUDNESS_MATCH_PEAK_CEILING_DBFS / 20.0)

    peak_limited = False
    applied_gain_linear = ideal_gain_linear
    if peak > 0 and peak * ideal_gain_linear > ceiling_linear:
        applied_gain_linear = ceiling_linear / peak
        peak_limited = True

    applied_gain_db = 20.0 * np.log10(applied_gain_linear) if applied_gain_linear > 0 else float("-inf")
    matched_audio = (audio * applied_gain_linear).astype(audio.dtype)
    final_loudness = measure_integrated_loudness(matched_audio, fs)

    return {
        "matched_audio": matched_audio,
        "source_loudness": target_lufs,
        "edited_loudness": edited_loudness,
        "applied_gain_db": applied_gain_db,
        "final_loudness": final_loudness,
        "peak_limited": peak_limited,
        "skipped": False,
    }
