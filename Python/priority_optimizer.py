"""
priority_optimizer.py

Run via main.py (the stable entry point) -- this file can still be run
directly too, both do the same thing. Pulls in everything else:
timbral_target.py (objective function), spectral_optimizer.py (band
editing), and sensitivity_analysis.py (relevance ranking).

Combines sensitivity_analysis.py with spectral_optimizer.py: instead of
letting Nelder-Mead search over all N_BANDS blindly, first measure which
bands actually move the parameters the user has set real targets for
(their "active" targets -- see is_active() below), then restrict the
search to just the most relevant ones.

This is the concrete version of "spectral strategies should depend on
what the user prioritised, and on the actual sound" -- it doesn't assume
which bands matter, it measures it on this sound, weighted by which
parameters the user actually cares about and how much (priority).

Two direct benefits over the plain band-gain optimizer:
  1. Fewer search dimensions -> fewer objective evaluations -> faster,
     directly attacking the performance issue.
  2. The bands left OUT of the search are frequency regions with little
     measured effect on anything the user asked for -- so nothing is
     spent (or risked, in terms of unwanted side effects) on them.

Bands not selected are held fixed at 0 dB (unchanged), not removed from
the signal.

No GUI. No Max/OSC. All user-editable settings live in config.py -- edit
that file, then re-run main.py.
"""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import minimize

from timbral_target import (
    load_tonal,
    load_noise,
    build_targets,
    describe_target_resolution,
    analyse,
    total_error,
    report,
    PARAM_NAMES,
    _fmt_num,
)
from spectral_optimizer import band_edges, mix_channels, build_initial_simplex
from sensitivity_analysis import measure_sensitivity
from attack_shaping import detect_attack_and_decay, crossfade_edit
from loudness_match import measure_integrated_loudness, match_loudness
import config

# config.py's values are read live via `config.X` at the point of use,
# never cached into a module-level name here -- see spectral_optimizer.py's
# equivalent note for why (the short version: OSC can rewrite config.py's
# attributes on the fly between runs -- see osc_interface_Max.py -- and a
# name bound once at import time would silently stop tracking that).


# ============================================================
# Core functions — shouldn't need to touch below here to test scenarios.
# All the values that WOULD normally need editing live in config.py.
# ============================================================

def is_active(target_min: float, target_max: float, priority: float, full_range: float = 100.0) -> bool:
    """A target counts as 'active' (worth steering the search toward) only
    if BOTH: its band is narrower than the full 0-100 native range (a
    (0, 100) target can never contribute error), AND its priority is > 0
    (priority 0 means the user explicitly doesn't want it influencing
    anything, regardless of what range is set)."""
    return (target_max - target_min) < full_range and priority > 0


def compute_band_relevance(sensitivity_matrix: np.ndarray, targets: dict) -> np.ndarray:
    """relevance[candidate] = sum over ACTIVE parameters of priority *
    |sensitivity|. A candidate scores high if it strongly moves parameters
    the user actually set a real (non-full-range), priority>0 target for,
    weighted by how much they said that parameter matters. Follows
    sensitivity_matrix's own row count (N_BANDS for tonal-only, 2*N_BANDS
    for tonal+noise -- see measure_sensitivity()) rather than assuming
    N_BANDS, so this needs no changes to handle either case."""
    relevance = np.zeros(sensitivity_matrix.shape[0])
    for param_j, name in enumerate(PARAM_NAMES):
        t = targets[name]
        if not is_active(t.target_min, t.target_max, t.priority):
            continue
        relevance += t.priority * np.abs(sensitivity_matrix[:, param_j])
    return relevance


def select_bands(relevance: np.ndarray, ratio_threshold: float, min_bands: int, max_bands: int,
                  noise_active: bool = False) -> np.ndarray:
    """Indices of candidates whose relevance is at least `ratio_threshold`
    of the top candidate's relevance, in candidate order (not relevance
    order) -- keeps downstream code simpler. Adaptive: how many are
    returned depends on how concentrated relevance actually is for this
    sound/target combo, not a fixed count. Clamped to [min_bands,
    max_bands] as guardrails on this PRIMARY selection -- against the
    total candidate count, tonal and noise combined: which channel fills
    that budget is left entirely to measured relevance, not hardcoded, so
    a noise-dominant source (a field recording, say) can end up almost
    entirely noise candidates, and a clean tonal source can end up
    entirely tonal ones -- same mechanism either way, no special-casing
    per source type.

    Tonal/noise pairing (only when noise_active): both channels share the
    same underlying frequency region for a given band index, and they get
    SUMMED before anything is measured (mix_channels()) -- so if one
    channel's band in a region is selected because it's clearly relevant,
    and the other channel is ALSO measurably relevant there (its own
    relevance > 0, however small next to its sibling's), that sibling is
    pulled in too, even though its relevance alone wouldn't have cleared
    ratio_threshold. Otherwise the search could shape only, say, the
    tonal content of a band while leaving that same band's noise energy
    untouched -- which is exactly the kind of ceiling that shows up as a
    band pinning against the ideal-range edge without reaching the
    target, when the other channel in that same region actually had room
    to help. A sibling with EXACTLY zero measured relevance (no response
    to the perturbation probe at all) is left out -- there's nothing to
    gain from searching it. This pairing step runs after the
    ratio/min/max selection above, so it can push the final candidate
    count past max_bands -- max_bands bounds how many band REGIONS get
    chosen, not the total candidates once a chosen region's other channel
    is added in."""
    n_candidates = len(relevance)
    max_bands = min(max_bands, n_candidates)
    ranked = np.argsort(relevance)[::-1]

    if relevance[ranked[0]] <= 0:
        # nothing measurably relevant at all -- just take the floor
        kept = ranked[:min_bands]
    else:
        cutoff = relevance[ranked[0]] * ratio_threshold
        kept = ranked[relevance[ranked] >= cutoff]

        if len(kept) < min_bands:
            kept = ranked[:min_bands]
        elif len(kept) > max_bands:
            kept = ranked[:max_bands]

    if noise_active:
        paired = set(int(i) for i in kept)
        for idx in list(paired):
            sibling = idx + config.N_BANDS if idx < config.N_BANDS else idx - config.N_BANDS
            if relevance[sibling] > 1e-9 and sibling not in paired:
                paired.add(sibling)
        kept = np.array(sorted(paired))

    return np.sort(kept)


def candidate_channel_band(idx: int, noise_active: bool) -> tuple:
    """Maps a flat candidate index back to (channel, band_index).
    Candidates [0, N_BANDS) are tonal bands; [N_BANDS, 2*N_BANDS) -- only
    meaningful when noise_active -- are noise bands. Both channels share
    the SAME N_BANDS log-spaced split of the spectrum (band_edges()
    depends only on fs and N_BANDS, not on which channel), so a band
    index means the same frequency range in either channel."""
    idx = int(idx)
    if noise_active and idx >= config.N_BANDS:
        return "noise", idx - config.N_BANDS
    return "tonal", idx


def candidate_label(idx: int, noise_active: bool) -> str:
    """Human-readable label for a flat candidate index, e.g. 'tonal band 5'
    or 'noise band 3' -- for print statements that don't have `edges` on
    hand to also show a frequency range (see candidate_channel_band())."""
    channel, band = candidate_channel_band(idx, noise_active)
    return f"{channel} band {band}"


def sensitivity_x0(active_bands: np.ndarray, sensitivity_matrix: np.ndarray, targets: dict,
                    starting_achieved: dict) -> np.ndarray:
    """First-order initial gain guess, one value per active band, computed
    entirely from data ALREADY measured during sensitivity analysis --
    zero extra objective evaluations. This replaces guessing blind from
    0 dB: sensitivity_matrix[band, param] is a real measured slope (from
    the +-PERTURBATION_DB probe around 0 dB in sensitivity_analysis.py),
    i.e. exactly the "try a gain, see how much the error changes" data
    point -- we already paid for it, we just weren't using it to aim
    anywhere before.

    For each band, this finds whichever ACTIVE (priority>0, real-range)
    target parameter that band's measured sensitivity affects most, then
    solves that local slope for the gain which would move the parameter
    from its baseline value to the middle of its target band -- treating
    the response as locally linear (a first-order approximation, not an
    assumption that it's linear everywhere). Bands are solved
    independently (each assuming every other band stays at 0 dB), so this
    can't account for cross-band interaction -- that's what the joint
    Nelder-Mead search downstream is for; this just starts it in the
    right neighbourhood instead of at 0 dB. Clipped to the ideal range,
    so a steep target that a small local slope implausibly extrapolates
    past the bound (e.g. "raise this band 40 dB") lands exactly on that
    bound instead -- which is the same as trying the extreme and seeing
    what happens, just resolved analytically instead of by brute force."""
    x0 = np.zeros(len(active_bands))

    for i, band in enumerate(active_bands):
        best_name, best_slope = None, 0.0
        for param_j, name in enumerate(PARAM_NAMES):
            t = targets[name]
            if not is_active(t.target_min, t.target_max, t.priority):
                continue
            slope = sensitivity_matrix[band, param_j]
            if abs(slope) > abs(best_slope):
                best_name, best_slope = name, slope

        if best_name is None or abs(best_slope) < 1e-9:
            continue  # nothing measurably active for this band -- leave at 0 dB

        baseline = starting_achieved.get(best_name)
        if baseline is None:
            continue

        t = targets[best_name]
        target_mid = (t.target_min + t.target_max) / 2.0
        gain_guess = (target_mid - baseline) / best_slope
        x0[i] = np.clip(gain_guess, config.IDEAL_GAIN_MIN_DB, config.IDEAL_GAIN_MAX_DB)

    return x0


def warmstart_x0(tonal_audio, fs, targets, active_bands: np.ndarray, relevance: np.ndarray,
                  warmstart_bands: int, warmstart_max_iter: int, x0_seed: np.ndarray = None,
                  noise_audio: np.ndarray = None) -> np.ndarray:
    """Quick low-dimensional pre-solve on just the most relevant of the
    active bands (tonal or noise -- active_bands is a flat candidate
    list, see candidate_channel_band()), used to REFINE the full joint
    search's starting point. x0_seed (see sensitivity_x0) seeds every
    active band, not just the ones this pre-solve searches directly --
    bands outside warmstart_bands pass through with their seed value
    untouched rather than reverting to 0 dB. Nelder-Mead's evaluation
    count is sensitive to how close the initial simplex is to the
    optimum, so both this pre-solve AND seeding it well (instead of
    starting from 0 dB) cut iterations needed in the full search. Returns
    a vector the same length as active_bands, in active_bands' order
    (candidate order, not relevance order)."""
    x0 = np.zeros(len(active_bands)) if x0_seed is None else np.array(x0_seed, dtype=float)
    n = min(warmstart_bands, len(active_bands))
    if n <= 0:
        return x0

    active_relevance = relevance[active_bands]
    top_within_active = np.argsort(active_relevance)[::-1][:n]
    warm_band_ids = active_bands[top_within_active]

    objective = make_restricted_objective(tonal_audio, fs, targets, warm_band_ids, noise_audio=noise_audio)
    warm_bounds = [(config.IDEAL_GAIN_MIN_DB, config.IDEAL_GAIN_MAX_DB)] * n
    x0_sub = x0[top_within_active]
    result = minimize(
        objective,
        x0_sub,
        method="Nelder-Mead",
        bounds=warm_bounds,
        options={
            "maxiter": warmstart_max_iter, "fatol": config.FATOL, "xatol": config.XATOL,
            "initial_simplex": build_initial_simplex(x0_sub, config.NELDER_MEAD_STEP_DB, warm_bounds),
        },
    )

    x0[top_within_active] = result.x
    return x0


def make_restricted_objective(tonal_audio, fs, targets, active_bands: np.ndarray, noise_audio: np.ndarray = None):
    """Same idea as spectral_optimizer.make_objective, but the function it
    returns only takes len(active_bands) parameters (in dB) -- everything
    else stays fixed at 0 dB (unchanged). active_bands indexes a flat
    candidate space: [0, N_BANDS) are tonal bands, and -- only when
    noise_audio is given -- [N_BANDS, 2*N_BANDS) are noise bands (see
    candidate_channel_band()). The two channels are edited independently
    with their own gains, then mixed (summed) via mix_channels() before
    being measured -- that's the actual audible signal the timbral models
    hear, not two isolated stems. noise_audio=None collapses this back to
    the original tonal-only behaviour exactly (mix_channels() just
    returns the edited tonal signal).

    Also skips computing any priority<=0 parameter on every call, same
    reasoning as spectral_optimizer.py's version -- this is the hot loop,
    so that's where skipping actually saves real time. Memoized for the
    same reason as before: each evaluation is expensive, don't pay for
    the same point twice."""
    cache = {}
    priorities_map = {name: t.priority for name, t in targets.items()}
    n_candidates = config.N_BANDS * (2 if noise_audio is not None else 1)

    def objective(sub_gains_db: np.ndarray) -> float:
        key = tuple(np.round(sub_gains_db, 6))
        if key in cache:
            return cache[key]

        full_gains_db = np.zeros(n_candidates)
        full_gains_db[active_bands] = sub_gains_db

        tonal_gains_db = full_gains_db[:config.N_BANDS]
        noise_gains_db = full_gains_db[config.N_BANDS:] if noise_audio is not None else None

        mixed = mix_channels(tonal_audio, fs, tonal_gains_db, noise_audio, noise_gains_db)
        achieved = analyse(mixed, fs, priorities=priorities_map)
        error = total_error(targets, achieved)

        cache[key] = error
        return error

    return objective


def run_search_pass(objective, x0_pass: np.ndarray, bounds_pass: list, label: str = None):
    """Runs one Nelder-Mead search given an already-built objective,
    starting point, and bounds -- prints a progress line per iteration.
    Factored out of run_priority_optimizer() (rather than kept as a local
    closure there) so refine_attack()'s second, attack-scoped search pass
    can reuse the exact same search mechanics instead of duplicating
    them."""
    history = []

    def callback(xk):
        err = objective(xk)
        history.append(err)
        prefix = f"  [{label}] " if label else "  "
        print(f"{prefix}iter {len(history):>4}  total_error = {err:.5f}")

    return minimize(
        objective,
        x0_pass,
        method="Nelder-Mead",
        bounds=bounds_pass,
        callback=callback,
        options={
            "maxiter": config.MAX_ITER, "fatol": config.FATOL, "xatol": config.XATOL,
            "initial_simplex": build_initial_simplex(x0_pass, config.NELDER_MEAD_STEP_DB, bounds_pass),
        },
    )


def run_priority_optimizer(tonal_audio, fs, targets, active_bands: np.ndarray, relevance: np.ndarray,
                            sensitivity_matrix: np.ndarray, starting_achieved: dict,
                            noise_audio: np.ndarray = None):
    """Runs the joint search over active_bands within the ideal dB range
    first. If it converges with any band pinned against an ideal-range
    edge AND the target is still missed -- i.e. the ideal range is
    provably the thing standing in the way -- it retries, warm-started
    from that result, with the limiter range substituted in ONLY for the
    pinned band(s). A band that wasn't pinned, or whose target was
    already hit, never sees the limiter range.

    The starting point for all of this is no longer 0 dB: sensitivity_x0()
    first turns the sensitivity data already measured for these bands into
    a direct per-band gain guess (0 extra evaluations), which warmstart_x0()
    then refines with a short joint pre-solve on just the most relevant
    bands. Both are just about getting the full search's starting simplex
    close to the answer -- neither replaces it, since only the full joint
    search actually accounts for how the active bands interact together.

    active_bands is a flat candidate list (tonal + noise, when noise_audio
    is given -- see candidate_channel_band()), and the gains this returns
    are split back into their two channels and mixed via mix_channels()
    before being measured or written -- exactly like every other search
    step here, so tonal and noise are always evaluated in the combined
    signal the timbral models actually hear."""
    noise_active = noise_audio is not None
    n_candidates = config.N_BANDS * (2 if noise_active else 1)
    objective = make_restricted_objective(tonal_audio, fs, targets, active_bands, noise_audio=noise_audio)

    k = len(active_bands)

    x0 = sensitivity_x0(active_bands, sensitivity_matrix, targets, starting_achieved)
    print(f"  first-order guess from existing sensitivity data: {np.round(x0, 2)} dB (0 extra evaluations)")

    if config.WARMSTART_BANDS > 0:
        print(f"  refining with a {min(config.WARMSTART_BANDS, k)}-band pre-solve, seeded from that guess...")
        x0 = warmstart_x0(
            tonal_audio, fs, targets, active_bands, relevance,
            config.WARMSTART_BANDS, config.WARMSTART_MAX_ITER, x0_seed=x0, noise_audio=noise_audio,
        )

    ideal_bounds = [(config.IDEAL_GAIN_MIN_DB, config.IDEAL_GAIN_MAX_DB)] * k

    result = run_search_pass(objective, x0, ideal_bounds)

    pinned = np.array([
        (abs(g - config.IDEAL_GAIN_MIN_DB) <= config.PINNED_EPSILON_DB)
        or (abs(g - config.IDEAL_GAIN_MAX_DB) <= config.PINNED_EPSILON_DB)
        for g in result.x
    ])

    escalated_band_ids = set()
    if pinned.any() and result.fun > 1e-9:
        pinned_labels = [candidate_label(active_bands[i], noise_active) for i in range(k) if pinned[i]]
        print(
            f"\n  {len(pinned_labels)} band(s) pinned at the ideal-range edge and target "
            f"still missed (error={result.fun:.5f}) -- retrying with the limiter range "
            f"for: {', '.join(pinned_labels)}\n"
        )
        escalated_bounds = [
            (config.LIMITER_GAIN_MIN_DB, config.LIMITER_GAIN_MAX_DB) if pinned[i] else ideal_bounds[i]
            for i in range(k)
        ]
        escalated_band_ids = {int(active_bands[i]) for i in range(k) if pinned[i]}
        result = run_search_pass(objective, result.x, escalated_bounds, label="limiter")

    # best_full_gains_db is a flat candidate vector, same layout as
    # active_bands/relevance/sensitivity_matrix: [0, N_BANDS) tonal,
    # [N_BANDS, 2*N_BANDS) noise (only when noise_active). Split it back
    # into the two channels' own gain vectors right here, once, so both
    # the final measurement below AND whatever the caller does with the
    # result (writing audio, reporting) see an explicit noise_gains_db
    # rather than having to know the flat layout themselves.
    best_full_gains_db = np.zeros(n_candidates)
    best_full_gains_db[active_bands] = result.x
    tonal_gains_db = best_full_gains_db[:config.N_BANDS]
    noise_gains_db = best_full_gains_db[config.N_BANDS:] if noise_active else None

    best_edited = mix_channels(tonal_audio, fs, tonal_gains_db, noise_audio, noise_gains_db)
    best_achieved = analyse(best_edited, fs)
    best_error = total_error(targets, best_achieved)

    return best_full_gains_db, best_edited, best_error, best_achieved, result, escalated_band_ids


def refine_attack(tonal_audio, fs, targets, active_bands: np.ndarray, pass1_gains_db: np.ndarray,
                   noise_audio: np.ndarray = None):
    """Second search pass, run ONLY when hardness has a real (active)
    target (see is_active() / main()) -- hardness is the one parameter,
    of the 7, whose formula is attack-window-specific (see
    attack_shaping.py's module docstring), so it's the only reason this
    tool needs anything more than a single whole-signal edit.

    Pass 1 (run_priority_optimizer(), already run by the time this is
    called) found the best single gain vector for the WHOLE signal --
    good for every parameter, including hardness's own whole-signal
    features, but it can't move hardness's attack-specific feature any
    further without also reshaping the sustain, which the rest of the
    search already settled on being correct. This pass searches a SECOND
    gain vector meant only for the attack, and composites it against
    pass 1's finished edit via crossfade_edit() -- so what changes here
    is confined to the attack, and everything pass 1 already got right
    elsewhere is left alone.

    Reuses active_bands/targets from pass 1 rather than re-selecting bands
    from scratch: the bands already measured as relevant to the active
    targets include hardness's own sensitivity (see
    compute_band_relevance()), so there's no separate band-selection step
    needed here -- same candidates, a second gain vector for them.

    Attack/decay is detected on the BASELINE (unedited, 0dB-mixed) audio,
    not on pass 1's edit -- the physical attack/decay structure being
    located belongs to the original sound object, and shouldn't shift
    just because pass 1 happened to reshape the envelope somewhat.

    Returns (attack_gains_db, final_edited, final_error, final_achieved,
    result, xfade_info) -- xfade_info is detect_attack_and_decay()'s own
    dict, useful for the run summary."""
    noise_active = noise_audio is not None
    n_candidates = config.N_BANDS * (2 if noise_active else 1)

    baseline_audio = mix_channels(
        tonal_audio, fs, np.zeros(config.N_BANDS),
        noise_audio, np.zeros(config.N_BANDS) if noise_active else None,
    )
    xfade_info = detect_attack_and_decay(baseline_audio, fs)

    tonal_pass1_db = pass1_gains_db[:config.N_BANDS]
    noise_pass1_db = pass1_gains_db[config.N_BANDS:] if noise_active else None

    priorities_map = {name: t.priority for name, t in targets.items()}

    def build_composite(sub_gains_db: np.ndarray) -> np.ndarray:
        full_gains_db = np.zeros(n_candidates)
        full_gains_db[active_bands] = sub_gains_db
        tonal_attack_db = full_gains_db[:config.N_BANDS]
        noise_attack_db = full_gains_db[config.N_BANDS:] if noise_active else None
        return crossfade_edit(
            tonal_audio, fs, tonal_attack_db, tonal_pass1_db,
            noise_audio, noise_attack_db, noise_pass1_db,
            xfade_info["xfade_start_idx"], xfade_info["xfade_duration_samples"],
        )

    cache = {}

    def objective(sub_gains_db: np.ndarray) -> float:
        key = tuple(np.round(sub_gains_db, 6))
        if key in cache:
            return cache[key]
        composite = build_composite(sub_gains_db)
        achieved = analyse(composite, fs, priorities=priorities_map)
        error = total_error(targets, achieved)
        cache[key] = error
        return error

    # Warm-started from pass 1's own gains, not 0 dB -- pass 1 is already
    # a reasonable edit for the attack too, this pass only needs to find
    # how much further to push it there specifically.
    x0 = pass1_gains_db[active_bands]
    bounds = [(config.IDEAL_GAIN_MIN_DB, config.IDEAL_GAIN_MAX_DB)] * len(active_bands)
    result = run_search_pass(objective, x0, bounds, label="attack")

    best_full_gains_db = np.zeros(n_candidates)
    best_full_gains_db[active_bands] = result.x

    final_edited = build_composite(result.x)
    final_achieved = analyse(final_edited, fs)
    final_error = total_error(targets, final_achieved)

    return best_full_gains_db, final_edited, final_error, final_achieved, result, xfade_info


def build_run_report(targets, starting_achieved, best_achieved, edges,
                      active_bands, relevance, best_gains_db, escalated_band_ids,
                      noise_active: bool = False, converged: bool = None, convergence_message: str = None,
                      attack_info: dict = None, loudness_info: dict = None, output_audio_path=None) -> dict:
    """Everything a finished "edit" run needs to report -- to the console
    (via print_report() below) AND, unchanged, to Max over OSC as the
    final message before the server goes back to listening (see
    osc_interface_Max.py) -- assembled ONCE into a single plain,
    JSON-serialisable dict (only str/int/float/bool/None/list/dict), so
    there is exactly one source of truth for "what happened in this run"
    rather than the console text and the OSC report drifting apart.

    Shape:
      mode: "edit"
      converged, convergence_message: from the (last) Nelder-Mead result
      total_error: float
      parameters: [{name, source, target_min, target_max, achieved, hit,
                     priority, weighted_error}, ...] -- one per PARAM_NAMES,
                    in that order; achieved/hit are None for a priority<=0
                    parameter analyse() skipped computing.
      active_bands: [{candidate_index, channel, band, freq_min, freq_max,
                       relevance, final_gain_db, range_used}, ...] -- one
                     per candidate the search was allowed to touch.
      bands_not_selected: int -- candidates held fixed at 0 dB.
      attack_refinement: None, or {decay_detected, used_fallback,
                    attack_start_ms, peak_ms, xfade_start_ms, xfade_end_ms,
                    bands: [{channel, band, freq_min, freq_max,
                              attack_gain_db, pass1_gain_db}, ...]}
                  -- only present when hardness was an active target (see
                  refine_attack() / main()).
      loudness_match: None, or {skipped, gate_lufs} when skipped, else
                    {skipped: False, source_loudness_lufs,
                     edited_loudness_lufs, applied_gain_db,
                     final_loudness_lufs, peak_limited,
                     peak_ceiling_dbfs (only when peak_limited)} -- only
                  present when MATCH_SOURCE_LOUDNESS was on.
      output_audio_path: str or None."""
    parameters = []
    for name in PARAM_NAMES:
        t = targets[name]
        src = starting_achieved.get(name)
        val = best_achieved.get(name)
        hit = None
        if val is not None:
            # bool(...) matters here: val/target_min/target_max can mix in a
            # numpy float (from best_achieved), and a chained comparison
            # against one produces a numpy bool scalar rather than a plain
            # Python bool -- harmless everywhere except json.dumps() below,
            # which only accepts the real builtin bool and raises "Object of
            # type bool is not JSON serializable" otherwise (numpy's scalar
            # bool type's class name prints as plain "bool", which is what
            # makes that error look like it's complaining about a real bool).
            hit = bool(t.target_min <= val <= t.target_max) if t.priority > 0 else None
        parameters.append({
            "name": name,
            "source": src,
            "target_min": t.target_min,
            "target_max": t.target_max,
            "achieved": val,
            "hit": hit,
            "priority": t.priority,
            "weighted_error": t.weighted_error(val) if val is not None else 0.0,
        })

    active_band_reports = []
    for idx in active_bands:
        idx = int(idx)
        channel, band = candidate_channel_band(idx, noise_active)
        active_band_reports.append({
            "candidate_index": idx,
            "channel": channel,
            "band": band,
            "freq_min": float(edges[band]),
            "freq_max": float(edges[band + 1]),
            "relevance": float(relevance[idx]),
            "final_gain_db": float(best_gains_db[idx]),
            "range_used": "limiter" if idx in escalated_band_ids else "ideal",
        })

    n_candidates = config.N_BANDS * (2 if noise_active else 1)

    attack_refinement = None
    if attack_info is not None:
        xfade_info = attack_info["xfade_info"]
        fs_local = attack_info["fs"]
        attack_gains_db = attack_info["attack_gains_db"]
        attack_refinement = {
            "decay_detected": bool(xfade_info["decay_detected"]),
            "used_fallback": bool(xfade_info["used_fallback"]),
            "attack_start_ms": xfade_info["attack_start_idx"] / fs_local * 1000,
            "peak_ms": xfade_info["peak_idx"] / fs_local * 1000,
            "xfade_start_ms": xfade_info["xfade_start_idx"] / fs_local * 1000,
            "xfade_end_ms": (xfade_info["xfade_start_idx"] + xfade_info["xfade_duration_samples"]) / fs_local * 1000,
            "bands": [],
        }
        for idx in active_bands:
            idx = int(idx)
            channel, band = candidate_channel_band(idx, noise_active)
            attack_refinement["bands"].append({
                "channel": channel,
                "band": band,
                "freq_min": float(edges[band]),
                "freq_max": float(edges[band + 1]),
                "attack_gain_db": float(attack_gains_db[idx]),
                "pass1_gain_db": float(best_gains_db[idx]),
            })

    loudness_match = None
    if loudness_info is not None:
        if loudness_info["skipped"]:
            loudness_match = {"skipped": True, "gate_lufs": loudness_info.get("gate", -70.0)}
        else:
            loudness_match = {
                "skipped": False,
                "source_loudness_lufs": loudness_info["source_loudness"],
                "edited_loudness_lufs": loudness_info["edited_loudness"],
                "applied_gain_db": loudness_info["applied_gain_db"],
                "final_loudness_lufs": loudness_info["final_loudness"],
                "peak_limited": bool(loudness_info["peak_limited"]),
            }
            if loudness_info["peak_limited"]:
                loudness_match["peak_ceiling_dbfs"] = config.LOUDNESS_MATCH_PEAK_CEILING_DBFS

    return {
        "mode": "edit",
        "converged": bool(converged) if converged is not None else None,
        "convergence_message": convergence_message,
        "total_error": total_error(targets, best_achieved),
        "parameters": parameters,
        "active_bands": active_band_reports,
        "bands_not_selected": n_candidates - len(active_bands),
        "attack_refinement": attack_refinement,
        "loudness_match": loudness_match,
        "output_audio_path": Path(output_audio_path).as_posix() if output_audio_path is not None else None,
    }


def build_scan_report(achieved: dict, fs: int, noise_active: bool) -> dict:
    """Same idea as build_run_report(), for MODE="scan" -- just the 7
    raw timbral readings plus the sources/sample rate that were analysed,
    no targets/search/output file to report since scan mode has none."""
    return {
        "mode": "scan",
        "tonal_source": config.TONAL_AUDIO_PATH.as_posix(),
        "noise_source": config.NOISE_AUDIO_PATH.as_posix() if noise_active else None,
        "sample_rate_hz": fs,
        "parameters": [{"name": name, "value": achieved[name]} for name in PARAM_NAMES],
    }


def print_report(report: dict) -> None:
    """Prints build_run_report()'s or build_scan_report()'s dict to the
    console -- the exact same information the OSC final report carries
    (see osc_interface_Max.py), just formatted for a human reading the
    terminal instead of a Max patch parsing OSC. Pure formatter: takes no
    argument this dict doesn't already carry, so console output and the
    OSC report can never drift apart from each other."""
    if report["mode"] == "scan":
        print("--- scan mode: analysis only (no targets, no search, no output file) ---\n")
        print(f"tonal source: {report['tonal_source']}")
        print(f"noise source: {report['noise_source'] or '(none -- tonal-only)'}")
        print(f"sample rate:  {report['sample_rate_hz']} Hz\n")

        header = f"{'parameter':<12}{'value':<10}"
        print(header)
        print("-" * len(header))
        for p in report["parameters"]:
            print(f"{p['name']:<12}{p['value']:<10.2f}")
        return

    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)

    print("\n-- parameters --")
    header = f"{'parameter':<12}{'source':<10}{'target':<16}{'achieved':<10}{'hit?':<6}{'priority':<10}{'weighted err':<14}"
    print(header)
    print("-" * len(header))
    for p in report["parameters"]:
        band = f"{_fmt_num(p['target_min'])}-{_fmt_num(p['target_max'])}"
        src_str = f"{p['source']:.2f}" if p["source"] is not None else "--"
        if p["achieved"] is None:
            print(f"{p['name']:<12}{src_str:<10}{band:<16}{'--':<10}{'N/A':<6}{p['priority']:<10.1f}{'0.00000':<14}")
            continue
        hit_str = "N/A" if p["hit"] is None else ("yes" if p["hit"] else "no")
        print(
            f"{p['name']:<12}{src_str:<10}{band:<16}{p['achieved']:<10.2f}{hit_str:<6}"
            f"{p['priority']:<10.1f}{p['weighted_error']:<14.5f}"
        )
    print("-" * len(header))
    print(f"TOTAL ERROR: {report['total_error']:.5f}")

    print("\n-- active bands --")
    bheader = f"{'channel':<9}{'band':<6}{'freq (Hz)':<18}{'relevance':<12}{'final gain (dB)':<18}{'range used':<12}"
    print(bheader)
    print("-" * len(bheader))
    for b in report["active_bands"]:
        freq = f"{b['freq_min']:.0f}-{b['freq_max']:.0f}"
        print(f"{b['channel']:<9}{b['band']:<6}{freq:<18}{b['relevance']:<12.3f}{b['final_gain_db']:<18.2f}{b['range_used']:<12}")
    print("-" * len(bheader))
    print(f"({report['bands_not_selected']} candidate(s) not selected, held fixed at 0 dB)")

    attack_refinement = report["attack_refinement"]
    if attack_refinement is not None:
        print("\n-- attack refinement (hardness is active) --")
        if attack_refinement["decay_detected"] and not attack_refinement["used_fallback"]:
            print(f"decay stage detected: peak -> stabilised over "
                  f"{attack_refinement['xfade_end_ms'] - attack_refinement['xfade_start_ms']:.1f} ms -- "
                  f"crossfade rides the sound's own decay")
        else:
            reason = ("no decay stage found" if not attack_refinement["decay_detected"]
                       else "detected decay was shorter than the minimum usable crossfade")
            print(f"{reason} -- fell back to config defaults "
                  f"(offset={config.ATTACK_CENTROID_WINDOW_MS:.0f}ms after attack start, "
                  f"duration={config.DEFAULT_XFADE_DURATION_MS:.0f}ms)")
        print(f"attack start: {attack_refinement['attack_start_ms']:.1f} ms   "
              f"peak: {attack_refinement['peak_ms']:.1f} ms   "
              f"crossfade: {attack_refinement['xfade_start_ms']:.1f} ms "
              f"-> {attack_refinement['xfade_end_ms']:.1f} ms")

        print(f"\n{'channel':<9}{'band':<6}{'freq (Hz)':<18}{'attack gain (dB)':<18}{'pass-1 gain (dB)':<18}")
        print("-" * 69)
        for b in attack_refinement["bands"]:
            freq = f"{b['freq_min']:.0f}-{b['freq_max']:.0f}"
            print(f"{b['channel']:<9}{b['band']:<6}{freq:<18}{b['attack_gain_db']:<18.2f}{b['pass1_gain_db']:<18.2f}")

    loudness_match = report["loudness_match"]
    if loudness_match is not None:
        print("\n-- loudness matching (final broadband gain, applied last) --")
        if loudness_match["skipped"]:
            print("skipped -- source and/or edited signal too quiet to measure "
                  f"a meaningful loudness (<= {loudness_match['gate_lufs']:.0f} LUFS); audio left unchanged")
        else:
            print(f"source (unedited) loudness:  {loudness_match['source_loudness_lufs']:.2f} LUFS-I")
            print(f"edited, pre-match loudness:  {loudness_match['edited_loudness_lufs']:.2f} LUFS-I")
            print(f"applied gain:                {loudness_match['applied_gain_db']:+.2f} dB")
            print(f"final (written) loudness:    {loudness_match['final_loudness_lufs']:.2f} LUFS-I")
            if loudness_match["peak_limited"]:
                shortfall = loudness_match["source_loudness_lufs"] - loudness_match["final_loudness_lufs"]
                print(f"NOTE: gain was limited by the {loudness_match['peak_ceiling_dbfs']:.1f} dBFS peak ceiling -- "
                      f"landed {shortfall:.2f} LU short of an exact match rather than risk clipping")


def run_scan(tonal_audio: np.ndarray, fs: int, noise_audio: np.ndarray = None) -> dict:
    """MODE="scan": the analysis half of the pipeline only -- no TARGETS,
    no sensitivity probing, no search, nothing written to OUTPUT_AUDIO_PATH.
    Just "where does this sound sit right now?". Analyses the same signal
    edit mode would start from: tonal + noise summed at 0 dB (unchanged)
    when a noise stem is active, tonal alone otherwise -- via
    mix_channels(), same as main()'s baseline, so a scan and the "source"
    row of an edit run always agree.

    Returns the structured report dict (see build_scan_report()) after
    printing it -- same convention as main()'s edit-mode return, so a
    caller (main.py, or the OSC /run handler in osc_interface_Max.py) gets
    the same kind of value regardless of which MODE actually ran."""
    noise_active = noise_audio is not None
    audio = mix_channels(
        tonal_audio, fs, np.zeros(config.N_BANDS),
        noise_audio, np.zeros(config.N_BANDS) if noise_active else None,
    )
    achieved = analyse(audio, fs)

    scan_report = build_scan_report(achieved, fs, noise_active)
    print_report(scan_report)
    return scan_report


def main() -> dict:
    """Returns the structured run report (build_run_report() for MODE="edit",
    build_scan_report() for MODE="scan") after printing it -- this is what
    lets a caller other than __main__ (the OSC /run handler in
    osc_interface_Max.py, in particular) get the exact same "what happened
    in this run" data that was printed to the console, to forward to Max
    as the final report."""
    if not config.TONAL_AUDIO_PATH.exists():
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {config.TONAL_AUDIO_PATH}")

    tonal_audio, fs = load_tonal(config.TONAL_AUDIO_PATH)
    noise_audio, noise_fs = load_noise(config.NOISE_AUDIO_PATH)
    if noise_audio is not None and noise_fs != fs:
        print(f"note: noise sample rate ({noise_fs} Hz) != tonal ({fs} Hz) -- ignoring noise component\n")
        noise_audio = None

    if config.MODE == "scan":
        return run_scan(tonal_audio, fs, noise_audio)

    noise_active = noise_audio is not None
    n_candidates = config.N_BANDS * (2 if noise_active else 1)
    edges = band_edges(fs, config.N_BANDS)

    print("--- before optimisation ---")
    # Baseline is the UNEDITED mix (both channels at 0 dB, i.e. unchanged)
    # -- when noise is active this is tonal+noise summed, not tonal alone,
    # since that sum is the actual audible source the search is trying to
    # move away from, and it's also what "relative" TARGET_MODE resolves
    # its deltas against (see build_targets()).
    starting_audio = mix_channels(
        tonal_audio, fs, np.zeros(config.N_BANDS),
        noise_audio, np.zeros(config.N_BANDS) if noise_active else None,
    )
    starting_achieved = analyse(starting_audio, fs)
    targets = build_targets(config.TARGETS, config.PRIORITIES, baseline=starting_achieved)
    describe_target_resolution(config.TARGETS, config.TARGET_MODE, starting_achieved)
    print()
    report(targets, starting_achieved)

    channel_note = " x 2 channels (tonal+noise)" if noise_active else ""
    print(f"\n--- measuring sensitivity ({config.N_BANDS} bands{channel_note}, perturbation ±{config.PERTURBATION_DB} dB) ---")
    priorities_map = {name: t.priority for name, t in targets.items()}
    sensitivity_matrix = measure_sensitivity(tonal_audio, fs, noise_audio=noise_audio, priorities=priorities_map)
    relevance = compute_band_relevance(sensitivity_matrix, targets)

    print("\ncandidate relevance (higher = matters more for your active targets):")
    for idx in range(n_candidates):
        channel, band = candidate_channel_band(idx, noise_active)
        band_label = f"{edges[band]:.0f}-{edges[band + 1]:.0f} Hz"
        print(f"  {channel:<6} band {band} ({band_label:<15}) relevance = {relevance[idx]:.3f}")

    active_bands = select_bands(
        relevance, config.RELEVANCE_RATIO_THRESHOLD, config.MIN_ACTIVE_BANDS, config.MAX_ACTIVE_BANDS,
        noise_active=noise_active
    )
    active_labels = [candidate_label(idx, noise_active) for idx in active_bands]
    print(f"\nsearching only the top {len(active_bands)} candidates: {list(zip(active_bands.tolist(), active_labels))}")
    print(f"(remaining {n_candidates - len(active_bands)} candidates held fixed at 0 dB)")

    # Time one evaluation in the RESTRICTED search space for an honest
    # estimate -- fewer dimensions means fewer evaluations needed overall.
    probe = make_restricted_objective(tonal_audio, fs, targets, active_bands, noise_audio=noise_audio)
    t0 = time.time()
    probe(np.zeros(len(active_bands)))
    seconds_per_eval = time.time() - t0
    est_evals = (len(active_bands) + 1) + 2 * config.MAX_ITER
    print(
        f"\n~{seconds_per_eval:.1f}s per evaluation -> up to ~{est_evals * seconds_per_eval / 60:.1f} min "
        f"worst case for {config.MAX_ITER} iterations across {len(active_bands)} active candidates "
        "(often stops earlier; a limiter-range retry, if triggered, adds up to "
        "another pass on top). Ctrl+C to abort.\n"
    )

    print(f"--- optimising ---")
    best_gains, best_edited, best_error, best_achieved, result, escalated_band_ids = run_priority_optimizer(
        tonal_audio, fs, targets, active_bands, relevance, sensitivity_matrix, starting_achieved,
        noise_audio=noise_audio,
    )

    print(f"\nconverged: {result.success}  ({result.message})")

    # Hardness is the only one of the 7 parameters whose formula is
    # attack-window-specific (see attack_shaping.py's module docstring),
    # so it's the only reason a second, attack-scoped search pass is ever
    # worth running -- skipped entirely (zero extra cost) otherwise.
    hardness_target = targets["hardness"]
    attack_info = None
    final_edited, final_achieved = best_edited, best_achieved

    if is_active(hardness_target.target_min, hardness_target.target_max, hardness_target.priority):
        print(f"\n--- refining attack (hardness target is active) ---")
        (attack_gains_db, attack_edited, attack_error, attack_achieved,
         attack_result, xfade_info) = refine_attack(
            tonal_audio, fs, targets, active_bands, best_gains, noise_audio=noise_audio,
        )
        print(f"\nattack refinement converged: {attack_result.success}  ({attack_result.message})")
        print(f"total error before attack refinement: {best_error:.5f}  ->  after: {attack_error:.5f}")

        final_edited, final_achieved = attack_edited, attack_achieved
        attack_info = {"xfade_info": xfade_info, "attack_gains_db": attack_gains_db, "fs": fs}

    # Last step, after everything above: match the delivered file's
    # overall loudness to the ORIGINAL source's (starting_audio -- the
    # unedited tonal+noise mix already measured for starting_achieved).
    # None of the 7 timbral targets are about overall level, so nothing
    # upstream of this corrects for the level shift a spectral edit
    # causes as a side effect -- see loudness_match.py's module docstring.
    loudness_info = None
    if config.MATCH_SOURCE_LOUDNESS:
        source_loudness = measure_integrated_loudness(starting_audio, fs)
        loudness_result = match_loudness(final_edited, fs, source_loudness)
        final_edited = loudness_result["matched_audio"]
        loudness_info = loudness_result
        if not loudness_result["skipped"]:
            # The gain can nudge the 7 timbral readings (some of the real
            # models have some absolute-level sensitivity), so the report
            # below should reflect what's actually being written, not the
            # pre-gain values -- one more full analyse() call, off the hot
            # loop, same convention as run_priority_optimizer/refine_attack.
            final_achieved = analyse(final_edited, fs)

    sf.write(config.OUTPUT_AUDIO_PATH, final_edited, fs)

    run_report = build_run_report(
        targets, starting_achieved, final_achieved, edges,
        active_bands, relevance, best_gains, escalated_band_ids,
        noise_active=noise_active, converged=result.success, convergence_message=result.message,
        attack_info=attack_info, loudness_info=loudness_info, output_audio_path=config.OUTPUT_AUDIO_PATH,
    )
    print_report(run_report)
    print(f"\nedited audio written to: {config.OUTPUT_AUDIO_PATH}")

    return run_report


if __name__ == "__main__":
    main()
