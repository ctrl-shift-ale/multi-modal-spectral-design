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

import numpy as np
import soundfile as sf
from scipy.optimize import minimize

from timbral_target import (
    TONAL_AUDIO_PATH,
    TARGETS,
    PRIORITIES,
    TARGET_MODE,
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
from spectral_optimizer import N_BANDS, band_edges, mix_channels, build_initial_simplex
from sensitivity_analysis import PERTURBATION_DB, measure_sensitivity
from config import (
    OUTPUT_AUDIO_PATH, NOISE_AUDIO_PATH,
    RELEVANCE_RATIO_THRESHOLD, MIN_ACTIVE_BANDS, MAX_ACTIVE_BANDS,
    WARMSTART_BANDS, WARMSTART_MAX_ITER,
    IDEAL_GAIN_MIN_DB, IDEAL_GAIN_MAX_DB,
    LIMITER_GAIN_MIN_DB, LIMITER_GAIN_MAX_DB, PINNED_EPSILON_DB,
    NELDER_MEAD_STEP_DB,
    MAX_ITER, FATOL, XATOL,
)


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
            sibling = idx + N_BANDS if idx < N_BANDS else idx - N_BANDS
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
    if noise_active and idx >= N_BANDS:
        return "noise", idx - N_BANDS
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
        x0[i] = np.clip(gain_guess, IDEAL_GAIN_MIN_DB, IDEAL_GAIN_MAX_DB)

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
    warm_bounds = [(IDEAL_GAIN_MIN_DB, IDEAL_GAIN_MAX_DB)] * n
    x0_sub = x0[top_within_active]
    result = minimize(
        objective,
        x0_sub,
        method="Nelder-Mead",
        bounds=warm_bounds,
        options={
            "maxiter": warmstart_max_iter, "fatol": FATOL, "xatol": XATOL,
            "initial_simplex": build_initial_simplex(x0_sub, NELDER_MEAD_STEP_DB, warm_bounds),
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
    n_candidates = N_BANDS * (2 if noise_audio is not None else 1)

    def objective(sub_gains_db: np.ndarray) -> float:
        key = tuple(np.round(sub_gains_db, 6))
        if key in cache:
            return cache[key]

        full_gains_db = np.zeros(n_candidates)
        full_gains_db[active_bands] = sub_gains_db

        tonal_gains_db = full_gains_db[:N_BANDS]
        noise_gains_db = full_gains_db[N_BANDS:] if noise_audio is not None else None

        mixed = mix_channels(tonal_audio, fs, tonal_gains_db, noise_audio, noise_gains_db)
        achieved = analyse(mixed, fs, priorities=priorities_map)
        error = total_error(targets, achieved)

        cache[key] = error
        return error

    return objective


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
    n_candidates = N_BANDS * (2 if noise_active else 1)
    objective = make_restricted_objective(tonal_audio, fs, targets, active_bands, noise_audio=noise_audio)

    k = len(active_bands)

    x0 = sensitivity_x0(active_bands, sensitivity_matrix, targets, starting_achieved)
    print(f"  first-order guess from existing sensitivity data: {np.round(x0, 2)} dB (0 extra evaluations)")

    if WARMSTART_BANDS > 0:
        print(f"  refining with a {min(WARMSTART_BANDS, k)}-band pre-solve, seeded from that guess...")
        x0 = warmstart_x0(
            tonal_audio, fs, targets, active_bands, relevance,
            WARMSTART_BANDS, WARMSTART_MAX_ITER, x0_seed=x0, noise_audio=noise_audio,
        )

    ideal_bounds = [(IDEAL_GAIN_MIN_DB, IDEAL_GAIN_MAX_DB)] * k

    def run_pass(x0_pass, bounds_pass, label=None):
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
                "maxiter": MAX_ITER, "fatol": FATOL, "xatol": XATOL,
                "initial_simplex": build_initial_simplex(x0_pass, NELDER_MEAD_STEP_DB, bounds_pass),
            },
        )

    result = run_pass(x0, ideal_bounds)

    pinned = np.array([
        (abs(g - IDEAL_GAIN_MIN_DB) <= PINNED_EPSILON_DB) or (abs(g - IDEAL_GAIN_MAX_DB) <= PINNED_EPSILON_DB)
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
            (LIMITER_GAIN_MIN_DB, LIMITER_GAIN_MAX_DB) if pinned[i] else ideal_bounds[i]
            for i in range(k)
        ]
        escalated_band_ids = {int(active_bands[i]) for i in range(k) if pinned[i]}
        result = run_pass(result.x, escalated_bounds, label="limiter")

    # best_full_gains_db is a flat candidate vector, same layout as
    # active_bands/relevance/sensitivity_matrix: [0, N_BANDS) tonal,
    # [N_BANDS, 2*N_BANDS) noise (only when noise_active). Split it back
    # into the two channels' own gain vectors right here, once, so both
    # the final measurement below AND whatever the caller does with the
    # result (writing audio, reporting) see an explicit noise_gains_db
    # rather than having to know the flat layout themselves.
    best_full_gains_db = np.zeros(n_candidates)
    best_full_gains_db[active_bands] = result.x
    tonal_gains_db = best_full_gains_db[:N_BANDS]
    noise_gains_db = best_full_gains_db[N_BANDS:] if noise_active else None

    best_edited = mix_channels(tonal_audio, fs, tonal_gains_db, noise_audio, noise_gains_db)
    best_achieved = analyse(best_edited, fs)
    best_error = total_error(targets, best_achieved)

    return best_full_gains_db, best_edited, best_error, best_achieved, result, escalated_band_ids


def print_run_summary(targets, starting_achieved, best_achieved, edges,
                       active_bands, relevance, best_gains_db, escalated_band_ids,
                       noise_active: bool = False):
    """One consolidated block at the end of the run, instead of having to
    piece the picture together from the scattered before/after reports and
    the raw gains array: per-parameter source/target/achieved/error, and
    per-candidate (channel + frequency range)/relevance/final gain for
    every candidate the search was actually allowed to touch. Two tables,
    not one -- there are 7 parameters and a different number of active
    candidates, so cramming both into one row-aligned table would
    misrepresent the data, not simplify it."""
    print("\n" + "=" * 78)
    print("RUN SUMMARY")
    print("=" * 78)

    print("\n-- parameters --")
    header = f"{'parameter':<12}{'source':<10}{'target':<16}{'achieved':<10}{'hit?':<6}{'priority':<10}{'weighted err':<14}"
    print(header)
    print("-" * len(header))
    for name in PARAM_NAMES:
        t = targets[name]
        src = starting_achieved.get(name)
        val = best_achieved.get(name)
        band = f"{_fmt_num(t.target_min)}-{_fmt_num(t.target_max)}"
        src_str = f"{src:.2f}" if src is not None else "--"
        if val is None:
            print(f"{name:<12}{src_str:<10}{band:<16}{'--':<10}{'N/A':<6}{t.priority:<10.1f}{'0.00000':<14}")
            continue
        # priority 0 -- hit/miss is meaningless (never affects total_error
        # regardless of where it lands), so N/A rather than a misleading
        # yes/no.
        if t.priority <= 0:
            hit_str = "N/A"
        else:
            hit_str = "yes" if t.target_min <= val <= t.target_max else "no"
        werr = t.weighted_error(val)
        print(
            f"{name:<12}{src_str:<10}{band:<16}{val:<10.2f}{hit_str:<6}"
            f"{t.priority:<10.1f}{werr:<14.5f}"
        )
    print("-" * len(header))
    print(f"TOTAL ERROR: {total_error(targets, best_achieved):.5f}")

    print("\n-- active bands --")
    bheader = f"{'channel':<9}{'band':<6}{'freq (Hz)':<18}{'relevance':<12}{'final gain (dB)':<18}{'range used':<12}"
    print(bheader)
    print("-" * len(bheader))
    for idx in active_bands:
        idx = int(idx)
        channel, band = candidate_channel_band(idx, noise_active)
        freq = f"{edges[band]:.0f}-{edges[band + 1]:.0f}"
        rng = "limiter" if idx in escalated_band_ids else "ideal"
        print(f"{channel:<9}{band:<6}{freq:<18}{relevance[idx]:<12.3f}{best_gains_db[idx]:<18.2f}{rng:<12}")
    print("-" * len(bheader))
    n_candidates = N_BANDS * (2 if noise_active else 1)
    print(f"({n_candidates - len(active_bands)} candidate(s) not selected, held fixed at 0 dB)")


def main():
    if not TONAL_AUDIO_PATH.exists():
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    tonal_audio, fs = load_tonal(TONAL_AUDIO_PATH)
    noise_audio, noise_fs = load_noise(NOISE_AUDIO_PATH)
    if noise_audio is not None and noise_fs != fs:
        print(f"note: noise sample rate ({noise_fs} Hz) != tonal ({fs} Hz) -- ignoring noise component\n")
        noise_audio = None
    noise_active = noise_audio is not None
    n_candidates = N_BANDS * (2 if noise_active else 1)
    edges = band_edges(fs, N_BANDS)

    print("--- before optimisation ---")
    # Baseline is the UNEDITED mix (both channels at 0 dB, i.e. unchanged)
    # -- when noise is active this is tonal+noise summed, not tonal alone,
    # since that sum is the actual audible source the search is trying to
    # move away from, and it's also what "relative" TARGET_MODE resolves
    # its deltas against (see build_targets()).
    starting_audio = mix_channels(
        tonal_audio, fs, np.zeros(N_BANDS),
        noise_audio, np.zeros(N_BANDS) if noise_active else None,
    )
    starting_achieved = analyse(starting_audio, fs)
    targets = build_targets(TARGETS, PRIORITIES, baseline=starting_achieved)
    describe_target_resolution(TARGETS, TARGET_MODE, starting_achieved)
    print()
    report(targets, starting_achieved)

    channel_note = " x 2 channels (tonal+noise)" if noise_active else ""
    print(f"\n--- measuring sensitivity ({N_BANDS} bands{channel_note}, perturbation ±{PERTURBATION_DB} dB) ---")
    priorities_map = {name: t.priority for name, t in targets.items()}
    sensitivity_matrix = measure_sensitivity(tonal_audio, fs, noise_audio=noise_audio, priorities=priorities_map)
    relevance = compute_band_relevance(sensitivity_matrix, targets)

    print("\ncandidate relevance (higher = matters more for your active targets):")
    for idx in range(n_candidates):
        channel, band = candidate_channel_band(idx, noise_active)
        band_label = f"{edges[band]:.0f}-{edges[band + 1]:.0f} Hz"
        print(f"  {channel:<6} band {band} ({band_label:<15}) relevance = {relevance[idx]:.3f}")

    active_bands = select_bands(
        relevance, RELEVANCE_RATIO_THRESHOLD, MIN_ACTIVE_BANDS, MAX_ACTIVE_BANDS, noise_active=noise_active
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
    est_evals = (len(active_bands) + 1) + 2 * MAX_ITER
    print(
        f"\n~{seconds_per_eval:.1f}s per evaluation -> up to ~{est_evals * seconds_per_eval / 60:.1f} min "
        f"worst case for {MAX_ITER} iterations across {len(active_bands)} active candidates "
        "(often stops earlier; a limiter-range retry, if triggered, adds up to "
        "another pass on top). Ctrl+C to abort.\n"
    )

    print(f"--- optimising ---")
    best_gains, best_edited, best_error, best_achieved, result, escalated_band_ids = run_priority_optimizer(
        tonal_audio, fs, targets, active_bands, relevance, sensitivity_matrix, starting_achieved,
        noise_audio=noise_audio,
    )

    print(f"\nconverged: {result.success}  ({result.message})")

    print_run_summary(
        targets, starting_achieved, best_achieved, edges,
        active_bands, relevance, best_gains, escalated_band_ids,
        noise_active=noise_active,
    )

    sf.write(OUTPUT_AUDIO_PATH, best_edited, fs)
    print(f"\nedited audio written to: {OUTPUT_AUDIO_PATH}")


if __name__ == "__main__":
    main()
