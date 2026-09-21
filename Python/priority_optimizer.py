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
    build_targets,
    describe_target_resolution,
    analyse,
    total_error,
    report,
    PARAM_NAMES,
    _fmt_num,
)
from spectral_optimizer import N_BANDS, band_edges, apply_band_gains, db_to_linear, build_initial_simplex
from sensitivity_analysis import PERTURBATION_DB, measure_sensitivity
from config import (
    OUTPUT_AUDIO_PATH,
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
    """relevance[band] = sum over ACTIVE parameters of priority * |sensitivity|.
    A band scores high if it strongly moves parameters the user actually
    set a real (non-full-range), priority>0 target for, weighted by how
    much they said that parameter matters."""
    relevance = np.zeros(N_BANDS)
    for param_j, name in enumerate(PARAM_NAMES):
        t = targets[name]
        if not is_active(t.target_min, t.target_max, t.priority):
            continue
        relevance += t.priority * np.abs(sensitivity_matrix[:, param_j])
    return relevance


def select_bands(relevance: np.ndarray, ratio_threshold: float, min_bands: int, max_bands: int) -> np.ndarray:
    """Indices of bands whose relevance is at least `ratio_threshold` of the
    top band's relevance, in band order (not relevance order) -- keeps
    downstream code simpler. Adaptive: how many bands this returns depends
    on how concentrated relevance actually is on this sound/target combo,
    not a fixed count. Clamped to [min_bands, max_bands] as guardrails."""
    max_bands = min(max_bands, N_BANDS)
    ranked = np.argsort(relevance)[::-1]

    if relevance[ranked[0]] <= 0:
        # nothing measurably relevant at all -- just take the floor
        return np.sort(ranked[:min_bands])

    cutoff = relevance[ranked[0]] * ratio_threshold
    kept = ranked[relevance[ranked] >= cutoff]

    if len(kept) < min_bands:
        kept = ranked[:min_bands]
    elif len(kept) > max_bands:
        kept = ranked[:max_bands]

    return np.sort(kept)


def warmstart_x0(tonal_audio, fs, targets, active_bands: np.ndarray, relevance: np.ndarray,
                  warmstart_bands: int, warmstart_max_iter: int) -> np.ndarray:
    """Quick low-dimensional pre-solve on just the most relevant of the
    active bands, used to seed the full joint search's starting point
    instead of 0 dB everywhere. Nelder-Mead's evaluation count is
    sensitive to how close the initial simplex is to the optimum, so this
    usually cuts iterations needed in the full search. Returns a vector
    the same length as active_bands, in active_bands' order (band order,
    not relevance order)."""
    x0 = np.zeros(len(active_bands))
    n = min(warmstart_bands, len(active_bands))
    if n <= 0:
        return x0

    active_relevance = relevance[active_bands]
    top_within_active = np.argsort(active_relevance)[::-1][:n]
    warm_band_ids = active_bands[top_within_active]

    objective = make_restricted_objective(tonal_audio, fs, targets, warm_band_ids)
    warm_bounds = [(IDEAL_GAIN_MIN_DB, IDEAL_GAIN_MAX_DB)] * n
    result = minimize(
        objective,
        np.zeros(n),
        method="Nelder-Mead",
        bounds=warm_bounds,
        options={
            "maxiter": warmstart_max_iter, "fatol": FATOL, "xatol": XATOL,
            "initial_simplex": build_initial_simplex(np.zeros(n), NELDER_MEAD_STEP_DB, warm_bounds),
        },
    )

    x0[top_within_active] = result.x
    return x0


def make_restricted_objective(tonal_audio, fs, targets, active_bands: np.ndarray):
    """Same idea as spectral_optimizer.make_objective, but the function it
    returns only takes len(active_bands) parameters (in dB) -- everything
    else stays fixed at 0 dB (unchanged). Also skips computing any
    priority<=0 parameter on every call, same reasoning as
    spectral_optimizer.py's version -- this is the hot loop, so that's
    where skipping actually saves real time. Memoized for the same reason
    as before: each evaluation is expensive, don't pay for the same point
    twice."""
    cache = {}
    priorities_map = {name: t.priority for name, t in targets.items()}

    def objective(sub_gains_db: np.ndarray) -> float:
        key = tuple(np.round(sub_gains_db, 6))
        if key in cache:
            return cache[key]

        full_gains_db = np.zeros(N_BANDS)
        full_gains_db[active_bands] = sub_gains_db

        edited = apply_band_gains(tonal_audio, fs, db_to_linear(full_gains_db))
        achieved = analyse(edited, fs, priorities=priorities_map)
        error = total_error(targets, achieved)

        cache[key] = error
        return error

    return objective


def run_priority_optimizer(tonal_audio, fs, targets, active_bands: np.ndarray, relevance: np.ndarray):
    """Runs the joint search over active_bands within the ideal dB range
    first. If it converges with any band pinned against an ideal-range
    edge AND the target is still missed -- i.e. the ideal range is
    provably the thing standing in the way -- it retries, warm-started
    from that result, with the limiter range substituted in ONLY for the
    pinned band(s). A band that wasn't pinned, or whose target was
    already hit, never sees the limiter range."""
    objective = make_restricted_objective(tonal_audio, fs, targets, active_bands)

    k = len(active_bands)
    if WARMSTART_BANDS > 0:
        print(f"  warm-starting from a {min(WARMSTART_BANDS, k)}-band pre-solve...")
        x0 = warmstart_x0(tonal_audio, fs, targets, active_bands, relevance, WARMSTART_BANDS, WARMSTART_MAX_ITER)
    else:
        x0 = np.zeros(k)

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
        pinned_labels = [f"band {active_bands[i]}" for i in range(k) if pinned[i]]
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

    best_full_gains_db = np.zeros(N_BANDS)
    best_full_gains_db[active_bands] = result.x
    best_edited = apply_band_gains(tonal_audio, fs, db_to_linear(best_full_gains_db))
    best_achieved = analyse(best_edited, fs)
    best_error = total_error(targets, best_achieved)

    return best_full_gains_db, best_edited, best_error, best_achieved, result, escalated_band_ids


def print_run_summary(targets, starting_achieved, best_achieved, edges,
                       active_bands, relevance, best_gains_db, escalated_band_ids):
    """One consolidated block at the end of the run, instead of having to
    piece the picture together from the scattered before/after reports and
    the raw gains array: per-parameter source/target/achieved/error, and
    per-band frequency range/relevance/final gain for every band the
    search was actually allowed to touch. Two tables, not one -- there are
    7 parameters and a different number of active bands, so cramming both
    into one row-aligned table would misrepresent the data, not simplify it."""
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
    bheader = f"{'band':<6}{'freq (Hz)':<18}{'relevance':<12}{'final gain (dB)':<18}{'range used':<12}"
    print(bheader)
    print("-" * len(bheader))
    for i in active_bands:
        i = int(i)
        freq = f"{edges[i]:.0f}-{edges[i + 1]:.0f}"
        rng = "limiter" if i in escalated_band_ids else "ideal"
        print(f"{i:<6}{freq:<18}{relevance[i]:<12.3f}{best_gains_db[i]:<18.2f}{rng:<12}")
    print("-" * len(bheader))
    print(f"({N_BANDS - len(active_bands)} bands not selected, held fixed at 0 dB)")


def main():
    if not TONAL_AUDIO_PATH.exists():
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    tonal_audio, fs = load_tonal(TONAL_AUDIO_PATH)
    edges = band_edges(fs, N_BANDS)

    print("--- before optimisation ---")
    starting_achieved = analyse(tonal_audio, fs)
    targets = build_targets(TARGETS, PRIORITIES, baseline=starting_achieved)
    describe_target_resolution(TARGETS, TARGET_MODE, starting_achieved)
    print()
    report(targets, starting_achieved)

    print(f"\n--- measuring sensitivity ({N_BANDS} bands, perturbation ±{PERTURBATION_DB} dB) ---")
    priorities_map = {name: t.priority for name, t in targets.items()}
    sensitivity_matrix = measure_sensitivity(tonal_audio, fs, priorities=priorities_map)
    relevance = compute_band_relevance(sensitivity_matrix, targets)

    print("\nband relevance (higher = matters more for your active targets):")
    for i in range(N_BANDS):
        band_label = f"{edges[i]:.0f}-{edges[i + 1]:.0f} Hz"
        print(f"  band {i} ({band_label:<15}) relevance = {relevance[i]:.3f}")

    active_bands = select_bands(relevance, RELEVANCE_RATIO_THRESHOLD, MIN_ACTIVE_BANDS, MAX_ACTIVE_BANDS)
    active_labels = [f"{edges[i]:.0f}-{edges[i + 1]:.0f} Hz" for i in active_bands]
    print(f"\nsearching only the top {len(active_bands)} bands: {list(zip(active_bands.tolist(), active_labels))}")
    print(f"(remaining {N_BANDS - len(active_bands)} bands held fixed at 0 dB)")

    # Time one evaluation in the RESTRICTED search space for an honest
    # estimate -- fewer dimensions means fewer evaluations needed overall.
    probe = make_restricted_objective(tonal_audio, fs, targets, active_bands)
    t0 = time.time()
    probe(np.zeros(len(active_bands)))
    seconds_per_eval = time.time() - t0
    est_evals = (len(active_bands) + 1) + 2 * MAX_ITER
    print(
        f"\n~{seconds_per_eval:.1f}s per evaluation -> up to ~{est_evals * seconds_per_eval / 60:.1f} min "
        f"worst case for {MAX_ITER} iterations across {len(active_bands)} active bands "
        "(often stops earlier; a limiter-range retry, if triggered, adds up to "
        "another pass on top). Ctrl+C to abort.\n"
    )

    print(f"--- optimising ---")
    best_gains, best_edited, best_error, best_achieved, result, escalated_band_ids = run_priority_optimizer(
        tonal_audio, fs, targets, active_bands, relevance
    )

    print(f"\nconverged: {result.success}  ({result.message})")

    print_run_summary(
        targets, starting_achieved, best_achieved, edges,
        active_bands, relevance, best_gains, escalated_band_ids,
    )

    sf.write(OUTPUT_AUDIO_PATH, best_edited, fs)
    print(f"\nedited audio written to: {OUTPUT_AUDIO_PATH}")


if __name__ == "__main__":
    main()
