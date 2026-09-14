"""
priority_optimizer.py

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

Bands not selected are held fixed at gain 1.0 (unchanged), not removed
from the signal.

No GUI. No Max/OSC. Everything you'd want to change to test a scenario is
in the CONFIG block directly below (plus TARGETS/PRIORITIES, which live in
timbral_target.py) -- edit and re-run.
"""

import time

import numpy as np
import soundfile as sf
from scipy.optimize import minimize

from timbral_target import (
    REPO_ROOT,
    TONAL_AUDIO_PATH,
    TARGETS,
    PRIORITIES,
    load_tonal,
    build_targets,
    analyse,
    total_error,
    report,
)
from spectral_optimizer import N_BANDS, band_edges, apply_band_gains, GAIN_MIN, GAIN_MAX
from sensitivity_analysis import PARAM_NAMES, PERTURBATION, measure_sensitivity


# ============================================================
# CONFIG — edit everything below this line to test a scenario
# ============================================================

OUTPUT_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_tonal_edited.wav"

# How many of the N_BANDS bands (ranked by measured relevance) the
# optimizer is actually allowed to touch. The rest stay fixed at gain 1.0.
# Smaller = faster (fewer dimensions for Nelder-Mead to search) but less
# flexible; this is the main lever for trading search quality vs runtime.
TOP_K_BANDS = 3

MAX_ITER = 60
FATOL = 1e-5
XATOL = 1e-3


# ============================================================
# Core functions — shouldn't need to touch below here to test scenarios
# ============================================================

def is_active(target_min: float, target_max: float, full_range: float = 100.0) -> bool:
    """A target counts as 'active' (the user actually cares about it) if
    its band is narrower than the full 0-100 native range. A (0, 100)
    target can never contribute error (see TimbralTarget.raw_error), so
    it shouldn't influence which bands get selected either."""
    return (target_max - target_min) < full_range


def compute_band_relevance(sensitivity_matrix: np.ndarray, targets: dict) -> np.ndarray:
    """relevance[band] = sum over ACTIVE parameters of priority * |sensitivity|.
    A band scores high if it strongly moves parameters the user actually
    set a real (non-full-range) target for, weighted by how much they
    said that parameter matters."""
    relevance = np.zeros(N_BANDS)
    for param_j, name in enumerate(PARAM_NAMES):
        t = targets[name]
        if not is_active(t.target_min, t.target_max):
            continue
        relevance += t.priority * np.abs(sensitivity_matrix[:, param_j])
    return relevance


def select_bands(relevance: np.ndarray, top_k: int) -> np.ndarray:
    """Indices of the top_k highest-relevance bands, in band order (not
    relevance order) -- keeps downstream code simpler."""
    top_k = min(top_k, N_BANDS)
    ranked = np.argsort(relevance)[::-1][:top_k]
    return np.sort(ranked)


def make_restricted_objective(tonal_audio, fs, targets, active_bands: np.ndarray):
    """Same idea as spectral_optimizer.make_objective, but the function it
    returns only takes len(active_bands) parameters -- everything else
    stays fixed at gain 1.0. Memoized for the same reason as before:
    each evaluation is expensive, don't pay for the same point twice."""
    cache = {}

    def objective(sub_gains: np.ndarray) -> float:
        key = tuple(np.round(sub_gains, 6))
        if key in cache:
            return cache[key]

        full_gains = np.ones(N_BANDS)
        full_gains[active_bands] = sub_gains

        edited = apply_band_gains(tonal_audio, fs, full_gains)
        achieved = analyse(edited, fs)
        error = total_error(targets, achieved)

        cache[key] = error
        return error

    return objective


def run_priority_optimizer(tonal_audio, fs, targets, active_bands: np.ndarray):
    objective = make_restricted_objective(tonal_audio, fs, targets, active_bands)

    k = len(active_bands)
    x0 = np.ones(k)
    bounds = [(GAIN_MIN, GAIN_MAX)] * k

    history = []

    def callback(xk):
        err = objective(xk)
        history.append(err)
        print(f"  iter {len(history):>4}  total_error = {err:.5f}")

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        callback=callback,
        options={"maxiter": MAX_ITER, "fatol": FATOL, "xatol": XATOL},
    )

    best_full_gains = np.ones(N_BANDS)
    best_full_gains[active_bands] = result.x
    best_edited = apply_band_gains(tonal_audio, fs, best_full_gains)
    best_achieved = analyse(best_edited, fs)
    best_error = total_error(targets, best_achieved)

    return best_full_gains, best_edited, best_error, best_achieved, result


def main():
    if not TONAL_AUDIO_PATH.exists():
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    tonal_audio, fs = load_tonal(TONAL_AUDIO_PATH)
    targets = build_targets(TARGETS, PRIORITIES)
    edges = band_edges(fs, N_BANDS)

    print("--- before optimisation ---")
    starting_achieved = analyse(tonal_audio, fs)
    report(targets, starting_achieved)

    print(f"\n--- measuring sensitivity ({N_BANDS} bands, perturbation ±{PERTURBATION}) ---")
    sensitivity_matrix = measure_sensitivity(tonal_audio, fs)
    relevance = compute_band_relevance(sensitivity_matrix, targets)

    print("\nband relevance (higher = matters more for your active targets):")
    for i in range(N_BANDS):
        band_label = f"{edges[i]:.0f}-{edges[i + 1]:.0f} Hz"
        print(f"  band {i} ({band_label:<15}) relevance = {relevance[i]:.3f}")

    active_bands = select_bands(relevance, TOP_K_BANDS)
    active_labels = [f"{edges[i]:.0f}-{edges[i + 1]:.0f} Hz" for i in active_bands]
    print(f"\nsearching only the top {len(active_bands)} bands: {list(zip(active_bands.tolist(), active_labels))}")
    print(f"(remaining {N_BANDS - len(active_bands)} bands held fixed at gain 1.0)")

    # Time one evaluation in the RESTRICTED search space for an honest
    # estimate -- fewer dimensions means fewer evaluations needed overall.
    probe = make_restricted_objective(tonal_audio, fs, targets, active_bands)
    t0 = time.time()
    probe(np.ones(len(active_bands)))
    seconds_per_eval = time.time() - t0
    est_evals = (len(active_bands) + 1) + 2 * MAX_ITER
    print(
        f"\n~{seconds_per_eval:.1f}s per evaluation -> up to ~{est_evals * seconds_per_eval / 60:.1f} min "
        f"worst case for {MAX_ITER} iterations across {len(active_bands)} active bands "
        "(often stops earlier). Ctrl+C to abort.\n"
    )

    print(f"--- optimising ---")
    best_gains, best_edited, best_error, best_achieved, result = run_priority_optimizer(
        tonal_audio, fs, targets, active_bands
    )

    print("\n--- after optimisation ---")
    report(targets, best_achieved)

    print(f"\nconverged: {result.success}  ({result.message})")
    print("full band gains (fixed bands shown as 1.0):", np.round(best_gains, 3))

    sf.write(OUTPUT_AUDIO_PATH, best_edited, fs)
    print(f"\nedited audio written to: {OUTPUT_AUDIO_PATH}")


if __name__ == "__main__":
    main()