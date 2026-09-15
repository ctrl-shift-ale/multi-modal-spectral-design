"""
spectral_optimizer.py

Step 2 of the spectral design engine: the OPTIMIZER.

Uses the objective function from timbral_target.py (TARGETS, PRIORITIES,
TimbralTarget, total_error) and actually changes the spectrum to try and
hit it, via a gradient-free search (Nelder-Mead).

WHAT IT EDITS, FOR NOW: this is deliberately coarser than "individual
harmonics" -- true harmonic-level editing needs the tonal/noise
decomposition and harmonic tracking work that's still on the roadmap.
As an interim, working, testable version: the spectrum is split into
N_BANDS logarithmically-spaced frequency bands, and the optimizer searches
for a magnitude gain multiplier per band. That's enough to prove the whole
closed loop -- analyse, edit, re-analyse, converge -- end to end. When
harmonic-level editing exists, it slots in as a drop-in replacement for
apply_band_gains() below; nothing in the objective function or the search
loop itself needs to change.

No GUI. No Max/OSC. All user-editable settings live in config.py -- edit
that file, then re-run this one.
"""

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from scipy.optimize import minimize

from timbral_target import (
    TONAL_AUDIO_PATH,
    TARGETS,
    PRIORITIES,
    TARGET_MODE,
    build_targets,
    describe_target_resolution,
    analyse,
    total_error,
)
from config import N_BANDS, FMIN_HZ, GAIN_MIN, GAIN_MAX, MAX_ITER, FATOL, XATOL, NPERSEG, NOVERLAP, OUTPUT_AUDIO_PATH


# ============================================================
# Core functions — shouldn't need to touch below here to test scenarios.
# All the values that WOULD normally need editing live in config.py.
# ============================================================

def band_edges(fs: int, n_bands: int = N_BANDS, fmin: float = FMIN_HZ) -> np.ndarray:
    """N_BANDS+1 log-spaced edges from fmin to Nyquist."""
    nyquist = fs / 2
    return np.geomspace(fmin, nyquist, n_bands + 1)


def apply_band_gains(audio: np.ndarray, fs: int, gains: np.ndarray) -> np.ndarray:
    """STFT the signal, multiply each frequency bin's magnitude by its
    band's gain (phase untouched), ISTFT back to audio. Same gain is
    applied across the whole duration -- no time-varying shaping yet."""
    edges = band_edges(fs, len(gains))
    f, _, Zxx = stft(audio, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)

    band_idx = np.digitize(f, edges) - 1
    band_idx = np.clip(band_idx, 0, len(gains) - 1)
    gain_per_bin = gains[band_idx]

    Zxx_edited = Zxx * gain_per_bin[:, None]
    _, edited = istft(Zxx_edited, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)

    # ISTFT can return a slightly different length than the input due to
    # framing -- trim or zero-pad back to match so downstream length checks
    # (and the output file) stay predictable.
    if len(edited) > len(audio):
        edited = edited[: len(audio)]
    elif len(edited) < len(audio):
        edited = np.pad(edited, (0, len(audio) - len(edited)))

    return edited.astype(audio.dtype)


def make_objective(tonal_audio: np.ndarray, fs: int, targets: dict):
    """Returns a function(gains) -> scalar error, closing over the fixed
    audio/fs/targets so scipy.optimize only has to deal with the gains
    vector it's actually searching over.

    Skips computing any priority<=0 parameter on every call (derived from
    targets itself, so it always matches whatever priorities were actually
    used to build targets) -- this is the hot loop, called potentially
    hundreds of times, so avoiding a wasted model evaluation here is where
    the real payoff is (unlike the one-off before/after reports, which
    compute everything for full visibility).

    Memoized: Nelder-Mead (and our own progress callback) sometimes ask for
    the score at a point it's already evaluated. Each evaluation costs
    several seconds, so a plain dict cache keyed on the rounded gains
    avoids paying for that twice."""
    cache = {}
    priorities_map = {name: t.priority for name, t in targets.items()}

    def objective(gains: np.ndarray) -> float:
        key = tuple(np.round(gains, 6))
        if key in cache:
            return cache[key]
        edited = apply_band_gains(tonal_audio, fs, gains)
        achieved = analyse(edited, fs, priorities=priorities_map)
        error = total_error(targets, achieved)
        cache[key] = error
        return error

    return objective


def run_optimizer(tonal_audio: np.ndarray, fs: int, targets: dict):
    """Search for the per-band gains that minimise total_error. Returns
    (best_gains, best_edited_audio, best_error, achieved_dict)."""
    objective = make_objective(tonal_audio, fs, targets)

    x0 = np.ones(N_BANDS)
    bounds = [(GAIN_MIN, GAIN_MAX)] * N_BANDS

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

    best_gains = result.x
    best_edited = apply_band_gains(tonal_audio, fs, best_gains)
    best_achieved = analyse(best_edited, fs)
    best_error = total_error(targets, best_achieved)

    return best_gains, best_edited, best_error, best_achieved, result


def main():
    import time
    from timbral_target import load_tonal, report

    if not TONAL_AUDIO_PATH.exists():
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    tonal_audio, fs = load_tonal(TONAL_AUDIO_PATH)

    print("--- before optimisation ---")
    starting_achieved = analyse(tonal_audio, fs)
    targets = build_targets(TARGETS, PRIORITIES, baseline=starting_achieved)
    describe_target_resolution(TARGETS, TARGET_MODE, starting_achieved)
    print()
    report(targets, starting_achieved)

    # Time one real evaluation so the runtime estimate below is measured,
    # not guessed -- this single call also warms nothing, it's just a
    # regular objective evaluation, so it isn't wasted.
    probe_objective = make_objective(tonal_audio, fs, targets)
    t0 = time.time()
    probe_objective(np.ones(N_BANDS))
    seconds_per_eval = time.time() - t0

    # Nelder-Mead needs N_BANDS+1 evaluations just for its starting simplex;
    # each further iteration typically costs roughly one more evaluation,
    # occasionally two or three (reflect/expand/contract). This is a rough
    # worst-case, not a promise -- fatol/xatol often stop it earlier.
    est_worst_case_evals = (N_BANDS + 1) + 2 * MAX_ITER
    est_seconds = est_worst_case_evals * seconds_per_eval
    print(
        f"\n~{seconds_per_eval:.1f}s per evaluation -> up to ~{est_seconds/60:.1f} min "
        f"worst case for {MAX_ITER} iterations across {N_BANDS} bands "
        "(often stops earlier). Ctrl+C to abort.\n"
    )

    print(f"--- optimising ({N_BANDS} bands, up to {MAX_ITER} iterations) ---")
    best_gains, best_edited, best_error, best_achieved, result = run_optimizer(
        tonal_audio, fs, targets
    )

    print("\n--- after optimisation ---")
    report(targets, best_achieved)

    print(f"\nconverged: {result.success}  ({result.message})")
    print("band gains:", np.round(best_gains, 3))

    sf.write(OUTPUT_AUDIO_PATH, best_edited, fs)
    print(f"\nedited audio written to: {OUTPUT_AUDIO_PATH}")


if __name__ == "__main__":
    main()
