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

No GUI. No Max/OSC. Everything you'd want to change to test a scenario is
in the CONFIG block directly below (plus TARGETS/PRIORITIES, which live in
timbral_target.py) -- edit and re-run.
"""

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from scipy.optimize import minimize

from timbral_target import (
    REPO_ROOT,
    TONAL_AUDIO_PATH,
    TARGETS,
    PRIORITIES,
    build_targets,
    analyse,
    total_error,
)


# ============================================================
# CONFIG — edit everything below this line to test a scenario
# ============================================================

# Where to write the edited result so you can actually listen to it.
OUTPUT_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_tonal_edited.wav"

# How many frequency bands to split the spectrum into, log-spaced from
# FMIN_HZ to Nyquist. More bands = finer control, but a bigger search space
# for Nelder-Mead to explore -- and each evaluation of the objective
# (spectral edit + all 7 timbral models) costs several seconds, so the
# search-space size directly drives wall-clock runtime. Nelder-Mead needs
# N_BANDS+1 evaluations just to build its starting simplex, before any
# real searching happens. Start small and raise it once the loop itself is
# proven out.
N_BANDS = 6
FMIN_HZ = 20.0

# Per-band gain is a multiplier on magnitude (1.0 = unchanged). These bounds
# stop the optimizer from finding a "solution" that silences a band
# entirely or blows it out to absurd levels.
GAIN_MIN = 0.1
GAIN_MAX = 3.0

# Search budget. Each objective evaluation is expensive (~4-6s on a typical
# machine, dominated by the timbral models themselves, not the STFT edit) --
# main() times one evaluation up front and prints an estimated worst-case
# runtime before starting, so you can judge whether to raise or lower this
# rather than finding out by waiting. Nelder-Mead does stop early if
# fatol/xatol are satisfied, so actual runtime is often well under the
# worst case.
MAX_ITER = 60
FATOL = 1e-5   # stop if total_error changes by less than this between steps
XATOL = 1e-3   # stop if gain values change by less than this between steps

# STFT window settings. 2048 @ typical 44.1/48kHz sample rates gives
# ~20-40ms frames -- fine time resolution isn't the point here since we're
# editing a whole sustained tone uniformly across its duration, not
# shaping it over time.
NPERSEG = 2048
NOVERLAP = 1536


# ============================================================
# Core functions — shouldn't need to touch below here to test scenarios
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

    Memoized: Nelder-Mead (and our own progress callback) sometimes ask for
    the score at a point it's already evaluated. Each evaluation costs
    several seconds, so a plain dict cache keyed on the rounded gains
    avoids paying for that twice."""
    cache = {}

    def objective(gains: np.ndarray) -> float:
        key = tuple(np.round(gains, 6))
        if key in cache:
            return cache[key]
        edited = apply_band_gains(tonal_audio, fs, gains)
        achieved = analyse(edited, fs)
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
    targets = build_targets(TARGETS, PRIORITIES)

    print("--- before optimisation ---")
    starting_achieved = analyse(tonal_audio, fs)
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