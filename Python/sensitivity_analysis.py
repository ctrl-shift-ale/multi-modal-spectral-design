"""
sensitivity_analysis.py

Answers: "which frequency bands actually move which timbral parameters,
for THIS sound?" -- rather than guessing (e.g. assuming "sharpness = high
frequencies") from general intuition.

Method: for each band, nudge its gain up and down a little from baseline
(1.0, i.e. unchanged) and measure how much each of the 7 timbral
parameters moves in response. That gives a sensitivity value per
(band, parameter) pair -- a measured slope, not an assumption. Negative
means "raising this band's gain pushes the parameter down."

This is a diagnostic tool, not an editor -- it doesn't change the audio
or try to hit any targets. Its output (the sensitivity matrix) is what
lets a future optimizer restrict its search to the bands that actually
matter for whichever parameters the user has prioritised, instead of
searching blindly across all bands -- which is also the mechanism behind
the "spectral strategies" idea (e.g. "prioritise the most psychoacoustically
significant harmonics" becomes concrete: the bands with the highest
measured sensitivity for the parameters in play).

No GUI. No Max/OSC. All user-editable settings live in config.py -- edit
that file, then re-run this one.
"""

import time

import numpy as np

from timbral_target import TONAL_AUDIO_PATH, load_tonal, analyse, PARAM_NAMES
from spectral_optimizer import N_BANDS, band_edges, apply_band_gains, GAIN_MIN, GAIN_MAX
from config import PERTURBATION


# ============================================================
# Core functions — shouldn't need to touch below here to test scenarios.
# All the values that WOULD normally need editing live in config.py.
# ============================================================

def measure_sensitivity(tonal_audio: np.ndarray, fs: int, priorities: dict = None) -> np.ndarray:
    """Returns an (N_BANDS x 7) matrix. matrix[i, j] = how much parameter j
    changes per unit change in band i's gain (a measured slope), using a
    central-difference probe (up vs down) around baseline.

    priorities: optional, passed straight through to analyse() -- skips
    computing (and therefore measuring sensitivity for) any priority<=0
    parameter, leaving its column at 0. That's safe: priority_optimizer.py's
    compute_band_relevance() never reads a column for a priority<=0
    parameter anyway (see is_active()), so a column of zeros there costs
    nothing and is never mistaken for "not sensitive"."""
    gain_up = 1.0 + PERTURBATION
    gain_down = max(GAIN_MIN, 1.0 - PERTURBATION)
    gain_span = gain_up - gain_down  # denominator for the slope

    matrix = np.zeros((N_BANDS, len(PARAM_NAMES)))

    for band_i in range(N_BANDS):
        gains_up = np.ones(N_BANDS)
        gains_up[band_i] = min(GAIN_MAX, gain_up)
        gains_down = np.ones(N_BANDS)
        gains_down[band_i] = gain_down

        edited_up = apply_band_gains(tonal_audio, fs, gains_up)
        edited_down = apply_band_gains(tonal_audio, fs, gains_down)

        achieved_up = analyse(edited_up, fs, priorities=priorities)
        achieved_down = analyse(edited_down, fs, priorities=priorities)

        for param_j, name in enumerate(PARAM_NAMES):
            if achieved_up[name] is None or achieved_down[name] is None:
                continue  # priority 0 -- skipped, left at 0.0
            matrix[band_i, param_j] = (achieved_up[name] - achieved_down[name]) / gain_span

    return matrix


def report(matrix: np.ndarray, edges: np.ndarray):
    """Print the full matrix, then a 'most sensitive band' summary per
    parameter -- the part that's actually useful for steering an
    optimizer's search."""
    col_w = 11
    header = f"{'band (Hz)':<18}" + "".join(f"{p:<{col_w}}" for p in PARAM_NAMES)
    print(header)
    print("-" * len(header))

    for band_i in range(N_BANDS):
        band_label = f"{edges[band_i]:.0f}-{edges[band_i + 1]:.0f}"
        row = f"{band_label:<18}"
        for param_j in range(len(PARAM_NAMES)):
            row += f"{matrix[band_i, param_j]:<{col_w}.2f}"
        print(row)

    print("-" * len(header))
    print("\nMost sensitive band per parameter (by |slope|):")
    for param_j, name in enumerate(PARAM_NAMES):
        col = matrix[:, param_j]
        best_band = int(np.argmax(np.abs(col)))
        band_label = f"{edges[best_band]:.0f}-{edges[best_band + 1]:.0f} Hz"
        direction = "raises" if col[best_band] > 0 else "lowers"
        print(f"  {name:<12} -> band {best_band} ({band_label}), raising its gain {direction} {name}")


def main():
    if not TONAL_AUDIO_PATH.exists():
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    tonal_audio, fs = load_tonal(TONAL_AUDIO_PATH)
    edges = band_edges(fs, N_BANDS)

    # Time one evaluation (reuse analyse() directly, same cost as the
    # optimizer's objective) so the estimate below is measured, not guessed.
    t0 = time.time()
    analyse(tonal_audio, fs)
    seconds_per_eval = time.time() - t0
    total_evals = 2 * N_BANDS  # up + down probe per band
    print(
        f"~{seconds_per_eval:.1f}s per evaluation -> ~{total_evals * seconds_per_eval / 60:.1f} min "
        f"for {N_BANDS} bands (2 probes each). Ctrl+C to abort.\n"
    )

    matrix = measure_sensitivity(tonal_audio, fs)
    report(matrix, edges)


if __name__ == "__main__":
    main()
