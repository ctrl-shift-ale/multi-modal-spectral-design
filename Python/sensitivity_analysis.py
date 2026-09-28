"""
sensitivity_analysis.py

Answers: "which frequency bands actually move which timbral parameters,
for THIS sound?" -- rather than guessing (e.g. assuming "sharpness = high
frequencies") from general intuition.

Method: for each band, nudge its gain up and down a little (in dB) from
baseline (0 dB, i.e. unchanged) and measure how much each of the 7
timbral parameters moves in response. That gives a sensitivity value per
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

from timbral_target import TONAL_AUDIO_PATH, load_tonal, load_noise, analyse, PARAM_NAMES
from spectral_optimizer import N_BANDS, band_edges, mix_channels, db_to_linear
from config import PERTURBATION_DB, IDEAL_GAIN_MIN_DB, IDEAL_GAIN_MAX_DB, NOISE_AUDIO_PATH


# ============================================================
# Core functions — shouldn't need to touch below here to test scenarios.
# All the values that WOULD normally need editing live in config.py.
# ============================================================

def measure_sensitivity(tonal_audio: np.ndarray, fs: int, noise_audio: np.ndarray = None,
                         priorities: dict = None) -> np.ndarray:
    """Returns an (N_BANDS x 7) matrix -- or (2*N_BANDS x 7) when
    noise_audio is given. matrix[i, j] = how much parameter j changes per
    unit change in candidate i's gain (a measured slope), using a
    central-difference probe (up vs down) around baseline. Rows
    [0, N_BANDS) are tonal bands; rows [N_BANDS, 2*N_BANDS) -- only
    present when noise_audio is given -- are noise bands, probed the same
    way with the tonal component held at baseline. Negative means
    "raising this candidate's gain pushes the parameter down."

    Probing a tonal band mixes it with noise held at 0 dB (unchanged,
    not silenced), and vice versa for a noise band -- each candidate's
    slope is measured in the actual mixed context the search will edit
    it in, not in isolation.

    priorities: optional, passed straight through to analyse() -- skips
    computing (and therefore measuring sensitivity for) any priority<=0
    parameter, leaving its column at 0. That's safe: priority_optimizer.py's
    compute_band_relevance() never reads a column for a priority<=0
    parameter anyway (see is_active()), so a column of zeros there costs
    nothing and is never mistaken for "not sensitive"."""
    gain_up_db = min(IDEAL_GAIN_MAX_DB, PERTURBATION_DB)
    gain_down_db = max(IDEAL_GAIN_MIN_DB, -PERTURBATION_DB)
    gain_span_db = gain_up_db - gain_down_db  # denominator for the slope, in dB now

    n_candidates = N_BANDS * (2 if noise_audio is not None else 1)
    matrix = np.zeros((n_candidates, len(PARAM_NAMES)))

    def probe(tonal_gains_db, noise_gains_db):
        mixed = mix_channels(tonal_audio, fs, tonal_gains_db, noise_audio, noise_gains_db)
        return analyse(mixed, fs, priorities=priorities)

    zero_tonal = np.zeros(N_BANDS)
    zero_noise = np.zeros(N_BANDS) if noise_audio is not None else None

    for band_i in range(N_BANDS):
        gains_up_db = zero_tonal.copy()
        gains_up_db[band_i] = gain_up_db
        gains_down_db = zero_tonal.copy()
        gains_down_db[band_i] = gain_down_db

        achieved_up = probe(gains_up_db, zero_noise)
        achieved_down = probe(gains_down_db, zero_noise)

        for param_j, name in enumerate(PARAM_NAMES):
            if achieved_up[name] is None or achieved_down[name] is None:
                continue  # priority 0 -- skipped, left at 0.0
            matrix[band_i, param_j] = (achieved_up[name] - achieved_down[name]) / gain_span_db

    if noise_audio is not None:
        for band_i in range(N_BANDS):
            gains_up_db = zero_noise.copy()
            gains_up_db[band_i] = gain_up_db
            gains_down_db = zero_noise.copy()
            gains_down_db[band_i] = gain_down_db

            achieved_up = probe(zero_tonal, gains_up_db)
            achieved_down = probe(zero_tonal, gains_down_db)

            for param_j, name in enumerate(PARAM_NAMES):
                if achieved_up[name] is None or achieved_down[name] is None:
                    continue
                matrix[N_BANDS + band_i, param_j] = (achieved_up[name] - achieved_down[name]) / gain_span_db

    return matrix


def _candidate_row_label(idx: int, edges: np.ndarray, noise_active: bool) -> str:
    """'tonal 115-207' or 'noise 664-1191' for a flat candidate row index.
    See priority_optimizer.candidate_channel_band() for the canonical
    version of this split -- duplicated here in miniature so this file
    stays runnable standalone without importing priority_optimizer.py."""
    if noise_active and idx >= N_BANDS:
        band, channel = idx - N_BANDS, "noise"
    else:
        band, channel = idx, "tonal"
    return f"{channel} {edges[band]:.0f}-{edges[band + 1]:.0f}"


def report(matrix: np.ndarray, edges: np.ndarray, noise_active: bool = False):
    """Print the full matrix, then a 'most sensitive candidate' summary
    per parameter -- the part that's actually useful for steering an
    optimizer's search. noise_active=True labels rows past N_BANDS as
    noise candidates instead of tonal (see measure_sensitivity())."""
    col_w = 11
    header = f"{'candidate (Hz)':<24}" + "".join(f"{p:<{col_w}}" for p in PARAM_NAMES)
    print(header)
    print("-" * len(header))

    for row_i in range(matrix.shape[0]):
        row_label = _candidate_row_label(row_i, edges, noise_active)
        row = f"{row_label:<24}"
        for param_j in range(len(PARAM_NAMES)):
            row += f"{matrix[row_i, param_j]:<{col_w}.2f}"
        print(row)

    print("-" * len(header))
    print("\nMost sensitive candidate per parameter (by |slope|):")
    for param_j, name in enumerate(PARAM_NAMES):
        col = matrix[:, param_j]
        best_row = int(np.argmax(np.abs(col)))
        row_label = _candidate_row_label(best_row, edges, noise_active)
        direction = "raises" if col[best_row] > 0 else "lowers"
        print(f"  {name:<12} -> {row_label} Hz, raising its gain {direction} {name}")


def main():
    if not TONAL_AUDIO_PATH.exists():
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    tonal_audio, fs = load_tonal(TONAL_AUDIO_PATH)
    noise_audio, noise_fs = load_noise(NOISE_AUDIO_PATH)
    if noise_audio is not None and noise_fs != fs:
        print(f"note: noise sample rate ({noise_fs} Hz) != tonal ({fs} Hz) -- ignoring noise component\n")
        noise_audio = None
    edges = band_edges(fs, N_BANDS)

    # Time one evaluation (reuse analyse() directly, same cost as the
    # optimizer's objective) so the estimate below is measured, not guessed.
    t0 = time.time()
    analyse(tonal_audio, fs)
    seconds_per_eval = time.time() - t0
    total_evals = 2 * N_BANDS * (2 if noise_audio is not None else 1)  # up + down probe per candidate
    print(
        f"~{seconds_per_eval:.1f}s per evaluation -> ~{total_evals * seconds_per_eval / 60:.1f} min "
        f"for {N_BANDS} bands" + (" x 2 channels (tonal+noise)" if noise_audio is not None else "") +
        " (2 probes each). Ctrl+C to abort.\n"
    )

    matrix = measure_sensitivity(tonal_audio, fs, noise_audio=noise_audio)
    report(matrix, edges, noise_active=(noise_audio is not None))


if __name__ == "__main__":
    main()
