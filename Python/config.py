"""
config.py

EVERY user-editable parameter for the spectral design pipeline lives here,
and only here. timbral_target.py, spectral_optimizer.py,
sensitivity_analysis.py, and priority_optimizer.py all import their
settings from this file instead of defining their own -- so testing a
different scenario (different audio, different targets, different search
budget) means editing exactly one file, not hunting across four.

Run whichever script you actually want (priority_optimizer.py is the main
one -- see its own docstring) after editing values here.
"""

from pathlib import Path


# ============================================================
# Paths
# ============================================================

# Repo root, worked out from this file's own location on disk (Python/ is
# one level below the repo root) rather than from the current working
# directory. This means paths built from it stay correct however a script
# is launched (double-click, terminal in a different folder, a different
# machine after git clone), and on Windows/Mac/Linux alike -- no manual
# backslash-escaping needed.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Tonal-only for now (see "Tonal vs. noise" below). Point this at your
# real tonal split before running anything.
TONAL_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_tonal.wav"

# Not used yet -- kept here so the path is already in place for when noise
# handling (and tonal/noise weighting) is built.
NOISE_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_noise.wav"

# Where the optimizers write the edited result so you can listen to it.
OUTPUT_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_tonal_edited.wav"


# ============================================================
# Targets & priorities
# ============================================================

# One (min, max) range per parameter, on each model's native 0-100 scale.
# A tight range (e.g. (60, 60)) behaves like a fixed-point target.
# A full-width range (0, 100) means "don't care" -- it can never contribute
# error, regardless of priority, and priority_optimizer.py treats it as
# inactive when ranking which bands matter.
TARGETS = {
    "warmth":     (55, 65),
    "brightness": (0, 100),
    "depth":      (0, 100),
    "hardness":   (25, 35),
    "roughness":  (0, 100),
    "sharpness":  (65, 75),
    "booming":    (0, 100),
}

# Relative importance of hitting each target. Larger = more important.
# These are WEIGHTS on a weighted-sum error, not a hard ordering: a very
# large miss on a low-priority parameter can still outweigh a small miss
# on a high-priority one. (A strict "never sacrifice hardness for
# sharpness" guarantee would need a different, lexicographic scheme --
# not implemented.)
PRIORITIES = {
    "warmth":     1.0,
    "brightness": 1.0,
    "depth":      1.0,
    "hardness":   3.0,
    "roughness":  1.0,
    "sharpness":  1.0,
    "booming":    1.0,
}


# ============================================================
# Spectral band structure (spectral_optimizer.py, sensitivity_analysis.py,
# priority_optimizer.py)
# ============================================================

# How many frequency bands to split the spectrum into, log-spaced from
# FMIN_HZ to Nyquist. More bands = finer control, but a bigger search space
# for Nelder-Mead to explore -- and each evaluation of the objective
# (spectral edit + all 7 timbral models) costs several seconds, so the
# search-space size directly drives wall-clock runtime.
N_BANDS = 6
FMIN_HZ = 20.0

# Per-band gain is a multiplier on magnitude (1.0 = unchanged). These bounds
# stop the optimizer from finding a "solution" that silences a band
# entirely or blows it out to absurd levels.
GAIN_MIN = 0.1
GAIN_MAX = 3.0

# STFT window settings. 2048 @ typical 44.1/48kHz sample rates gives
# ~20-40ms frames -- fine time resolution isn't the point here since edits
# are applied uniformly across a whole sustained tone, not shaped over time
# (see "Time segmentation" in the project notes -- deferred).
NPERSEG = 2048
NOVERLAP = 1536


# ============================================================
# Optimizer search budget (spectral_optimizer.py, priority_optimizer.py)
# ============================================================

# Each objective evaluation is expensive (several seconds, dominated by
# the timbral models themselves, not the STFT edit) -- both optimizer
# scripts time one evaluation up front and print an estimated worst-case
# runtime before starting, so you can judge whether to raise or lower this
# rather than finding out by waiting. Nelder-Mead does stop early if
# fatol/xatol are satisfied, so actual runtime is often well under the
# worst case.
MAX_ITER = 60
FATOL = 1e-5   # stop if total_error changes by less than this between steps
XATOL = 1e-3   # stop if gain values change by less than this between steps


# ============================================================
# Priority-aware search (priority_optimizer.py)
# ============================================================

# How many of the N_BANDS bands (ranked by measured relevance to your
# ACTIVE targets) the optimizer is actually allowed to touch. The rest
# stay fixed at gain 1.0. Smaller = faster but less flexible.
TOP_K_BANDS = 3


# ============================================================
# Sensitivity analysis (sensitivity_analysis.py, priority_optimizer.py)
# ============================================================

# How far up/down from baseline (1.0 = unchanged) to nudge each band's
# gain when probing it. Bigger = a stronger, easier-to-see signal, but
# also a less "local" measurement (less like a true derivative, more like
# a coarse average over a wide swing). 0.3 means testing gains of 0.7 and
# 1.3 for each band in turn.
PERTURBATION = 0.3
