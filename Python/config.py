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

# ------------------------------------------------------------
# STEP 1: pick a mode. This is a single switch -- it changes how EVERY
# number in TARGETS below is read.
#
#   "absolute" -- each (min, max) is a literal band on the model's native
#                 0-100 scale. (0, 100) means "don't care".
#   "relative" -- each (min, max) is a DELTA range relative to the SOURCE
#                 AUDIO's own measured value for that parameter (measured
#                 by analysing TONAL_AUDIO_PATH before any editing).
#                 e.g. warmth = (10, 20) means "10 to 20 points warmer
#                 than the source sound, whatever that starts at";
#                 hardness = (-40, -30) means "30 to 40 points lower than
#                 the source". Resolved bands are clipped to 0-100, so an
#                 intentionally wide delta like (-100, 100) still means
#                 "don't care" regardless of the source's value.
# Order within a tuple doesn't matter in either mode -- (-40, -30) and
# (-30, -40) resolve identically.
# ------------------------------------------------------------
TARGET_MODE = "relative"   # <-- change to "relative" to use the deltas below


# STEP 2: fill in TARGETS to match whichever mode you picked above. Only
# ONE of the two blocks below is active at a time -- whichever one is
# actually named TARGETS. The other is commented out as a ready-to-use
# example: to switch, comment out the active block, uncomment the other,
# and set TARGET_MODE to match.

# --- ABSOLUTE example (active by default) ---
TARGETS = {
    "warmth":     (0, 100),
    "brightness": (0, 100),
    "depth":      (0, 100),
    "hardness":   (0, 100),
    "roughness":  (20, 50),
    "sharpness":  (0, 100),
    "booming":    (0, 100),
}

# --- RELATIVE example (commented out) -- uncomment this AND comment out
# the TARGETS block above AND set TARGET_MODE = "relative" to use it:
# TARGETS = {
#     "warmth":     (10, 20),     # 10-20 points warmer than the source
#     "brightness": (-100, 100),  # don't care
#     "depth":      (-100, 100),  # don't care
#     "hardness":   (-40, -30),   # 30-40 points lower than the source
#     "roughness":  (-100, 100),  # don't care
#     "sharpness":  (-100, 100),  # don't care
#     "booming":    (-100, 100),  # don't care
# }

# A tight range (e.g. (60, 60) in absolute mode, or (0, 0) in relative
# mode) behaves like a fixed-point target.

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
    "hardness":   1.0,
    "roughness":  3.0,
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
N_BANDS = 12 #was 6
FMIN_HZ = 20.0

# Per-band gain, in dB (0 dB = unchanged -- this replaces the old linear
# multiplier; the previous GAIN_MIN=0.1/GAIN_MAX=3.0 is roughly -20/+9.5 dB).
#
# Two ranges, not one:
#   IDEAL_GAIN_*_DB   -- normal operating range. The optimizer stays inside
#                        this unless it genuinely can't reach your targets
#                        without going further.
#   LIMITER_GAIN_*_DB -- a wider fallback range. priority_optimizer.py only
#                        reaches into this for a band that (a) ended its
#                        ideal-range search pinned against an ideal edge,
#                        AND (b) the target is still missed -- i.e. only
#                        when overriding the ideal range is the only
#                        viable way to get closer, and only for the
#                        specific band(s) that actually need it, not
#                        across the board.
# Keep LIMITER at least as wide as IDEAL on both ends (checked below).
IDEAL_GAIN_MIN_DB = -20.0
IDEAL_GAIN_MAX_DB = 9.5

LIMITER_GAIN_MIN_DB = -40.0
LIMITER_GAIN_MAX_DB = 18.0

assert LIMITER_GAIN_MIN_DB <= IDEAL_GAIN_MIN_DB, "LIMITER_GAIN_MIN_DB must be <= IDEAL_GAIN_MIN_DB"
assert LIMITER_GAIN_MAX_DB >= IDEAL_GAIN_MAX_DB, "LIMITER_GAIN_MAX_DB must be >= IDEAL_GAIN_MAX_DB"

# How close (in dB) a band's final gain has to land to an ideal-range edge
# to count as "pinned" there and become eligible for the limiter-range
# retry (priority_optimizer.py only).
PINNED_EPSILON_DB = 0.1

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
MAX_ITER = 100 # was 60
FATOL = 1e-5   # stop if total_error changes by less than this between steps
XATOL = 1e-3   # stop if gain values change by less than this between steps


# ============================================================
# Priority-aware search (priority_optimizer.py)
# ============================================================

# Which of the N_BANDS bands (ranked by measured relevance to your ACTIVE
# targets) the optimizer is actually allowed to touch. The rest stay
# fixed at 0 dB.
#
# Selection is ratio-based, not a fixed count: a band is kept if its
# relevance is at least RELEVANCE_RATIO_THRESHOLD * the single most
# relevant band's relevance. This adapts to how concentrated relevance
# actually is for this sound/target combo -- e.g. one run had band 5 at
# 27.7 vs band 8 at 3.4 (an 8x gap) and band 11 at 0.049 (a 560x gap); a
# fixed top_k searches all of those equally, a ratio cutoff won't.
# MIN_ACTIVE_BANDS / MAX_ACTIVE_BANDS are floor/ceiling guardrails so a
# pathological run can't collapse to nothing or blow back up to everything.
RELEVANCE_RATIO_THRESHOLD = 0.05   # drop anything more than 20x below the top band
MIN_ACTIVE_BANDS = 2
MAX_ACTIVE_BANDS = 6               # was: TOP_K_BANDS

# How many of the selected active bands (the most relevant among THEM)
# get a fast, low-dimensional pre-solve before the full joint search --
# its result seeds the full search's starting point instead of every
# band starting at 0 dB. Nelder-Mead's evaluation count is sensitive
# to how close the initial simplex is to the optimum, so this usually
# cuts iterations needed in the full search. 0 disables warm-starting.
WARMSTART_BANDS = 2
WARMSTART_MAX_ITER = 30


# ============================================================
# Sensitivity analysis (sensitivity_analysis.py, priority_optimizer.py)
# ============================================================

# How far up/down from baseline (0 dB = unchanged) to nudge each band's
# gain when probing it, in dB. Bigger = a stronger, easier-to-see signal,
# but also a less "local" measurement (less like a true derivative, more
# like a coarse average over a wide swing). 3.0 means testing -3dB and
# +3dB for each band in turn (roughly the old PERTURBATION=0.3's swing,
# just expressed in dB now). Clamped to the ideal range, same as before.
PERTURBATION_DB = 3.0