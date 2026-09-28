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

# Point this at your real tonal split before running anything.
TONAL_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_tonal.wav"

# Optional. When this file exists, priority_optimizer.py edits it
# alongside the tonal component -- its own independent per-band gains,
# summed back with the edited tonal signal before anything is measured
# (see spectral_optimizer.mix_channels()). When it doesn't exist (or its
# sample rate doesn't match TONAL_AUDIO_PATH's), the tool falls back to
# tonal-only automatically -- no config flag needed, nothing breaks for
# a source that was never decomposed. This is what lets the SAME search
# machinery cover a resonant musical note (tonal-dominant) and a field
# recording or noisy sound design source (noise-dominant, maybe even
# tonal-free) without treating either as the default case.
NOISE_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_noise.wav"

# Where the optimizers write the edited result so you can listen to it.
OUTPUT_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_tonal_edited.wav"


# ============================================================
# Run mode (priority_optimizer.py)
# ============================================================

# "scan"  -- analysis only. Loads the audio (tonal + noise, summed if a
#            decomposed noise stem is present -- same as everywhere else
#            in this tool) and prints the 7 timbral models' raw values to
#            the console. No targets, no sensitivity analysis, no search,
#            no output file written -- just "where does this sound sit
#            right now?". Useful before you've decided what TARGETS to
#            set below, or just to check a sound.
# "edit"  -- the full pipeline, as it's always worked: sensitivity
#            analysis, priority-aware band selection, and the Nelder-Mead
#            search against TARGETS/PRIORITIES below, writing the result
#            to OUTPUT_AUDIO_PATH. This is the original, default behaviour.
MODE = "edit"   # "scan" or "edit"
assert MODE in ("scan", "edit"), f"MODE must be 'scan' or 'edit', got {MODE!r}"


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
#                 by analysing TONAL_AUDIO_PATH -- plus NOISE_AUDIO_PATH,
#                 summed in, when it exists -- before any editing).
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
    "warmth":     0.0,
    "brightness": 0.0,
    "depth":      0.0,
    "hardness":   0.0,
    "roughness":  3.0,
    "sharpness":  0.0,
    "booming":    0.0,
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
IDEAL_GAIN_MIN_DB = -18.0
IDEAL_GAIN_MAX_DB = 10.0

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

# How far apart (in dB) Nelder-Mead's starting simplex vertices are. This
# MATTERS now that gains are dB and "unchanged" is 0: scipy's default
# simplex step is ~5% of x0, which silently collapses to a near-zero
# absolute step (0.00025) whenever x0 is exactly 0 in a dimension -- and
# since every search here starts at 0 dB (or warm-starts from a previous
# 0-dB-rooted search), that default was producing a simplex too flat to
# register any real movement in the timbral models, so the search
# "converged" after 1 iteration without actually searching. Passing an
# explicit initial_simplex (see spectral_optimizer.build_initial_simplex)
# sidesteps the problem outright regardless of how close to 0 dB the
# starting point is.
NELDER_MEAD_STEP_DB = 2.0


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
#
# When a decomposed noise stem is in play (see NOISE_AUDIO_PATH above),
# this ratio/floor/ceiling selection runs over tonal+noise candidates
# combined, and MAX_ACTIVE_BANDS bounds how many frequency REGIONS get
# picked this way -- not the final candidate count. If a region's tonal
# (or noise) candidate is picked and its sibling in the SAME region is
# also measurably relevant (nonzero, even if far smaller), that sibling
# is pulled in too, since the two get summed before anything is measured
# -- shaping only one channel in a region the other channel also affects
# would leave real headroom on the table (see select_bands() in
# priority_optimizer.py). So the true number of active candidates can run
# a bit past MAX_ACTIVE_BANDS once noise is active.
RELEVANCE_RATIO_THRESHOLD = 0.1   # drop anything more than 10x below the top band
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
PERTURBATION_DB = 1.0


# ============================================================
# Attack/decay-aware editing (attack_shaping.py, priority_optimizer.py)
# ============================================================

# hardness (timbral_hardness) is the one model, of the 7, whose formula
# gives real weight to a short window right after the attack transient
# specifically (its "attack centroid" -- verified against the real
# timbral_models source, not assumed). Editing this tool has always done
# is a single per-band gain applied uniformly across the WHOLE signal --
# fine for every other parameter, since they're whole-signal or
# per-note-segment averages, but wasteful for hardness: the same gain
# change that shapes the attack window also reshapes the sustain/decay
# equally, even when only the attack needed to move.
#
# So when (and only when) hardness is an active target (see is_active()
# in priority_optimizer.py), a second search runs AFTER the normal
# whole-signal search: it finds a separate set of band gains meant only
# for the attack, then crosses that attack-edited signal into the
# whole-signal edit over a short window, so the transition is inaudible.
# Zero extra cost when hardness isn't active -- this pass is skipped
# entirely.

# Envelope resolution for attack/decay detection: ATTACK_ENVELOPE_HOP_MS
# is the STEP between estimates (small, for precise placement),
# ATTACK_ENVELOPE_WINDOW_MS is how much audio each RMS estimate actually
# averages over (longer than the hop -- overlapping windows). The window
# needs to be longer than the hop specifically so a low-pitched, sustained
# fundamental (a bassoon's A3 is 220Hz -- about 4.5ms per cycle) doesn't
# alias into a false "decay" ripple: a plain non-overlapping 5ms block
# covers barely one cycle of a tone that low, and where each block happens
# to start relative to the waveform's phase shifts its RMS by a fraction
# of a dB block to block -- enough to look like real envelope movement.
# Averaging over ~20ms instead covers several cycles even for a low
# fundamental, so that phase-dependent wobble washes out.
ATTACK_ENVELOPE_HOP_MS = 5.0
ATTACK_ENVELOPE_WINDOW_MS = 20.0

# The attack is considered to "start" once the envelope first rises above
# this fraction of its own peak value (measured from a quiet lead-in) --
# a simple, robust onset threshold rather than a fixed absolute level,
# so it adapts to how loud the source recording happens to be.
ATTACK_ONSET_THRESHOLD_FRAC = 0.05

# Safety bound on how far past the detected onset to search for the
# attack's peak (loudest point), in ms. Prevents a noisy or very slowly
# swelling source from making the "attack" search run away.
ATTACK_PEAK_SEARCH_MAX_MS = 500.0

# Fallback crossfade anchor, used when no usable decay stage is found:
# attack_start + this many ms. Matches timbral_hardness's own 125ms
# attack-centroid integration window (measured from attack start, NOT
# from the peak), so the fallback placement still overlaps the actual
# window hardness's formula reads from.
ATTACK_CENTROID_WINDOW_MS = 125.0

# Fallback / minimum crossfade duration, in ms. Used whenever a decay
# stage either isn't detected at all, or is detected but shorter than
# this -- a crossfade forced into an unusably short window would sound
# abrupt, so this floor takes over instead of shrinking to fit it.
DEFAULT_XFADE_DURATION_MS = 80.0

# Decay-stage detection: starting from the attack's peak, walk the
# envelope hop by hop and track the change between consecutive hops.
# The decay stage is however long that keeps changing; it's considered
# "stabilised" (i.e. decay has ended, sustain has begun) once consecutive
# hops stay within this many dB of each other...
DECAY_STABILIZATION_TOLERANCE_DB = 0.5

# ...for at least this many consecutive ms -- long enough that a brief,
# incidental flat spot partway through a real decay doesn't get mistaken
# for having reached the sustain plateau.
DECAY_STABILIZATION_HOLD_MS = 50.0

# How far past the peak to search for stabilisation before giving up and
# falling back to ATTACK_CENTROID_WINDOW_MS / DEFAULT_XFADE_DURATION_MS
# above -- covers a source with no real decay stage at all (a hard onset
# straight into a flat sustain, or a one-shot that just keeps ringing).
DECAY_SEARCH_MAX_MS = 500.0