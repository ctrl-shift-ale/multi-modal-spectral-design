"""
timbral_target.py

Step 1 of the spectral design engine: the TARGET / OBJECTIVE FUNCTION.

This script does NOT edit or synthesise anything yet. It answers one question:
"given a candidate sound, and a set of target ranges + priorities for the
7 timbral parameters, how far off is it, and where?"

That "how far off" number is what a future optimizer (Nelder-Mead / CMA-ES)
will try to minimise by changing the spectrum. Building and testing the
objective function on its own first means we can trust the score before we
ever start editing audio with it.

TONAL ONLY, for now: the noise component is deliberately ignored at this
stage of development, not deleted from the design. Planned future feature:
let the user choose whether spectral edits prioritise the tonal or noise
domain, or set a weighting/range between the two. Until that exists, mixing
tonal + noise back together here would be pointless (see NOISE_AUDIO_PATH
below) -- if the tonal signal is going to be the only thing edited and
analysed, feeding it noise-diluted would just distort the target-matching
without buying anything.

No GUI. No Max/OSC. Everything you'd want to change to test a scenario is
in the CONFIG block directly below — edit and re-run.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from timbral_models import (
    timbral_warmth,
    timbral_brightness,
    timbral_depth,
    timbral_hardness,
    timbral_roughness,
    timbral_sharpness,
    timbral_booming,
)


# ============================================================
# CONFIG — edit everything below this line to test a scenario
# ============================================================

# Repo root, worked out from this file's own location on disk (Python/ is
# one level below the repo root) rather than from the current working
# directory. This means paths built from it stay correct however the
# script is launched (double-click, terminal in a different folder, a
# different machine after git clone), and on Windows/Mac/Linux alike --
# no manual backslash-escaping needed.
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Input audio -------------------------------------------------
# Tonal-only for now. Point this at your real tonal split before running.
TONAL_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_tonal.wav"

# Not used yet -- kept here so the path is already in place for when noise
# handling (and tonal/noise weighting) is built. See module docstring.
NOISE_AUDIO_PATH = REPO_ROOT / "samples" / "Deconstructed" / "Bassoon_A3_MF" / "Bassoon_A3_MF_noise.wav"

# --- Targets -------------------------------------------------------
# One (min, max) range per parameter, on the model's native 0-100 scale.
# A tight range (e.g. (60, 60)) behaves like a fixed-point target.
# A full-width range (0, 100) means "don't care" -- it can never contribute
# error, regardless of priority.
TARGETS = {
    "warmth":     (55, 65),
    "brightness": (0, 100),
    "depth":      (0, 100),
    "hardness":   (25, 35),
    "roughness":  (0, 100),
    "sharpness":  (65, 75),
    "booming":    (0, 100),
}

# --- Priorities ------------------------------------------------------
# Relative importance of hitting each target. Larger = more important.
# These are WEIGHTS on a weighted-sum error, not a hard ordering: a very
# large miss on a low-priority parameter can still outweigh a small miss
# on a high-priority one. (If we later want a strict "never sacrifice
# hardness for sharpness" guarantee, that needs a different, lexicographic
# scheme -- worth a separate conversation once this loop is working.)
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
# Core objects — shouldn't need to touch below here to test scenarios
# ============================================================

TIMBRAL_MODELS = {
    "warmth":     timbral_warmth,
    "brightness": timbral_brightness,
    "depth":      timbral_depth,
    "hardness":   timbral_hardness,
    "roughness":  timbral_roughness,
    "sharpness":  timbral_sharpness,
    "booming":    timbral_booming,
}

# All 7 models clip their output to this native range (see output_clip in
# timbral_util.py) -- used to normalise errors onto a common 0-1 scale so
# priorities behave consistently across parameters.
MODEL_OUTPUT_RANGE = 100.0


@dataclass
class TimbralTarget:
    """A target band + priority for a single timbral parameter."""

    name: str
    target_min: float
    target_max: float
    priority: float = 1.0

    def __post_init__(self):
        if self.target_min > self.target_max:
            raise ValueError(
                f"{self.name}: target_min ({self.target_min}) is greater than "
                f"target_max ({self.target_max})"
            )

    @property
    def is_hit(self):
        return True  # placeholder, set properly in .evaluate()

    def raw_error(self, achieved: float) -> float:
        """Deadzone error: 0 inside the band, distance outside it."""
        if achieved > self.target_max:
            return achieved - self.target_max
        if achieved < self.target_min:
            return self.target_min - achieved
        return 0.0

    def weighted_error(self, achieved: float) -> float:
        """Normalised, squared, priority-weighted error -- what the future
        optimizer will actually minimise the sum of."""
        normalised = self.raw_error(achieved) / MODEL_OUTPUT_RANGE
        return self.priority * (normalised ** 2)


def build_targets(targets=TARGETS, priorities=PRIORITIES) -> dict:
    """Turn the CONFIG dicts into TimbralTarget objects, keyed by name."""
    result = {}
    for name, (lo, hi) in targets.items():
        result[name] = TimbralTarget(
            name=name,
            target_min=lo,
            target_max=hi,
            priority=priorities.get(name, 1.0),
        )
    return result


def load_tonal(tonal_path) -> tuple:
    """Load the tonal-only audio file. Noise handling deliberately excluded
    for now -- see module docstring."""
    tonal, sr = sf.read(tonal_path, always_2d=False)
    if tonal.ndim > 1:
        tonal = np.mean(tonal, axis=1)
    return tonal, sr


def analyse(audio: np.ndarray, fs: int) -> dict:
    """Run all 7 timbral models on an in-memory audio array and return a
    plain dict of {name: value}."""
    achieved = {}
    for name, model_fn in TIMBRAL_MODELS.items():
        if name == "warmth":
            # timbral_warmth defaults verbose=True and prints its internals
            achieved[name] = model_fn(audio, fs=fs, clip_output=True, verbose=False)
        else:
            achieved[name] = model_fn(audio, fs=fs, clip_output=True)
    return achieved


def total_error(targets: dict, achieved: dict) -> float:
    """Single scalar a future optimizer minimises."""
    return sum(t.weighted_error(achieved[name]) for name, t in targets.items())


def report(targets: dict, achieved: dict):
    """Print a per-parameter breakdown: target band, achieved value, hit/miss,
    and this parameter's contribution to the total error."""
    header = f"{'parameter':<12}{'target':<12}{'achieved':<10}{'hit?':<6}{'priority':<10}{'weighted err':<14}"
    print(header)
    print("-" * len(header))

    for name, t in targets.items():
        val = achieved[name]
        hit = t.target_min <= val <= t.target_max
        band = f"{t.target_min:g}-{t.target_max:g}"
        werr = t.weighted_error(val)
        print(
            f"{name:<12}{band:<12}{val:<10.2f}{'yes' if hit else 'no':<6}"
            f"{t.priority:<10.1f}{werr:<14.5f}"
        )

    print("-" * len(header))
    print(f"TOTAL ERROR: {total_error(targets, achieved):.5f}")


def main():
    if not os.path.exists(TONAL_AUDIO_PATH):
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    audio, fs = load_tonal(TONAL_AUDIO_PATH)
    achieved = analyse(audio, fs)
    targets = build_targets()
    report(targets, achieved)


if __name__ == "__main__":
    main()