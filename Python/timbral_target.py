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
in config.py) -- if the tonal signal is going to be the only thing edited
and analysed, feeding it noise-diluted would just distort the
target-matching without buying anything.

No GUI. No Max/OSC. All user-editable settings live in config.py -- edit
that file, then re-run this one.
"""

import os
from dataclasses import dataclass

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

from config import TONAL_AUDIO_PATH, TARGETS, PRIORITIES, TARGET_MODE


# ============================================================
# Core objects — shouldn't need to touch below here to test scenarios.
# All the values that WOULD normally need editing live in config.py.
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

# The order the 7 parameters are reported/iterated in throughout the
# pipeline -- derived from TIMBRAL_MODELS so there's one source of truth,
# rather than separately-hardcoded lists risking drifting out of sync.
PARAM_NAMES = list(TIMBRAL_MODELS.keys())

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


def build_targets(targets=TARGETS, priorities=PRIORITIES, mode=None, baseline=None) -> dict:
    """Turn the CONFIG dicts into TimbralTarget objects, keyed by name.

    mode: "absolute" (targets are literal 0-100 bands) or "relative"
    (targets are (delta_min, delta_max) offsets from the SOURCE AUDIO's
    own value for that parameter -- e.g. hardness=(-40, -30) means
    "30 to 40 points lower than the source sound's hardness"). Defaults
    to TARGET_MODE from config.py if not given.

    In relative mode, baseline (a dict of {name: achieved_value}, from
    analysing the unedited source audio) is required. Resolved bands are
    clipped to each model's native 0-100 range, so an intentionally wide
    delta like (-100, 100) still resolves to "don't care" regardless of
    the source's value -- same convention as absolute mode's (0, 100).
    """
    mode = mode or TARGET_MODE
    if mode == "relative" and baseline is None:
        raise ValueError(
            "build_targets(mode='relative') requires baseline= "
            "(the source audio's own achieved values -- analyse it before "
            "calling this)"
        )

    result = {}
    for name, (a, b) in targets.items():
        lo, hi = min(a, b), max(a, b)  # tolerate either order in CONFIG
        if mode == "relative":
            if baseline[name] is None:
                # priority 0 -- this model was skipped during analyse(),
                # so there's no source value to offset from. Doesn't
                # matter what the resolved band is since priority 0
                # excludes it from total_error regardless.
                lo, hi = 0.0, 100.0
            else:
                lo = min(max(baseline[name] + lo, 0.0), 100.0)
                hi = min(max(baseline[name] + hi, 0.0), 100.0)
        result[name] = TimbralTarget(
            name=name,
            target_min=lo,
            target_max=hi,
            priority=priorities.get(name, 1.0),
        )
    return result


def describe_target_resolution(targets_raw: dict, mode: str = None, baseline: dict = None):
    """Print how each CONFIG target was resolved into an absolute band.
    Trivial in absolute mode, but worth seeing explicitly in relative mode
    since the resolved band depends on the source audio's own values --
    this is what lets you check the resolution matches what you intended
    before trusting the rest of the run."""
    mode = mode or TARGET_MODE
    print(f"Target mode: {mode}")
    for name, (a, b) in targets_raw.items():
        lo, hi = min(a, b), max(a, b)
        if mode == "relative" and baseline is not None and baseline.get(name) is not None:
            abs_lo = min(max(baseline[name] + lo, 0.0), 100.0)
            abs_hi = min(max(baseline[name] + hi, 0.0), 100.0)
            print(
                f"  {name:<12} source={baseline[name]:<8.2f} delta=({lo:g},{hi:g}) "
                f"-> band {abs_lo:.2f}-{abs_hi:.2f}"
            )
        elif baseline is not None and baseline.get(name) is None:
            print(f"  {name:<12} priority 0 -- skipped, not computed")
        else:
            print(f"  {name:<12} band {lo:g}-{hi:g}")


def load_tonal(tonal_path) -> tuple:
    """Load the tonal-only audio file. Noise handling deliberately excluded
    for now -- see module docstring."""
    tonal, sr = sf.read(tonal_path, always_2d=False)
    if tonal.ndim > 1:
        tonal = np.mean(tonal, axis=1)
    return tonal, sr


def analyse(audio: np.ndarray, fs: int, priorities: dict = None) -> dict:
    """Run the 7 timbral models on an in-memory audio array and return a
    plain dict of {name: value}.

    priorities: optional {name: priority} dict. When given, any model
    with priority <= 0 is SKIPPED entirely (stored as None) rather than
    computed and then multiplied away -- each model costs real time, and
    priority 0 means it will never affect total_error anyway. Left as
    None (the default) computes all 7 unconditionally, which is what you
    want for a one-off/diagnostic read (e.g. this module's own main(), or
    sensitivity_analysis.py run standalone) where seeing every value
    matters more than the small time saving. The optimizers' hot loops
    (spectral_optimizer.py, priority_optimizer.py) pass priorities=targets'
    priority map explicitly, since that's where the repeated cost -- and
    therefore the payoff -- actually is.
    """
    achieved = {}
    for name, model_fn in TIMBRAL_MODELS.items():
        if priorities is not None and priorities.get(name, 1.0) <= 0:
            achieved[name] = None
            continue
        if name == "warmth":
            # timbral_warmth defaults verbose=True and prints its internals
            achieved[name] = model_fn(audio, fs=fs, clip_output=True, verbose=False)
        else:
            achieved[name] = model_fn(audio, fs=fs, clip_output=True)
    return achieved


def total_error(targets: dict, achieved: dict) -> float:
    """Single scalar a future optimizer minimises. Skips priority<=0
    parameters explicitly (rather than relying on weighted_error's
    multiply-by-zero) so it's also safe when analyse() was called with
    priority-based skipping and achieved[name] is None for those."""
    total = 0.0
    for name, t in targets.items():
        if t.priority <= 0:
            continue
        total += t.weighted_error(achieved[name])
    return total


def _fmt_num(x: float) -> str:
    """Whole numbers print clean ('55'), everything else to 1 decimal
    ('59.6') -- keeps the report table's fixed-width columns from
    overflowing when relative-mode resolution produces long floats."""
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def report(targets: dict, achieved: dict):
    """Print a per-parameter breakdown: target band, achieved value, hit/miss,
    and this parameter's contribution to the total error."""
    header = f"{'parameter':<12}{'target':<16}{'achieved':<10}{'hit?':<6}{'priority':<10}{'weighted err':<14}"
    print(header)
    print("-" * len(header))

    for name, t in targets.items():
        val = achieved[name]
        band = f"{_fmt_num(t.target_min)}-{_fmt_num(t.target_max)}"
        if val is None:
            # priority 0 -- skipped during analyse(), never computed
            print(f"{name:<12}{band:<16}{'--':<10}{'--':<6}{t.priority:<10.1f}{'0.00000':<14}")
            continue
        hit = t.target_min <= val <= t.target_max
        werr = t.weighted_error(val)
        print(
            f"{name:<12}{band:<16}{val:<10.2f}{'yes' if hit else 'no':<6}"
            f"{t.priority:<10.1f}{werr:<14.5f}"
        )

    print("-" * len(header))
    print(f"TOTAL ERROR: {total_error(targets, achieved):.5f}")


def main():
    if not os.path.exists(TONAL_AUDIO_PATH):
        raise FileNotFoundError(f"TONAL_AUDIO_PATH not found: {TONAL_AUDIO_PATH}")

    audio, fs = load_tonal(TONAL_AUDIO_PATH)
    achieved = analyse(audio, fs)
    targets = build_targets(TARGETS, PRIORITIES, baseline=achieved)
    describe_target_resolution(TARGETS, TARGET_MODE, achieved)
    print()
    report(targets, achieved)


if __name__ == "__main__":
    main()
