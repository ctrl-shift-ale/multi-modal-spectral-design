"""
config_registry.py

Bridge between config.py's plain module-level attributes and anything
that wants to read/write them BY NAME at runtime -- concretely,
osc_interface_Max.py, which receives generic "/config/set <name> <value>"
messages from Max and needs to know, for an arbitrary name: what type is
this, how do I parse the OSC argument into it, and where do I actually
write it.

config.py itself stays free of any of this (see its own "No GUI. No
Max/OSC" note on every file in this pipeline) -- it's still just a plain
module of settings, importable and readable exactly as it always was for
a console-only run. This file is the ONLY thing that knows how to reach
into it by name from outside, and every other module keeps reading it the
normal way (`import config` + `config.X` at point of use -- see
spectral_optimizer.py's note) so a write made here is visible to the rest
of the pipeline immediately, with no extra plumbing.

Two categories of registry entry:

  SCALAR -- a config.py attribute that's already a single value
      (int/float/bool/str/Path) -- read/written directly via
      getattr/setattr.

  TARGET / PRIORITY -- TARGETS and PRIORITIES are dicts keyed by the 7
      timbral parameter names, not flat attributes. Flattened here into
      three OSC-addressable names per parameter: "<param>_target_min",
      "<param>_target_max", "<param>_priority" (e.g. "warmth_target_min")
      -- so a single generic "/config/set <name> <value>" message can
      reach them exactly like any scalar, with no nested addressing
      scheme needed on the Max side. Setting *_target_min/_max only
      touches that one side of the (min, max) tuple; the other side is
      left as it was. Order within the tuple doesn't matter either way
      -- see timbral_target.build_targets(), which sorts it.

Every setter here is intentionally strict and atomic per call: an unknown
name, an unparseable value, or a value that fails one of the few
cross-field invariants config.py itself asserts at import time (e.g.
LIMITER_GAIN_MIN_DB <= IDEAL_GAIN_MIN_DB) raises RegistryError and rolls
back to the pre-call value, rather than leaving config.py in a state
those assertions would have rejected had they run again. A malformed OSC
message should come back to Max as an error reply, not silently corrupt
the next run or crash the server.
"""

from pathlib import Path

import config
from timbral_target import PARAM_NAMES  # the 7 timbral parameter names -- single source of truth


class RegistryError(ValueError):
    """Unknown parameter name, a value of the wrong type/shape, or a
    value that fails a cross-field invariant. Callers (osc_interface_Max.py)
    catch this specifically to turn it into an OSC error reply instead of
    letting the exception crash the server thread."""


# ------------------------------------------------------------
# Value parsing helpers -- OSC hands us a python-osc-decoded arg (str,
# int, or float depending on the OSC type tag Max sent), never guaranteed
# to already be the python type config.py wants.
# ------------------------------------------------------------

def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"cannot interpret {value!r} as a bool")


def _to_path(value) -> Path:
    return Path(str(value))


# ------------------------------------------------------------
# Scalar registry: name -> {kind, type, parser}
# ------------------------------------------------------------

def _scalar(type_name: str, parser) -> dict:
    return {"kind": "scalar", "type": type_name, "parser": parser}


_SCALAR_SPECS = {
    # Paths
    "TONAL_AUDIO_PATH":  _scalar("path", _to_path),
    "NOISE_AUDIO_PATH":  _scalar("path", _to_path),
    "OUTPUT_AUDIO_PATH": _scalar("path", _to_path),

    # Run mode / target mode -- constrained strings, checked against
    # _ENUM_CHOICES below (mirrors config.py's own `assert MODE in (...)`).
    "MODE":        _scalar("enum:scan,edit", str),
    "TARGET_MODE": _scalar("enum:absolute,relative", str),

    # Spectral band structure
    "N_BANDS": _scalar("int", int),
    "FMIN_HZ": _scalar("float", float),
    "IDEAL_GAIN_MIN_DB": _scalar("float", float),
    "IDEAL_GAIN_MAX_DB": _scalar("float", float),
    "LIMITER_GAIN_MIN_DB": _scalar("float", float),
    "LIMITER_GAIN_MAX_DB": _scalar("float", float),
    "PINNED_EPSILON_DB": _scalar("float", float),
    "NPERSEG": _scalar("int", int),
    "NOVERLAP": _scalar("int", int),

    # Optimizer search budget
    "MAX_ITER": _scalar("int", int),
    "FATOL": _scalar("float", float),
    "XATOL": _scalar("float", float),
    "NELDER_MEAD_STEP_DB": _scalar("float", float),

    # Priority-aware search
    "RELEVANCE_RATIO_THRESHOLD": _scalar("float", float),
    "MIN_ACTIVE_BANDS": _scalar("int", int),
    "MAX_ACTIVE_BANDS": _scalar("int", int),
    "WARMSTART_BANDS": _scalar("int", int),
    "WARMSTART_MAX_ITER": _scalar("int", int),

    # Targets & priorities
    "TARGET_CENTERING_PULL": _scalar("float", float),

    # Sensitivity analysis
    "PERTURBATION_DB": _scalar("float", float),

    # Attack/decay-aware editing
    "ATTACK_ENVELOPE_HOP_MS": _scalar("float", float),
    "ATTACK_ENVELOPE_WINDOW_MS": _scalar("float", float),
    "ATTACK_ONSET_THRESHOLD_FRAC": _scalar("float", float),
    "ATTACK_PEAK_SEARCH_MAX_MS": _scalar("float", float),
    "ATTACK_CENTROID_WINDOW_MS": _scalar("float", float),
    "DEFAULT_XFADE_DURATION_MS": _scalar("float", float),
    "DECAY_STABILIZATION_TOLERANCE_DB": _scalar("float", float),
    "DECAY_STABILIZATION_HOLD_MS": _scalar("float", float),
    "DECAY_SEARCH_MAX_MS": _scalar("float", float),

    # Final loudness matching
    "MATCH_SOURCE_LOUDNESS": _scalar("bool", _to_bool),
    "LOUDNESS_MATCH_PEAK_CEILING_DBFS": _scalar("float", float),
}

_ENUM_CHOICES = {
    "MODE": ("scan", "edit"),
    "TARGET_MODE": ("absolute", "relative"),
}

# Cross-field invariants config.py itself asserts at import time (gain
# ranges) or otherwise relies on implicitly (NOVERLAP < NPERSEG is a hard
# scipy.signal.stft requirement; MIN_ACTIVE_BANDS <= MAX_ACTIVE_BANDS isn't
# asserted in config.py but select_bands() assumes it). Re-checked after
# every set() that touches a name on either side of one of these, so a bad
# OSC-driven combination is rejected immediately -- with the just-changed
# attribute rolled back -- rather than surfacing as a confusing failure
# partway through the next run.
_CROSS_FIELD_CHECKS = {
    "IDEAL_GAIN_MIN_DB": [
        (lambda: config.LIMITER_GAIN_MIN_DB <= config.IDEAL_GAIN_MIN_DB,
         "LIMITER_GAIN_MIN_DB must be <= IDEAL_GAIN_MIN_DB"),
    ],
    "IDEAL_GAIN_MAX_DB": [
        (lambda: config.LIMITER_GAIN_MAX_DB >= config.IDEAL_GAIN_MAX_DB,
         "LIMITER_GAIN_MAX_DB must be >= IDEAL_GAIN_MAX_DB"),
    ],
    "LIMITER_GAIN_MIN_DB": [
        (lambda: config.LIMITER_GAIN_MIN_DB <= config.IDEAL_GAIN_MIN_DB,
         "LIMITER_GAIN_MIN_DB must be <= IDEAL_GAIN_MIN_DB"),
    ],
    "LIMITER_GAIN_MAX_DB": [
        (lambda: config.LIMITER_GAIN_MAX_DB >= config.IDEAL_GAIN_MAX_DB,
         "LIMITER_GAIN_MAX_DB must be >= IDEAL_GAIN_MAX_DB"),
    ],
    "NPERSEG": [
        (lambda: config.NOVERLAP < config.NPERSEG, "NOVERLAP must be < NPERSEG"),
    ],
    "NOVERLAP": [
        (lambda: config.NOVERLAP < config.NPERSEG, "NOVERLAP must be < NPERSEG"),
    ],
    "MIN_ACTIVE_BANDS": [
        (lambda: config.MIN_ACTIVE_BANDS <= config.MAX_ACTIVE_BANDS,
         "MIN_ACTIVE_BANDS must be <= MAX_ACTIVE_BANDS"),
    ],
    "MAX_ACTIVE_BANDS": [
        (lambda: config.MIN_ACTIVE_BANDS <= config.MAX_ACTIVE_BANDS,
         "MIN_ACTIVE_BANDS must be <= MAX_ACTIVE_BANDS"),
    ],
}


# ------------------------------------------------------------
# TARGETS / PRIORITIES: flattened per-timbral-parameter names
# ------------------------------------------------------------

def _build_target_specs() -> dict:
    names = {}
    for p in PARAM_NAMES:
        names[f"{p}_target_min"] = ("target", p, "min")
        names[f"{p}_target_max"] = ("target", p, "max")
        names[f"{p}_priority"] = ("priority", p, None)
    return names


_TARGET_SPECS = _build_target_specs()


def all_names() -> list:
    """Every OSC-settable parameter name -- scalar config attributes plus
    the flattened per-timbral-parameter target/priority names. Used by
    osc_interface_Max.py's /config/dump to report the tool's full current
    state so a Max GUI can initialise its controls to it."""
    return sorted(_SCALAR_SPECS.keys()) + sorted(_TARGET_SPECS.keys())


def get(name: str):
    """Current value of a registry entry, by name."""
    if name in _SCALAR_SPECS:
        return getattr(config, name)
    if name in _TARGET_SPECS:
        kind, param, side = _TARGET_SPECS[name]
        if kind == "priority":
            return config.PRIORITIES[param]
        lo, hi = config.TARGETS[param]
        return lo if side == "min" else hi
    raise RegistryError(f"unknown config parameter: {name!r}")


def set(name: str, raw_value) -> None:
    """Parse raw_value (whatever OSC handed us -- typically a str, int, or
    float argument, depending on the OSC type tag Max sent) and write it
    into config.py's live module state. Atomic per call: on any failure
    (unknown name, unparseable value, or a cross-field invariant this
    change would break) the previous value is restored and RegistryError
    is raised -- config.py is never left half-updated."""
    if name in _SCALAR_SPECS:
        spec = _SCALAR_SPECS[name]
        try:
            value = spec["parser"](raw_value)
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"{name}: cannot parse {raw_value!r} as {spec['type']}: {exc}") from exc

        if name in _ENUM_CHOICES and value not in _ENUM_CHOICES[name]:
            raise RegistryError(f"{name}: must be one of {_ENUM_CHOICES[name]}, got {value!r}")

        previous = getattr(config, name)
        setattr(config, name, value)
        for check, message in _CROSS_FIELD_CHECKS.get(name, []):
            if not check():
                setattr(config, name, previous)
                raise RegistryError(f"{name}: rejected ({message})")
        return

    if name in _TARGET_SPECS:
        kind, param, side = _TARGET_SPECS[name]
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"{name}: cannot parse {raw_value!r} as float: {exc}") from exc

        if kind == "priority":
            config.PRIORITIES[param] = value
        else:
            lo, hi = config.TARGETS[param]
            lo, hi = float(lo), float(hi)
            config.TARGETS[param] = (value, hi) if side == "min" else (lo, value)
        return

    raise RegistryError(f"unknown config parameter: {name!r}")


def describe(name: str) -> dict:
    """{"name", "type", "value"} for one registry entry -- what
    /config/dump sends per-parameter to Max, address-agnostic (Max never
    needs to know whether a name is a plain config.py attribute or a
    flattened TARGETS/PRIORITIES entry)."""
    if name in _SCALAR_SPECS:
        value = get(name)
        if isinstance(value, Path):
            value = value.as_posix()
        return {"name": name, "type": _SCALAR_SPECS[name]["type"], "value": value}
    if name in _TARGET_SPECS:
        return {"name": name, "type": "float", "value": get(name)}
    raise RegistryError(f"unknown config parameter: {name!r}")


def dump_all() -> list:
    """describe() for every registered name, in a stable (sorted) order --
    the full state osc_interface_Max.py's /config/dump replies with."""
    return [describe(n) for n in all_names()]
