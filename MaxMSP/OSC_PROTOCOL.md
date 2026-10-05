# Max <-> Python OSC protocol

This documents the OSC interface implemented in `Python/osc_interface_Max.py`
(the persistent server) and backed by `Python/config_registry.py` (the
live config.py parameter registry). It's written so a new Max patch can
be built against it directly -- nothing here requires editing
`MaxMSP/PythonTest/pythonTest.maxpat`'s raw JSON; that prototype patch is
left as reference only (its `/analyse` message has no equivalent in the
new interface -- see below).

## Starting it up

```
python Python/osc_interface_Max.py
```

Runs until Ctrl+C. Prints how many config parameters it registered and
then goes quiet except for a line per message received/sent (set
`VERBOSE = False` at the top of the file to silence that).

## Transport

Same as the prototype (`MaxMSP/PythonTest/pythonTest.maxpat`,
`oscTest.py`) -- unchanged so existing Max-side `[udpsend]`/`[udpreceive]`
objects don't need touching:

- Max -> Python: `[udpsend 127.0.0.1 7400]`
- Python -> Max: `[udpreceive 8800]`
- Every reply is **one OSC message with exactly one argument**, a plain
  string. Where that string packs multiple pieces of information (most
  replies), they're **space-separated** and read left to right in a
  fixed order -- pull it apart in Max with `[unpack s s s]` /
  `[tosymbol]` / `[fromsymbol]`, or a `[regexp]` split, depending on how
  many fields a given reply has (see each message below). This is the
  same "flatten to one string" convention the prototype's
  `/return-analysis` used.
- The one exception is `/run/report`, whose single string argument is a
  **JSON document**, not space-separated fields -- see that section.

## Messages Max sends

### `/test`
No arguments. Liveness/status check.
**Reply:** `/test-reply "<text>"` -- e.g.
`"osc_interface_Max.py alive -- status=idle mode=edit"`. Watch for
`status=busy` vs `status=idle`.

### `/config/set <name> <value>`
The **one generic message that reaches every parameter in `config.py`**
-- this is the whole point of this phase: nothing about the pipeline's
settings requires a dedicated OSC address per parameter. `<name>` is a
symbol (string), `<value>` is whatever Max naturally sends for that
control (int, float, or symbol/string) -- the Python side parses it
according to that parameter's type.

**Reply:** `/config/set-reply "<name> ok <new value>"` on success, or
`/config/set-reply "<name> error <message>"` on failure (unknown name,
value of the wrong type, a value outside a fixed choice like `MODE`, or
a value that would break one of a few cross-field rules `config.py`
itself enforces, e.g. `LIMITER_GAIN_MIN_DB <= IDEAL_GAIN_MIN_DB`). A
rejected `/config/set` changes nothing -- the previous value stays in
effect.

Also rejected, with `"<name> error busy -- ..."`, if a `/run` is
currently executing -- see **Busy behaviour** below.

**Full parameter list, with types:** send `/config/dump` (below) rather
than hardcoding this doc's list into the patch, so the patch always
matches whatever `config_registry.py` actually exposes. The categories:

- **Paths** (type `path`) -- `TONAL_AUDIO_PATH`, `NOISE_AUDIO_PATH`,
  `OUTPUT_AUDIO_PATH`. Value is a filesystem path string.
- **Enums** (type `enum:...`) -- `MODE` (`scan` or `edit`), `TARGET_MODE`
  (`absolute` or `relative`).
- **Ints** (type `int`) -- `N_BANDS`, `NPERSEG`, `NOVERLAP`, `MAX_ITER`,
  `MIN_ACTIVE_BANDS`, `MAX_ACTIVE_BANDS`, `WARMSTART_BANDS`,
  `WARMSTART_MAX_ITER`.
- **Bools** (type `bool`) -- `MATCH_SOURCE_LOUDNESS`. Accepts
  `0`/`1`/`true`/`false`/`on`/`off` (case-insensitive) as well as an
  actual OSC int/float.
- **Floats** (type `float`) -- everything else numeric: `FMIN_HZ`,
  `IDEAL_GAIN_MIN_DB`/`MAX_DB`, `LIMITER_GAIN_MIN_DB`/`MAX_DB`,
  `PINNED_EPSILON_DB`, `FATOL`, `XATOL`, `NELDER_MEAD_STEP_DB`,
  `RELEVANCE_RATIO_THRESHOLD`, `TARGET_CENTERING_PULL`, `PERTURBATION_DB`,
  `ATTACK_ENVELOPE_HOP_MS`, `ATTACK_ENVELOPE_WINDOW_MS`,
  `ATTACK_ONSET_THRESHOLD_FRAC`, `ATTACK_PEAK_SEARCH_MAX_MS`,
  `ATTACK_CENTROID_WINDOW_MS`, `DEFAULT_XFADE_DURATION_MS`,
  `DECAY_STABILIZATION_TOLERANCE_DB`, `DECAY_STABILIZATION_HOLD_MS`,
  `DECAY_SEARCH_MAX_MS`, `LOUDNESS_MATCH_PEAK_CEILING_DBFS`.
- **Per-timbral-parameter targets/priorities** (type `float`) -- `TARGETS`
  and `PRIORITIES` in `config.py` are dicts keyed by the 7 timbral
  parameter names (`warmth`, `brightness`, `depth`, `hardness`,
  `roughness`, `sharpness`, `booming`), not flat names -- so each is
  exposed as three separate settable names instead of one nested
  address: `<param>_target_min`, `<param>_target_max`, `<param>_priority`
  (e.g. `warmth_target_min`, `roughness_priority`). Setting `_target_min`
  or `_target_max` only touches that one side of the (min, max) pair;
  the other side is left alone. Order doesn't matter -- `config.py`
  sorts it internally.

### `/config/get <name>`
**Reply:** `/config/get-reply "<name> <value>"`, or
`/config/get-reply "<name> error <message>"` for an unknown name.

### `/config/dump`
No arguments. Asks Python to report every registered parameter's current
value -- use this once at patch startup to initialise every GUI control
to the tool's actual current state (rather than hardcoding defaults into
the patch that can drift from `config.py`), and again any time the patch
wants to resync.

**Reply:** a burst of `/config/dump-reply "<name> <type> <value>"`
messages, one per parameter (56 as of this writing -- don't hardcode
that count either, it grows if `config.py` gains parameters), followed
by exactly one `/config/dump-done <count>` (a plain OSC int) marking the
end of the burst. In Max: `[route /config/dump-reply /config/dump-done]`
-> feed `dump-reply` lines into `[unpack s s s]` to populate a
`[coll]`/`[dict]` of current values, and use `dump-done`'s count to know
when the burst is finished.

### `/run`
No arguments. Executes the pipeline (`priority_optimizer.main()`) using
whatever `config.py` currently holds -- i.e. exactly the state Max has
built up via `/config/set` calls since the server started (or since the
last `/run`). Runs `MODE="edit"` (the full search) or `MODE="scan"`
(analysis only) depending on the current `MODE` parameter.

**Immediate reply** (before the run itself has necessarily finished):
- `/run-started "<mode>"` -- accepted, now executing on a background
  thread. The server stays responsive to `/test`, `/config/get`,
  `/config/dump` while this runs; `/config/set` and a second `/run` are
  rejected until this one finishes (see **Busy behaviour**).
- `/run-busy "<message>"` -- a run was already in progress; this `/run`
  was ignored. Wait for that run's `/run/report`/`/run/error` and send
  `/run` again.

**Final reply, once the run actually finishes** -- this is the message
that answers "send all the information about the bands edited, gain,
values reached, etc." It's the LAST thing Python does for a given
`/run`, after which the server is idle again:
- `/run/report "<JSON>"` on success -- see **The run report** below.
- `/run/error "<ExceptionType>: <message>"` if the run raised (missing
  audio file, an invalid config combination that slipped past
  `/config/set`'s validation, etc). Nothing is left half-written in this
  case; check the message and fix the offending `/config/set` (or the
  file paths) before retrying.

## Busy behaviour

Only one `/run` executes at a time (mutating shared pipeline state and
writing the same `OUTPUT_AUDIO_PATH` concurrently would be meaningless).
While one is in flight:
- A second `/run` gets `/run-busy` instead of starting.
- Any `/config/set` gets `"<name> error busy -- ..."` instead of taking
  effect -- config.py's values are read live at many separate points
  across a single run, so a parameter changing mid-run could leave that
  run internally inconsistent (e.g. band arrays already sized to the old
  `N_BANDS` meeting code that's since started reading a new one).
- `/config/get`, `/config/dump`, `/test` all still work normally during
  a run -- they only read, never mutate.

## The run report (`/run/report`)

The single JSON string is exactly what `priority_optimizer.build_run_report()`
(mode `"edit"`) or `build_scan_report()` (mode `"scan"`) produced --
the same data the console's own "RUN SUMMARY" printout is built from, so
the OSC report and the terminal output can never disagree.

**Max-side tip:** feed the JSON string straight into `[dict.deserialize]`
-- that turns it into a queryable Max `dict` object with no manual
parsing needed; pull fields out with `[dict]`'s usual getters, or dump it
to a `[js]` object if you'd rather walk it in JavaScript.

### Shape when `mode == "edit"`

```
{
  "mode": "edit",
  "converged": true | false,
  "convergence_message": "<scipy Nelder-Mead status message>",
  "total_error": <float>,
  "parameters": [
    {
      "name": "warmth" | "brightness" | "depth" | "hardness" |
              "roughness" | "sharpness" | "booming",
      "source": <float or null>,       // unedited source's value
      "target_min": <float>,
      "target_max": <float>,
      "achieved": <float or null>,     // null only if priority <= 0 (skipped)
      "hit": true | false | null,      // null if priority <= 0 (meaningless)
      "priority": <float>,
      "weighted_error": <float>
    },
    ... one entry per timbral parameter, always 7, always in this order ...
  ],
  "active_bands": [
    {
      "candidate_index": <int>,        // flat index -- see config_registry naming
      "channel": "tonal" | "noise",
      "band": <int>,                   // band index within that channel
      "freq_min": <float>, "freq_max": <float>,   // Hz
      "relevance": <float>,
      "final_gain_db": <float>,
      "range_used": "ideal" | "limiter"
    },
    ... one entry per candidate the search was allowed to touch ...
  ],
  "bands_not_selected": <int>,         // candidates held fixed at 0 dB
  "attack_refinement": null | {
    "decay_detected": true | false,
    "used_fallback": true | false,
    "attack_start_ms": <float>, "peak_ms": <float>,
    "xfade_start_ms": <float>, "xfade_end_ms": <float>,
    "bands": [
      {"channel": ..., "band": ..., "freq_min": ..., "freq_max": ...,
       "attack_gain_db": <float>, "pass1_gain_db": <float>},
      ...
    ]
  },                                    // present only when hardness had an
                                         // active target (see is_active() in
                                         // priority_optimizer.py)
  "loudness_match": null | {
    "skipped": true, "gate_lufs": <float>
  } | {
    "skipped": false,
    "source_loudness_lufs": <float>, "edited_loudness_lufs": <float>,
    "applied_gain_db": <float>, "final_loudness_lufs": <float>,
    "peak_limited": true | false,
    "peak_ceiling_dbfs": <float>       // present only if peak_limited
  },                                    // present only when
                                         // MATCH_SOURCE_LOUDNESS was on
  "output_audio_path": "<path>" | null
}
```

### Shape when `mode == "scan"`

```
{
  "mode": "scan",
  "tonal_source": "<path>",
  "noise_source": "<path>" | null,
  "sample_rate_hz": <int>,
  "parameters": [
    {"name": "warmth", "value": <float>},
    ... 7 entries, one per timbral parameter, no targets/search involved ...
  ]
}
```

## What's retired from the prototype

`pythonTest.maxpat` / `oscTest.py`'s `/analyse` message (raw
`timbral_*` calls on a file path, independent of the actual editing
pipeline) has no equivalent address here. The same job -- "read this
sound's timbral values without editing it" -- is now
`/config/set MODE scan` followed by `/run`, which goes through the same
loader/mixer the real edit pipeline uses and reports back over the same
`/run/report` channel (scan shape above), so a scan reading and an edit
run's "source" row always agree, by construction, rather than being two
separate code paths that could quietly drift apart.
