"""
osc_interface_Max.py

The Python side of the Max MSP <-> Python bridge, adapted from the
earlier prototype under MaxMSP/PythonTest/ (oscTest.py / hello.py /
pythonTest.maxpat) for a different job: that prototype ran one-shot
timbral analysis on a fixed handful of parameters via a single /analyse
message. This version is the live control surface for the whole
pipeline -- every setting in config.py, settable individually from Max
while the server keeps running, plus a /run trigger that executes the
actual search (priority_optimizer.main()) and reports everything it did
back to Max when it finishes.

Kept from the prototype, unchanged, so Ale's existing Max-side objects
still point at the right place: local_IP="127.0.0.1", receive_port=7400
(Max -> Python, an [udpsend 127.0.0.1 7400] object on the Max side),
send_port=8800 (Python -> Max, an [udpreceive 8800] object), and the
"flatten every reply into ONE string argument" convention (see
send_osc()) -- so existing [udpreceive]/[route]/[textedit]-style Max
objects need no rework to receive it.

Persistent server (this is deliberate, not a placeholder): this process
is meant to be launched once and left running for a whole working
session -- Max can send any number of /config/set messages, trigger any
number of /run passes, and get a report back after each one, all over
the one connection. It does NOT exit after a run; it goes back to
listening. python-osc's ThreadingOSCUDPServer already gives each
incoming UDP datagram its own handler thread, so a /config/set arriving
while a /run is executing is still processed immediately rather than
queued behind it.

Why every config.py parameter is reachable, and how: see config_registry.py
for the full explanation -- in short, config.py's values are read LIVE
(`import config` + `config.X` at point of use, everywhere in this
pipeline; see spectral_optimizer.py's note) rather than cached at import
time, specifically so a change made here via OSC, between runs, actually
takes effect on the next /run. config_registry.py is the only thing that
knows how to reach config.py's attributes BY NAME; this file just wires
that up to OSC addresses. All of config.py's parameters -- every scalar,
plus the 7 timbral parameters' TARGETS/PRIORITIES flattened into
<param>_target_min / <param>_target_max / <param>_priority -- are
reachable through the single generic /config/set address (see
config_registry.all_names() for the exact list, or send /config/dump to
have Python enumerate them over OSC).

Superseded from the prototype: /analyse (which ran the 7 raw timbral_*
functions directly on a file path, bypassing the actual editing pipeline
entirely) has no equivalent here -- the same "just tell me where this
sound sits" job is now MODE="scan" (`/config/set MODE scan` then `/run`),
which goes through the real pipeline's loader/mixer and reports back
over /run/report exactly like an edit run does, just with no search, no
targets, and no file written. Keeping a second, parallel "quick analyse"
path that reads the file differently would just be two ways to get a
number that should always agree.

Message reference (see also the protocol doc this pipeline ships
alongside this file):
  /test                                -> /test-reply "<status text>"
  /config/set   <name> <value>         -> /config/set-reply  "<name> ok <new value>" | "<name> error <message>"
  /config/get   <name>                 -> /config/get-reply  "<name> <value>" | "<name> error <message>"
  /config/dump  (no args)              -> one /config/dump-reply "<name> <type> <value>" per registered
                                           parameter, then /config/dump-done <count>
  /run          (no args)              -> /run-started "<mode>"  or  /run-busy "<message>"
                                           (immediate ack; the run itself happens on a background thread)
                                        -> LAST message once the run actually finishes:
                                           /run/report "<report JSON>"   (mode="edit" or "scan", see
                                              priority_optimizer.build_run_report()/build_scan_report())
                                           or /run/error "<exception type>: <message>" if it raised.
                                           Max-side: [dict.deserialize] on the JSON string turns it
                                           straight into a queryable Max dict.
"""

import json
import threading
import traceback

from pythonosc import dispatcher, osc_server, osc_message_builder, udp_client

import config
import config_registry
import priority_optimizer

VERBOSE = True

local_IP = "127.0.0.1"
receive_port = 7400   # Max -> Python
send_port = 8800       # Python -> Max

maxMSP_client = udp_client.SimpleUDPClient(local_IP, send_port)

# Guards against two /run passes stepping on each other -- both mutate
# config.py-driven global pipeline state and write to the same
# OUTPUT_AUDIO_PATH, so a second /run has to wait for the first to finish
# rather than run concurrently. Also used to reject a /config/set that
# arrives mid-run: config.py's values are read live at many separate
# points across a single run (see config_registry.py's note), so a
# parameter changing partway through would leave that run internally
# inconsistent -- e.g. arrays already sized to the OLD N_BANDS meeting
# code that's since started reading the NEW one.
_run_lock = threading.Lock()
_run_busy = False


def send_osc(address: str, arg) -> None:
    """Sends ONE OSC message with ONE argument -- every reply in this
    file is a single flattened string, same convention the prototype
    used (see this file's module docstring), so nothing on the Max side
    has to handle a variable-length OSC arg list."""
    msg = osc_message_builder.OscMessageBuilder(address=address)
    msg.add_arg(arg)
    maxMSP_client.send(msg.build())
    if VERBOSE:
        print(f"  -> {address}  {arg}")


def _log(address: str, args) -> None:
    if VERBOSE:
        print(f"<- {address}  {args}")


# ------------------------------------------------------------
# /test -- simple liveness/status check, no side effects.
# ------------------------------------------------------------

def handle_test(address, *args):
    _log(address, args)
    status = "busy" if _run_busy else "idle"
    send_osc("/test-reply", f"osc_interface_Max.py alive -- status={status} mode={config.MODE}")


# ------------------------------------------------------------
# /config/set, /config/get, /config/dump -- the live parameter surface,
# backed entirely by config_registry.py.
# ------------------------------------------------------------

def handle_config_set(address, *args):
    _log(address, args)
    if len(args) < 2:
        send_osc("/config/set-reply", "error: /config/set needs <name> <value>")
        return

    name = str(args[0])
    raw_value = args[1]

    if _run_busy:
        send_osc("/config/set-reply", f"{name} error busy -- a run is in progress, wait for /run/report")
        return

    try:
        config_registry.set(name, raw_value)
        send_osc("/config/set-reply", f"{name} ok {config_registry.get(name)}")
    except config_registry.RegistryError as exc:
        send_osc("/config/set-reply", f"{name} error {exc}")


def handle_config_get(address, *args):
    _log(address, args)
    if not args:
        send_osc("/config/get-reply", "error: /config/get needs <name>")
        return

    name = str(args[0])
    try:
        send_osc("/config/get-reply", f"{name} {config_registry.get(name)}")
    except config_registry.RegistryError as exc:
        send_osc("/config/get-reply", f"{name} error {exc}")


def handle_config_dump(address, *args):
    """Reports every registered parameter's current value, one OSC
    message at a time (rather than one giant blob) -- lets a Max GUI
    walk them with a plain [route]/[unpack] chain instead of needing a
    JSON parser just to initialise its controls to the tool's actual
    current state. /config/dump-done marks the end of the burst so Max
    knows when it's seen everything."""
    _log(address, args)
    names = config_registry.all_names()
    for entry in config_registry.dump_all():
        send_osc("/config/dump-reply", f"{entry['name']} {entry['type']} {entry['value']}")
    send_osc("/config/dump-done", len(names))


# ------------------------------------------------------------
# /run -- executes priority_optimizer.main() (the real pipeline) on a
# background thread, so this handler (and therefore the OSC server's
# ability to answer /config/get, /config/dump, /test) returns
# immediately rather than blocking for however long the search takes.
# ------------------------------------------------------------

def handle_run(address, *args):
    global _run_busy
    _log(address, args)

    with _run_lock:
        if _run_busy:
            send_osc("/run-busy", "a run is already in progress -- wait for /run/report or /run/error")
            return
        _run_busy = True

    send_osc("/run-started", config.MODE)
    threading.Thread(target=_execute_run, daemon=True).start()


def _execute_run() -> None:
    """Runs the pipeline and sends its report back over OSC -- this is
    the LAST thing that happens for a given /run before the server goes
    back to idle (equivalent to "before the script ends" now that the
    server is persistent rather than a one-shot script): success sends
    /run/report with the full structured report (see
    priority_optimizer.build_run_report() / build_scan_report() --
    bands edited, gains, achieved parameter values, loudness match, etc,
    as one JSON string); any exception sends /run/error instead, so a
    bad config combination or a missing audio file is reported back to
    Max rather than silently leaving it waiting forever. Either way,
    _run_busy is always released in `finally`, so a failed run doesn't
    permanently wedge the server against ever running again."""
    global _run_busy
    try:
        report = priority_optimizer.main()
        send_osc("/run/report", json.dumps(report))
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure must still reply, not hang Max
        if VERBOSE:
            traceback.print_exc()
        send_osc("/run/error", f"{type(exc).__name__}: {exc}")
    finally:
        with _run_lock:
            _run_busy = False


def start_osc_server():
    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/test", handle_test)
    osc_dispatcher.map("/config/set", handle_config_set)
    osc_dispatcher.map("/config/get", handle_config_get)
    osc_dispatcher.map("/config/dump", handle_config_dump)
    osc_dispatcher.map("/run", handle_run)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", receive_port), osc_dispatcher)
    server.daemon_threads = True  # so a run/handler thread never blocks process exit on Ctrl+C

    print(f"osc_interface_Max.py listening on {local_IP}:{receive_port}, replying on {local_IP}:{send_port}")
    print(f"{len(config_registry.all_names())} config parameters reachable via /config/set -- "
          f"send /config/dump to enumerate them, /run to execute the pipeline. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")


if __name__ == "__main__":
    start_osc_server()
