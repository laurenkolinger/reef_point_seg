"""Save on quit: SIGTERM, SIGINT and normal interpreter exit flush the active
session to disk before the annotator process ends, and a closed browser tab's
last edit reaches disk through a pagehide sendBeacon.

VEGA annotation safety net, 2026-09-27 (LO). Decisions "Save on quit" and
"Save when the tab closes" in
docs/superpowers/plans/2026-09-27-vega-annotation-safety-net.md. Covers both
combinedAnnotate and editMasks, which share this app.py shape (mirrors
test_polish_20260901.py's own combinedAnnotate+editMasks pattern from this
same tests/ folder), plus one real-process subprocess proof that a live
SIGTERM actually reaches the handler.

No pytest, GPU-free: CUDA_VISIBLE_DEVICES is forced empty below, and nothing
here imports sam_engine or instantiates SAM3Engine. Importing app.py as a
module never does either, since the model load sits behind `if __name__ ==
'__main__':`, which is exactly the guard this suite checks stays in place.

Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_save_on_quit.py
"""
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
CA = os.path.join(_SCRIPTS, "TCRMPclip_combinedAnnotate")
EM = os.path.join(_SCRIPTS, "TCRMPclip_editMasks")

_MODS = ("app", "config", "mask_ops", "sam_engine", "export_yolo",
         "render_overlay", "provenance", "custom_imports")

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _import_app(app_dir):
    """Import one app module in a clean sys.path, mirroring
    test_polish_20260901.py's own exercise() helper: no other test's stale
    module objects leak in, and this file's own imports do not leak out."""
    src = os.path.join(app_dir, "src")
    for mod in _MODS:
        sys.modules.pop(mod, None)
    sys.path.insert(0, src)
    try:
        import app as A
    finally:
        sys.path.remove(src)
    return A


def _release(app_dir):
    for mod in _MODS:
        sys.modules.pop(mod, None)


def exercise_quit_flush(app_dir, label):
    """Drive flush_before_exit() directly against one app module."""
    A = _import_app(app_dir)
    try:
        check(hasattr(A, "flush_before_exit"),
              f"{label}: app defines flush_before_exit")
        if not hasattr(A, "flush_before_exit"):
            return
        _orig_flush_all = A.flush_all
        _orig_log = A.log

        # ── an unconfigured session (no project ever opened): quit writes
        #    nothing, never an empty file over a real one ─────────────────
        A._quit_flushed = False
        A.session["export_dir"] = ""
        calls = []
        A.flush_all = lambda: calls.append(1)
        A.flush_before_exit()
        check(calls == [],
              f"{label}: an unconfigured session's quit never calls flush_all")
        A.flush_all = _orig_flush_all

        d = tempfile.mkdtemp(prefix="quitflush_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            os.makedirs(export_dir, exist_ok=True)
            seg_path = A._seg_path(export_dir)

            # ── a configured session with an in-memory, never-flushed
            #    change: the quit flush writes it to disk ─────────────────
            A._quit_flushed = False
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"FRAME.jpg": {"masks": [], "tag": "unsaved"}}
            A.session["class_map"] = {}
            # A direct dict mutation (this test bypasses every route), so mark
            # dirty by hand the way @_dirty_on_success would after a real edit.
            A._mark_dirty()
            check(not os.path.exists(seg_path),
                  f"{label}: nothing on disk before the quit flush")
            A.flush_before_exit()
            check(os.path.exists(seg_path),
                  f"{label}: the quit flush wrote segmentations.json")
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk.get("FRAME.jpg", {}).get("tag") == "unsaved",
                  f"{label}: the in-memory change reached disk")

            # ── idempotent: calling it twice in a row writes only once ────
            calls = []
            A.flush_all = lambda: calls.append(1)
            A.flush_before_exit()
            A.flush_before_exit()
            check(calls == [],
                  f"{label}: a second consecutive call is a no-op "
                  f"(already flushed this process)")
            A.flush_all = _orig_flush_all

            # ── the session lock held elsewhere past the timeout: the
            #    handler returns promptly and logs, it does not hang ──────
            # The prior save cleared dirty; mark it again so this attempt
            # actually reaches the lock instead of skipping via the dirty gate.
            A._quit_flushed = False
            A._mark_dirty()
            A._QUIT_FLUSH_LOCK_TIMEOUT_S = 0.2
            holder_ready = threading.Event()
            release_holder = threading.Event()

            def hold_lock():
                A._store_write_lock.acquire()
                holder_ready.set()
                release_holder.wait(5)
                A._store_write_lock.release()

            holder = threading.Thread(target=hold_lock, daemon=True)
            holder.start()
            check(holder_ready.wait(2),
                  f"{label}: the contending thread took the session lock")
            logged = []
            A.log = lambda msg: logged.append(msg)
            started = time.time()
            A.flush_before_exit()
            elapsed = time.time() - started
            A.log = _orig_log
            release_holder.set()
            holder.join(timeout=5)
            check(elapsed < 2.0,
                  f"{label}: a busy lock returns promptly instead of "
                  f"blocking forever (took {elapsed:.2f}s)")
            check(bool(logged) and export_dir in logged[0],
                  f"{label}: a busy-lock quit logs one line naming the "
                  f"export dir, got {logged}")
            A._QUIT_FLUSH_LOCK_TIMEOUT_S = 4.0
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def exercise_handlers_installed_only_in_main(app_dir, label):
    """Importing app.py must never install a signal handler or an atexit
    hook, since only the __main__ path does, so a test importing app.py (this
    file, or any other) never wires a real quit handler into the pytest/test
    process itself. Read as source text: the absence of a side effect from a
    plain import cannot be asserted any other way without a subprocess per
    case, which the later subprocess test already covers end to end.
    """
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    parts = src.split("if __name__ == '__main__':", 1)
    check(len(parts) == 2, f"{label}: app.py has a __main__ guard")
    if len(parts) != 2:
        return
    before, after = parts
    check("signal.signal(" not in before,
          f"{label}: no signal.signal registration runs at import time")
    check("atexit.register(" not in before,
          f"{label}: no atexit.register call runs at import time")
    check("signal.signal(signal.SIGTERM" in after,
          f"{label}: __main__ installs a SIGTERM handler")
    check("signal.signal(signal.SIGINT" in after,
          f"{label}: __main__ installs a SIGINT handler")
    check("atexit.register(" in after,
          f"{label}: __main__ registers an atexit flush")


def exercise_beacon_shaped_post(app_dir, label):
    """POST /api/flush must accept a sendBeacon-shaped request: no JSON body,
    text/plain content type. sendBeacon cannot set an Authorization header or
    choose JSON; the route must work with whatever a browser's beacon API
    actually sends."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        A.session["export_dir"] = ""
        with A.app.test_client() as c:
            r = c.post("/api/flush", data=b"",
                       content_type="text/plain;charset=UTF-8")
            check(r.status_code == 200,
                  f"{label}: POST /api/flush with a beacon's text/plain "
                  f"body and no JSON still returns 200 (got {r.status_code})")
            check(r.get_json() == {"ok": True},
                  f"{label}: the beacon POST gets the normal {{ok: true}} body")
    finally:
        _release(app_dir)


for _app_dir, _label in ((CA, "combinedAnnotate"), (EM, "editMasks")):
    print(f"\n[{_label}] flush_before_exit")
    exercise_quit_flush(_app_dir, _label)
    print(f"\n[{_label}] handlers installed only in __main__")
    exercise_handlers_installed_only_in_main(_app_dir, _label)
    print(f"\n[{_label}] POST /api/flush accepts a beacon")
    exercise_beacon_shaped_post(_app_dir, _label)


# ── pagehide beacon (combinedAnnotate template) ──────────────────────────────
# editMasks is opened for one correction pass at a time from the orchestrator,
# which already flushes it on its own stop path (task T2); the plan scopes
# the tab-close beacon to combinedAnnotate's own long batch-review sessions.
print("\n[combinedAnnotate] pagehide beacon in the template")
with open(os.path.join(CA, "src", "templates", "index.html")) as f:
    _tpl = f.read()
_ph = _tpl.split("addEventListener('pagehide'", 1)
check(len(_ph) == 2, "combinedAnnotate: a pagehide listener is registered")
if len(_ph) == 2:
    check(_tpl.count("addEventListener('pagehide'") == 1,
          "combinedAnnotate: exactly one pagehide listener")
    _body = _ph[1].split("});", 1)[0]
    check("sendBeacon(" in _body,
          "combinedAnnotate: the pagehide handler calls navigator.sendBeacon")
    check("/api/flush" in _body,
          "combinedAnnotate: the beacon targets the flush route")


# ── subprocess proof: a real SIGTERM flushes before the process exits ───────
print("\n[subprocess] a real SIGTERM flushes an unsaved change before exit")

_DRIVER = """
import os
import signal
import sys
import time

sys.path.insert(0, {src!r})
os.environ.setdefault("TCRMP_TARGET_SPECIES", "PA")
os.environ.setdefault("TCRMP_MANUAL_ANNOTATE", "1")
import app as A

A.session["export_dir"] = {export_dir!r}
A.session["segmentations"] = {{"FRAME.jpg": {{"masks": [], "tag": "unsaved"}}}}
A.session["class_map"] = {{}}
# A direct dict mutation (no route ran), so mark dirty by hand the way
# @_dirty_on_success would after a real edit.
A._mark_dirty()

signal.signal(signal.SIGTERM, A._handle_quit_signal)
signal.signal(signal.SIGINT, A._handle_quit_signal)

print("READY", flush=True)
while True:
    time.sleep(0.05)
"""


def _wait_for_ready(proc, timeout_s=20):
    """Wait up to timeout_s for the driver process to print "READY".

    Reads proc.stdout on a background thread into a queue, rather than
    pairing select.select() with the stream's own buffered readline()
    (2026-09-27, found while diagnosing this test's own new flakiness):
    a single readline() call can pull several already-flushed lines out of
    the OS pipe in one internal read, so select() reports the file
    descriptor not ready on the next pass even though a later line, such as
    "READY", is already sitting unread in Python's own buffer. /proc/<pid>/
    status showed the driver process already asleep past its READY print
    while this test still waited on a select() that would never fire again.
    A dedicated reader thread draining every line into a queue sidesteps
    that mismatch entirely: the queue always holds whatever has been read,
    in order, and a queue.Empty on a timed get() means exactly "no line has
    arrived yet", not "a line arrived but nothing will tell me".
    """
    lines = queue.Queue()

    def _drain():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # stdout closed with no further lines

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            continue
        if line is None:
            return False
        if line.strip() == "READY":
            return True
    return False


def exercise_subprocess_sigterm(app_dir, label):
    d = tempfile.mkdtemp(prefix="quitproc_%s_" % label)
    proc = None
    try:
        export_dir = os.path.join(d, "out")
        os.makedirs(export_dir, exist_ok=True)
        driver_path = os.path.join(d, "driver.py")
        with open(driver_path, "w") as f:
            f.write(_DRIVER.format(src=os.path.join(app_dir, "src"),
                                    export_dir=export_dir))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.Popen([sys.executable, driver_path], env=env,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        booted = _wait_for_ready(proc)
        check(booted, f"{label}: the driver process booted and armed its handlers")
        if not booted:
            return
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rc = None
        check(rc == 0, f"{label}: the process exits cleanly on SIGTERM (rc={rc})")
        seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")
        check(os.path.exists(seg_path),
              f"{label}: SIGTERM flushed segmentations.json before the process exited")
        if os.path.exists(seg_path):
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk.get("FRAME.jpg", {}).get("tag") == "unsaved",
                  f"{label}: the flushed file carries the unsaved in-memory change")
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        shutil.rmtree(d, ignore_errors=True)


exercise_subprocess_sigterm(CA, "combinedAnnotate")
exercise_subprocess_sigterm(EM, "editMasks")


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
