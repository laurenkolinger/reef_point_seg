"""Closing the launch window cancels the routing pass.

The orchestrator UI is the only way these tools are ever launched, so a launch
window the operator closed must not leave work running: no routing sub-app left
holding the GPU, and above all no annotator kill+relaunch fired by a pass nobody
is watching (that is how an abandoned pass killed a live annotator on
2026-07-14).

Covers:
  cancel endpoint  - sets the flag, kills the ROUTING stage (not just the
                     annotator), and moves an in-flight pass to phase "cancelled".
  stop endpoint    - stops BOTH the annotator and the routing pass ("Stop" that
                     leaves the GPU pegged is not a stop).
  the tail gate    - a pass cancelled after its OCR loop must NOT launch an
                     annotator, even though routing itself completed.
  fresh pass       - starting a route clears a cancel left by the previous one.
  no clobber       - cancelling when nothing is routing must not wreck a
                     ui_ready/idle phase.

No-pytest harness: run with github_repo/env/bin/python <this file>.
"""
import os
import sys
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

import app as appmod
import project_manager as pm

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _make_project(tmp):
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    sel = os.path.join(steps["3"]["dir"], "selected_frames.csv")
    with open(sel, "w") as f:
        f.write("basename,year\nTCRMP20180225_clip_BIX_T111,2018\n")
    steps["3"]["outputs"]["selected_frames"] = sel
    os.makedirs(os.path.join(tmp, pm.STEP_DIRS["step4test"], "routed_input"), exist_ok=True)
    return {"id": os.path.basename(tmp), "name": "t", "project_dir": tmp,
            "created_at": "x", "updated_at": "x", "current_step": 4, "steps": steps}


class _FakeRunner:
    """Records kills instead of touching real processes."""

    def __init__(self):
        self.killed = []
        self.ports = {}

    def kill(self, step):
        self.killed.append(step)

    def health_check(self, port):
        return False

    def poll_status(self, step):
        return {"running": False, "exit_code": None, "log_tail": []}


def _client(tmp):
    appmod.current_project = _make_project(tmp)
    appmod.step4test_cfg = {}
    return appmod.create_app().test_client()


# ── cancel kills the ROUTING stage and marks the pass cancelled ──────────────
tmp = tempfile.mkdtemp(prefix="cancel_")
_real_runner = appmod.runner
fake = _FakeRunner()
appmod.runner = fake
try:
    client = _client(tmp)
    appmod.route_cancel.clear()
    appmod._route_set(phase="routing_ocr", processed=102, total=321, error=None)

    r = client.post("/api/step/step4test/cancel", json={})
    data = r.get_json() or {}

    check(data.get("success") is True, f"cancel returns success, got {data}")
    check(appmod.route_cancel.is_set(), "cancel sets the route_cancel flag")
    check("step4test_route" in fake.killed,
          f"cancel kills the ROUTING stage (frees the GPU); killed={fake.killed}")
    check(appmod.route_status.get("phase") == "cancelled",
          f'in-flight pass moves to phase "cancelled", got {appmod.route_status.get("phase")!r}')
finally:
    appmod.runner = _real_runner
    appmod.current_project = None
    shutil.rmtree(tmp, ignore_errors=True)


# ── Stop stops BOTH the annotator and the routing pass ───────────────────────
tmp = tempfile.mkdtemp(prefix="stop_")
fake = _FakeRunner()
appmod.runner = fake
try:
    client = _client(tmp)
    appmod.route_cancel.clear()
    appmod._route_set(phase="routing_ocr", processed=10, total=321, error=None)

    client.post("/api/step/step4test/stop", json={})

    check("step4test" in fake.killed, f"stop kills the annotator; killed={fake.killed}")
    check("step4test_route" in fake.killed,
          f"stop ALSO kills the routing pass (a Stop that leaves the GPU pegged is not "
          f"a stop); killed={fake.killed}")
    check(appmod.route_cancel.is_set(), "stop cancels the routing pass")
finally:
    appmod.runner = _real_runner
    appmod.current_project = None
    shutil.rmtree(tmp, ignore_errors=True)


# ── cancelling when nothing is routing must not clobber a good phase ─────────
tmp = tempfile.mkdtemp(prefix="noclobber_")
fake = _FakeRunner()
appmod.runner = fake
try:
    client = _client(tmp)
    appmod.route_cancel.clear()
    appmod._route_set(phase="ui_ready", ui_port=5080, error=None)

    client.post("/api/step/step4test/cancel", json={})

    check(appmod.route_status.get("phase") == "ui_ready",
          f'cancel must not rewrite a ui_ready phase, got '
          f'{appmod.route_status.get("phase")!r}')
finally:
    appmod.runner = _real_runner
    appmod.current_project = None
    appmod.route_cancel.clear()
    appmod._route_set(phase="idle", ui_port=None)
    shutil.rmtree(tmp, ignore_errors=True)


# ── the tail gate: a cancelled pass must NOT launch an annotator ─────────────
# The incident: an abandoned pass reached its tail, killed the running annotator
# (_launch_step4test_ui kills every annotator first), and launched its own. The
# tail must now refuse outright when the pass was cancelled.
src = open(os.path.join(_ORCH, "app.py")).read()

tail = src.split("Free the GPU before the combined annotator")[-1]
tail_before_launch = tail.split("_launch_step4test_ui()")[0]
check("route_cancel.is_set()" in tail_before_launch,
      "the route tail checks route_cancel BEFORE launching the annotator")

check(src.count("_abort_if_cancelled()") >= 4,
      f"the routing driver polls for cancellation between its steps "
      f"(found {src.count('_abort_if_cancelled()')} checks, want >= 4)")
check("route_cancel.clear()" in src,
      "a fresh routing pass clears a cancel left by the previous one")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
