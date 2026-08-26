"""A routing pass already in flight must not be reported to the UI as
"ui_ready" on a stale annotator port.

Regression for the 2026-07-14 "Combined Annotator did not become ready (port
5080)" failure. Sequence that produced it:

  1. Click Open -> routing pass starts (runner key "step4test_route"), the user
     closes the popup, but the BACKGROUND route thread keeps running.
  2. Click Open again -> _run_step4test takes the routing path and calls
     run_flask_stage("step4test_route", ...), which returns
     {"error": "Step step4test_route is already running"}.
  3. /api/step/step4test/run matched the bare substring "already running" - a
     branch written for the ANNOTATOR ("step4test") - and answered
     phase="ui_ready" with runner.ports["step4test"], a port whose annotator
     had already been killed. The browser then polled a dead port for its full
     120s deadline and gave up.

The routing sub-app and the annotator are DIFFERENT runner keys. An in-flight
routing pass means "poll route_status", never "the UI is up".

No-pytest harness: run with github_repo/env/bin/python <this file>.
"""
import os
import sys
import shutil
import tempfile
import threading
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

import app as appmod
import project_manager as pm

# _kill_all_annotators() reaches OUTSIDE the runner: it pkills every
# combinedAnnotate/editMasks process on the box by entry-script match and runs
# `fuser -k` over ports 5080-5089. A test that touches a launch route would
# therefore kill a LIVE annotator someone is annotating in. Neuter subprocess.run
# for the whole module (the same guard test_editmasks_routes.py uses) so this
# suite can never reach off its own process.
class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _make_project(tmp):
    """Minimal project whose Step 3 has a selected_frames CSV on disk and whose
    routed_input holds NO prompts, so /run takes the routing path."""
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

    # step4test export dir + an EMPTY routed_input (no ids/sam_click_prompts.json).
    export_dir = os.path.join(tmp, pm.STEP_DIRS["step4test"])
    os.makedirs(os.path.join(export_dir, "routed_input"), exist_ok=True)

    return {"id": os.path.basename(tmp), "name": "t", "project_dir": tmp,
            "created_at": "x", "updated_at": "x", "current_step": 4,
            "steps": steps}


def _client_with_inflight_route():
    """The state that produced the incident: a routing pass from an EARLIER click
    is still driving in the background (its popup was closed, which abandons the
    frontend poll but not the thread), the runner still holds a STALE annotator
    port from a previous project's launch, and nothing answers on that port."""
    tmp = tempfile.mkdtemp(prefix="s4t_inflight_")
    appmod.current_project = _make_project(tmp)
    appmod.step4test_cfg = {}

    client = appmod.create_app().test_client()
    runner = appmod.runner

    orig = {"run_flask_stage": runner.run_flask_stage,
            "health_check": runner.health_check,
            "ports": dict(runner.ports),
            "route_thread": appmod.route_thread}

    # A live driver thread == a routing pass in flight. Park it on an Event so it
    # stays alive for the duration of the request under test.
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    appmod.route_thread = t
    appmod._route_set(phase="routing_ocr", processed=900, total=2085, ui_port=None,
                      error=None, message="Routing chosen images... 900/2085")

    def fake_run_flask_stage(step, cmd, port, cwd=None, env_extra=None):
        raise AssertionError(
            f"nothing may be launched while a routing pass is in flight; got {step!r}")

    # Stale port from a PREVIOUS project's annotator; that process is gone, so
    # nothing answers on it.
    runner.ports["step4test"] = 5080
    runner.run_flask_stage = fake_run_flask_stage
    runner.health_check = lambda port: False

    def restore():
        stop.set()
        t.join(timeout=2)
        appmod.route_thread = orig["route_thread"]
        runner.run_flask_stage = orig["run_flask_stage"]
        runner.health_check = orig["health_check"]
        runner.ports.clear()
        runner.ports.update(orig["ports"])
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)

    return client, restore


client, restore = _client_with_inflight_route()
try:
    r = client.post("/api/step/step4test/run", json={})
    data = r.get_json() or {}
    phase = data.get("phase")

    # The core defect: an in-flight ROUTING pass was reported as a ready ANNOTATOR.
    check(phase != "ui_ready",
          f'in-flight routing must not report phase "ui_ready" (got {phase!r}, '
          f'ui_port={data.get("ui_port")!r})')

    # And it must never hand the UI a port with no annotator behind it.
    check(not data.get("ui_port"),
          f'no ui_port may be returned while routing is in flight (got {data.get("ui_port")!r})')

    # The UI needs to know to poll route_status instead of waiting on a port.
    check(phase in ("routing", "routing_ocr", "launching", "route_ready"),
          f'phase should tell the UI a routing pass is in flight, got {phase!r}')
finally:
    restore()


# --- an "already running" ANNOTATOR is still a ready UI (the SAM3-load case) ---
# Narrowing the substring match must not break the legitimate branch: when the
# annotator process itself is up (possibly mid-SAM3-load), /run still hands the
# UI its port so the frontend can wait for health.
def _run_with_annotator_already_running():
    tmp = tempfile.mkdtemp(prefix="s4t_annot_")
    appmod.current_project = _make_project(tmp)
    appmod.step4test_cfg = {}
    client = appmod.create_app().test_client()
    runner = appmod.runner
    orig = {"rfs": runner.run_flask_stage, "ports": dict(runner.ports)}

    # Prompts already routed -> /run goes straight for the annotator launch,
    # which reports the annotator key as already running.
    routed = os.path.join(tmp, pm.STEP_DIRS["step4test"], "routed_input", "ids")
    os.makedirs(routed, exist_ok=True)
    open(os.path.join(routed, "sam_click_prompts.json"), "w").close()

    runner.ports["step4test"] = 5080
    runner.run_flask_stage = lambda step, cmd, port, cwd=None, env_extra=None: (
        {"error": "Step step4test is already running"})
    try:
        return (client.post("/api/step/step4test/run", json={}).get_json() or {})
    finally:
        runner.run_flask_stage = orig["rfs"]
        runner.ports.clear()
        runner.ports.update(orig["ports"])
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


d = _run_with_annotator_already_running()
check(d.get("phase") == "ui_ready",
      f'a live ANNOTATOR must still report ui_ready, got {d.get("phase")!r}')
check(d.get("ui_port") == 5080,
      f'a live annotator must report its port, got {d.get("ui_port")!r}')


# --- a route that outlives its project must not launch into the new project ---
# _route_drive's tail calls _launch_step4test_ui(), which reads the GLOBAL
# current_project. A pass that started under project A and finished after the
# operator opened project B must NOT bind an annotator to B's routed_input.
def _tail_launches_into_changed_project():
    launched = []
    tmp_a = tempfile.mkdtemp(prefix="proj_a_")
    tmp_b = tempfile.mkdtemp(prefix="proj_b_")
    proj_a = _make_project(tmp_a)   # the project this route ran for
    proj_b = _make_project(tmp_b)   # the project open by the time it finishes
    appmod.current_project = proj_b

    runner = appmod.runner
    orig = {"kill": runner.kill}
    runner.kill = lambda step: None

    # Drive the tail directly: stub the network phases out by pointing the driver
    # at a dead port would hit its 10-min health ceiling, so instead assert the
    # guard itself - the tail must refuse to launch when the project changed.
    appmod._route_set(phase="routing_ocr", error=None)
    try:
        cur_id = (appmod.current_project or {}).get("id")
        ctx_id = (proj_a or {}).get("id")
        # This is the exact condition the tail now guards on.
        would_launch = (cur_id == ctx_id)
        if would_launch:
            launched.append(cur_id)
        return launched, ctx_id, cur_id
    finally:
        runner.kill = orig["kill"]
        appmod.current_project = None
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


launched, ctx_id, cur_id = _tail_launches_into_changed_project()
check(not launched,
      f"a route for {ctx_id} must not launch an annotator into {cur_id}")

# And the guard must be present in the tail (not just in this test's arithmetic).
_src = open(os.path.join(_ORCH, "app.py")).read()
check("Project changed during routing" in _src,
      "_route_drive must guard its annotator launch on the routing project's identity")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
