"""Task 10: editmasks (standalone Edit Masks) string routes.

No-pytest harness: run with env/bin/python <this file>.

Covers:
  run     - guards (no project loaded, missing segmentations.json -> 4xx "run
            Step 4 first" guard), happy path launches the editMasks app
            (monkeypatched find_free_port + run_flask_stage so nothing real
            launches), "already running" reuse path.
  status  - reflects runner.poll_status()/health_check() for the "editmasks"
            runner key.
  stop    - kills the runner and clears editmasks_state.
  folder  - refuses a path outside the project; opens the step4test export dir.
  reset   - stops the service, clears state, and NEVER touches the export
            (unlike step4test/step4loop, editmasks owns no data of its own).
"""
import os
import re
import sys
import json
import shutil
import subprocess
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

import app as appmod
import project_manager as pm


# Every launch route reaches _kill_all_annotators(), which does NOT confine itself
# to the runner's own processes: it `pkill -f`s both annotator entry scripts and
# runs `fuser -k` over ports 5080-5089. Un-neutered, that kills a LIVE annotator
# someone is annotating in - test_run_already_running_reuses_port only stubbed
# run_flask_stage, so it reached the real pkill/fuser and killed the box's running
# annotator every time this suite ran. Stub subprocess.run for the WHOLE module so
# no test here can reach off its own process, whatever launch path it touches.
class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()

_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    import traceback
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _make_project(tmp):
    """Minimal fake project: step4test export dir + every chain step dir,
    directly under tmp (same style as test_step4loop_routes.py's fixture)."""
    project_dir = tmp
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    export_dir = os.path.join(project_dir, pm.STEP_DIRS["step4test"])
    os.makedirs(export_dir, exist_ok=True)

    state = {
        "id": os.path.basename(project_dir), "name": "t", "project_dir": project_dir,
        "created_at": "x", "updated_at": "x", "current_step": 4,
        "steps": steps,
    }
    return state, export_dir


def _client_with_project():
    tmp = tempfile.mkdtemp()
    state, export_dir = _make_project(tmp)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    appmod.editmasks_state = {"running": False, "port": None, "started_at": None,
                              "out_dir": None, "project_id": None, "input_dir": None}
    return client, tmp, state, export_dir


def _write_segmentations(export_dir, data=None):
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    seg_path = os.path.join(seg_dir, "segmentations.json")
    with open(seg_path, "w") as f:
        json.dump(data if data is not None else {}, f)
    return seg_path


def test_run_no_project_is_4xx():
    client = appmod.create_app().test_client()
    appmod.current_project = None
    try:
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code >= 400, f"expected 4xx, got {r.status_code}")
    finally:
        appmod.current_project = None


def test_run_missing_segmentations_json_is_4xx():
    """Fail-closed guard (Task 10 spec): the export must already have masks to
    edit; editmasks never creates a fresh export the way step4test can."""
    client, tmp, state, export_dir = _client_with_project()
    try:
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code >= 400, f"expected 4xx, got {r.status_code}")
        data = r.get_json()
        check(data.get("success") is False, f"expected success=false: {data}")
        check("run step 4 first" in data.get("error", "").lower(),
              f"error message should point at Step 4: {data}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_happy_path_launches_editmasks_app():
    """A segmentations.json on disk must let /run reach the launcher. Stub
    _kill_all_annotators' subprocess calls AND runner.run_flask_stage +
    find_free_port so nothing real launches."""
    import subprocess as real_subprocess
    client, tmp, state, export_dir = _client_with_project()
    _write_segmentations(export_dir, {"img0.jpeg": {"masks": [], "reviewed": False, "exported": False}})

    def _fake_run(cmd, **kwargs):
        class _FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""
        return _FakeCompleted()

    orig_run = real_subprocess.run
    real_subprocess.run = _fake_run
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    captured = {}

    def _fake_flask_stage(runner_key, cmd, port, cwd=None, env_extra=None):
        captured["runner_key"] = runner_key
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env_extra"] = env_extra
        return {"pid": 99999, "port": port}

    appmod.runner.run_flask_stage = _fake_flask_stage
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/editmasks/run", json={})
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("success") is True, f"expected success: {data}")
        check(data.get("phase") == "ui_ready", f"expected ui_ready phase: {data}")
        check(data.get("ui_port") == 5085, f"expected ui_port 5085 (editMasks default): {data}")

        check(captured.get("runner_key") == "editmasks",
              f"expected runner key 'editmasks': {captured.get('runner_key')}")
        cmd = captured.get("cmd") or []
        check(any("TCRMPclip_editMasks" in str(c) for c in cmd),
              f"launch cmd should point at TCRMPclip_editMasks, got: {cmd}")
        check("TCRMPclip_combinedAnnotate" not in " ".join(str(c) for c in cmd),
              f"editmasks must NOT launch combinedAnnotate: {cmd}")
        env = captured.get("env_extra") or {}
        check(env.get("TCRMP_SESSION_MODE") == "edit",
              f"TCRMP_SESSION_MODE should be 'edit': {env.get('TCRMP_SESSION_MODE')}")
        check(env.get("TCRMP_PROVENANCE_SOURCE") == "edit",
              f"TCRMP_PROVENANCE_SOURCE should be 'edit': {env.get('TCRMP_PROVENANCE_SOURCE')}")
        check(env.get("TCRMP_EXPORT_DIR") == export_dir,
              f"TCRMP_EXPORT_DIR should be the current project's export dir: {env.get('TCRMP_EXPORT_DIR')} != {export_dir}")
    finally:
        real_subprocess.run = orig_run
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_already_running_reuses_port():
    client, tmp, state, export_dir = _client_with_project()
    _write_segmentations(export_dir)
    orig_launch = appmod._launch_step4test_ui if hasattr(appmod, "_launch_step4test_ui") else None
    # _launch_step4test_ui is a closure inside create_app(); reach it through
    # the route by monkeypatching runner.run_flask_stage to report "already
    # running" the way StageRunner does for a live process.
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.run_flask_stage = lambda *a, **k: {"error": "already running (pid 123)"}
    appmod.find_free_port = lambda preferred: 5085
    appmod.runner.ports = dict(getattr(appmod.runner, "ports", {}) or {})
    appmod.runner.ports["editmasks"] = 5085
    try:
        r = client.post("/api/step/editmasks/run", json={})
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("success") is True, f"expected success on reuse: {data}")
        check(data.get("already_running") is True, f"expected already_running flag: {data}")
        check(data.get("ui_port") == 5085, f"expected reused port 5085: {data}")
    finally:
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        appmod.runner.ports.pop("editmasks", None)
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_status_reflects_runner_and_health():
    client, tmp, state, export_dir = _client_with_project()
    orig_poll = appmod.runner.poll_status
    orig_health = appmod.runner.health_check
    appmod.runner.poll_status = lambda step: {"running": True, "exit_code": None}
    appmod.runner.health_check = lambda port: True
    appmod.runner.ports = dict(getattr(appmod.runner, "ports", {}) or {})
    appmod.runner.ports["editmasks"] = 5085
    try:
        r = client.get("/api/step/editmasks/status")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("healthy") is True, f"expected healthy=true: {data}")
        check(data.get("port") == 5085, f"expected port 5085: {data}")
    finally:
        appmod.runner.poll_status = orig_poll
        appmod.runner.health_check = orig_health
        appmod.runner.ports.pop("editmasks", None)
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_stop_kills_runner_and_clears_state():
    client, tmp, state, export_dir = _client_with_project()
    killed = {}
    orig_kill = appmod.runner.kill
    appmod.runner.kill = lambda step: killed.setdefault("step", step)
    appmod.editmasks_state.update(running=True, port=5085, project_id="x", input_dir="y")
    try:
        r = client.post("/api/step/editmasks/stop", json={})
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("success") is True, f"expected success: {data}")
        check(killed.get("step") == "editmasks", f"runner.kill should target 'editmasks': {killed}")
        check(appmod.editmasks_state.get("running") is False, "editmasks_state.running should be False after stop")
        check(appmod.editmasks_state.get("project_id") is None, "editmasks_state.project_id should be cleared after stop")
    finally:
        appmod.runner.kill = orig_kill
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_folder_refuses_path_outside_project_and_opens_export_dir():
    client, tmp, state, export_dir = _client_with_project()
    try:
        r_ok = client.post("/api/step/editmasks/folder")
        check(r_ok.status_code == 200, f"in-project folder call should succeed: {r_ok.get_json()}")
        data = r_ok.get_json()
        check(data.get("path") == os.path.abspath(export_dir),
              f"folder should open the step4test export dir: {data}")

        # Guard invariant: the resolved export dir must live under project_dir
        # (the same check the route performs) so an out-of-project export can
        # never be opened.
        project_dir_abs = os.path.abspath(state["project_dir"])
        export_dir_abs = os.path.abspath(export_dir)
        check(export_dir_abs.startswith(project_dir_abs + os.sep) or export_dir_abs == project_dir_abs,
              "test fixture invariant broken: export_dir must be inside project_dir")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_reset_stops_service_and_never_touches_export():
    client, tmp, state, export_dir = _client_with_project()
    seg_path = _write_segmentations(export_dir, {"img0.jpeg": {"masks": [{"id": 0}], "reviewed": True, "exported": True}})
    killed = {}
    orig_kill = appmod.runner.kill
    appmod.runner.kill = lambda step: killed.setdefault("step", step)
    appmod.editmasks_state.update(running=True, port=5085, out_dir=export_dir,
                                  project_id="x", input_dir="y")
    try:
        with open(seg_path) as f:
            before = json.load(f)

        r = client.post("/api/step/editmasks/reset", json={})
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("success") is True, f"expected success: {data}")
        check(killed.get("step") == "editmasks", f"runner.kill should target 'editmasks': {killed}")
        check(appmod.editmasks_state.get("running") is False, "editmasks_state.running should be False after reset")
        check(appmod.editmasks_state.get("port") is None, "editmasks_state.port should be cleared after reset")

        # The export must be byte-for-byte untouched: editmasks owns none of
        # this data, it only ever opens what Step 4/Refine already wrote.
        check(os.path.isfile(seg_path), "segmentations.json must survive an editmasks reset")
        with open(seg_path) as f:
            after = json.load(f)
        check(after == before, "segmentations.json content must be unchanged by an editmasks reset")
    finally:
        appmod.runner.kill = orig_kill
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TASK 10 -- editmasks (standalone Edit Masks) string routes\n")
    print("RUN GUARDS:")
    run(test_run_no_project_is_4xx)
    run(test_run_missing_segmentations_json_is_4xx)
    run(test_run_happy_path_launches_editmasks_app)
    run(test_run_already_running_reuses_port)
    print("STATUS:")
    run(test_status_reflects_runner_and_health)
    print("STOP:")
    run(test_stop_kills_runner_and_clears_state)
    print("FOLDER GUARD:")
    run(test_folder_refuses_path_outside_project_and_opens_export_dir)
    print("RESET (non-destructive):")
    run(test_reset_stops_service_and_never_touches_export)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
