"""Open-folder routes (2026-08-26, "I can not open folder").

Root cause: step4test_state["out_dir"] survived a project switch, so project
B's "Open Folder" resolved project A's export dir and the guard 403'd. Covers:
  - _reset_transient_status forgets every annotator launch binding
  - /api/step/step4test/folder ignores a remembered out_dir from another project
  - every error body carries the resolved path
  - _open_in_file_manager reports a fast xdg-open failure, treats a
    still-running launch as success, and strips snap vars / puts /usr/bin first
  - a temp-dir fixture and VICARIUS_HOST_OPEN=0 are refused at the route
    before any spawn (2026-09-02 hotfix), so the spawn-path fixtures below
    live under this tests dir, not the system temp dir

No-pytest harness: run with env/bin/python <this file>. subprocess.run and
subprocess.Popen are stubbed module-wide (standing rule: never reach a live
annotator, never open a host window).
"""
# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import atexit
import os
import sys
import shutil
import subprocess
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import app as appmod
import project_manager as pm


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()

_RESULTS = []
_POPENS = []


class _FakePopen:
    """Stands in for subprocess.Popen: records argv/env, simulates exit."""
    mode = "running"   # "running" -> TimeoutExpired; "fail" -> rc 1; "ok" -> rc 0

    def __init__(self, argv, **kw):
        _POPENS.append({"argv": argv, "env": kw.get("env"), "kw": kw})
        self.returncode = None

    def communicate(self, timeout=None):
        if _FakePopen.mode == "running":
            raise subprocess.TimeoutExpired(cmd=["xdg-open"], timeout=timeout)
        self.returncode = 1 if _FakePopen.mode == "fail" else 0
        return b"", (b"no file manager registered" if _FakePopen.mode == "fail" else b"")


subprocess.Popen = _FakePopen

# This suite exercises the spawn path itself (argv, detached, GUI env), so it
# runs with host opening enabled: pop VICARIUS_HOST_OPEN (run_all_tests.sh
# exports 0) the way test_lock_gate.py pops VICARIUS_LOCK_BYPASS. The Popen
# stub above is what keeps the desktop safe, not the env var.
os.environ.pop("VICARIUS_HOST_OPEN", None)

# host_open refuses any path under the system temp dir, so fixture projects
# that must reach the (stubbed) spawn live under this tests dir instead. The
# tests dir sits inside REPO_DIR, which _folder_response accepts.
_FIXTURE_ROOT = os.path.join(_HERE, "_fixtures_open_folder")


def _fixture_dir(prefix="open_folder_"):
    os.makedirs(_FIXTURE_ROOT, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=_FIXTURE_ROOT)


atexit.register(lambda: shutil.rmtree(_FIXTURE_ROOT, ignore_errors=True))


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


def _make_project(tmp, pid):
    project_dir = os.path.join(tmp, f"run_{pid}")
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    export_dir = os.path.join(project_dir, pm.STEP_DIRS["step4test"])
    os.makedirs(export_dir, exist_ok=True)
    state = {"id": pid, "name": pid, "project_dir": project_dir,
             "created_at": "x", "updated_at": "x", "current_step": 4, "steps": steps}
    return state, export_dir


def _reset_states():
    for st in (appmod.step4test_state, appmod.step4loop_state, appmod.editmasks_state):
        st.update(running=False, port=None, started_at=None, out_dir=None,
                  project_id=None, input_dir=None)


def test_reset_transient_status_forgets_launch_bindings():
    _reset_states()
    appmod.step4test_state.update(out_dir="/x/a", project_id="A", input_dir="/x/in",
                                  running=True, port=5080)
    appmod.editmasks_state.update(out_dir="/x/b", project_id="A")
    appmod._reset_transient_status("Project switched.")
    for st in (appmod.step4test_state, appmod.step4loop_state, appmod.editmasks_state):
        check(st["out_dir"] is None and st["project_id"] is None and st["input_dir"] is None,
              f"launch binding not cleared: {st}")
    check(appmod.step4test_state["running"] is True and appmod.step4test_state["port"] == 5080,
          "process claims (running/port) must survive the reset for the liveness reconcile")
    _reset_states()


def test_step4test_folder_after_project_switch_opens_current_project():
    tmp = _fixture_dir()
    try:
        _reset_states()
        _POPENS.clear(); _FakePopen.mode = "running"
        state_a, export_a = _make_project(tmp, "A")
        state_b, export_b = _make_project(tmp, "B")
        client = appmod.create_app().test_client()
        # Launched under A, then the reader opened B (as the 21:40 log shows).
        appmod.current_project = state_b
        appmod.step4test_state.update(out_dir=export_a, project_id="A")
        r = client.post("/api/step/step4test/folder", json={})
        j = r.get_json() or {}
        check(r.status_code == 200, f"expected 200 after switch, got {r.status_code} {j}")
        check(j.get("path") == os.path.abspath(export_b),
              f"must open project B's export dir, got {j.get('path')!r}")
        check(_POPENS and _POPENS[-1]["argv"] == ["xdg-open", os.path.abspath(export_b)],
              f"xdg-open argv wrong: {_POPENS[-1:]}")
        # Same project: the remembered launch dir is honored.
        appmod.step4test_state.update(out_dir=export_b, project_id="B")
        r = client.post("/api/step/step4test/folder", json={})
        check((r.get_json() or {}).get("path") == os.path.abspath(export_b), "same-project out_dir honored")
    finally:
        appmod.current_project = None
        _reset_states()
        shutil.rmtree(tmp, ignore_errors=True)


def test_fs_open_errors_carry_the_path():
    tmp = _fixture_dir()
    outside = tempfile.mkdtemp(prefix="open_folder_outside_")
    try:
        _reset_states()
        state, export = _make_project(tmp, "A")
        client = appmod.create_app().test_client()
        appmod.current_project = state
        r = client.post("/api/fs/open", json={"path": outside})
        j = r.get_json() or {}
        check(r.status_code == 403 and j.get("path") == os.path.abspath(outside),
              f"403 body must carry the resolved path: {r.status_code} {j}")
        r = client.post("/api/fs/open", json={"path": os.path.join(tmp, "nope")})
        j = r.get_json() or {}
        check(r.status_code == 400 and "path" in j, f"400 body must carry path: {j}")
        r = client.post("/api/fs/open", json={"path": ""})
        check(r.status_code == 400, "empty path -> 400")
        _FakePopen.mode = "fail"
        r = client.post("/api/fs/open", json={"path": export})
        j = r.get_json() or {}
        check(r.status_code == 500 and "no file manager registered" in j.get("error", "")
              and j.get("path") == os.path.abspath(export),
              f"a fast xdg-open failure must surface stderr + path: {r.status_code} {j}")
        _FakePopen.mode = "ok"
        r = client.post("/api/fs/open", json={"path": export})
        check(r.status_code == 200 and (r.get_json() or {}).get("success") is True, "rc 0 -> success")
        _FakePopen.mode = "running"
        r = client.post("/api/fs/open", json={"path": export})
        check(r.status_code == 200, "still running after 3 s -> success")
    finally:
        appmod.current_project = None
        _FakePopen.mode = "running"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_other_folder_routes_share_the_guard():
    tmp = _fixture_dir()
    try:
        _reset_states()
        state, export = _make_project(tmp, "A")
        client = appmod.create_app().test_client()
        appmod.current_project = state
        for route in ("/api/step/step4loop/folder", "/api/step/editmasks/folder"):
            r = client.post(route, json={})
            j = r.get_json() or {}
            check(r.status_code == 200 and j.get("path") == os.path.abspath(export),
                  f"{route}: {r.status_code} {j}")
        appmod.current_project = None
        for route in ("/api/fs/open", "/api/step/step4test/folder",
                      "/api/step/step4loop/folder", "/api/step/editmasks/folder"):
            r = client.post(route, json={"path": export})
            check(r.status_code == 400, f"{route} without a project -> 400, got {r.status_code}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_gui_env_strips_snap_and_prefers_usr_bin():
    saved = dict(os.environ)
    try:
        os.environ.update({
            "SNAP": "/snap/code/257", "SNAP_NAME": "code", "SNAP_INSTANCE_NAME": "code",
            "GTK_PATH": "/snap/code/257/usr/lib/gtk-3.0",
            "GIO_MODULE_DIR": "/home/bizon/snap/code/common/.cache/gio-modules",
            "SNAP_USER_COMMON": "/home/bizon/snap/code/common",
            "LOCPATH_VSCODE_SNAP_ORIG": "/usr/lib/locale",
            "LOCPATH": "/snap/code/257/usr/lib/locale",
            "PATH": "/home/bizon/anaconda3/bin:/usr/local/bin:/usr/bin:/bin",
            "DISPLAY": ":0",
        })
        env = appmod._gui_env()
        check("SNAP" not in env and "SNAP_NAME" not in env, "SNAP_* vars must be dropped")
        check("GTK_PATH" not in env and "GIO_MODULE_DIR" not in env,
              "vars pointing into the snap must be dropped")
        check(env.get("LOCPATH") == "/usr/lib/locale", "_VSCODE_SNAP_ORIG originals restored")
        check(env.get("DISPLAY") == ":0", "DISPLAY kept")
        check(env["PATH"].split(os.pathsep)[0] == "/usr/bin"
              and env["PATH"].count("/usr/bin") == 1,
              f"/usr/bin must lead PATH exactly once: {env['PATH']}")
    finally:
        os.environ.clear(); os.environ.update(saved)


def test_popen_is_detached_with_gui_env():
    _POPENS.clear(); _FakePopen.mode = "running"
    tmp = _fixture_dir()
    try:
        ok, err = appmod._open_in_file_manager(tmp)
        check(ok and err == "", f"running launch is success: {ok} {err!r}")
        kw = _POPENS[-1]["kw"]
        check(kw.get("start_new_session") is True, "must detach (start_new_session)")
        check(isinstance(kw.get("env"), dict) and "PATH" in kw["env"], "must pass the GUI env")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_temp_fixture_is_refused_at_the_route():
    """2026-09-02 hotfix: a project whose export dir sits under the system
    temp dir is refused before any spawn, the error body carries the path,
    and VICARIUS_HOST_OPEN=0 refuses even a real folder inside the repo."""
    tmp = tempfile.mkdtemp(prefix="open_folder_tmp_")
    try:
        _reset_states()
        _POPENS.clear(); _FakePopen.mode = "running"
        state, export = _make_project(tmp, "A")
        client = appmod.create_app().test_client()
        appmod.current_project = state
        r = client.post("/api/fs/open", json={"path": export})
        j = r.get_json() or {}
        check(r.status_code == 500 and "temporary" in j.get("error", "")
              and j.get("path") == os.path.abspath(export),
              f"temp export dir must be refused with the path: {r.status_code} {j}")
        check(not _POPENS, f"refusal must happen before any spawn: {_POPENS}")
        os.environ["VICARIUS_HOST_OPEN"] = "0"
        try:
            r = client.post("/api/fs/open", json={"path": _fixture_dir()})
            j = r.get_json() or {}
            check(r.status_code == 500 and "VICARIUS_HOST_OPEN=0" in j.get("error", ""),
                  f"VICARIUS_HOST_OPEN=0 refuses a real repo folder at the route: {r.status_code} {j}")
            check(not _POPENS, f"still no spawn under VICARIUS_HOST_OPEN=0: {_POPENS}")
        finally:
            os.environ.pop("VICARIUS_HOST_OPEN", None)
    finally:
        appmod.current_project = None
        _reset_states()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("open-folder routes\n")
    for fn in (test_reset_transient_status_forgets_launch_bindings,
               test_step4test_folder_after_project_switch_opens_current_project,
               test_fs_open_errors_carry_the_path,
               test_other_folder_routes_share_the_guard,
               test_gui_env_strips_snap_and_prefers_usr_bin,
               test_popen_is_detached_with_gui_env,
               test_temp_fixture_is_refused_at_the_route):
        run(fn)
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    for name, _, tb in failed:
        print(f"--- {name}\n{tb}")
    sys.exit(1 if failed else 0)
