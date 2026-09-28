"""Standalone Annotate Images launch routes. subprocess.run is stubbed for the
whole module (a LIVE annotator may be running on 5080; nothing here may reach it).
Run: env/bin/python scripts/pipeline_orchestrator/tests/test_annotate_routes.py
"""
import os, sys, subprocess, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__)); _ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)
class _FakeCompleted:
    returncode = 0; stdout = ""; stderr = ""
subprocess.run = lambda *a, **k: _FakeCompleted()
os.environ["VICARIUS_LOCK_BYPASS"] = "1"
import app as appmod

# Test isolation (2026-09-28 review): _kill_all_annotators' flush-before-kill
# (POST /api/flush, pipeline_orchestrator/app.py) reaches the network even
# when subprocess.run/Popen are stubbed, since it goes through urllib, not
# subprocess. Every launch/stop route in this file can drive it (the fake
# ss/pgrep data further down names real-looking annotator listeners on
# 5080/5081), so appmod._flush_annotator is stubbed here (never a real save,
# never a real socket) and socket.socket is guarded so any code path that
# still tries to reach the orchestrator (5050) or the annotator port range
# (5080-5089) fails the test loudly instead of making a live network call.
appmod._flush_annotator = lambda port: (True, None)
import socket as _socket
_REAL_SOCKET = _socket.socket
_GUARDED_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}
_GUARDED_PORTS = {5050} | set(range(5080, 5090))


class _NoLivePortSocket(_REAL_SOCKET):
    def connect(self, address, *a, **k):
        if (isinstance(address, tuple) and len(address) >= 2
                and address[0] in _GUARDED_HOSTS and address[1] in _GUARDED_PORTS):
            raise AssertionError(
                f"test tried to open a real socket to {address[0]}:{address[1]} "
                f"(guarded: 5050, 5080-5089); stub the code path that reached this")
        return super().connect(address, *a, **k)


_socket.socket = _NoLivePortSocket

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

launched = []
def fake_run_flask_stage(step, cmd, port, cwd=None, env_extra=None):
    launched.append({"step": step, "cmd": cmd, "port": port, "cwd": cwd, "env": env_extra or {}})
    appmod.runner.ports[step] = port
    return {"ok": True}
appmod.runner.run_flask_stage = fake_run_flask_stage
appmod.find_free_port = lambda preferred: preferred
kills = []
appmod._kill_all_annotators = lambda: kills.append("global")
appmod.runner.kill = lambda step: kills.append(step)
appmod.runner.poll_status = lambda step: {"running": step in appmod.runner.ports}
appmod.runner.health_check = lambda port: True

client = appmod.create_app().test_client()
appmod.current_project = None

r = client.get("/annotate")
check(r.status_code == 200 and b"Annotate Images" in r.data, "GET /annotate serves the launch page")
check(b"/api/annotate/run" in r.data and b"/api/annotate/status" in r.data, "page wires run + status")

# D5 (2026-09-02, final-fix wave): the boot page had no display role - it
# loaded only Inter and Space Mono, so the page title "Annotate Images"
# rendered in Inter where the theme contract assigns page titles to Syne.
# Every other window in the family loads all three; this page now does too,
# with Syne scoped to the title only (not the whole page).
_html = r.data.decode()
check("family=Syne" in _html, "annotate.html must load Syne (D5: the page had no display role)")
check("family=Inter" in _html and "family=Space+Mono" in _html,
      "annotate.html must still load Inter and Space Mono")
_i = _html.index("<style>"); _j = _html.index("</style>")
_style = _html[_i:_j]
check("'Syne'" in _style, "the Syne font must actually be used somewhere in the page style")
_body_rule = _style[_style.index("body{"):_style.index("}", _style.index("body{"))]
check("Syne" not in _body_rule, "Syne belongs to the page title only, not the whole body")
_title_rule = _style[_style.index("#title{"):_style.index("}", _style.index("#title{"))]
check("Syne" in _title_rule, "#title must use Syne (the page title is the display role)")

r = client.post("/api/annotate/run", json={})
j = r.get_json()
check(r.status_code == 200 and j.get("phase") == "ui_ready" and j.get("ui_port") == appmod.ANNOTATE_PORT,
      f"run launches on ANNOTATE_PORT, got {j}")
check(len(launched) == 1 and launched[0]["step"] == "annotate", "runner key is 'annotate'")
env = launched[0]["env"]
check(env.get("TCRMP_SESSION_MODE") == "standalone" and env.get("TCRMP_PROVENANCE_SOURCE") == "standalone",
      "env: standalone session + provenance")
check(env.get("TCRMP_STANDALONE_ROOT", "").endswith(os.path.join("inprocess", "_standalone")), "env: standalone root")
check("TCRMP_INPUT_DIR" not in env and "TCRMP_PROJECT_ID" not in env, "env: no project vars")
check(launched[0]["cmd"][1].endswith(os.path.join("TCRMPclip_combinedAnnotate", "src", "app.py")), "entry is combinedAnnotate")
check("global" not in kills, "standalone launch never calls _kill_all_annotators")
check(kills == ["annotate"], f"only its own previous runner is stopped, got {kills}")

r = client.get("/api/annotate/status"); j = r.get_json()
check(j.get("port") == appmod.ANNOTATE_PORT and j.get("healthy") is True, "status reports port + health")

r = client.post("/api/annotate/run", json={}); j = r.get_json()
check(j.get("phase") == "ui_ready" and j.get("already_running") is True and len(launched) == 1,
      "second run reuses the healthy instance")

r = client.post("/api/annotate/stop"); check(r.status_code == 200 and "annotate" in kills, "stop kills the runner")

check(appmod.ANNOTATE_PORT not in range(5080, 5090),
      "ANNOTATE_PORT sits outside the annotator port range _kill_all_annotators sweeps")

# Fix round 1: _kill_all_annotators (called by every PROJECT-mode launch, e.g.
# editmasks/run below) must spare a standalone session instead of killing it
# by name/port. Drive it through the editmasks/run route (same route
# test_editmasks_routes.py uses) since _kill_all_annotators is a private
# closure with no direct module-level handle. pgrep is faked to report two
# pids for the combinedAnnotate entry match; one pid's /proc/<pid>/environ
# (via a patched _proc_environ) carries TCRMP_SESSION_MODE=standalone and
# must be spared, the other carries nothing and must be killed.
import project_manager as pm
import shutil as _shutil
import signal as _signal

_tmp = tempfile.mkdtemp(prefix="annotate_kill_test_")
try:
    _steps = {}
    for _s in pm.STEP_KEYS:
        _d = os.path.join(_tmp, pm.STEP_DIRS[_s])
        os.makedirs(_d, exist_ok=True)
        _steps[_s] = {"status": "pending", "name": pm.STEP_NAMES[_s], "dir": _d,
                      "config": {}, "outputs": {}}
    _export_dir = os.path.join(_tmp, pm.STEP_DIRS["step4test"])
    _seg_dir = os.path.join(_export_dir, "segmentations")
    os.makedirs(_seg_dir, exist_ok=True)
    with open(os.path.join(_seg_dir, "segmentations.json"), "w") as fh:
        fh.write("{}")
    appmod.current_project = {
        "id": os.path.basename(_tmp), "name": "t", "project_dir": _tmp,
        "created_at": "x", "updated_at": "x", "current_step": 4, "steps": _steps,
    }
    appmod.editmasks_state = {"running": False, "port": None, "started_at": None,
                              "out_dir": None, "project_id": None, "input_dir": None}

    def _fake_pgrep_run(cmd, *a, **k):
        if cmd and cmd[0] == "pgrep" and len(cmd) > 2 and "combinedAnnotate" in cmd[2]:
            return type("R", (), {"returncode": 0, "stdout": "4001\n4002\n", "stderr": ""})()
        return _FakeCompleted()

    def _fake_proc_environ(pid):
        return b"TCRMP_SESSION_MODE=standalone\0" if pid == 4001 else b""

    killed = []
    _orig_subprocess_run = subprocess.run
    _orig_proc_environ = appmod._proc_environ
    _orig_os_kill = os.kill
    subprocess.run = _fake_pgrep_run
    appmod._proc_environ = _fake_proc_environ
    os.kill = lambda pid, sig: killed.append((pid, sig))
    try:
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code == 200,
              f"editmasks/run (drives _kill_all_annotators) succeeds, got {r.status_code} {r.get_json()}")
    finally:
        subprocess.run = _orig_subprocess_run
        appmod._proc_environ = _orig_proc_environ
        os.kill = _orig_os_kill

    check(killed == [(4002, _signal.SIGTERM)],
          f"_kill_all_annotators spares the standalone pid and kills only the other, got {killed}")
finally:
    appmod.current_project = None
    _shutil.rmtree(_tmp, ignore_errors=True)

# ── Port half of _kill_all_annotators: cmdline-checked, never a blind sweep ──
# 5080-5089 is shared with the rest of the platform (vicariusLLM autostarts on
# 5085; that WAS also the Edit Masks preferred port until TODO T20260826-223636
# moved Edit Masks to 5083), so the old `fuser -k` loop over the range killed
# vicariusLLM on 2026-08-26. Only a listener whose /proc/<pid>/cmdline names an
# annotator entry script may be signalled now. The sample below still uses 5085
# to stand in for a live vicariusLLM listener, which the sweep must leave alone.
SS_SAMPLE = "\n".join([
    "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process",
    'LISTEN 0      128          127.0.0.1:5085       0.0.0.0:*    users:(("python",pid=9000001,fd=4))',
    'LISTEN 0      128            0.0.0.0:5080       0.0.0.0:*    users:(("python",pid=9000002,fd=49))',
    'LISTEN 0      128            0.0.0.0:5050       0.0.0.0:*    users:(("python",pid=9000003,fd=19))',
])
check(appmod._port_listener_pids(SS_SAMPLE, 5080) == [9000002],
      f"_port_listener_pids reads the pid off an ss line, got {appmod._port_listener_pids(SS_SAMPLE, 5080)}")
check(appmod._port_listener_pids(SS_SAMPLE, 5085) == [9000001], "and off a loopback-bound line")
check(appmod._port_listener_pids(SS_SAMPLE, 5081) == [], "a port with no listener yields nothing")
check(appmod._port_listener_pids("", 5080) == [] and appmod._port_listener_pids(None, 5080) == [],
      "empty or missing ss output never raises")

_tmp2 = tempfile.mkdtemp(prefix="annotate_sweep_test_")
try:
    _steps2 = {}
    for _s in pm.STEP_KEYS:
        _d = os.path.join(_tmp2, pm.STEP_DIRS[_s])
        os.makedirs(_d, exist_ok=True)
        _steps2[_s] = {"status": "pending", "name": pm.STEP_NAMES[_s], "dir": _d,
                       "config": {}, "outputs": {}}
    _seg_dir2 = os.path.join(_tmp2, pm.STEP_DIRS["step4test"], "segmentations")
    os.makedirs(_seg_dir2, exist_ok=True)
    with open(os.path.join(_seg_dir2, "segmentations.json"), "w") as fh:
        fh.write("{}")
    appmod.current_project = {
        "id": os.path.basename(_tmp2), "name": "t", "project_dir": _tmp2,
        "created_at": "x", "updated_at": "x", "current_step": 4, "steps": _steps2,
    }
    appmod.editmasks_state = {"running": False, "port": None, "started_at": None,
                              "out_dir": None, "project_id": None, "input_dir": None}

    _ANN_ENTRY = os.path.join("TCRMPclip_combinedAnnotate", "src", "app.py")
    _CMDLINES = {
        9000001: "/mnt/rip/vicarius_drive/vicarius/modules/vicarius_llm/env/bin/python "
                 "/mnt/rip/vicarius_drive/vicarius/modules/vicarius_llm/src/app.py",
        9000002: "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python "
                 "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/" + _ANN_ENTRY,
    }

    def _fake_ss_run(cmd, *a, **k):
        if cmd and cmd[0] == "ss":
            return type("R", (), {"returncode": 0, "stdout": SS_SAMPLE, "stderr": ""})()
        return _FakeCompleted()   # pgrep finds nothing: only the port half runs

    killed2 = []
    _orig_run2 = subprocess.run
    _orig_cmdline = appmod._proc_cmdline
    _orig_environ2 = appmod._proc_environ
    _orig_kill2 = os.kill
    subprocess.run = _fake_ss_run
    appmod._proc_cmdline = lambda pid: _CMDLINES.get(pid, "")
    appmod._proc_environ = lambda pid: b""
    os.kill = lambda pid, sig: killed2.append((pid, sig))
    try:
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code == 200,
              f"editmasks/run (drives the port sweep) succeeds, got {r.status_code} {r.get_json()}")
    finally:
        subprocess.run = _orig_run2
        appmod._proc_cmdline = _orig_cmdline
        appmod._proc_environ = _orig_environ2
        os.kill = _orig_kill2

    check(killed2 == [(9000002, _signal.SIGTERM)],
          f"only the annotator listener is killed; vicariusLLM on 5085 is left alone, got {killed2}")
finally:
    appmod.current_project = None
    _shutil.rmtree(_tmp2, ignore_errors=True)

# An unreadable cmdline fails closed: the listener is left running.
_tmp3 = tempfile.mkdtemp(prefix="annotate_failclosed_test_")
try:
    _steps3 = {}
    for _s in pm.STEP_KEYS:
        _d = os.path.join(_tmp3, pm.STEP_DIRS[_s])
        os.makedirs(_d, exist_ok=True)
        _steps3[_s] = {"status": "pending", "name": pm.STEP_NAMES[_s], "dir": _d,
                       "config": {}, "outputs": {}}
    _seg_dir3 = os.path.join(_tmp3, pm.STEP_DIRS["step4test"], "segmentations")
    os.makedirs(_seg_dir3, exist_ok=True)
    with open(os.path.join(_seg_dir3, "segmentations.json"), "w") as fh:
        fh.write("{}")
    appmod.current_project = {
        "id": os.path.basename(_tmp3), "name": "t", "project_dir": _tmp3,
        "created_at": "x", "updated_at": "x", "current_step": 4, "steps": _steps3,
    }
    appmod.editmasks_state = {"running": False, "port": None, "started_at": None,
                              "out_dir": None, "project_id": None, "input_dir": None}
    killed3 = []
    _orig_run3 = subprocess.run
    _orig_cmdline3 = appmod._proc_cmdline
    _orig_kill3 = os.kill
    subprocess.run = _fake_ss_run
    appmod._proc_cmdline = lambda pid: ""
    os.kill = lambda pid, sig: killed3.append((pid, sig))
    try:
        client.post("/api/step/editmasks/run", json={})
    finally:
        subprocess.run = _orig_run3
        appmod._proc_cmdline = _orig_cmdline3
        os.kill = _orig_kill3
    check(killed3 == [], f"an unreadable cmdline fails closed and kills nothing, got {killed3}")
finally:
    appmod.current_project = None
    _shutil.rmtree(_tmp3, ignore_errors=True)

# lock gate: locked registry -> 423 on the API and the lock page on /annotate
os.environ.pop("VICARIUS_LOCK_BYPASS", None)
import lock_status
orig = lock_status.is_locked; lock_status.is_locked = lambda: True
try:
    r = client.post("/api/annotate/run", json={}); check(r.status_code == 423, "locked module: run is 423")
    r = client.get("/annotate"); check(b"vic-lock-page" in r.data, "locked module: /annotate shows the lock page")
finally:
    lock_status.is_locked = orig; os.environ["VICARIUS_LOCK_BYPASS"] = "1"

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
