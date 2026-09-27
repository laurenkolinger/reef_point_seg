"""2026-09-27 annotation safety-net plan, Task T2: the orchestrator saves
before it stops.

No-pytest harness: run with env/bin/python <this file>.

Covers:
  _flush_annotator: the POST /api/flush helper, success, a non-2xx status, a
                     connection failure, an HTTPError, and the "no port"
                     fail-closed case. Never raises.
  _listener_port_for_pid: the pid to port reverse lookup _kill_all_annotators
                     uses to flush a zombie process (found only by pgrep, not
                     tracked in runner.ports) by port before it is signalled.
  _kill_all_annotators: every kill (a runner-tracked port, a pgrep-matched
                     zombie pid, a port-swept listener) is preceded by a
                     flush of that same port; a standalone session and an
                     unrelated listener (vicariusLLM on 5085) are never
                     flushed or killed; the same port is flushed at most once
                     per call even when both the pgrep pass and the port
                     sweep name it.
  the alive check: a tracked port is only flushed when runner.is_running(key)
                     confirms that key still owns a live process (a dead
                     key's stale port, never cleared by kill(), can be reused
                     by an unrelated process); getattr-safe for a runner
                     without is_running, which fails closed (no flush).
  stop-path routes: step4test/stop, step4test/reset, step4loop/reset,
                     editmasks/stop, editmasks/reset and /api/project/quit
                     each flush the annotator's port before calling
                     runner.kill; a failed flush is logged (port and reason)
                     and the stop still succeeds.
  step4loop/run: flushes AND fully stops any tracked step4test/step4loop
                     annotator BEFORE the seeder subprocess runs (not merely
                     before the later kill inside _launch_step4test_ui),
                     because the seeder rewrites segmentations.json directly
                     and the annotator's own flush is a blind dump of its
                     in-memory dict, so a flush reached only after the seed
                     would erase the newly seeded masks.
  restart_reef_point_seg.sh, run_pipeline.sh: both are valid bash and
                     (checked statically, not by executing them, see the
                     comment above the shell-script tests below) flush only
                     an annotator-owned port (its listening pid's cmdline
                     matches one of the two entry scripts) before the first
                     kill/fuser/pkill/lsof line reachable from a normal run.

subprocess.run/Popen are stubbed at module level throughout (mirrors every
other file in this directory): _kill_all_annotators reaches pgrep/ss/fuser
outside the runner, and a live annotator on this box must never be touched by
running this suite. The one exception is the shell scripts' `bash -n` syntax
check below, which uses the real Popen captured before the stub, since
parsing a script's syntax never executes it, so nothing on the box is at
risk.
"""
import os
import re
import sys
import json
import signal
import shutil
import subprocess
import tempfile
import urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)
_REAL_POPEN = subprocess.Popen


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


def _blocked_popen(*a, **k):
    raise AssertionError(f"subprocess.Popen reached: {a[0] if a else k.get('args')!r}")


subprocess.run = lambda *a, **k: _FakeCompleted()
subprocess.Popen = _blocked_popen

import app as appmod
import project_manager as pm

# _live_annotator_port only flushes a tracked port when runner.is_running(key)
# confirms that key's process is still alive (the alive-check review fix,
# 2026-09-27). Every fixture below defaults the real runner to "always alive"
# so existing flush-before-kill assertions do not have to restate this on
# every test; a test that means to exercise the gate itself overrides it.
_ORIG_RUNNER_IS_RUNNING = appmod.runner.is_running

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
    directly under tmp (same style as the other pipeline_orchestrator tests)."""
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


def _fresh_state_dict():
    return {"running": False, "port": None, "started_at": None,
            "out_dir": None, "project_id": None, "input_dir": None}


def _client_with_project():
    tmp = tempfile.mkdtemp(prefix="flush_before_kill_test_")
    state, export_dir = _make_project(tmp)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    appmod.step4test_state = _fresh_state_dict()
    appmod.step4loop_state = _fresh_state_dict()
    appmod.editmasks_state = _fresh_state_dict()
    appmod.step4loop_cfg = {}
    appmod.runner.is_running = lambda step: True
    return client, tmp, state, export_dir


def _teardown(tmp):
    appmod.current_project = None
    appmod.runner.ports = {}
    appmod.runner.is_running = _ORIG_RUNNER_IS_RUNNING
    shutil.rmtree(tmp, ignore_errors=True)


# ── _flush_annotator: the POST /api/flush helper, never raises ──────────────

def test_flush_annotator_no_port_fails_closed_without_network():
    ok, reason = appmod._flush_annotator(None)
    check(ok is False and reason == "no port", f"expected (False, 'no port'), got {(ok, reason)}")
    ok2, reason2 = appmod._flush_annotator(0)
    check(ok2 is False, f"a falsy port must fail closed too, got {(ok2, reason2)}")


def test_flush_annotator_success_posts_to_api_flush_with_named_timeout():
    class _FakeResp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    seen = {}

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["timeout"] = timeout
        return _FakeResp()

    orig_urlopen = appmod.urlopen
    appmod.urlopen = _fake_urlopen
    try:
        ok, reason = appmod._flush_annotator(5080)
        check(ok is True and reason is None, f"expected success, got {(ok, reason)}")
        check(seen["url"] == "http://127.0.0.1:5080/api/flush", f"wrong url: {seen}")
        check(seen["method"] == "POST", f"expected POST, got {seen.get('method')}")
        check(seen["timeout"] == appmod._ANNOTATOR_FLUSH_TIMEOUT_S,
              f"expected the named timeout constant ({appmod._ANNOTATOR_FLUSH_TIMEOUT_S}), "
              f"got {seen.get('timeout')}")
    finally:
        appmod.urlopen = orig_urlopen


def test_flush_annotator_reports_non_2xx_status():
    class _FakeResp:
        status = 500
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    orig_urlopen = appmod.urlopen
    appmod.urlopen = lambda req, timeout=None: _FakeResp()
    try:
        ok, reason = appmod._flush_annotator(5080)
        check(ok is False and reason == "HTTP 500", f"expected (False, 'HTTP 500'), got {(ok, reason)}")
    finally:
        appmod.urlopen = orig_urlopen


def test_flush_annotator_reports_http_error_code_without_raising():
    def _raise(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    orig_urlopen = appmod.urlopen
    appmod.urlopen = _raise
    try:
        ok, reason = appmod._flush_annotator(5080)
        check(ok is False and reason == "HTTP 404", f"expected (False, 'HTTP 404'), got {(ok, reason)}")
    finally:
        appmod.urlopen = orig_urlopen


def test_flush_annotator_reports_connection_failure_without_raising():
    def _raise(req, timeout=None):
        raise appmod.URLError("Connection refused")

    orig_urlopen = appmod.urlopen
    appmod.urlopen = _raise
    try:
        ok, reason = appmod._flush_annotator(5080)
        check(ok is False, f"expected failure, got {(ok, reason)}")
        check(bool(reason), "a failure must carry a non-empty reason to log")
    finally:
        appmod.urlopen = orig_urlopen


# ── _listener_port_for_pid: the pid -> port reverse lookup ───────────────────

SS_SAMPLE = "\n".join([
    "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process",
    'LISTEN 0      128          127.0.0.1:5085       0.0.0.0:*    users:(("python",pid=9000001,fd=4))',
    'LISTEN 0      128            0.0.0.0:5080       0.0.0.0:*    users:(("python",pid=9000002,fd=49))',
    'LISTEN 0      128            0.0.0.0:5050       0.0.0.0:*    users:(("python",pid=9000003,fd=19))',
])


def test_listener_port_for_pid_reverse_lookup():
    check(appmod._listener_port_for_pid(SS_SAMPLE, 9000002) == 5080,
          f"expected port 5080 for pid 9000002, got {appmod._listener_port_for_pid(SS_SAMPLE, 9000002)}")
    check(appmod._listener_port_for_pid(SS_SAMPLE, 9000001) == 5085,
          "and off a loopback-bound line")
    check(appmod._listener_port_for_pid(SS_SAMPLE, 424242) is None,
          "an unlisted pid returns None")
    check(appmod._listener_port_for_pid("", 9000002) is None
          and appmod._listener_port_for_pid(None, 9000002) is None,
          "empty or missing ss output never raises")


def test_listener_port_for_pid_no_numeric_prefix_false_match():
    """pid=900 must never match a line that actually carries pid=9000."""
    sample = 'LISTEN 0 128 0.0.0.0:5080 0.0.0.0:* users:(("python",pid=9000,fd=1))'
    check(appmod._listener_port_for_pid(sample, 900) is None,
          "a numeric prefix must not match a longer pid")
    check(appmod._listener_port_for_pid(sample, 9000) == 5080,
          "the exact pid still matches")


# ── _kill_all_annotators: flush before every kill, on every path ────────────
# Driven through /api/step/editmasks/run (same technique test_annotate_routes.py
# uses): _kill_all_annotators is a private closure with no module-level handle,
# but _flush_annotator, _proc_environ, _proc_cmdline and _annotator_pids are
# plain module-level functions the closure looks up by name at call time, so
# patching appmod.<name> reaches inside it.

def test_kill_all_annotators_flushes_runner_tracked_ports_before_kill():
    client, tmp, state, export_dir = _client_with_project()
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        f.write("{}")

    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.ports = {"step4test": 5080, "editmasks": 5083}
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill = lambda step: calls.append(("kill", step))
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5083}
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code == 200, f"expected 200, got {r.status_code}: {r.get_json()}")

        check(("flush", 5080) in calls, f"step4test's tracked port must be flushed: {calls}")
        check(("flush", 5083) in calls, f"editmasks' tracked port must be flushed: {calls}")
        check(calls.index(("flush", 5080)) < calls.index(("kill", "step4test")),
              f"step4test flush must precede its kill: {calls}")
        check(calls.index(("flush", 5083)) < calls.index(("kill", "editmasks")),
              f"editmasks flush must precede its kill: {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp)


def test_kill_all_annotators_flushes_zombie_pid_by_port_and_spares_standalone():
    """A pid found only via pgrep (not runner-tracked) is flushed by its own
    listening port, reverse-looked-up from the ss snapshot, before its
    SIGTERM. A standalone session (pid 4001) gets neither."""
    client, tmp, state, export_dir = _client_with_project()
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        f.write("{}")

    ss_sample = "\n".join([
        'LISTEN 0 128 0.0.0.0:5080 0.0.0.0:* users:(("python",pid=4001,fd=1))',
        'LISTEN 0 128 0.0.0.0:5081 0.0.0.0:* users:(("python",pid=4002,fd=1))',
    ])

    def _fake_subprocess_run(cmd, *a, **k):
        if cmd and cmd[0] == "pgrep" and len(cmd) > 2 and "combinedAnnotate" in cmd[2]:
            return type("R", (), {"returncode": 0, "stdout": "4001\n4002\n", "stderr": ""})()
        if cmd and cmd[0] == "ss":
            return type("R", (), {"returncode": 0, "stdout": ss_sample, "stderr": ""})()
        return _FakeCompleted()

    def _fake_proc_environ(pid):
        return b"TCRMP_SESSION_MODE=standalone\0" if pid == 4001 else b""

    calls = []
    orig_subprocess_run = subprocess.run
    orig_proc_environ = appmod._proc_environ
    orig_flush = appmod._flush_annotator
    orig_os_kill = os.kill
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.ports = {}
    subprocess.run = _fake_subprocess_run
    appmod._proc_environ = _fake_proc_environ
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    os.kill = lambda pid, sig: calls.append(("kill", pid, sig))
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5083}
    appmod.find_free_port = lambda preferred: preferred
    try:
        # _proc_cmdline is left real on purpose (as test_annotate_routes.py's
        # own equivalent scenario does): pids 4001/4002 do not exist on this
        # box, so /proc/<pid>/cmdline reads "" and the SEPARATE port-sweep
        # loop's cmdline check skips both, isolating this assertion to the
        # pgrep pass alone.
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code == 200, f"expected 200, got {r.status_code}: {r.get_json()}")

        check(("flush", 5080) not in calls, f"the standalone pid's port must never be flushed: {calls}")
        check(("kill", 4001, signal.SIGTERM) not in calls, f"the standalone pid must never be killed: {calls}")
        check(("flush", 5081) in calls, f"the non-standalone pid's port must be flushed: {calls}")
        check(calls.index(("flush", 5081)) < calls.index(("kill", 4002, signal.SIGTERM)),
              f"flush must precede kill for the non-standalone pid: {calls}")
    finally:
        subprocess.run = orig_subprocess_run
        appmod._proc_environ = orig_proc_environ
        appmod._flush_annotator = orig_flush
        os.kill = orig_os_kill
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp)


def test_kill_all_annotators_port_sweep_flushes_before_kill_spares_vicariusllm():
    """Mirrors test_annotate_routes.py's port-sweep regression: only the
    cmdline-matched annotator listener (9000002 on 5080) is flushed and
    killed; vicariusLLM on 5085 (9000001) is left alone."""
    client, tmp, state, export_dir = _client_with_project()
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        f.write("{}")

    ann_entry = os.path.join("TCRMPclip_combinedAnnotate", "src", "app.py")
    cmdlines = {
        9000001: "/mnt/rip/vicarius_drive/vicarius/modules/vicarius_llm/env/bin/python "
                 "/mnt/rip/vicarius_drive/vicarius/modules/vicarius_llm/src/app.py",
        9000002: "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python "
                 "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/" + ann_entry,
    }

    def _fake_ss_run(cmd, *a, **k):
        if cmd and cmd[0] == "ss":
            return type("R", (), {"returncode": 0, "stdout": SS_SAMPLE, "stderr": ""})()
        return _FakeCompleted()  # pgrep finds nothing: only the port half runs

    calls = []
    orig_run = subprocess.run
    orig_cmdline = appmod._proc_cmdline
    orig_environ = appmod._proc_environ
    orig_kill = os.kill
    orig_flush = appmod._flush_annotator
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.ports = {}
    subprocess.run = _fake_ss_run
    appmod._proc_cmdline = lambda pid: cmdlines.get(pid, "")
    appmod._proc_environ = lambda pid: b""
    os.kill = lambda pid, sig: calls.append(("kill", pid, sig))
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5083}
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code == 200, f"expected 200, got {r.status_code}: {r.get_json()}")

        check(("flush", 5085) not in calls, f"vicariusLLM's port must never be flushed: {calls}")
        check(("kill", 9000001, signal.SIGTERM) not in calls, f"vicariusLLM must never be killed: {calls}")
        check(("flush", 5080) in calls, f"the annotator listener's port must be flushed: {calls}")
        check(calls.index(("flush", 5080)) < calls.index(("kill", 9000002, signal.SIGTERM)),
              f"flush must precede kill for the annotator listener: {calls}")
    finally:
        subprocess.run = orig_run
        appmod._proc_cmdline = orig_cmdline
        appmod._proc_environ = orig_environ
        os.kill = orig_kill
        appmod._flush_annotator = orig_flush
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp)


def test_kill_all_annotators_flushes_each_port_at_most_once():
    """The runner-tracked port for step4test (5080) is also the port a zombie
    pid is found listening on via the port sweep, flushed once, not twice."""
    client, tmp, state, export_dir = _client_with_project()
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        f.write("{}")

    ann_entry = os.path.join("TCRMPclip_combinedAnnotate", "src", "app.py")
    ss_sample = ('LISTEN 0 128 0.0.0.0:5080 0.0.0.0:* '
                 'users:(("python",pid=7001,fd=1))')

    def _fake_ss_run(cmd, *a, **k):
        if cmd and cmd[0] == "ss":
            return type("R", (), {"returncode": 0, "stdout": ss_sample, "stderr": ""})()
        return _FakeCompleted()

    flush_calls = []
    orig_run = subprocess.run
    orig_cmdline = appmod._proc_cmdline
    orig_environ = appmod._proc_environ
    orig_kill = os.kill
    orig_runner_kill = appmod.runner.kill
    orig_flush = appmod._flush_annotator
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.ports = {"step4test": 5080}
    subprocess.run = _fake_ss_run
    appmod._proc_cmdline = lambda pid: (
        "/mnt/rip/.../scripts/" + ann_entry if pid == 7001 else "")
    appmod._proc_environ = lambda pid: b""
    os.kill = lambda pid, sig: None
    appmod.runner.kill = lambda step: None
    appmod._flush_annotator = lambda port: (flush_calls.append(port) or (True, None))
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5083}
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code == 200, f"expected 200, got {r.status_code}: {r.get_json()}")
        check(flush_calls.count(5080) == 1,
              f"port 5080 must be flushed exactly once per call, got {flush_calls}")
    finally:
        subprocess.run = orig_run
        appmod._proc_cmdline = orig_cmdline
        appmod._proc_environ = orig_environ
        os.kill = orig_kill
        appmod.runner.kill = orig_runner_kill
        appmod._flush_annotator = orig_flush
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp)


# ── Stop-path routes: flush before kill, a failed flush never blocks ────────

def test_step4test_stop_flushes_before_kill():
    client, tmp, state, export_dir = _client_with_project()
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    appmod.runner.ports = {"step4test": 5080}
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill = lambda step: calls.append(("kill", step))
    try:
        r = client.post("/api/step/step4test/stop", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(calls == [("flush", 5080), ("kill", "step4test_route"), ("kill", "step4test")]
              or calls.index(("flush", 5080)) < calls.index(("kill", "step4test")),
              f"flush must precede the annotator kill: {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        _teardown(tmp)


def test_step4test_stop_failed_flush_is_logged_and_stop_still_succeeds():
    client, tmp, state, export_dir = _client_with_project()
    logged = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    orig_note = appmod.vic.note
    killed = []
    appmod.runner.ports = {"step4test": 5080}
    appmod._flush_annotator = lambda port: (False, "timed out")
    appmod.runner.kill = lambda step: killed.append(step)
    appmod.vic.note = lambda msg, study=None: logged.append(msg)
    try:
        r = client.post("/api/step/step4test/stop", json={})
        data = r.get_json()
        check(r.status_code == 200, f"expected 200: {data}")
        check(data.get("success") is True, f"a failed flush must never block the stop: {data}")
        check("step4test" in killed, f"the kill must still happen: {killed}")
        check(any("5080" in m and "timed out" in m for m in logged),
              f"the failure must be logged with port and reason: {logged}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        appmod.vic.note = orig_note
        _teardown(tmp)


def test_step4test_stop_skips_flush_when_state_dict_remembers_a_port_but_runner_never_tracked_it():
    """A remembered step4test_state['port'] alone proves nothing: runner never
    tracked the key at all here (is_running would be False on a real runner),
    so no flush is attempted, only the kill (a harmless no-op on nothing
    running). Closes the "reused port" review finding: flushing on the
    state dict's say-so, with no runner confirmation, could POST to whatever
    unrelated process later bound that same port number."""
    client, tmp, state, export_dir = _client_with_project()
    appmod.step4test_state["port"] = 5080
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    appmod.runner.ports = {}
    appmod.runner.is_running = lambda step: False
    appmod._flush_annotator = lambda port: (calls.append(port) or (True, None))
    appmod.runner.kill = lambda step: None
    try:
        r = client.post("/api/step/step4test/stop", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(calls == [], f"an unconfirmed-alive port must never be flushed, got {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        _teardown(tmp)


def test_step4test_stop_skips_flush_when_runner_ports_has_a_dead_key():
    """runner.ports[step] is never cleared by kill() (stage_runner.py), so a
    dead process's OLD port can still sit in the dict after it exits and be
    reused by an unrelated listener. is_running(step) False must suppress
    the flush even though .ports still names a port."""
    client, tmp, state, export_dir = _client_with_project()
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    appmod.runner.ports = {"step4test": 5080}
    appmod.runner.is_running = lambda step: False
    appmod._flush_annotator = lambda port: (calls.append(port) or (True, None))
    appmod.runner.kill = lambda step: None
    try:
        r = client.post("/api/step/step4test/stop", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(calls == [], f"a dead-but-still-tracked port must never be flushed, got {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        _teardown(tmp)


class _RunnerWithoutIsRunning:
    """Delegates every attribute to the wrapped runner EXCEPT is_running,
    which it genuinely does not have (not merely set to None), so
    getattr(runner, "is_running", None) truly falls back to its default.
    Proves the getattr-safe fallback fails closed for a runner-like object
    that never implements a liveness probe at all."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        if name == "is_running":
            raise AttributeError(name)
        return getattr(self._real, name)


def test_step4test_stop_skips_flush_when_runner_has_no_is_running_at_all():
    """A test double or a future runner implementation without is_running is
    treated as "cannot confirm alive", so nothing is flushed: the
    getattr-safe fallback the review asked for is fail-closed, not
    fail-open."""
    client, tmp, state, export_dir = _client_with_project()
    calls = []
    orig_flush = appmod._flush_annotator
    real_runner = appmod.runner
    orig_kill = real_runner.kill
    real_runner.ports = {"step4test": 5080}
    real_runner.kill = lambda step: None
    appmod._flush_annotator = lambda port: (calls.append(port) or (True, None))
    appmod.runner = _RunnerWithoutIsRunning(real_runner)
    try:
        r = client.post("/api/step/step4test/stop", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(calls == [], f"a runner with no is_running must never be flushed against, got {calls}")
    finally:
        appmod.runner = real_runner
        appmod.runner.kill = orig_kill
        appmod._flush_annotator = orig_flush
        _teardown(tmp)


def test_step4test_reset_flushes_before_kill():
    client, tmp, state, export_dir = _client_with_project()
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    appmod.runner.ports = {"step4test": 5081}
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill = lambda step: calls.append(("kill", step))
    try:
        r = client.post("/api/step/step4test/reset", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(("flush", 5081) in calls, f"expected a flush of port 5081: {calls}")
        check(calls.index(("flush", 5081)) < calls.index(("kill", "step4test")),
              f"flush must precede the annotator kill: {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        _teardown(tmp)


def test_step4loop_reset_flushes_before_kill():
    client, tmp, state, export_dir = _client_with_project()
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    appmod.runner.ports = {"step4loop": 5081}
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill = lambda step: calls.append(("kill", step))
    try:
        r = client.post("/api/step/step4loop/reset", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(("flush", 5081) in calls, f"expected a flush of port 5081: {calls}")
        check(calls.index(("flush", 5081)) < calls.index(("kill", "step4loop")),
              f"flush must precede the Refine annotator kill: {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        _teardown(tmp)


def test_editmasks_stop_flushes_before_kill():
    client, tmp, state, export_dir = _client_with_project()
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    appmod.runner.ports = {"editmasks": 5083}
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill = lambda step: calls.append(("kill", step))
    try:
        r = client.post("/api/step/editmasks/stop", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(("flush", 5083) in calls, f"expected a flush of port 5083: {calls}")
        check(calls.index(("flush", 5083)) < calls.index(("kill", "editmasks")),
              f"flush must precede the Edit Masks kill: {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        _teardown(tmp)


def test_editmasks_reset_flushes_before_kill_and_never_touches_export():
    client, tmp, state, export_dir = _client_with_project()
    seg_path = os.path.join(export_dir, "segmentations", "segmentations.json")
    os.makedirs(os.path.dirname(seg_path), exist_ok=True)
    with open(seg_path, "w") as f:
        json.dump({"img0.jpeg": {"masks": [{"id": 0}], "reviewed": True, "exported": True}}, f)
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    appmod.runner.ports = {"editmasks": 5083}
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill = lambda step: calls.append(("kill", step))
    try:
        with open(seg_path) as f:
            before = json.load(f)
        r = client.post("/api/step/editmasks/reset", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        check(("flush", 5083) in calls, f"expected a flush of port 5083: {calls}")
        check(calls.index(("flush", 5083)) < calls.index(("kill", "editmasks")),
              f"flush must precede the Edit Masks kill: {calls}")
        with open(seg_path) as f:
            after = json.load(f)
        check(after == before, "editmasks reset must still never touch the export")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        _teardown(tmp)


def test_project_quit_flushes_every_tracked_annotator_before_kill_all():
    client, tmp, state, export_dir = _client_with_project()
    calls = []
    orig_flush = appmod._flush_annotator
    orig_kill_all = appmod.runner.kill_all
    appmod.runner.ports = {"step4test": 5080, "step4loop": 5081, "editmasks": 5083}
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill_all = lambda: calls.append(("kill_all",))
    try:
        r = client.post("/api/project/quit", json={})
        check(r.status_code == 200, f"expected 200: {r.get_json()}")
        for expected_port in (5080, 5081, 5083):
            check(("flush", expected_port) in calls, f"expected a flush of port {expected_port}: {calls}")
            check(calls.index(("flush", expected_port)) < calls.index(("kill_all",)),
                  f"every tracked annotator must be flushed before kill_all: {calls}")
    finally:
        appmod._flush_annotator = orig_flush
        appmod.runner.kill_all = orig_kill_all
        _teardown(tmp)


# ── step4loop/run: flush AND fully stop before the seeder writes ───────────

def _make_predictions(step8_dir, name, n_items=1):
    rdir = os.path.join(step8_dir, name)
    os.makedirs(rdir, exist_ok=True)
    items = [{"filename": f"img{i}.jpeg", "raw": os.path.join(rdir, f"img{i}.jpeg"),
              "width": 100, "height": 100, "detections": []} for i in range(n_items)]
    for it in items:
        open(it["raw"], "w").close()
    with open(os.path.join(rdir, "predictions.json"), "w") as f:
        json.dump({"schema_version": 1, "generated_at": "2026-09-27T00:00:00-04:00",
                   "items": items}, f)
    return rdir


def _project_with_frozen_split():
    tmp = tempfile.mkdtemp(prefix="flush_before_kill_step4loop_")
    state, export_dir = _make_project(tmp)
    dataset_dir = os.path.join(state["steps"]["6"]["dir"], "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, "split_manifest.json"), "w") as f:
        json.dump({"holdout_mode": "transect", "holdout_transects": {"val": [5], "test": [6]},
                   "pinned": {}}, f)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    appmod.step4loop_cfg = {}
    appmod.step4test_state = _fresh_state_dict()
    appmod.step4loop_state = _fresh_state_dict()
    appmod.runner.is_running = lambda step: True
    return client, tmp, state, export_dir


def test_step4loop_run_flushes_and_stops_live_annotators_before_seeder_runs():
    """The seeder rewrites segmentations.json directly; flush_all() in the
    annotator is a blind dump of its in-memory dict, so a live step4test/
    step4loop process must be flushed AND fully stopped before the seeder
    ever runs, not merely flushed after."""
    client, tmp, state, export_dir = _project_with_frozen_split()
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        json.dump({"img0.jpeg": {"masks": [], "reviewed": False, "exported": False}}, f)

    calls = []

    def _fake_subprocess_run(cmd, **kwargs):
        if cmd and len(cmd) > 1 and "seed_from_predictions.py" in str(cmd[1]):
            calls.append(("seed_run",))
            class _R:
                returncode = 0
                stdout = json.dumps({
                    "seeded_frames": 1, "seeded_masks": 3, "skipped_existing": 0,
                    "skipped_below_conf": 0, "skipped_missing_raw": 0,
                    "skipped_holdout": 0, "round_manifest": "loop_rounds/round_x.json",
                })
                stderr = ""
            return _R()
        return _FakeCompleted()

    orig_subprocess_run = subprocess.run
    orig_flush = appmod._flush_annotator
    orig_kill = appmod.runner.kill
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.ports = {"step4test": 5080, "step4loop": 5081}
    subprocess.run = _fake_subprocess_run
    appmod._flush_annotator = lambda port: (calls.append(("flush", port)) or (True, None))
    appmod.runner.kill = lambda step: calls.append(("kill", step))
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5081}
    appmod.find_free_port = lambda preferred: preferred
    try:
        step8_dir = state["steps"]["8"]["dir"]
        pred_dir = _make_predictions(step8_dir, "infer_x")
        r = client.post("/api/step/step4loop/run", json={"config": {"predictions_dir": pred_dir}})
        data = r.get_json()
        check(r.status_code == 200, f"expected 200: {data}")
        check(data.get("success") is True, f"expected success: {data}")

        check(("flush", 5080) in calls, f"step4test's tracked port must be flushed: {calls}")
        check(("flush", 5081) in calls, f"step4loop's tracked port must be flushed: {calls}")
        check(("kill", "step4test") in calls, f"step4test must be fully stopped: {calls}")
        check(("kill", "step4loop") in calls, f"step4loop must be fully stopped: {calls}")
        check(("seed_run",) in calls, f"the seeder must still run: {calls}")

        seed_index = calls.index(("seed_run",))
        for tag in (("flush", 5080), ("flush", 5081), ("kill", "step4test"), ("kill", "step4loop")):
            check(calls.index(tag) < seed_index,
                  f"{tag} must happen before the seeder runs, got order {calls}")
    finally:
        subprocess.run = orig_subprocess_run
        appmod._flush_annotator = orig_flush
        appmod.runner.kill = orig_kill
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp)


def test_step4loop_run_with_no_live_annotator_skips_flush_and_still_seeds():
    """When neither step4test nor step4loop is tracked (the common case: the
    operator closed the annotator before choosing a new inference run to
    seed), no flush is attempted and the seeder still runs normally."""
    client, tmp, state, export_dir = _project_with_frozen_split()
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        json.dump({"img0.jpeg": {"masks": [], "reviewed": False, "exported": False}}, f)

    flush_calls = []

    def _fake_subprocess_run(cmd, **kwargs):
        if cmd and len(cmd) > 1 and "seed_from_predictions.py" in str(cmd[1]):
            class _R:
                returncode = 0
                stdout = json.dumps({
                    "seeded_frames": 1, "seeded_masks": 1, "skipped_existing": 0,
                    "skipped_below_conf": 0, "skipped_missing_raw": 0,
                    "skipped_holdout": 0, "round_manifest": None,
                })
                stderr = ""
            return _R()
        return _FakeCompleted()

    orig_subprocess_run = subprocess.run
    orig_flush = appmod._flush_annotator
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.ports = {}
    subprocess.run = _fake_subprocess_run
    appmod._flush_annotator = lambda port: (flush_calls.append(port) or (True, None))
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5080}
    appmod.find_free_port = lambda preferred: preferred
    try:
        step8_dir = state["steps"]["8"]["dir"]
        pred_dir = _make_predictions(step8_dir, "infer_x")
        r = client.post("/api/step/step4loop/run", json={"config": {"predictions_dir": pred_dir}})
        data = r.get_json()
        check(r.status_code == 200, f"expected 200: {data}")
        check(data.get("success") is True, f"expected success: {data}")
        check(flush_calls == [], f"no annotator is tracked, so no flush should fire: {flush_calls}")
    finally:
        subprocess.run = orig_subprocess_run
        appmod._flush_annotator = orig_flush
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp)


# ── Shell scripts: flush precedes every kill (checked statically) ──────────
# restart_reef_point_seg.sh and run_pipeline.sh are launchers, not something
# this suite may execute for real: both hard-code real, fixed ports (5055,
# 5065, 5066, 5070, 5075, 5080) that they fuser-kill/lsof-kill regardless of
# any $PORT argument, and restart_reef_point_seg.sh spawns a real orchestrator
# process via setsid. Stubbing every external command each one calls (ss,
# fuser, pkill, curl, setsid, lsof, kill, xdg-open, tail, date, sleep) well
# enough to run them for real, on a box where another session's GPU training
# job is the live orchestrator's own parent process, is more risk than the
# assertion is worth, so this checks the shipped script TEXT instead: each
# flush loop's line number must come before every kill/fuser/pkill/lsof
# line's. `bash -n` (below) only parses the script; it never executes a line
# of it, so that one check uses the real subprocess.run, captured above
# before the module-wide stub.

SCRIPTS_DIR = os.path.dirname(_ORCH)
ANNOTATOR_FLUSH_PORTS_LITERAL = (
    "(5080 5081 5082 5083 5084 5085 5086 5087 5088 5089 5096)"
)


def _read_script(name):
    with open(os.path.join(SCRIPTS_DIR, name)) as f:
        return f.read()


def _first_line_with(lines, needle):
    for i, line in enumerate(lines):
        if needle in line:
            return i
    return None


def _bash_syntax_check(path):
    """`bash -n <path>` (parse only, never executes a line) via the real
    Popen captured before the module-wide stub. Returns (returncode, stderr)."""
    proc = _REAL_POPEN(["bash", "-n", path], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)
    _, stderr = proc.communicate(timeout=10)
    return proc.returncode, stderr


def test_restart_script_is_valid_bash_and_flushes_before_every_kill():
    path = os.path.join(SCRIPTS_DIR, "restart_reef_point_seg.sh")
    returncode, stderr = _bash_syntax_check(path)
    check(returncode == 0, f"restart_reef_point_seg.sh must be valid bash: {stderr}")

    text = _read_script("restart_reef_point_seg.sh")
    check(ANNOTATOR_FLUSH_PORTS_LITERAL in text,
          "expected the full 5080-5089 + 5096 flush port list")
    check("curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST \"http://127.0.0.1:${fp}/api/flush\""
          in text, "expected a best-effort POST /api/flush per candidate port")

    lines = text.splitlines()
    flush_idx = _first_line_with(lines, "ANNOTATOR_FLUSH_PORTS=")
    check(flush_idx is not None, "expected an ANNOTATOR_FLUSH_PORTS flush block")
    kill_needles = ('pkill -9 -f "run_pipeline.py"',
                     'pkill -9 -f "pipeline_orchestrator/app.py"',
                     'fuser -k -9 "${p}/tcp"')
    for needle in kill_needles:
        kill_idx = _first_line_with(lines, needle)
        check(kill_idx is not None, f"expected to find the kill line: {needle}")
        check(flush_idx < kill_idx,
              f"the flush block (line {flush_idx}) must precede '{needle}' (line {kill_idx})")

    # 5096 (the standalone "Annotate Images" session) is flushed but never
    # killed by this script: MODULE_PORTS is the actual kill target list.
    check("5096" not in _read_script("restart_reef_point_seg.sh").split("MODULE_PORTS=(")[1].split(")")[0],
          "5096 must stay out of MODULE_PORTS: this restart never kills a standalone session")


def test_run_pipeline_script_is_valid_bash_and_flushes_before_every_kill():
    path = os.path.join(SCRIPTS_DIR, "run_pipeline.sh")
    returncode, stderr = _bash_syntax_check(path)
    check(returncode == 0, f"run_pipeline.sh must be valid bash: {stderr}")

    text = _read_script("run_pipeline.sh")
    check(ANNOTATOR_FLUSH_PORTS_LITERAL in text,
          "expected the full 5080-5089 + 5096 flush port list")
    check("curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST \"http://127.0.0.1:${fp}/api/flush\""
          in text, "expected a best-effort POST /api/flush per candidate port")

    lines = text.splitlines()
    flush_idx = _first_line_with(lines, "ANNOTATOR_FLUSH_PORTS=")
    check(flush_idx is not None, "expected an ANNOTATOR_FLUSH_PORTS flush block")
    kill_needles = ('kill "$pid" 2>/dev/null && echo "    killed pid $pid ($pat)"',
                     'kill "$pid" 2>/dev/null && echo "    killed pid $pid (port $p)"',
                     'pgrep -f "$pat" 2>/dev/null | xargs -r kill -9 2>/dev/null')
    for needle in kill_needles:
        kill_idx = _first_line_with(lines, needle)
        check(kill_idx is not None, f"expected to find the kill line: {needle}")
        check(flush_idx < kill_idx,
              f"the flush block (line {flush_idx}) must precede '{needle}' (line {kill_idx})")


def test_shell_scripts_only_flush_annotator_owned_ports():
    """Neither script may flush a port on runner.ports' say-so alone: each
    must resolve the listening pid (ss -ltnp) and check its
    /proc/<pid>/cmdline against the same two entry scripts
    _kill_all_annotators checks in pipeline_orchestrator/app.py, before the
    curl. This is the review fix for flushing an unrelated listener sharing
    the port range (benthic_species_viewer's serve_edit.py on 5081 today)."""
    for name in ("restart_reef_point_seg.sh", "run_pipeline.sh"):
        text = _read_script(name)
        check("ss -ltnp" in text, f"{name}: expected a pid-carrying ss -ltnp lookup")
        check('TCRMPclip_combinedAnnotate/src/app.py' in text and
              'TCRMPclip_editMasks/src/app.py' in text,
              f"{name}: expected both annotator entry scripts in the owner check")
        check('/proc/${fp_pid}/cmdline' in text,
              f"{name}: expected the listening pid's cmdline to be read")

        lines = text.splitlines()
        ss_idx = _first_line_with(lines, "ss -ltnp")
        cmdline_idx = _first_line_with(lines, '/proc/${fp_pid}/cmdline')
        curl_idx = _first_line_with(lines, "curl -s -m 5 -o /dev/null -w '%{http_code}'")
        for idx, label in ((ss_idx, "the ss lookup"), (cmdline_idx, "the cmdline read")):
            check(idx is not None, f"{name}: expected to find {label}")
            check(idx < curl_idx,
                  f"{name}: {label} (line {idx}) must precede the curl (line {curl_idx})")


def test_flush_helper_quotes_its_shell_variables():
    """Every use of the new variables must be inside double quotes, so an
    unset or space-carrying value can never split into extra curl/ss/sed
    words."""
    for name in ("restart_reef_point_seg.sh", "run_pipeline.sh"):
        text = _read_script(name)
        for bad in ("=$fp", " $fp ", "=$flush_code", " $flush_code ",
                    "=$fp_pid", " $fp_pid ", "=$fp_cmdline", " $fp_cmdline ",
                    "=$fp_ss_line", " $fp_ss_line "):
            check(bad not in text, f"{name}: {bad!r} looks unquoted")


if __name__ == "__main__":
    print("2026-09-27 SAFETY NET T2 -- orchestrator saves before it stops\n")
    print("_flush_annotator:")
    run(test_flush_annotator_no_port_fails_closed_without_network)
    run(test_flush_annotator_success_posts_to_api_flush_with_named_timeout)
    run(test_flush_annotator_reports_non_2xx_status)
    run(test_flush_annotator_reports_http_error_code_without_raising)
    run(test_flush_annotator_reports_connection_failure_without_raising)
    print("_listener_port_for_pid:")
    run(test_listener_port_for_pid_reverse_lookup)
    run(test_listener_port_for_pid_no_numeric_prefix_false_match)
    print("_kill_all_annotators:")
    run(test_kill_all_annotators_flushes_runner_tracked_ports_before_kill)
    run(test_kill_all_annotators_flushes_zombie_pid_by_port_and_spares_standalone)
    run(test_kill_all_annotators_port_sweep_flushes_before_kill_spares_vicariusllm)
    run(test_kill_all_annotators_flushes_each_port_at_most_once)
    print("stop-path routes:")
    run(test_step4test_stop_flushes_before_kill)
    run(test_step4test_stop_failed_flush_is_logged_and_stop_still_succeeds)
    print("alive-check gate (is_running before a tracked-port flush):")
    run(test_step4test_stop_skips_flush_when_state_dict_remembers_a_port_but_runner_never_tracked_it)
    run(test_step4test_stop_skips_flush_when_runner_ports_has_a_dead_key)
    run(test_step4test_stop_skips_flush_when_runner_has_no_is_running_at_all)
    print("stop-path routes (continued):")
    run(test_step4test_reset_flushes_before_kill)
    run(test_step4loop_reset_flushes_before_kill)
    run(test_editmasks_stop_flushes_before_kill)
    run(test_editmasks_reset_flushes_before_kill_and_never_touches_export)
    run(test_project_quit_flushes_every_tracked_annotator_before_kill_all)
    print("step4loop/run (flush+stop precede the seeder):")
    run(test_step4loop_run_flushes_and_stops_live_annotators_before_seeder_runs)
    run(test_step4loop_run_with_no_live_annotator_skips_flush_and_still_seeds)
    print("shell scripts (static order check, never executed):")
    run(test_restart_script_is_valid_bash_and_flushes_before_every_kill)
    run(test_run_pipeline_script_is_valid_bash_and_flushes_before_every_kill)
    run(test_shell_scripts_only_flush_annotator_owned_ports)
    run(test_flush_helper_quotes_its_shell_variables)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
