"""A finished run is recorded against the project that launched it.

The stage runner holds ONE process per step for the whole orchestrator, so a
training started in one project finishes while another may be open. Until
2026-09-16 the status poll wrote that result into whichever project happened
to be open: open project A, start its training, open project B while it runs,
and B's own poller recorded A's training in B's project.json as completed
work B never did.

The same poll is where a restart used to lie. The restart script kills the
orchestrator with SIGKILL and a training child spawned in its own session
survives it, so after a restart the runner holds no process and reports the
step as not running with no exit code at all. Reading that as a failure
marked a training still writing into its run folder as errored on the first
poll after every restart.

Covered here:
  ATTRIBUTION  a step launched by another project is recorded there, not on
               the open one, for both a success and a failure; a project
               polling its own launch still completes normally; a launch
               record left by an earlier press that nobody polled is not read
               as the owner of a later process.
  RESTART      a step whose process the runner no longer holds is neither
               completed nor failed; the kernel is consulted for the process
               and the poll reports what is actually known.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_run_attribution.py

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level and the stage runner's launch is replaced by a
capture that registers a finished stand-in process. Fixtures live under one
temporary folder each and are removed at the end, which is verified.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []
_FIXTURES = []


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# Standing rule: no orchestrator test may reach a real subprocess. A launch
# route in this file must never spawn anything, and must never be able to
# reach a live trainer.
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402


class _FakeVic:
    @staticmethod
    def process_start(*_a, **_k):
        return None

    @staticmethod
    def process_end(*_a, **_k):
        return None

    @staticmethod
    def note(*_a, **_k):
        return None


appmod.vic = _FakeVic()

_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()
# A pid no real process on this box carries, so nothing here can name a live
# one. The stand-in process below never runs; only its pid is read.
FAKE_PID = 987654321
STEP = 1


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


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

class _FinishedProcess:
    """A stand-in for a subprocess that has already exited.

    The stage runner's poll_status only calls poll() and reads pid, so this
    is everything a finished process needs to be. Nothing is spawned.
    """

    def __init__(self, pid, returncode):
        self.pid = pid
        self._returncode = returncode

    def poll(self):
        return self._returncode


def _make_project(name):
    """(state, tmp) for a project on disk with every step folder created."""
    tmp = tempfile.mkdtemp(prefix=f"attr_{name}_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    state = {"id": name, "name": name, "project_dir": tmp,
             "created_at": "2026-09-16T09:00:00-04:00",
             "updated_at": "2026-09-16T09:00:00-04:00",
             "current_step": 1, "steps": steps}
    pm.save_project(state)
    return state, tmp


def _launch(client, state, returncode=0, pid=FAKE_PID):
    """Press Run on STEP for `state`, with the runner replaced by a capture
    that registers a finished stand-in process. Returns that process."""
    proc = _FinishedProcess(pid, returncode)
    real = appmod.runner.run_cli_stage

    def _fake_run(step, cmd, cwd=None, env_extra=None):
        appmod.runner.processes[step] = proc
        return {"pid": proc.pid}

    appmod.runner.run_cli_stage = _fake_run
    appmod.current_project = state
    try:
        r = client.post(f"/api/step/{STEP}/run")
        check(r.status_code == 200, f"launch refused: {r.status_code} {r.get_json()}")
    finally:
        appmod.runner.run_cli_stage = real
    return proc


def _saved(state):
    """project.json as it is on disk right now."""
    with open(os.path.join(state["project_dir"], "project.json")) as fh:
        return json.load(fh)


def _reset_module_state():
    appmod.current_project = None
    appmod.runner.processes.pop(STEP, None)
    appmod._step_launches.clear()


def _close(*tmps):
    _reset_module_state()
    for tmp in tmps:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# ATTRIBUTION
# ---------------------------------------------------------------------------

def test_a_finished_step_is_recorded_on_the_project_that_launched_it():
    a, tmp_a = _make_project("owner")
    b, tmp_b = _make_project("bystander")
    client = appmod.create_app().test_client()
    try:
        _launch(client, a)
        appmod.current_project = b
        b["steps"][str(STEP)]["status"] = "running"

        r = client.get(f"/api/step/{STEP}/status")
        data = r.get_json() or {}
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("completed") is False,
              f"the open project did no work, so nothing completed here: {data}")
        note = " ".join(data.get("warnings") or [])
        check("owner" in note, f"the warning names the project the result went to: {note!r}")

        check(_saved(a)["steps"][str(STEP)]["status"] == "completed",
              f"the launching project records the completion: {_saved(a)['steps'][str(STEP)]}")
        check(b["steps"][str(STEP)]["status"] == "running",
              f"the open project's own step is left exactly as it was: {b['steps'][str(STEP)]}")
        check(_saved(b)["steps"][str(STEP)]["status"] != "completed",
              f"and nothing was written to its project.json: {_saved(b)['steps'][str(STEP)]}")
    finally:
        _close(tmp_a, tmp_b)


def test_a_failed_step_is_recorded_on_the_project_that_launched_it():
    a, tmp_a = _make_project("owner")
    b, tmp_b = _make_project("bystander")
    client = appmod.create_app().test_client()
    try:
        _launch(client, a, returncode=1)
        appmod.current_project = b
        b["steps"][str(STEP)]["status"] = "running"

        data = client.get(f"/api/step/{STEP}/status").get_json() or {}
        check(data.get("completed") is False, f"nothing completed for the open project: {data}")
        check(_saved(a)["steps"][str(STEP)]["status"] == "error",
              f"the failure is recorded on the launching project: {_saved(a)['steps'][str(STEP)]}")
        check(b["steps"][str(STEP)]["status"] == "running",
              f"the open project is untouched: {b['steps'][str(STEP)]}")
    finally:
        _close(tmp_a, tmp_b)


def test_a_project_polling_its_own_launch_completes_normally():
    a, tmp_a = _make_project("owner")
    client = appmod.create_app().test_client()
    try:
        _launch(client, a)
        data = client.get(f"/api/step/{STEP}/status").get_json() or {}
        check(data.get("completed") is True, f"its own work still completes: {data}")
        check(not data.get("warnings"), f"and says nothing about another project: {data}")
        check(a["steps"][str(STEP)]["status"] == "completed",
              f"the open project's own step is recorded: {a['steps'][str(STEP)]}")
    finally:
        _close(tmp_a)


def test_an_unwritable_owner_is_reported_and_nothing_is_recorded():
    """The launching project's folder can be gone by the time the poll
    arrives. Nothing is recorded anywhere and the poll says why."""
    a, tmp_a = _make_project("owner")
    b, tmp_b = _make_project("bystander")
    client = appmod.create_app().test_client()
    try:
        _launch(client, a)
        shutil.rmtree(tmp_a)
        appmod.current_project = b
        b["steps"][str(STEP)]["status"] = "running"

        data = client.get(f"/api/step/{STEP}/status").get_json() or {}
        check(data.get("completed") is False, f"nothing completed for the open project: {data}")
        note = " ".join(data.get("warnings") or [])
        check("could not record" in note, f"the poll says it could not record the result: {note!r}")
        check("project.json" in note, f"and names the file it tried: {note!r}")
        check(b["steps"][str(STEP)]["status"] == "running",
              f"the open project is still untouched: {b['steps'][str(STEP)]}")
    finally:
        _close(tmp_a, tmp_b)


def test_a_stale_launch_record_is_not_read_as_the_owner():
    """A press whose result nobody ever polled leaves a record behind. It
    describes a process the runner no longer holds, so a later poll must not
    credit that project with someone else's process."""
    a, tmp_a = _make_project("owner")
    b, tmp_b = _make_project("bystander")
    client = appmod.create_app().test_client()
    try:
        _launch(client, a)
        # The runner's process table moves on to a different process, as it
        # does when the next press dispatches.
        appmod.runner.processes[STEP] = _FinishedProcess(FAKE_PID + 1, 0)
        appmod.current_project = b
        b["steps"][str(STEP)]["status"] = "running"

        data = client.get(f"/api/step/{STEP}/status").get_json() or {}
        check(data.get("completed") is True,
              f"the stale record is dropped and the open project is credited: {data}")
        check(_saved(a)["steps"][str(STEP)]["status"] != "completed",
              f"and the earlier project is not credited with it: {_saved(a)['steps'][str(STEP)]}")
    finally:
        _close(tmp_a, tmp_b)


# ---------------------------------------------------------------------------
# RESTART
# ---------------------------------------------------------------------------

def test_a_restart_never_marks_a_running_step_as_failed():
    a, tmp_a = _make_project("owner")
    client = appmod.create_app().test_client()
    try:
        appmod.current_project = a
        a["steps"][str(STEP)]["status"] = "running"
        # What a restart leaves: the runner holds no process for the step.
        appmod.runner.processes.pop(STEP, None)

        data = client.get(f"/api/step/{STEP}/status").get_json() or {}
        check(data.get("untracked") is True,
              f"the poll says the process is no longer tracked: {data}")
        check("completed" not in data, f"nothing is claimed as completed: {data}")
        check(a["steps"][str(STEP)]["status"] == "running",
              f"the step is not marked errored: {a['steps'][str(STEP)]}")
        note = " ".join(data.get("warnings") or [])
        check("restarted" in note.lower(), f"the poll says what happened: {note!r}")
        check(a["steps"][str(STEP)]["dir"] in note,
              f"and names the folder to check: {note!r}")
    finally:
        _close(tmp_a)


def test_a_process_that_outlived_the_restart_reads_as_running():
    """The kernel's own process list is what makes status survive a restart:
    a driver still writing into the step folder is reported as running, with
    the pid the orchestrator can no longer stop."""
    a, tmp_a = _make_project("owner")
    client = appmod.create_app().test_client()
    real_probe = appmod._step6_driver_alive_outside_runner
    try:
        appmod.current_project = a
        a["steps"][str(STEP)]["status"] = "running"
        appmod.runner.processes.pop(STEP, None)
        appmod._step6_driver_alive_outside_runner = lambda step_dir, proc_root=None: FAKE_PID

        data = client.get(f"/api/step/{STEP}/status").get_json() or {}
        check(data.get("running") is True, f"a live process reads as running: {data}")
        check(data.get("orphan_pid") == FAKE_PID, f"and names the pid: {data}")
        check(data.get("untracked") is not True,
              f"a process the kernel can see is not 'nobody knows': {data}")
        check(a["steps"][str(STEP)]["status"] == "running",
              f"the step's own record is left alone: {a['steps'][str(STEP)]}")
    finally:
        appmod._step6_driver_alive_outside_runner = real_probe
        _close(tmp_a)



# 2026-09-17 (LO): a finished Train run is stamped with the split it was scored on, at completion.
def test_a_finished_train_run_is_stamped_with_its_split_at_completion():
    import json, os, tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="stamp_at_finish_", dir=_TMP_ROOT if "_TMP_ROOT" in globals() else None)
    try:
        step6 = os.path.join(tmp, "run_p", "step6_trainModel")
        run_dir = os.path.join(step6, "runs", "R1"); os.makedirs(os.path.join(run_dir, "weights"))
        open(os.path.join(run_dir, "weights", "best.pt"), "wb").write(b"x")
        ds = os.path.join(step6, "dataset"); os.makedirs(ds)
        open(os.path.join(ds, "data.yaml"), "w").write("names: {0: a}\n")
        json.dump({"pins": {"a.jpg": "train", "b.jpg": "test"}, "split_fingerprint": "abc123def4567890", "split_fingerprint_scheme": "pins-v1"}, open(os.path.join(ds, "split_manifest.json"), "w"))
        owner = {"name": "P", "project_dir": os.path.join(tmp, "run_p"),
                 "steps": {"6": {"name": "trainModel", "status": "running", "dir": step6, "config": {"run_name": "R1"}}}}
        appmod._stamp_finished_training_run(owner, 6, owner["steps"]["6"])
        rec = appmod._labruns.run_split_record_path(run_dir)
        check(os.path.isfile(rec), "a split record is written into the finished run folder")
        data = json.load(open(rec)) if os.path.isfile(rec) else {}
        check(data.get(appmod._labruns.RUN_SPLIT_RECORDED_SOURCE_FIELD) == appmod._labruns.SPLIT_RECORD_SOURCE_TRAINING, f"the record says it was stamped at training time: {data.get(appmod._labruns.RUN_SPLIT_RECORDED_SOURCE_FIELD)!r}")
        owner["steps"]["6"]["config"]["run_name"] = "missing"
        appmod._stamp_finished_training_run(owner, 6, owner["steps"]["6"])
        check(True, "a run folder that is not on disk is noted, never raised")
        appmod._stamp_finished_training_run(owner, 7, owner["steps"]["6"])
        check(True, "a step other than Train is left alone")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("RUN ATTRIBUTION -- whose project a finished step belongs to\n")
    print("ATTRIBUTION:")
    run(test_a_finished_step_is_recorded_on_the_project_that_launched_it)
    run(test_a_failed_step_is_recorded_on_the_project_that_launched_it)
    run(test_a_project_polling_its_own_launch_completes_normally)
    run(test_an_unwritable_owner_is_reported_and_nothing_is_recorded)
    run(test_a_stale_launch_record_is_not_read_as_the_owner)
    print("RESTART:")
    run(test_a_restart_never_marks_a_running_step_as_failed)
    run(test_a_finished_train_run_is_stamped_with_its_split_at_completion)
    run(test_a_process_that_outlived_the_restart_reads_as_running)

    _reset_module_state()
    for path in _FIXTURES:
        shutil.rmtree(path, ignore_errors=True)
    left = [p for p in _FIXTURES if os.path.exists(p)]
    failed = [name for name, ok, _ in _RESULTS if not ok]
    print()
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAILED {name}:\n{detail}")
    print(f"{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed; fixtures removed: {not left}")
    sys.exit(1 if failed or left else 0)
