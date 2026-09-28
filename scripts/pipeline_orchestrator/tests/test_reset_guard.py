"""Reset and Stop tell the truth about a live training run.

Reset on the Train step empties that step's folder and every folder after it:
the dataset the trainer is reading and the run folder it is writing weights
into both go. Until 2026-09-16 it did that while a trainer was mid-epoch, and
the orphan probe that makes Start Training refuse over a live run was never
consulted on this path. Stop had the matching lie: it answered success while
killing nothing, because the runner forgets its children when the
orchestrator is restarted and kill() on a process it does not hold is a
no-op.

Covered here:
  RESET   refused while a run is live, naming the run, whether the process is
          one the runner holds or one that outlived a restart; refused as
          well when the reset starts at an earlier step, because the range it
          clears reaches Train; allowed, and really clearing, when nothing is
          alive.
  STOP    says what it really did: stopped a process the runner held, cleared
          a stale running mark when nothing was alive, or refused and named
          the pid it cannot stop.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_reset_guard.py

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, the orphan probe is replaced per test, and the
stand-in process the runner is handed is never spawned; the one test that
reaches the runner's kill path replaces kill() with a recorder, so no signal
can reach any process on this machine.
Fixtures are removed at the end and the removal is verified.
"""

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
# The Train step's internal key. The sidebar shows it as step 5; every
# message this suite reads goes through step_ref, so nothing here spells a
# sidebar number itself.
TRAIN_STEP = 6
# A pid no real process on this box carries.
FAKE_PID = 987654321
RUN_NAME = "coral_20260916_090000"


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

class _LiveProcess:
    """A stand-in for a process the runner is holding and that has not exited.

    poll() answering None is the whole of what is_running() reads, and
    nothing is ever spawned. The one test that reaches the runner's kill path
    replaces kill() with a recorder, so no signal can reach any process on
    this machine.
    """

    def __init__(self, pid=FAKE_PID):
        self.pid = pid

    def poll(self):
        return None


def _fixture():
    """A project whose Train step already holds a dataset and a run folder.

    Returns (state, tmp, paths) where paths are the two folders a reset would
    empty, so a test can prove they are still there afterwards.
    """
    tmp = tempfile.mkdtemp(prefix="reset_guard_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    step6 = steps[str(TRAIN_STEP)]["dir"]
    dataset = os.path.join(step6, "dataset")
    run_dir = os.path.join(step6, "runs", RUN_NAME, "weights")
    os.makedirs(dataset, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(dataset, "data.yaml"), "w") as fh:
        fh.write("names:\n  0: PA\n")
    with open(os.path.join(run_dir, "last.pt"), "w") as fh:
        fh.write("x")
    steps[str(TRAIN_STEP)]["config"] = {"run_name": RUN_NAME}
    steps[str(TRAIN_STEP)]["status"] = "running"
    state = {"id": os.path.basename(tmp), "name": "reset fixture", "project_dir": tmp,
             "created_at": "2026-09-16T09:00:00-04:00",
             "updated_at": "2026-09-16T09:00:00-04:00",
             "current_step": TRAIN_STEP, "steps": steps}
    pm.save_project(state)
    return state, tmp, {"dataset": dataset, "weights": run_dir, "step6": step6}


def _client(state):
    appmod.current_project = state
    return appmod.create_app().test_client()


def _no_orphan(_step_dir, proc_root=None):
    return None


def _an_orphan(_step_dir, proc_root=None):
    return FAKE_PID


def _close(tmp):
    appmod.current_project = None
    appmod.runner.processes.pop(TRAIN_STEP, None)
    appmod._step_launches.clear()
    appmod._step6_driver_alive_outside_runner = _REAL_PROBE
    shutil.rmtree(tmp, ignore_errors=True)


_REAL_PROBE = appmod._step6_driver_alive_outside_runner


# ---------------------------------------------------------------------------
# RESET
# ---------------------------------------------------------------------------

def test_reset_is_refused_while_the_runner_holds_a_trainer():
    state, tmp, paths = _fixture()
    try:
        appmod._step6_driver_alive_outside_runner = _no_orphan
        appmod.runner.processes[TRAIN_STEP] = _LiveProcess()
        client = _client(state)

        r = client.post(f"/api/step/{TRAIN_STEP}/reset")
        body = r.get_json() or {}
        check(r.status_code == 409, f"a live trainer refuses the reset: {r.status_code} {body}")
        check(RUN_NAME in str(body.get("error") or ""), f"the refusal names the run: {body}")
        check(body.get("busy") is True, f"and marks itself a busy refusal: {body}")
        check(os.path.isfile(os.path.join(paths["dataset"], "data.yaml")),
              "the dataset the trainer is reading is still there")
        check(os.path.isfile(os.path.join(paths["weights"], "last.pt")),
              "and the weights it is writing are still there")
    finally:
        _close(tmp)


def test_reset_is_refused_while_a_trainer_outlived_a_restart():
    state, tmp, paths = _fixture()
    try:
        appmod._step6_driver_alive_outside_runner = _an_orphan
        client = _client(state)

        r = client.post(f"/api/step/{TRAIN_STEP}/reset")
        body = r.get_json() or {}
        check(r.status_code == 409, f"an orphaned trainer refuses the reset: {r.status_code} {body}")
        check(body.get("orphan_pid") == FAKE_PID, f"the refusal names the pid: {body}")
        check(str(FAKE_PID) in str(body.get("error") or ""),
              f"and says it in the sentence the reader sees: {body}")
        check(os.path.isfile(os.path.join(paths["weights"], "last.pt")),
              "nothing was deleted")
    finally:
        _close(tmp)


def test_a_reset_at_an_earlier_step_is_refused_too():
    """Reset clears the step it is pressed on and every step after it, so a
    reset at Fetch Points would take the Train folder with it."""
    state, tmp, paths = _fixture()
    try:
        appmod._step6_driver_alive_outside_runner = _an_orphan
        client = _client(state)
        r = client.post("/api/step/1/reset")
        body = r.get_json() or {}
        check(r.status_code == 409, f"the whole range is guarded: {r.status_code} {body}")
        check(RUN_NAME in str(body.get("error") or ""), f"the refusal names the run: {body}")
        check(os.path.isfile(os.path.join(paths["dataset"], "data.yaml")),
              "the Train folder survives a reset pressed upstream of it")
    finally:
        _close(tmp)


def test_a_reset_after_the_train_step_is_not_guarded():
    """A reset at Evaluate clears nothing the trainer is using, so it goes
    through even while a run is alive."""
    state, tmp, paths = _fixture()
    try:
        appmod._step6_driver_alive_outside_runner = _an_orphan
        client = _client(state)
        r = client.post("/api/step/7/reset")
        check(r.status_code == 200, f"a later step resets normally: {r.status_code} {r.get_json()}")
        check(os.path.isfile(os.path.join(paths["weights"], "last.pt")),
              "and the live run's folder is untouched by it")
    finally:
        _close(tmp)


def test_reset_really_clears_when_nothing_is_alive():
    state, tmp, paths = _fixture()
    try:
        appmod._step6_driver_alive_outside_runner = _no_orphan
        client = _client(state)
        r = client.post(f"/api/step/{TRAIN_STEP}/reset")
        check(r.status_code == 200, f"a reset with nothing alive goes through: {r.status_code} {r.get_json()}")
        check(not os.path.exists(paths["dataset"]), "the dataset is cleared")
        check(os.path.isdir(paths["step6"]) and not os.listdir(paths["step6"]),
              f"the step folder is emptied and kept: {os.listdir(paths['step6'])}")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# STOP
# ---------------------------------------------------------------------------

def test_stop_reports_a_process_it_really_stopped():
    state, tmp, _paths = _fixture()
    real_kill = appmod.runner.kill
    stopped = []
    try:
        appmod._step6_driver_alive_outside_runner = _no_orphan
        appmod.runner.processes[TRAIN_STEP] = _LiveProcess()
        appmod.runner.kill = lambda step: stopped.append(step)
        client = _client(state)

        r = client.post(f"/api/step/{TRAIN_STEP}/stop")
        body = r.get_json() or {}
        check(r.status_code == 200, f"stop answers 200: {r.status_code} {body}")
        check(body.get("stopped") is True, f"and says it really stopped something: {body}")
        check(stopped == [TRAIN_STEP], f"the runner was asked to kill that step: {stopped}")
        check(state["steps"][str(TRAIN_STEP)]["status"] == "pending",
              f"the step is no longer marked running: {state['steps'][str(TRAIN_STEP)]}")
    finally:
        appmod.runner.kill = real_kill
        _close(tmp)


def test_stop_says_when_there_was_nothing_to_stop():
    state, tmp, _paths = _fixture()
    try:
        appmod._step6_driver_alive_outside_runner = _no_orphan
        client = _client(state)
        r = client.post(f"/api/step/{TRAIN_STEP}/stop")
        body = r.get_json() or {}
        check(r.status_code == 200, f"stop answers 200: {r.status_code} {body}")
        check(body.get("stopped") is False, f"and does not claim a kill: {body}")
        check("Nothing was running" in str(body.get("message") or ""),
              f"the message says what happened: {body}")
        check(state["steps"][str(TRAIN_STEP)]["status"] == "pending",
              "the stale running mark is cleared, because nothing is running")
    finally:
        _close(tmp)


def test_stop_refuses_to_claim_a_kill_it_cannot_make():
    """A trainer that outlived a restart is not the runner's to stop. Saying
    success here is how Stop left a training running while the panel showed
    it as stopped."""
    state, tmp, paths = _fixture()
    try:
        appmod._step6_driver_alive_outside_runner = _an_orphan
        client = _client(state)
        r = client.post(f"/api/step/{TRAIN_STEP}/stop")
        body = r.get_json() or {}
        check(r.status_code == 409, f"stop refuses: {r.status_code} {body}")
        check(body.get("stopped") is False and body.get("success") is False,
              f"and claims nothing: {body}")
        check(body.get("orphan_pid") == FAKE_PID, f"the pid rides with the refusal: {body}")
        check(str(FAKE_PID) in str(body.get("error") or ""),
              f"and is in the sentence the reader sees: {body}")
        check(state["steps"][str(TRAIN_STEP)]["status"] == "running",
              f"the step still reads running, because it is: {state['steps'][str(TRAIN_STEP)]}")
    finally:
        _close(tmp)


if __name__ == "__main__":
    print("RESET AND STOP -- never over a live training run\n")
    print("RESET:")
    run(test_reset_is_refused_while_the_runner_holds_a_trainer)
    run(test_reset_is_refused_while_a_trainer_outlived_a_restart)
    run(test_a_reset_at_an_earlier_step_is_refused_too)
    run(test_a_reset_after_the_train_step_is_not_guarded)
    run(test_reset_really_clears_when_nothing_is_alive)
    print("STOP:")
    run(test_stop_reports_a_process_it_really_stopped)
    run(test_stop_says_when_there_was_nothing_to_stop)
    run(test_stop_refuses_to_claim_a_kill_it_cannot_make)

    appmod.current_project = None
    appmod.runner.processes.pop(TRAIN_STEP, None)
    appmod._step6_driver_alive_outside_runner = _REAL_PROBE
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
