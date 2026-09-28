"""Task T1: the Step 6 launch guard (a second Start Training press).

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_launch_guard.py

Two defects this suite pins, both verified on this box (plan
docs/superpowers/plans/2026-09-04-vega-training-foldin.md, defects 1 and 4):

  1. _run_step6 built the detection dataset before anything asked whether a
     training run was already in flight. build_detect_dataset finishes with
     rmtree then rename on <step6>/dataset, so a second press removed the
     dataset the live trainer was reading.
  2. A launch refused as busy came back through the route's error branch,
     which flipped the FIRST run's recorded status to "error". The status
     poller completes only a step still marked running, so the live run
     could never finish.

Covers:
  GUARD     a second POST to /api/step/6/run answers 409 before the dataset
            builder or the stage runner is reached, and nothing under the
            step folder changes.
  STATE     the first run keeps its status, start time and VICARIUS event
            id, on disk too, and the poller still completes it when the
            process ends; the refused press's own event is closed.
  DATASET   the dataset directory keeps its inode and modification time
            through a refusal that runs the real builder and the real
            runner, the exact code that destroyed it before.
  EDGES     a stale "running" status with no runner process launches; a
            refusal with no dataset directory yet creates nothing; a
            different step running does not block Step 6; the runner's own
            refusal carries the busy marker; two presses serialise on the
            launch lock; the runner's own refusal reaching the route (the
            race the lock closes) is handled the same way; a driver that
            outlived an orchestrator restart is found in the kernel's
            process list and refused, and the refusal names its pid; a
            refusal whose process_start opened no event closes nothing; a
            busy Flask sub-app (Fetch Labels) is told to use Stop rather
            than wait and keeps its saved port; an ordinary failure still
            marks the step errored.

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, the VICARIUS bridge is replaced by a recorder, and
the "live trainer" is a stand-in whose poll() answers None until the test
finishes it.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []
# Every fixture root this run created, checked for removal at the end.
_FIXTURES = []


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# Standing rule: no orchestrator test may reach a real subprocess.
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import stage_runner  # noqa: E402
import project_manager as pm  # noqa: E402
from _reefreview.training_projects import mint_training_project  # noqa: E402

# A process id no real process can hold (Linux pid_max is 4194304), so even a
# stray kill on the stand-in below could never signal a real process group.
_FAKE_PID = 999999999
_STEP = 6
_OTHER_STEP = 7
_FLASK_STEP = 2
_ANNOTATOR_KEY = "step4test"
_ANNOTATOR_PORT = 5080
# Seconds between two presses, so a rewritten start stamp is detectable.
_CLOCK_TICK = 0.002
# How long the serialisation test waits to be sure a blocked press stays
# blocked, and how long it allows the released press to finish.
_HOLD_SECONDS = 0.5
_JOIN_SECONDS = 10.0
_BUILD_TMP_PREFIX = ".dataset_build_"
# Stand-in process ids for the fake process listing.
_ORPHAN_PID = 4242
_LOOKALIKE_PID = 4243
_OTHER_DIR_PID = 4244
_NO_CMDLINE_PID = 4245
_FIRST_CONFIG = {"run_name": "first", "device": "0", "batch": 8}
_SECOND_CONFIG = {"run_name": "second", "device": "0", "batch": 8}


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, print PASS or FAIL, and record the outcome."""
    import traceback
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ---------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------

class _RecordingVic:
    """Stands in for vicarius_bridge: hands out event ids and records every
    process_start and process_end call instead of writing to the log."""

    def __init__(self, start_limit=None):
        self.starts = []
        self.ends = []
        self.notes = []
        self.start_limit = start_limit

    def process_start(self, step_name, purpose, inputs=None, notes=""):
        """Record a start and return a fresh event id (evt-1, evt-2, ...),
        or None once start_limit ids have been handed out, the way the
        bridge answers when its write fails."""
        n = len(self.starts) + 1
        event_id = None if (self.start_limit is not None and n > self.start_limit) else f"evt-{n}"
        self.starts.append({"id": event_id, "step_name": step_name, "purpose": purpose})
        return event_id

    def process_end(self, step_name, parent_event_id, status, duration_sec,
                    outputs=None, notes=""):
        """Record an end linked to the event id it closes."""
        self.ends.append({"id": parent_event_id, "step_name": step_name,
                          "status": status, "notes": notes})

    def note(self, message, study=None):
        """Record a note."""
        self.notes.append(message)


def _fresh_vic(start_limit=None):
    """Install a new recorder on the app module and return it."""
    vic = _RecordingVic(start_limit=start_limit)
    appmod.vic = vic
    return vic


class _LiveProcess:
    """Stands in for the Popen of a trainer that is still running. poll()
    answers None until finish() is called, then the given exit code, so
    StageRunner.kill_all at interpreter exit never signals anything."""

    pid = _FAKE_PID

    def __init__(self):
        self.exit_code = None

    def poll(self):
        """None while the stand-in is alive, else its exit code."""
        return self.exit_code

    def finish(self, exit_code=0):
        """End the stand-in with exit_code."""
        self.exit_code = exit_code

    def wait(self, timeout=None):
        """Answer the exit code the way Popen.wait does; never blocks."""
        return self.exit_code


def _start_live_run(step, cmd, cwd=None, env_extra=None):
    """Launch stand-in: register a live trainer under step and answer like
    a started run."""
    appmod.runner.processes[step] = _LiveProcess()
    return {"pid": _FAKE_PID}


def _fake_build(source_dir, dataset_dir, seed_name, **_fractions):
    """Builder stand-in: writes one file into dataset_dir, removes nothing."""
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, "data.yaml"), "w") as fh:
        fh.write("path: %s\n" % dataset_dir)
    return {"n_pairs": 1, "n_unlabelled": 0,
            "counts": {"train": 1, "valid": 0, "test": 0}}


class _Stubs:
    """Count every call to the dataset builder and the runner's launch for
    one press, optionally replacing either, and put both back afterwards.
    With no replacement the REAL builder and the REAL run_cli_stage run."""

    def __init__(self, build=None, launch=None):
        self.build_calls = []
        self.launch_calls = []
        self._real_build = appmod._build_detect_dataset
        self._real_launch = appmod.runner.run_cli_stage
        self._build = build or self._real_build
        self._launch = launch or self._real_launch

    def __enter__(self):
        def _build(source_dir, dataset_dir, seed_name, **fractions):
            self.build_calls.append((source_dir, dataset_dir, seed_name))
            return self._build(source_dir, dataset_dir, seed_name, **fractions)

        def _launch(step, cmd, cwd=None, env_extra=None):
            self.launch_calls.append((step, list(cmd)))
            return self._launch(step, cmd, cwd=cwd, env_extra=env_extra)

        appmod._build_detect_dataset = _build
        appmod.runner.run_cli_stage = _launch
        return self

    def __exit__(self, *_exc):
        appmod._build_detect_dataset = self._real_build
        appmod.runner.run_cli_stage = self._real_launch
        return False


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

def _stage_dataset(root, name="coral_intel", pairs=4):
    """A staged detection dataset: images/ beside labels/, plus the manifest
    and data.yaml the boxes page writes."""
    d = os.path.join(root, "bbox_datasets", name)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    for i in range(pairs):
        with open(os.path.join(d, "images", f"img{i}.jpg"), "w") as fh:
            fh.write("x")
        with open(os.path.join(d, "labels", f"img{i}.txt"), "w") as fh:
            fh.write("0 0.5 0.5 0.2 0.2\n")
    with open(os.path.join(d, "classes.txt"), "w") as fh:
        fh.write("Alpha\n")
    with open(os.path.join(d, "data.yaml"), "w") as fh:
        fh.write("path: %s\ntrain: images\nval: images\nnc: 1\nnames:\n  0: Alpha\n" % d)
    with open(os.path.join(d, "ingest_manifest.json"), "w") as fh:
        json.dump({"n_images": pairs, "n_boxes": pairs, "classes": ["Alpha"]}, fh)
    return d


def _open_bbox_project(pairs=4):
    """(client, tmp_root, run_dir, dataset_dir) with a bbox project opened."""
    tmp = tempfile.mkdtemp(prefix="step6_guard_")
    _FIXTURES.append(tmp)
    inprocess = os.path.join(tmp, "inprocess")
    os.makedirs(inprocess, exist_ok=True)
    dataset = _stage_dataset(tmp, pairs=pairs)
    res = mint_training_project(inprocess, "bbox", "coral_intel", dataset, "LO",
                                extra={"task_default": "detect"})
    client = appmod.create_app().test_client()
    r = client.post("/api/project/open", json={"project_dir": res["run_dir"]})
    check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
    return client, tmp, res["run_dir"], dataset


def _open_plain_project():
    """(client, tmp) with an ordinary project (no training source) built in
    memory the way test_step6_task does, so a Flask stage can be pressed."""
    tmp = tempfile.mkdtemp(prefix="step6_guard_")
    _FIXTURES.append(tmp)
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    appmod.current_project = {
        "id": os.path.basename(tmp), "name": "plain", "project_dir": tmp,
        "created_at": "x", "updated_at": "x", "current_step": _FLASK_STEP, "steps": steps,
    }
    return appmod.create_app().test_client(), tmp


def _close(tmp):
    """Drop the project, forget every stand-in process and remove the fixture."""
    appmod.current_project = None
    for key in (_STEP, _OTHER_STEP, _ANNOTATOR_KEY, _FLASK_STEP):
        proc = appmod.runner.processes.pop(key, None)
        if isinstance(proc, _LiveProcess):
            proc.finish()
    shutil.rmtree(tmp, ignore_errors=True)


def _put_config(client, config):
    """Save a Step 6 config through the route the panel uses."""
    r = client.put(f"/api/project/config/{_STEP}", json=config)
    check(r.status_code == 200, f"config PUT failed: {r.get_json()}")


def _step6():
    """The open project's step 6 block."""
    return appmod.current_project["steps"][str(_STEP)]


def _step6_on_disk(run_dir):
    """The step 6 block as project.json holds it right now."""
    with open(os.path.join(run_dir, "project.json")) as fh:
        return json.load(fh)["steps"][str(_STEP)]


def _first_press(client):
    """The first Start Training press: the real builder writes the dataset
    and the launch stand-in leaves a live trainer registered for step 6.
    Returns the live process."""
    _put_config(client, _FIRST_CONFIG)
    with _Stubs(launch=_start_live_run) as s:
        r = client.post(f"/api/step/{_STEP}/run")
    body = r.get_json() or {}
    check(r.status_code == 200, f"the first press must launch: {r.status_code} {body}")
    check(len(s.build_calls) == 1, "the first press builds the dataset once")
    proc = appmod.runner.processes.get(_STEP)
    check(isinstance(proc, _LiveProcess), "the first press leaves a live trainer registered")
    check(_step6()["status"] == "running", "the first press marks the step running")
    return proc


def _tree(root):
    """{relative path: (inode, mtime_ns)} for root and everything under it."""
    out = {}
    st = os.lstat(root)
    out["."] = (st.st_ino, st.st_mtime_ns)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            st = os.lstat(p)
            out[os.path.relpath(p, root)] = (st.st_ino, st.st_mtime_ns)
    return out


def _build_temp_dirs(step_dir):
    """Any half-built dataset folders the builder left under step_dir."""
    return [n for n in os.listdir(step_dir) if n.startswith(_BUILD_TMP_PREFIX)]


# ---------------------------------------------------------------------------
# GUARD
# ---------------------------------------------------------------------------

def test_second_launch_is_refused_before_any_write():
    _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        _first_press(client)
        step_dir = _step6()["dir"]
        tree_before = _tree(step_dir)
        disk_before = _step6_on_disk(run_dir)
        time.sleep(_CLOCK_TICK)
        _put_config(client, _SECOND_CONFIG)
        with _Stubs(build=_fake_build, launch=_start_live_run) as s:
            r = client.post(f"/api/step/{_STEP}/run")
        body = r.get_json() or {}
        check(r.status_code == 409,
              f"a second press answers 409 (got {r.status_code}: {body})")
        check(body.get("busy") is True, f"the body carries the busy marker: {body}")
        err = str(body.get("error", ""))
        check("already running" in err, f"the existing busy wording is reused: {body}")
        check("Train Model" in err, f"the refusal names the tile the reader sees: {body}")
        check("Wait for it to finish" in err, f"the refusal says what to do: {body}")
        check("http_status" not in body, "http_status is a transport hint, never a body field")
        check(s.build_calls == [], f"the dataset builder never ran: {s.build_calls}")
        check(s.launch_calls == [], f"the runner never launched: {s.launch_calls}")
        check(_tree(step_dir) == tree_before, "nothing under the step folder changed")
        disk_after = _step6_on_disk(run_dir)
        for key in ("status", "started_at", "_vic_event_id", "dir"):
            check(disk_after.get(key) == disk_before.get(key),
                  f"project.json keeps {key}: {disk_before.get(key)!r} -> {disk_after.get(key)!r}")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

def test_refused_launch_leaves_the_first_run_running():
    vic = _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        proc = _first_press(client)
        first = dict(_step6())
        check(first.get("_vic_event_id") == "evt-1",
              f"the first press opened evt-1 (got {first.get('_vic_event_id')!r})")
        time.sleep(_CLOCK_TICK)
        _put_config(client, _SECOND_CONFIG)
        with _Stubs(build=_fake_build, launch=_start_live_run):
            r = client.post(f"/api/step/{_STEP}/run")
        check(r.status_code == 409, f"a second press answers 409 (got {r.status_code})")

        st = _step6()
        check(st["status"] == "running",
              f"the first run is still recorded as running (got {st['status']!r})")
        check(st["started_at"] == first["started_at"],
              f"the first run keeps its start time: {first['started_at']} -> {st['started_at']}")
        check(st.get("_vic_event_id") == "evt-1",
              f"the first run keeps its VICARIUS event id (got {st.get('_vic_event_id')!r})")
        disk = _step6_on_disk(run_dir)
        check(disk["status"] == "running" and disk["started_at"] == first["started_at"],
              f"project.json on disk agrees: {disk.get('status')} {disk.get('started_at')}")

        # The refused press opened its own event; that one is closed as
        # failed with the refusal as its note, and evt-1 stays open.
        check(len(vic.starts) == 2, f"two process_start calls (got {len(vic.starts)})")
        check([e["id"] for e in vic.ends] == ["evt-2"],
              f"only the refused press's event was closed (got {vic.ends})")
        check(vic.ends[0]["status"] == "failed" and "already running" in vic.ends[0]["notes"],
              f"the refused event says why: {vic.ends[0]}")

        # The poller can still complete the live run once it ends.
        proc.finish(0)
        r = client.get(f"/api/step/{_STEP}/status")
        body = r.get_json() or {}
        check(body.get("completed") is True, f"the poller completes the live run: {body}")
        check(_step6()["status"] == "completed",
              f"the step is recorded completed (got {_step6()['status']!r})")
        check(vic.ends[-1]["id"] == "evt-1" and vic.ends[-1]["status"] == "success",
              f"evt-1 is closed as a success: {vic.ends[-1]}")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

def test_dataset_dir_is_untouched_by_a_refused_launch():
    _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        _first_press(client)
        step_dir = _step6()["dir"]
        dataset_dir = os.path.join(step_dir, "dataset")
        check(os.path.isdir(dataset_dir), "the first press wrote the dataset")
        before = os.stat(dataset_dir)
        tree_before = _tree(dataset_dir)
        check(len(tree_before) > 1, "the dataset holds split folders and files")
        time.sleep(_CLOCK_TICK)
        _put_config(client, _SECOND_CONFIG)
        # The second press runs the REAL builder and the REAL runner: exactly
        # the code that removed and replaced the dataset before the guard.
        with _Stubs() as s:
            r = client.post(f"/api/step/{_STEP}/run")
        check(r.status_code == 409, f"a second press answers 409 (got {r.status_code})")
        after = os.stat(dataset_dir)
        check(after.st_ino == before.st_ino,
              f"same inode, so the directory was never removed and replaced "
              f"({before.st_ino} -> {after.st_ino})")
        check(after.st_mtime_ns == before.st_mtime_ns,
              f"same modification time ({before.st_mtime_ns} -> {after.st_mtime_ns})")
        check(_tree(dataset_dir) == tree_before, "every file under the dataset is unchanged")
        check(_build_temp_dirs(step_dir) == [],
              f"no half-built dataset folder was left behind: {_build_temp_dirs(step_dir)}")
        check(s.build_calls == [] and s.launch_calls == [],
              f"neither the builder nor the runner ran: {s.build_calls} {s.launch_calls}")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# EDGES
# ---------------------------------------------------------------------------

def test_stale_running_status_with_no_runner_process_allows_the_launch():
    _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        # The orchestrator restarted while project.json still said running:
        # the runner has no entry for step 6 at all.
        st = _step6()
        st["status"] = "running"
        st["started_at"] = "2026-09-04T09:00:00"
        check(_STEP not in appmod.runner.processes, "no runner entry for step 6")
        check(appmod.runner.is_running(_STEP) is False,
              "a missing runner key reads as not running, never as an error")
        _put_config(client, _SECOND_CONFIG)
        with _Stubs(build=_fake_build, launch=_start_live_run) as s:
            r = client.post(f"/api/step/{_STEP}/run")
        body = r.get_json() or {}
        check(r.status_code == 200, f"nothing is live, so the press launches: {r.status_code} {body}")
        check(len(s.build_calls) == 1, "the dataset is built once")
        check(len(s.launch_calls) == 1, "the runner is asked once")
        check(_step6()["status"] == "running", "the fresh launch is recorded as running")
    finally:
        _close(tmp)


def test_refusal_when_the_dataset_dir_does_not_exist_yet():
    _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        # A trainer is live for step 6 but this step folder holds no dataset,
        # for instance a project reopened while an older run is still going.
        appmod.runner.processes[_STEP] = _LiveProcess()
        st = _step6()
        st["status"] = "running"
        step_dir = st["dir"]
        dataset_dir = os.path.join(step_dir, "dataset")
        check(not os.path.exists(dataset_dir), "no dataset folder exists yet")
        _put_config(client, _SECOND_CONFIG)
        with _Stubs() as s:
            r = client.post(f"/api/step/{_STEP}/run")
        check(r.status_code == 409, f"the press is refused (got {r.status_code}: {r.get_json()})")
        check(not os.path.exists(dataset_dir), "the refusal created no dataset folder")
        check(_build_temp_dirs(step_dir) == [], "the refusal left no half-built folder")
        check(s.build_calls == [] and s.launch_calls == [], "neither the builder nor the runner ran")
        check(_step6()["status"] == "running", "the recorded status is untouched")
    finally:
        _close(tmp)


def test_launch_allowed_while_a_different_step_is_running():
    _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        appmod.runner.processes[_OTHER_STEP] = _LiveProcess()
        appmod.runner.processes[_ANNOTATOR_KEY] = _LiveProcess()
        _put_config(client, _SECOND_CONFIG)
        with _Stubs(build=_fake_build, launch=_start_live_run) as s:
            r = client.post(f"/api/step/{_STEP}/run")
        body = r.get_json() or {}
        check(r.status_code == 200,
              f"another step's process never blocks Step 6: {r.status_code} {body}")
        check(len(s.build_calls) == 1 and len(s.launch_calls) == 1,
              "the dataset is built and the runner is asked once each")
        check(appmod.runner.is_running(_OTHER_STEP) and appmod.runner.is_running(_ANNOTATOR_KEY),
              "the other processes are still registered")
    finally:
        _close(tmp)


def test_runner_refusal_carries_the_busy_marker():
    try:
        appmod.runner.processes[_STEP] = _LiveProcess()
        appmod.runner.processes[_ANNOTATOR_KEY] = _LiveProcess()
        cli = appmod.runner.run_cli_stage(_STEP, ["true"])
        check(cli == {"error": "Step 6 is already running", "busy": True},
              f"run_cli_stage refuses with the marker and the old wording: {cli}")
        flask_result = appmod.runner.run_flask_stage(_ANNOTATOR_KEY, ["true"], _ANNOTATOR_PORT)
        check(flask_result == {"error": "Step step4test is already running", "busy": True},
              f"run_flask_stage refuses the same way: {flask_result}")
        plain = stage_runner.busy_result(_STEP)
        check(plain == {"error": "Step 6 is already running", "busy": True},
              f"busy_result with no label keeps the runner's wording: {plain}")
        labelled = stage_runner.busy_result(_STEP, label="Train Model",
                                            advice="Wait for it to finish.")
        check(labelled["error"] == "Train Model is already running. Wait for it to finish.",
              f"a label and advice are folded into one message: {labelled}")
        check(labelled["busy"] is True, "the marker survives a label")
    finally:
        for key in (_STEP, _ANNOTATOR_KEY):
            proc = appmod.runner.processes.pop(key, None)
            if isinstance(proc, _LiveProcess):
                proc.finish()


def test_two_presses_serialise_on_the_launch_lock():
    _fresh_vic()
    lock = appmod._step6_launch_lock
    client, tmp, run_dir, dataset = _open_bbox_project()
    held = False
    try:
        _first_press(client)
        _put_config(client, _SECOND_CONFIG)
        outcome = {}

        def _second_press():
            r = client.post(f"/api/step/{_STEP}/run")
            outcome["status"] = r.status_code

        check(lock.acquire(timeout=1), "the launch lock is free before the test holds it")
        held = True
        with _Stubs(build=_fake_build, launch=_start_live_run) as s:
            t = threading.Thread(target=_second_press, daemon=True)
            t.start()
            t.join(_HOLD_SECONDS)
            check(t.is_alive(), "a press waits while another press holds the launch lock")
            check(s.build_calls == [] and s.launch_calls == [],
                  "and while it waits it has built and launched nothing")
            lock.release()
            held = False
            t.join(_JOIN_SECONDS)
            check(not t.is_alive(), "the waiting press finishes once the lock is released")
        check(outcome.get("status") == 409,
              f"the waiting press is then refused, the trainer being live: {outcome}")
        check(s.build_calls == [] and s.launch_calls == [],
              f"and it wrote nothing: {s.build_calls} {s.launch_calls}")
    finally:
        if held:
            lock.release()
        _close(tmp)


def test_runner_level_refusal_keeps_the_first_run_too():
    vic = _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    real_is_running = appmod.runner.is_running
    try:
        _first_press(client)
        first = dict(_step6())
        time.sleep(_CLOCK_TICK)
        _put_config(client, _SECOND_CONFIG)
        # The race the launch lock closes, replayed by hand: the probe says
        # nothing is live, so the press reaches the runner, whose own check
        # refuses it. That refusal has to be handled the same way.
        appmod.runner.is_running = lambda step: False
        with _Stubs(build=_fake_build) as s:
            r = client.post(f"/api/step/{_STEP}/run")
        body = r.get_json() or {}
        check(r.status_code == 409,
              f"the runner's own refusal answers 409 too: {r.status_code} {body}")
        check(body.get("busy") is True and "already running" in str(body.get("error", "")),
              f"and carries the marker and the wording: {body}")
        check("Train Model" in str(body.get("error", "")),
              f"and names the tile, not the internal step number: {body}")
        check(len(s.launch_calls) == 1, "the runner was reached once")
        st = _step6()
        check(st["status"] == "running" and st["started_at"] == first["started_at"]
              and st.get("_vic_event_id") == "evt-1",
              f"the first run's status, start time and event id are intact: {st}")
        check([e["id"] for e in vic.ends] == ["evt-2"],
              f"only the refused press's event was closed: {vic.ends}")
    finally:
        appmod.runner.is_running = real_is_running
        _close(tmp)


def test_a_refusal_whose_bridge_opened_no_event_closes_nothing():
    vic = _fresh_vic(start_limit=1)
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        proc = _first_press(client)
        check(_step6().get("_vic_event_id") == "evt-1", "the first press opened evt-1")
        time.sleep(_CLOCK_TICK)
        _put_config(client, _SECOND_CONFIG)
        # The bridge's write failed for the second press, so process_start
        # answered None and the step still carries the live run's evt-1.
        with _Stubs(build=_fake_build, launch=_start_live_run):
            r = client.post(f"/api/step/{_STEP}/run")
        check(r.status_code == 409, f"the press is refused (got {r.status_code})")
        check(len(vic.starts) == 2 and vic.starts[1]["id"] is None,
              f"the second start was attempted and answered None: {vic.starts}")
        check(vic.ends == [], f"evt-1 was not closed by the refusal: {vic.ends}")
        check(_step6().get("_vic_event_id") == "evt-1", "the live run keeps evt-1")
        proc.finish(0)
        client.get(f"/api/step/{_STEP}/status")
        check([(e["id"], e["status"]) for e in vic.ends] == [("evt-1", "success")],
              f"evt-1 is closed exactly once, as a success: {vic.ends}")
    finally:
        _close(tmp)


def _fake_proc(root, entries):
    """A stand-in for /proc: {pid: argv list or None} becomes
    <root>/<pid>/cmdline, None meaning a process folder with no readable
    cmdline. A non-numeric folder is added, as the real listing has."""
    os.makedirs(root, exist_ok=True)
    for pid, argv in entries.items():
        d = os.path.join(root, str(pid))
        os.makedirs(d, exist_ok=True)
        if argv is None:
            continue
        with open(os.path.join(d, "cmdline"), "wb") as fh:
            fh.write(b"\0".join(a.encode() for a in argv) + b"\0")
    os.makedirs(os.path.join(root, "sys"), exist_ok=True)
    return root


def test_a_driver_that_outlived_a_restart_is_found_and_refused():
    _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    real_proc_root = appmod.PROC_ROOT
    try:
        step_dir = _step6()["dir"]
        driver = appmod.ENTRY_POINTS[6]
        proc_root = _fake_proc(os.path.join(tmp, "proc"), {
            _ORPHAN_PID: ["/bin/bash", driver, "train", step_dir, "first", "500", "512", "yolo11m.pt"],
            _LOOKALIKE_PID: ["vim", os.path.join(step_dir, "dataset", "data.yaml")],
            _OTHER_DIR_PID: ["/bin/bash", driver, "train", os.path.join(tmp, "elsewhere"), "x"],
            _NO_CMDLINE_PID: None,
        })
        probe = appmod._step6_driver_alive_outside_runner
        found = probe(step_dir, proc_root=proc_root)
        check(found == _ORPHAN_PID, f"the driver over this step folder is found (got {found!r})")
        check(probe(step_dir + os.sep, proc_root=proc_root) == _ORPHAN_PID,
              "a trailing separator on the step folder still matches")
        check(probe(os.path.join(tmp, "nowhere"), proc_root=proc_root) is None,
              "a folder no driver names finds nothing")
        check(probe(step_dir, proc_root=os.path.join(tmp, "missing")) is None,
              "an unreadable process list finds nothing rather than raising")
        check(probe(step_dir) is None,
              "the real process list holds no driver over a fixture folder")

        # Through the route: the runner knows nothing, the process list does.
        check(_STEP not in appmod.runner.processes, "the runner has no entry for step 6")
        appmod.PROC_ROOT = proc_root
        _put_config(client, _SECOND_CONFIG)
        with _Stubs() as s:
            r = client.post(f"/api/step/{_STEP}/run")
        body = r.get_json() or {}
        check(r.status_code == 409, f"the press is refused (got {r.status_code}: {body})")
        check("Train Model" in str(body.get("error", "")), f"and names the tile: {body}")
        check(body.get("orphan_pid") == _ORPHAN_PID and str(_ORPHAN_PID) in str(body.get("error", "")),
              f"and names the process to stop, which the orchestrator no longer owns: {body}")
        check(s.build_calls == [] and s.launch_calls == [], "nothing was built or launched")
        check(not os.path.exists(os.path.join(step_dir, "dataset")), "no dataset folder appeared")
        check(_step6()["status"] == "pending", f"the recorded status is untouched (got {_step6()['status']!r})")
    finally:
        appmod.PROC_ROOT = real_proc_root
        _close(tmp)


def test_a_busy_flask_stage_gets_its_own_advice_and_keeps_its_port():
    _fresh_vic()
    client, tmp = _open_plain_project()
    real_ports = dict(appmod.runner.ports)
    try:
        # Fetch Labels is a sub-app launched once and used from the browser;
        # it never finishes on its own, so "wait for it" would be a dead end.
        appmod.runner.processes[_FLASK_STEP] = _LiveProcess()
        appmod.runner.ports[_FLASK_STEP] = _ANNOTATOR_PORT
        st = appmod.current_project["steps"][str(_FLASK_STEP)]
        st["status"] = "running"
        st["config"]["port"] = _ANNOTATOR_PORT
        r = client.post(f"/api/step/{_FLASK_STEP}/run")
        body = r.get_json() or {}
        check(r.status_code == 409, f"a busy sub-app is refused with 409: {r.status_code} {body}")
        err = str(body.get("error", ""))
        check("Fetch Labels" in err and "press Stop" in err,
              f"the refusal names the tile and points at Stop rather than waiting: {body}")
        check(st["status"] == "running", "the live sub-app keeps its status")
        check(st["config"]["port"] == _ANNOTATOR_PORT,
              f"the refused press saved no port nothing listens on (got {st['config'].get('port')})")
    finally:
        appmod.runner.ports.clear()
        appmod.runner.ports.update(real_ports)
        _close(tmp)


def test_an_ordinary_failure_still_marks_the_step_errored():
    vic = _fresh_vic()
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        _put_config(client, _SECOND_CONFIG)

        def _no_pairs(source_dir, dataset_dir, seed_name, **_fractions):
            raise ValueError(f"no image and label pairs found in {source_dir}")

        with _Stubs(build=_no_pairs, launch=_start_live_run) as s:
            r = client.post(f"/api/step/{_STEP}/run")
        body = r.get_json() or {}
        check(r.status_code == 400, f"the builder's refusal still answers 400: {r.status_code} {body}")
        check("busy" not in body, f"an ordinary failure carries no busy marker: {body}")
        check(_step6()["status"] == "error", f"the step is marked errored (got {_step6()['status']!r})")
        check(s.launch_calls == [], "the runner was never asked")
        check(vic.ends and vic.ends[-1]["status"] == "failed", f"the event is closed as failed: {vic.ends}")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GUARD:")
    run(test_second_launch_is_refused_before_any_write)
    print("STATE:")
    run(test_refused_launch_leaves_the_first_run_running)
    print("DATASET:")
    run(test_dataset_dir_is_untouched_by_a_refused_launch)
    print("EDGES:")
    run(test_stale_running_status_with_no_runner_process_allows_the_launch)
    run(test_refusal_when_the_dataset_dir_does_not_exist_yet)
    run(test_launch_allowed_while_a_different_step_is_running)
    run(test_runner_refusal_carries_the_busy_marker)
    run(test_two_presses_serialise_on_the_launch_lock)
    run(test_runner_level_refusal_keeps_the_first_run_too)
    run(test_a_refusal_whose_bridge_opened_no_event_closes_nothing)
    run(test_a_driver_that_outlived_a_restart_is_found_and_refused)
    run(test_a_busy_flask_stage_gets_its_own_advice_and_keeps_its_port)
    run(test_an_ordinary_failure_still_marks_the_step_errored)

    leftovers = [d for d in _FIXTURES if os.path.exists(d)]
    if leftovers:
        print(f"  FAIL fixture cleanup: {leftovers}")
        _RESULTS.append(("fixture_cleanup", False, str(leftovers)))
        for d in leftovers:
            shutil.rmtree(d, ignore_errors=True)

    failed = [n for n, ok, _ in _RESULTS if not ok]
    print()
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAILED {name}:\n{detail}")
    print("PASS" if not failed else f"{len(failed)} FAILED")
    sys.exit(1 if failed else 0)
