"""Task 9: Evaluate follows the run that was picked, not the open project.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step7_foreign_run.py

The Train step's runs card lists every run the module can see, including runs
in a model lab under inprocess/_models/<lab>/runs/, and offers Evaluate on
each one. The front end has kept a lab run selected since the 2026-09-05
fold-in; the server had not followed. This suite pins the server side of that
contract (design section 4.4 and section 11 of
/mnt/rip/vicarius_drive/docs/superpowers/specs/2026-09-14-vega-fish-box-training-set-design.md,
rule in docs/adr/0003-runs-compare-only-on-the-same-split-and-metric-family.md).

Covers, all through POST /api/step/7/run:
  OWN RUN     a run of the open project keeps every path it has always had:
              the project's dataset, the project's ledger, an eval_<run>
              folder under step 7, and the project's own Step 7 pointers.
              This is the regression guard.
  LAB RUN     a run in a model lab is scored on the lab's dataset, its ledger
              row goes to the lab's own rounds.csv, its report is written to
              inprocess/_models/_eval/<run>_<stamp> (where the report route
              looks for it), and the open project's Step 7 pointers are left
              exactly as they were.
  POINTERS    the two halves of the project's saved Step 7 pointer pair
              (steps.7.config.run_dir and steps.7.outputs.out_dir) always
              describe one and the same evaluation, so the report card asked
              without a run_dir never shows this project's numbers under
              another run's name.
  REFUSALS    a lab with no dataset folder, a run folder outside the module's
              inprocess folder, a run reached by a symlink from outside it, an
              empty selection, and a project run whose dataset has not been
              built yet.
  KNOBS       image size, confidence, IoU, batch, split and preview count
              reach the command identically for both kinds of run.

Task 11 added the other half of the same contract, through
GET /api/step/<step>/status, which is where a finished run is auto-detected:

  COMPLETION  a finished evaluation of someone else's run reports its success
              to the panel and records nothing in the open project: no
              completion, no stamp, no auto-linked outputs, no unlocked
              Inference step. The project's own run completes exactly as it
              always did (the regression guard), a failed run still reports
              its failure, every other step is untouched, and a poll that
              arrives with nothing left to say whose run was scored (an
              orchestrator restart during the evaluation) records nothing
              either.
  POINTERS    also covers a refused launch of one of the project's OWN runs,
              which used to leave the saved selection naming a different run
              from the saved evaluation folder.

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, the VICARIUS bridge is replaced by a recorder, and
the stage runner is replaced by a capture. Every fixture lives under one temp
folder that is removed in a finally block, and the removal is verified.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []
_FIXTURES = []


class _FakeCompleted:
    """What a stubbed subprocess.run answers: a finished, silent process."""
    returncode = 0
    stdout = ""
    stderr = ""


# Standing rule: no orchestrator test may reach a real subprocess.
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402


class _FakeVic:
    """The VICARIUS bridge with every call recorded and nothing written.

    process_end keeps its arguments, not just its count: the foreign path
    passes the evaluation folder it really wrote instead of the open
    project's own recorded outputs, and only a recorded argument list can
    prove the process log was not credited with work the project did not do.
    """
    notes = []
    ends = []

    @staticmethod
    def process_start(*_a, **_k):
        return None

    @classmethod
    def process_end(cls, step_name, event_id, status, duration,
                    outputs=None, notes="", *_a, **_k):
        """Record one closed process event and write nothing anywhere."""
        cls.ends.append({
            "step_name": str(step_name),
            "event_id": event_id,
            "status": str(status),
            "outputs": [str(o) for o in (outputs or [])],
            "notes": str(notes or ""),
        })
        return None

    @classmethod
    def note(cls, message, *_a, **_k):
        cls.notes.append(message)


appmod.vic = _FakeVic()
_REAL_INPROCESS_ROOT = appmod._inprocess_root

_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()
# The Step 7 output pointers the open project records for its own runs.
STEP7_OUTPUT_KEYS = ("out_dir", "report_md", "report_pdf", "metrics_json",
                     "rounds_csv", "champion_json")
# Where a foreign run's evaluation must be written, relative to inprocess/:
# the folder GET /api/step/7/report already searches (_eval_folders_for_run).
EVAL_ROOT_PARTS = ("_models", "_eval")


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function and record PASS or FAIL with its traceback."""
    import traceback
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write(path, data):
    """Write text or bytes to path, creating the parent folder first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(data)


def _make_run(run_dir, task="detect", imgsz=640):
    """One finished ultralytics run folder: weights/best.pt and args.yaml."""
    _write(os.path.join(run_dir, "weights", "best.pt"), b"weights")
    _write(os.path.join(run_dir, "args.yaml"), f"task: {task}\nimgsz: {imgsz}\n")
    return run_dir


def _make_project_state(project_dir, name="fx"):
    """A loaded-project dict of the shape app.py keeps, with every step dir made."""
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    return {"id": os.path.basename(project_dir), "name": name, "project_dir": project_dir,
            "created_at": "x", "updated_at": "x", "current_step": 7, "steps": steps}


def _make_fixture():
    """One inprocess root holding an open project, a model lab, a lab without a
    dataset, a second project, and a run folder outside the root entirely."""
    tmp = tempfile.mkdtemp(prefix="t9_foreign_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")

    project_dir = os.path.join(root, "run_26-09-15_LO_train_fx")
    state = _make_project_state(project_dir)
    step6 = state["steps"]["6"]["dir"]
    step7 = state["steps"]["7"]["dir"]
    own_run = _make_run(os.path.join(step6, "runs", "own_run"), task="segment", imgsz=512)
    # A second run of the same project, so a refused pick can be told apart
    # from the run the project has already evaluated.
    own_run_b = _make_run(os.path.join(step6, "runs", "own_run_b"), task="segment", imgsz=512)
    _write(os.path.join(step6, "dataset", "data.yaml"),
           "path: .\ntrain: train/images\nval: valid/images\nnc: 1\nnames:\n  0: PA\n")

    lab_dir = os.path.join(root, "_models", "fish_lab")
    lab_run = _make_run(os.path.join(lab_dir, "runs", "lab_run"), task="detect", imgsz=640)
    _write(os.path.join(lab_dir, "dataset", "data.yaml"),
           "path: .\ntrain: train/images\nval: valid/images\nnc: 1\nnames:\n  0: EPST_Dark\n")
    _write(os.path.join(lab_dir, "dataset", "split_manifest.json"), json.dumps({
        "schema_version": 2, "seed_name": "fish_photo_28feb25",
        "split_fingerprint": "0123456789abcdef",
        "split_fingerprint_scheme": "sha256-16/netstring-utf8-sorted-v1",
        "pins_source": "staged split_pins.json",
    }))

    bare_lab_dir = os.path.join(root, "_models", "bare_lab")
    bare_run = _make_run(os.path.join(bare_lab_dir, "runs", "bare_run"))

    other_project = os.path.join(root, "run_26-09-10_LO_train_other")
    other_state = _make_project_state(other_project, name="other")
    other_step6 = other_state["steps"]["6"]["dir"]
    other_run = _make_run(os.path.join(other_step6, "runs", "other_run"))
    _write(os.path.join(other_step6, "dataset", "data.yaml"), "nc: 1\nnames:\n  0: PA\n")

    outside_dir = os.path.join(tmp, "elsewhere")
    outside_run = _make_run(os.path.join(outside_dir, "outside_run"))
    link_run = os.path.join(outside_dir, "link_to_lab_run")
    os.symlink(lab_run, link_run)

    return {"tmp": tmp, "root": root, "state": state, "project_dir": project_dir,
            "step6": step6, "step7": step7, "own_run": own_run, "own_run_b": own_run_b,
            "lab_dir": lab_dir, "lab_run": lab_run,
            "bare_lab_dir": bare_lab_dir, "bare_run": bare_run,
            "other_state": other_state, "other_project": other_project,
            "other_step6": other_step6, "other_run": other_run,
            "outside_dir": outside_dir, "outside_run": outside_run, "link_run": link_run}


def _client(fx):
    """A test client with the inprocess root and the open project on the fixture.

    The recorded VICARIUS process events are emptied here, so each test counts
    only the events its own calls emitted.
    """
    appmod._inprocess_root = lambda: fx["root"]
    appmod.current_project = fx["state"]
    _FakeVic.ends.clear()
    return appmod.create_app().test_client()


def _step7_end_events():
    """Every VICARIUS process_end event recorded for the Evaluate step."""
    return [e for e in _FakeVic.ends if e["step_name"].startswith("step7")]


def _close(fx):
    """Restore the module state and remove the fixture tree.

    What the orchestrator process remembers about the last Evaluate launch is
    dropped too, so a launch one test never polled cannot be read by the next
    one; a real orchestrator forgets it the same way, by exiting.
    """
    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    appmod._evaluation_launches.clear()
    shutil.rmtree(fx["tmp"], ignore_errors=True)


def _evaluate(client, config, launch_result=None):
    """PUT the Step 7 config, POST the run with the stage runner captured.

    Parameters:
        client: the Flask test client.
        config: the Step 7 config dict to save before the run.
        launch_result: what the stage runner answers when the route hands it
            the command; None means a launch that started (a pid). Pass a
            busy result to model a press while the step's process is alive.
    Returns:
        (response, argv) where argv is the command the route would have
        launched, or [] when it refused before launching.
    """
    captured = {}
    real = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        return dict(launch_result) if launch_result else {"pid": 7}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/7", json=config)
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        return client.post("/api/step/7/run"), captured.get("cmd") or []
    finally:
        appmod.runner.run_cli_stage = real


def _argv_value(cmd, flag):
    """The value following flag in an argv list, or None when it is absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _error_of(response):
    """The error string of a refused response, or '' when it launched."""
    return str((response.get_json() or {}).get("error") or "")


def _outputs_snapshot():
    """A copy of the open project's Step 7 output pointers."""
    return dict(appmod.current_project["steps"]["7"].get("outputs") or {})


def _saved_project():
    """The open project as project.json holds it on disk."""
    path = os.path.join(appmod.current_project["project_dir"], "project.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class _FinishedRunner:
    """The stage runner as it answers once a step's process has exited.

    Swapped in for the poll only: the launch itself goes through the real
    runner object with run_cli_stage captured, so nothing here can spawn a
    process or touch a live one.
    """

    ports = {}

    def __init__(self, exit_code=0):
        """exit_code: what the finished process ended on; 0 means success."""
        self.exit_code = exit_code

    def poll_status(self, _step):
        """A process that is no longer running, with the code it exited on."""
        return {"running": False, "exit_code": self.exit_code, "log_tail": []}

    def health_check(self, _port):
        """No Flask sub-app is involved in an evaluation."""
        return False


def _poll_step(client, step, exit_code=0):
    """GET one step's status with its process reported as finished.

    Parameters:
        client: the Flask test client.
        step: the step key to poll.
        exit_code: the code the finished process exited on.
    Returns:
        the parsed status body the route answered with.
    """
    real = appmod.runner
    appmod.runner = _FinishedRunner(exit_code)
    try:
        return client.get(f"/api/step/{step}/status").get_json() or {}
    finally:
        appmod.runner = real


def _step_record(state, step="7"):
    """The fields of one step that say whether work was recorded for a project.

    Parameters:
        state: the project dict to read (the open project, or project.json as
            it was read back from disk).
        step: the step key to read.
    Returns:
        {"status", "completed_at", "outputs", "current_step", "next_status"},
        the five things pm.complete_step writes or unlocks: the step's own
        status and completion stamp, its output pointers, the project's
        current step, and the status of the step behind it.
    """
    st = (state.get("steps") or {}).get(step) or {}
    behind = (state.get("steps") or {}).get(str(int(step) + 1)) or {}
    return {"status": st.get("status"), "completed_at": st.get("completed_at"),
            "outputs": dict(st.get("outputs") or {}),
            "current_step": state.get("current_step"),
            "next_status": behind.get("status")}


def _open_record(step="7"):
    """_step_record for the project the orchestrator holds in memory."""
    return _step_record(appmod.current_project, step)


def _saved_record(step="7"):
    """_step_record for the project as project.json holds it on disk."""
    return _step_record(_saved_project(), step)


def _notes_of(status):
    """Every sentence a status body carries, joined into one string."""
    return " ".join(str(w) for w in (status.get("warnings") or []))


def _fake_evaluation(out_dir, run_dir, overall, family):
    """Stand in for the evaluator, which this suite never starts.

    Parameters:
        out_dir: the evaluation folder the route chose for this run.
        run_dir: the run that was scored; recorded under context.run_dir the
            way the evaluator records it.
        overall: the overall metrics block, for example {"box_map50": 0.9}.
        family: the metric family the evaluation reports, "box" or "mask".
    Returns:
        out_dir, so a caller can assert on the folder it just filled.
    """
    run_name = os.path.basename(os.path.normpath(run_dir))
    _write(os.path.join(out_dir, "report.md"), f"# Report for {run_name}\n")
    _write(os.path.join(out_dir, "metrics.json"), json.dumps({
        "metrics": {"overall": dict(overall), "per_class": {},
                    "per_class_instances": {}, "absent_classes": [],
                    "metric_family": family, "class_names": {},
                    "class_full_names": {}},
        "context": {"run_dir": run_dir},
        "previews": [], "args": {},
        "generated_at": "2026-09-15T01:00:00-04:00",
    }))
    return out_dir


# ---------------------------------------------------------------------------
# OWN RUN: the regression guard
# ---------------------------------------------------------------------------

def test_own_run_command_and_pointers_are_unchanged():
    """A run of the open project keeps the project's dataset, ledger, eval folder and pointers."""
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["own_run"], "imgsz": 512})
        check(r.status_code == 200, f"an own run still launches: {r.status_code} {r.get_json()}")
        check(_argv_value(cmd, "--dataset_dir") == os.path.join(fx["step6"], "dataset"),
              f"an own run is scored on the project's dataset: {cmd}")
        check(_argv_value(cmd, "--rounds_dir") == fx["step6"],
              f"an own run's ledger row goes to the project's step 6 folder: {cmd}")
        check(_argv_value(cmd, "--out_dir") == os.path.join(fx["step7"], "eval_own_run"),
              f"an own run's evaluation is written to <step 7>/eval_<run name>: {cmd}")
        check(_argv_value(cmd, "--run_dir") == fx["own_run"],
              f"the run folder reaches the evaluator as picked: {cmd}")
        check(os.path.isdir(os.path.join(fx["step7"], "eval_own_run")),
              "the evaluation folder is created before the evaluator starts")

        outputs = _outputs_snapshot()
        check(outputs.get("out_dir") == os.path.join(fx["step7"], "eval_own_run"),
              f"the project records its own evaluation folder: {outputs}")
        check(outputs.get("report_md") == os.path.join(fx["step7"], "eval_own_run", "report.md"),
              f"the project records its own report.md: {outputs}")
        check(outputs.get("report_pdf") == os.path.join(fx["step7"], "eval_own_run", "report.pdf"),
              f"the project records its own report.pdf: {outputs}")
        check(outputs.get("metrics_json") == os.path.join(fx["step7"], "eval_own_run", "metrics.json"),
              f"the project records its own metrics.json: {outputs}")
        check(outputs.get("rounds_csv") == os.path.join(fx["step6"], "rounds.csv"),
              f"the project records its own ledger: {outputs}")
        check(outputs.get("champion_json") == os.path.join(fx["step6"], "champion.json"),
              f"the project records its own champion file: {outputs}")
        check(appmod.current_project["steps"]["7"]["config"]["run_dir"] == fx["own_run"],
              "the panel remembers the selection")
    finally:
        _close(fx)


def test_own_run_gets_no_foreign_warning():
    """The note about another dataset appears only when the run really is another's."""
    fx = _make_fixture()
    try:
        r, _cmd = _evaluate(_client(fx), {"run_dir": fx["own_run"], "imgsz": 512})
        warnings = " ".join((r.get_json() or {}).get("warnings") or [])
        check("not part of the open project" not in warnings.lower(),
              f"an own run is not announced as foreign: {warnings!r}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# LAB RUN: the run the operator picked decides every path
# ---------------------------------------------------------------------------

def test_lab_run_is_scored_on_its_own_dataset_and_ledger():
    """A model lab's run uses the lab's dataset, the lab's ledger and the shared _eval folder."""
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["lab_run"], "imgsz": 640})
        check(r.status_code == 200, f"a lab run launches: {r.status_code} {r.get_json()}")
        check(_argv_value(cmd, "--dataset_dir") == os.path.join(fx["lab_dir"], "dataset"),
              f"a lab run is scored on the lab's own dataset: {cmd}")
        check(_argv_value(cmd, "--rounds_dir") == fx["lab_dir"],
              f"a lab run's ledger row goes to the lab's own rounds.csv: {cmd}")

        out_dir = _argv_value(cmd, "--out_dir") or ""
        eval_root = os.path.join(fx["root"], *EVAL_ROOT_PARTS)
        check(os.path.dirname(out_dir) == eval_root,
              f"a lab run's evaluation is written under inprocess/_models/_eval: {out_dir}")
        check(os.path.basename(out_dir).startswith("lab_run"),
              f"the evaluation folder is named for the run so the report route finds it: {out_dir}")
        check(os.path.isdir(out_dir), "the evaluation folder is created before the evaluator starts")
        saved = appmod.current_project["steps"]["7"]["config"].get("run_dir")
        check(saved != fx["lab_run"],
              f"a run that is not this project's must not stay in the project's saved "
              f"Step 7 selection, where it would be paired with this project's own "
              f"evaluation folder: {saved!r}")
    finally:
        _close(fx)


def test_lab_run_leaves_the_projects_step7_pointers_alone():
    """Evaluating someone else's run must not repoint the project's own report."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        before = _outputs_snapshot()
        check(before.get("out_dir"), "the project has its own pointers to protect")

        r, cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        check(r.status_code == 200, f"a lab run launches: {r.get_json()}")
        after = _outputs_snapshot()
        for key in STEP7_OUTPUT_KEYS:
            check(after.get(key) == before.get(key),
                  f"steps.7.outputs.{key} must survive a lab evaluation unchanged: "
                  f"{before.get(key)!r} became {after.get(key)!r}")
        check(fx["lab_dir"] not in json.dumps(after),
              f"no pointer outside the project is recorded: {after}")
    finally:
        _close(fx)


def test_lab_run_warning_names_the_dataset_and_the_ledger():
    """The operator is told, in one sentence, that this was not the project's data."""
    fx = _make_fixture()
    try:
        r, _cmd = _evaluate(_client(fx), {"run_dir": fx["lab_run"], "imgsz": 640})
        warnings = [str(w) for w in (r.get_json() or {}).get("warnings") or []]
        joined = " ".join(warnings)
        check(os.path.join(fx["lab_dir"], "dataset") in joined,
              f"the note names the dataset the run was scored on: {warnings!r}")
        check(os.path.join(fx["lab_dir"], "rounds.csv") in joined,
              f"the note names the ledger the row goes to: {warnings!r}")
        check(any("not part of the open project" in w.lower() for w in warnings),
              f"the note says plainly that the run is not this project's: {warnings!r}")
    finally:
        _close(fx)


def test_a_run_of_another_project_is_treated_as_foreign():
    """Another project's run is scored on that project's dataset and ledger, not this one's."""
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["other_run"], "imgsz": 640})
        check(r.status_code == 200, f"another project's run launches: {r.get_json()}")
        check(_argv_value(cmd, "--dataset_dir") == os.path.join(fx["other_step6"], "dataset"),
              f"it is scored on that project's dataset: {cmd}")
        check(_argv_value(cmd, "--rounds_dir") == fx["other_step6"],
              f"its ledger row goes to that project's step 6 folder: {cmd}")
        check(os.path.dirname(_argv_value(cmd, "--out_dir") or "")
              == os.path.join(fx["root"], *EVAL_ROOT_PARTS),
              f"its evaluation is written to the shared _eval folder: {cmd}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# REFUSALS
# ---------------------------------------------------------------------------

def test_a_lab_without_a_dataset_is_refused_by_name():
    """A lab whose dataset folder is missing names the run, never the open project's step 6."""
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["bare_run"], "imgsz": 640})
        err = _error_of(r)
        check(r.status_code == appmod.HTTP_BAD_REQUEST,
              f"a lab without a dataset is a bad request: {r.status_code} {err!r}")
        check(not cmd, f"nothing launched: {cmd}")
        check("bare_run" in err, f"the refusal names the run: {err!r}")
        check("dataset" in err.lower(), f"the refusal names the missing dataset: {err!r}")
        check(os.path.join(fx["bare_lab_dir"], "dataset") in err,
              f"the refusal says where the dataset folder was expected: {err!r}")
        check(fx["step6"] not in err,
              f"the refusal must not point at the open project's step 6 folder: {err!r}")
        check("has step" not in err.lower(),
              f"the refusal must not tell the operator to run a step that is not theirs: {err!r}")
    finally:
        _close(fx)


def test_a_run_outside_the_inprocess_root_is_refused():
    """A run folder elsewhere on the box is refused by containment, and tells nothing about it."""
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["outside_run"], "imgsz": 640})
        err = _error_of(r)
        check(r.status_code == appmod.HTTP_BAD_REQUEST,
              f"a run outside the module is a bad request: {r.status_code} {err!r}")
        check(not cmd, f"nothing launched: {cmd}")
        check("outside" in err.lower() and "root" in err.lower(),
              f"the refusal says the folder is outside the runs root: {err!r}")
        # The containment check runs before any look inside the folder, so the
        # refusal reports neither that the folder exists nor what it holds: a
        # typed path cannot be used to probe the box.
        for leak in ("weights", "best.pt", "not found"):
            check(leak not in err.lower(),
                  f"the refusal must not report the contents of an unrelated folder "
                  f"(found {leak!r} in {err!r})")
    finally:
        _close(fx)


def test_a_traversal_path_out_of_the_project_is_refused():
    """A '..' path that climbs out of the project cannot pass as one of its runs."""
    fx = _make_fixture()
    try:
        escape = os.path.join(fx["step6"], "runs", "..", "..", "..", "..",
                              "elsewhere", "outside_run")
        r, cmd = _evaluate(_client(fx), {"run_dir": escape, "imgsz": 640})
        check(r.status_code == appmod.HTTP_BAD_REQUEST,
              f"a traversal path is refused: {r.status_code} {_error_of(r)!r}")
        check(not cmd, f"nothing launched: {cmd}")
    finally:
        _close(fx)


def test_a_symlink_into_the_lab_from_outside_is_refused():
    """A link outside the module pointing at a lab run is refused, not followed."""
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["link_run"], "imgsz": 640})
        err = _error_of(r)
        check(r.status_code == appmod.HTTP_BAD_REQUEST,
              f"a link from outside the module is refused: {r.status_code} {err!r}")
        check(not cmd, f"nothing launched: {cmd}")
        check("link" in err.lower() or "outside" in err.lower(),
              f"the refusal says why the selection cannot be used: {err!r}")
    finally:
        _close(fx)


def test_empty_selection_keeps_the_existing_wording():
    """No run picked: the panel's own wording, naming where this project's runs live."""
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": "", "imgsz": 640})
        err = _error_of(r)
        check(not cmd, f"nothing launched: {cmd}")
        check(err == ("Select a training run to evaluate. Expected a path like "
                      f"{fx['step6']}/runs/<run_name>/"),
              f"the empty-selection wording is unchanged: {err!r}")
    finally:
        _close(fx)


def test_own_run_without_a_dataset_keeps_the_existing_message():
    """A project that has not built its split yet still reads 'No dataset at ...'."""
    fx = _make_fixture()
    try:
        shutil.rmtree(os.path.join(fx["step6"], "dataset"))
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["own_run"], "imgsz": 512})
        err = _error_of(r)
        check(not cmd, f"nothing launched: {cmd}")
        check(err == (f"No dataset at {os.path.join(fx['step6'], 'dataset')}. "
                      f"Has {appmod.step_ref(6)} run?"),
              f"the existing message is unchanged: {err!r}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# KNOBS: every evaluation setting reaches the command for both kinds of run
# ---------------------------------------------------------------------------

def test_every_knob_reaches_the_command_for_both_kinds_of_run():
    """Image size, split, batch, confidence, IoU and preview count are forwarded identically."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        knobs = {"imgsz": 320, "split": "val", "batch": 4, "conf_threshold": 0.001,
                 "iou_threshold": 0.7, "preview_count": 3}
        expected = {"--imgsz": "320", "--split": "val", "--batch": "4",
                    "--conf": "0.001", "--iou": "0.7", "--preview_count": "3"}
        for label, run_dir in (("own", fx["own_run"]), ("lab", fx["lab_run"])):
            r, cmd = _evaluate(client, dict(knobs, run_dir=run_dir))
            check(r.status_code == 200, f"{label} run launches: {r.get_json()}")
            for flag, value in expected.items():
                check(_argv_value(cmd, flag) == value,
                      f"{label} run forwards {flag}={value}: got "
                      f"{_argv_value(cmd, flag)!r} in {cmd}")
    finally:
        _close(fx)


def test_pdf_export_dir_is_forwarded_for_both_kinds_of_run():
    """An export folder typed in the panel still reaches the evaluator for a lab run."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        export = os.path.join(fx["tmp"], "exports")
        os.makedirs(export, exist_ok=True)
        for label, run_dir in (("own", fx["own_run"]), ("lab", fx["lab_run"])):
            _r, cmd = _evaluate(client, {"run_dir": run_dir, "imgsz": 512,
                                         "pdf_export_dir": export})
            check(_argv_value(cmd, "--pdf_export_dir") == export,
                  f"{label} run forwards the export folder: {cmd}")
    finally:
        _close(fx)


def test_blank_image_size_follows_the_picked_runs_args_yaml():
    """A blank size fills from the run that was picked, not from the project's own run."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": ""})
        check(_argv_value(own_cmd, "--imgsz") == "512",
              f"the own run's recorded size is used: {own_cmd}")
        _r, lab_cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": ""})
        check(_argv_value(lab_cmd, "--imgsz") == "640",
              f"the lab run's recorded size is used: {lab_cmd}")
    finally:
        _close(fx)


def test_the_report_card_finds_a_lab_runs_evaluation_where_it_was_written():
    """End to end: the folder the route chose is the folder GET /api/step/7/report reads."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        out_dir = _argv_value(cmd, "--out_dir") or ""
        check(out_dir, f"the lab evaluation chose an output folder: {cmd}")
        _fake_evaluation(out_dir, fx["lab_run"], {"box_map50": 0.904}, "box")
        r = client.get("/api/step/7/report", query_string={"run_dir": fx["lab_run"]})
        body = r.get_json() or {}
        check(r.status_code == 200, f"the report card finds the lab evaluation: {r.status_code} {body}")
        check(body.get("out_dir") == out_dir,
              f"the report card reads the folder the evaluation wrote: {body.get('out_dir')!r} "
              f"vs {out_dir!r}")
    finally:
        _close(fx)


def test_detector_family_note_still_appears_for_a_lab_run():
    """The box-family note the ledger needs is not lost on the foreign path."""
    fx = _make_fixture()
    try:
        r, _cmd = _evaluate(_client(fx), {"run_dir": fx["lab_run"], "imgsz": 640})
        warnings = " ".join((r.get_json() or {}).get("warnings") or []).lower()
        check("box" in warnings and "mask" in warnings,
              f"a detector lab run still says which metric family it is scored in: {warnings!r}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# POINTERS: the project's saved pair always describes one evaluation
# ---------------------------------------------------------------------------

def test_report_without_a_run_dir_names_the_projects_own_evaluation():
    """After evaluating someone else's run, this project still reports its own.

    The report card asks GET /api/step/7/report with no run_dir whenever the
    Evaluate panel opens with nothing selected, which is what a page reload
    gives: /api/step/6/list_runs holds only this project's runs, so a foreign
    path cannot be restored into the select. The answer must pair this
    project's own evaluation folder with this project's own run name and this
    project's own numbers.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        own_eval = _argv_value(own_cmd, "--out_dir") or ""
        _fake_evaluation(own_eval, fx["own_run"], {"mask_map50": 0.111}, "mask")

        _r, lab_cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        lab_eval = _argv_value(lab_cmd, "--out_dir") or ""
        check(lab_eval and lab_eval != own_eval, "the lab evaluation chose its own folder")
        _fake_evaluation(lab_eval, fx["lab_run"], {"box_map50": 0.999}, "box")
        on_disk = _saved_project()["steps"]["7"]["config"].get("run_dir")
        check(on_disk != fx["lab_run"],
              f"the lab pick is gone from project.json too, so a restart cannot "
              f"restore the wrong pair: {on_disk!r}")

        r = client.get("/api/step/7/report")
        body = r.get_json() or {}
        check(r.status_code == 200,
              f"the project's own report is still served: {r.status_code} {body}")
        check(body.get("out_dir") == own_eval,
              f"the folder served is the project's own evaluation: {body.get('out_dir')!r}")
        check(body.get("run_name") == "own_run",
              f"the name served is the project's own run: {body.get('run_name')!r}")
        check(body.get("metrics", {}).get("overall") == {"mask_map50": 0.111},
              f"the numbers served come from the same evaluation as the name: "
              f"{body.get('metrics', {}).get('overall')!r}")
    finally:
        _close(fx)


def test_a_refused_foreign_run_is_not_left_in_the_saved_selection():
    """A refused foreign pick must not stay in project.json as this project's selection.

    The client saves the picked run through PUT /api/project/config/7 before
    the run route sees it, so a refusal would otherwise leave another run's
    path beside this project's own evaluation folder, on disk and across a
    restart.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        own_eval = _argv_value(own_cmd, "--out_dir") or ""
        _fake_evaluation(own_eval, fx["own_run"], {"mask_map50": 0.111}, "mask")

        r, cmd = _evaluate(client, {"run_dir": fx["bare_run"], "imgsz": 640})
        check(r.status_code == appmod.HTTP_BAD_REQUEST, f"the pick is refused: {r.status_code}")
        check(not cmd, f"nothing launched: {cmd}")

        saved = _saved_project()["steps"]["7"]
        check(saved["config"].get("run_dir") != fx["bare_run"],
              f"the refused foreign pick is gone from project.json: {saved['config']!r}")
        check(saved["outputs"].get("out_dir") == own_eval,
              f"the project keeps its own evaluation folder: {saved['outputs']!r}")

        body = client.get("/api/step/7/report").get_json() or {}
        check(body.get("run_name") == "own_run",
              f"the report card still names this project's own run: {body.get('run_name')!r}")
        check(body.get("out_dir") == own_eval,
              f"and serves this project's own evaluation: {body.get('out_dir')!r}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# COMPLETION: only the project's own evaluation is recorded as its work
# ---------------------------------------------------------------------------

def test_a_foreign_evaluation_leaves_the_projects_step_exactly_as_it_was():
    """A finished lab evaluation writes nothing into the open project's history.

    The project has already evaluated its own run, so it carries a completed
    Evaluate step with a stamp, its own output pointers, Inference unlocked
    behind it and a current step past Evaluate. Scoring a lab run afterwards
    must leave every one of those exactly as the launch found them, in memory
    and in project.json.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        appmod.current_project["steps"]["8"]["status"] = "locked"
        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        _fake_evaluation(_argv_value(own_cmd, "--out_dir") or "", fx["own_run"],
                         {"mask_map50": 0.111}, "mask")
        first = _poll_step(client, 7)
        check(first.get("completed") is True,
              f"the project's own evaluation completes the step: {first}")
        before, before_disk = _open_record(), _saved_record()
        check(before["status"] == "completed" and before["completed_at"],
              f"the project starts this test with a completed, stamped step: {before}")

        r, cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        check(r.status_code == 200, f"the lab evaluation launches: {r.get_json()}")
        check(cmd, f"the lab evaluation was handed to the runner: {cmd}")
        check(_open_record()["status"] == "running",
              "the launch stamps the step running, as every launch does")

        status = _poll_step(client, 7)
        check(status.get("running") is False and status.get("exit_code") == 0,
              f"the poll sees a finished, successful process: {status}")
        after, after_disk = _open_record(), _saved_record()
        for key in ("status", "completed_at", "outputs", "current_step", "next_status"):
            check(after[key] == before[key],
                  f"the open project's {key} must survive a foreign evaluation unchanged: "
                  f"{before[key]!r} became {after[key]!r}")
            check(after_disk[key] == before_disk[key],
                  f"project.json's {key} must survive a foreign evaluation unchanged: "
                  f"{before_disk[key]!r} became {after_disk[key]!r}")
        check(status.get("completed") is not True,
              f"the poll must not report the project's step as completed: {status}")
    finally:
        _close(fx)


def test_a_foreign_evaluation_reports_success_and_names_the_run():
    """The panel is told the run finished, and why the project's step stands still."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        appmod.current_project["steps"]["8"]["status"] = "locked"
        before = _open_record()
        r, _cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        check(r.status_code == 200, f"the lab evaluation launches: {r.get_json()}")

        status = _poll_step(client, 7)
        check(status.get("running") is False and status.get("exit_code") == 0,
              f"the run is reported as finished and successful, so the panel stops "
              f"polling and reloads its report card: {status}")
        notes = _notes_of(status)
        check("lab_run" in notes, f"the sentence names the run that was scored: {notes!r}")
        check("open project" in notes.lower(),
              f"the sentence says the run is not this project's: {notes!r}")

        after = _open_record()
        check(after["status"] == before["status"] == "pending",
              f"a project that has never evaluated is still pending: {after}")
        check(after["completed_at"] is None, f"no completion stamp was written: {after}")
        check(after["outputs"] == {}, f"no output pointer was written: {after}")
        check(after["current_step"] == before["current_step"],
              f"the project's current step did not move: {after}")
        check(after["next_status"] == "locked",
              f"Inference stays locked behind an evaluation the project did not do: {after}")
        check(_saved_record() == after,
              f"project.json agrees with what the orchestrator holds: {_saved_record()}")
    finally:
        _close(fx)


def test_an_owned_evaluation_still_completes_the_step():
    """The regression guard: the project's own run completes exactly as before.

    Same completion, same stamp, same auto-linked runs folder, same unlocked
    Inference step, and the same response keys the poll has always answered
    with.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        runs_dir = os.path.join(fx["step6"], "runs")
        appmod.current_project["steps"]["6"]["outputs"]["runs_dir"] = runs_dir
        appmod.current_project["steps"]["8"]["status"] = "locked"
        r, cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        check(r.status_code == 200, f"the project's own run launches: {r.get_json()}")
        out_dir = _argv_value(cmd, "--out_dir") or ""

        status = _poll_step(client, 7)
        check(status.get("completed") is True, f"the poll reports the completion: {status}")
        check(sorted(status) == ["completed", "exit_code", "log_tail", "running"],
              f"the response keys are the ones this poll has always answered with: "
              f"{sorted(status)}")
        record = _open_record()
        check(record["status"] == "completed", f"the step is completed: {record}")
        check(record["completed_at"], f"the completion is stamped: {record}")
        datetime.fromisoformat(record["completed_at"])
        check(record["current_step"] == 8, f"the project moves on to Inference: {record}")
        check(record["next_status"] == "pending", f"Inference is unlocked: {record}")
        check(appmod.current_project["steps"]["8"]["config"].get("runs_dir") == runs_dir,
              "auto_link_outputs wired the runs folder into the Inference step")
        check(record["outputs"].get("out_dir") == out_dir,
              f"the project keeps its own evaluation folder: {record['outputs']}")
        check(_saved_record() == record,
              f"project.json records all of it: {_saved_record()}")
    finally:
        _close(fx)


def test_an_owned_evaluation_after_a_foreign_one_still_completes():
    """The remembered fact is spent by the poll that reads it, not kept."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, _cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        foreign = _poll_step(client, 7)
        check(foreign.get("completed") is not True,
              f"the foreign evaluation records nothing: {foreign}")

        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        owned = _poll_step(client, 7)
        check(owned.get("completed") is True,
              f"the project's own evaluation, run next, completes normally: {owned}")
        record = _open_record()
        check(record["status"] == "completed" and record["completed_at"],
              f"the step is completed and stamped: {record}")
        check(record["outputs"].get("out_dir") == (_argv_value(own_cmd, "--out_dir") or ""),
              f"pointing at the project's own evaluation: {record['outputs']}")
    finally:
        _close(fx)


def test_a_failed_foreign_evaluation_does_not_complete_the_step():
    """A non-zero exit is reported the way it always was, and completes nothing."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        before = _open_record()
        _r, _cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        status = _poll_step(client, 7, exit_code=3)
        check(status.get("exit_code") == 3 and status.get("running") is False,
              f"the failure is reported with its exit code: {status}")
        check(status.get("completed") is not True, f"nothing is completed: {status}")
        check(sorted(status) == ["exit_code", "log_tail", "running"],
              f"a failed poll answers the keys it always did: {sorted(status)}")
        record = _open_record()
        check(record["status"] == "error", f"the step reads as errored, as it does today: {record}")
        check(record["completed_at"] is None, f"no completion stamp: {record}")
        check(record["outputs"] == before["outputs"], f"no output pointer moved: {record}")
        check(record["current_step"] == before["current_step"],
              f"the project's current step did not move: {record}")
    finally:
        _close(fx)


def test_other_steps_are_untouched_by_the_evaluation_rule():
    """Train completes on a zero exit, and does not spend the evaluation's record."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        r, _cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        check(r.status_code == 200, f"a foreign evaluation is in flight: {r.get_json()}")
        appmod.current_project["steps"]["6"]["status"] = "running"

        six = _poll_step(client, 6)
        check(six.get("completed") is True, f"Train still completes on a zero exit: {six}")
        check(_open_record("6")["status"] == "completed",
              f"and is recorded as completed: {_open_record('6')}")
        check(_open_record("6")["completed_at"], "with its stamp")

        status = _poll_step(client, 7)
        check("lab_run" in _notes_of(status),
              f"the Train poll did not spend what the evaluation launch remembered: "
              f"{_notes_of(status)!r}")
    finally:
        _close(fx)


def test_a_forgotten_launch_does_not_complete_the_step():
    """Nothing is recorded when nothing can say whose run was scored.

    A real orchestrator restart also loses the process the runner was
    watching, so the poll that follows reads no exit code and the step is
    marked errored. This pins the answer for the one poll that could still
    arrive with a zero exit and no record: say plainly that the run cannot be
    identified and leave the project's step alone, because completing it
    would write the very claim this rule exists to prevent.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        before = _open_record()
        r, _cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        check(r.status_code == 200, f"the project's own run launches: {r.get_json()}")
        appmod._evaluation_launches.clear()  # stands in for the restart

        status = _poll_step(client, 7)
        check(status.get("completed") is not True, f"nothing is completed: {status}")
        record = _open_record()
        check(record["status"] != "completed", f"the step is not marked completed: {record}")
        check(record["completed_at"] is None, f"no completion stamp: {record}")
        check(record["current_step"] == before["current_step"],
              f"the project's current step did not move: {record}")
        notes = _notes_of(status).lower()
        check("cannot tell which" in notes,
              f"the sentence says what is not known: {notes!r}")
        check("restart" not in notes,
              f"and claims no cause it cannot prove, because a lost record also "
              f"follows a poll taken while another project was open: {notes!r}")
    finally:
        _close(fx)


def test_a_launch_remembered_for_another_project_never_completes_this_one():
    """A record made while another project was open cannot complete this one."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        r, _cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        check(r.status_code == 200, f"the first project's own run launches: {r.get_json()}")

        # The operator opens a different project while the evaluation runs.
        appmod.current_project = fx["other_state"]
        appmod.current_project["steps"]["7"]["status"] = "running"
        status = _poll_step(client, 7)
        check(status.get("completed") is not True,
              f"the second project is not credited with the first one's work: {status}")
        check(_open_record()["status"] != "completed",
              f"its Evaluate step is not completed: {_open_record()}")
        check(_open_record()["completed_at"] is None, "and is not stamped")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# POINTERS: a refused launch of the project's OWN run (the Task 11 second item)
# ---------------------------------------------------------------------------

def test_a_refused_own_run_leaves_the_saved_pair_describing_one_evaluation():
    """No foreign run in sight, and the saved pair must still agree.

    The project evaluates its own run. The operator then picks a second run
    of the same project and the launch is refused, here because the split has
    not been built. The client saved that pick before the route saw it, so
    the project would otherwise hold run own_run_b beside run own_run's
    evaluation folder, and the report card asked without a run_dir would
    serve own_run's numbers under own_run_b's name.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        own_eval = _argv_value(own_cmd, "--out_dir") or ""
        _fake_evaluation(own_eval, fx["own_run"], {"mask_map50": 0.111}, "mask")

        shutil.rmtree(os.path.join(fx["step6"], "dataset"))
        r, cmd = _evaluate(client, {"run_dir": fx["own_run_b"], "imgsz": 512})
        check(r.status_code == appmod.HTTP_BAD_REQUEST,
              f"the launch is refused: {r.status_code} {_error_of(r)!r}")
        check(not cmd, f"nothing launched: {cmd}")

        saved = _saved_project()["steps"]["7"]
        pointed = str(saved["outputs"].get("out_dir") or "")
        check(pointed == own_eval,
              f"the project keeps its own evaluation folder: {saved['outputs']!r}")
        selected = str(saved["config"].get("run_dir") or "")
        folder_leaf = os.path.basename(os.path.normpath(pointed))
        folder_run = (folder_leaf[len("eval_"):] if folder_leaf.startswith("eval_")
                      else folder_leaf)
        check(not selected or os.path.basename(os.path.normpath(selected)) == folder_run,
              f"the saved selection {selected!r} names a different evaluation from the "
              f"saved folder {pointed!r}")

        body = client.get("/api/step/7/report").get_json() or {}
        check(body.get("run_name") == "own_run",
              f"the report card names the run whose numbers it serves: "
              f"{body.get('run_name')!r}")
        check(body.get("out_dir") == own_eval,
              f"and serves the project's own evaluation: {body.get('out_dir')!r}")
        check(body.get("metrics", {}).get("overall") == {"mask_map50": 0.111},
              f"name and numbers come from the same evaluation: "
              f"{body.get('metrics', {}).get('overall')!r}")
    finally:
        _close(fx)


def test_a_busy_refusal_keeps_the_projects_evaluation_pointers():
    """A press while an evaluation is running repoints nothing.

    The runner refuses a launch whose step still has a live process. Nothing
    was spawned, so the project's saved evaluation folder must still be the
    one that holds numbers, not an empty folder named for the run that was
    refused.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        own_eval = _argv_value(own_cmd, "--out_dir") or ""
        _fake_evaluation(own_eval, fx["own_run"], {"mask_map50": 0.111}, "mask")

        busy = {"error": "Step 7 is already running", "busy": True}
        r, _cmd = _evaluate(client, {"run_dir": fx["own_run_b"], "imgsz": 512},
                            launch_result=busy)
        check(r.status_code == appmod.HTTP_CONFLICT,
              f"a press over a live evaluation answers 409: {r.status_code} {r.get_json()}")

        saved = _saved_project()["steps"]["7"]
        check(saved["outputs"].get("out_dir") == own_eval,
              f"the evaluation that holds numbers is still the one pointed at: "
              f"{saved['outputs']!r}")
        selected = str(saved["config"].get("run_dir") or "")
        check("own_run_b" not in selected,
              f"the refused pick is not left as this project's selection: {selected!r}")
        body = client.get("/api/step/7/report").get_json() or {}
        check(body.get("run_name") == "own_run" and body.get("out_dir") == own_eval,
              f"the report card still pairs one name with one folder: {body.get('run_name')!r} "
              f"{body.get('out_dir')!r}")
    finally:
        _close(fx)


def test_a_foreign_evaluation_over_a_running_step_does_not_re_enter():
    """A stale "running" mark is not written back, so the poll fires once.

    pm.load_project does not normalise a step left reading "running" by a
    crash, so an Evaluate press can be made over one. The launch remembers
    that prior status. Writing it back would leave the step reading "running"
    with nothing running, which is a false record of its own, and the very
    next poll would re-enter the completion branch: a second success row in
    the VICARIUS process log for one process, and a second sentence for the
    operator saying the run cannot be identified when it just was.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        appmod.current_project["steps"]["7"]["status"] = "running"
        appmod.current_project["steps"]["8"]["status"] = "locked"
        r, _cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        check(r.status_code == 200, f"the lab evaluation launches: {r.get_json()}")

        first = _poll_step(client, 7)
        check("lab_run" in _notes_of(first),
              f"the first poll names the run that was scored: {_notes_of(first)!r}")
        settled, settled_disk = _open_record(), _saved_record()
        check(settled["status"] == appmod.EVALUATE_NEUTRAL_STATUS,
              f"the step is left not done rather than running: {settled}")
        check(settled_disk["status"] == appmod.EVALUATE_NEUTRAL_STATUS,
              f"and project.json says the same: {settled_disk}")
        after_first = len(_step7_end_events())
        check(after_first == 1,
              f"one process, one closed VICARIUS event: {_step7_end_events()}")

        second = _poll_step(client, 7)
        check(not (second.get("warnings") or []),
              f"the second poll has nothing new to say: {second}")
        check(second.get("completed") is not True,
              f"and completes nothing: {second}")
        check(_open_record() == settled,
              f"the step did not move on the second poll: {_open_record()}")
        check(_saved_record() == settled_disk,
              f"nor did project.json: {_saved_record()}")
        check(len(_step7_end_events()) == after_first,
              f"and no second success row was written for one process: "
              f"{_step7_end_events()}")
    finally:
        _close(fx)


def test_the_process_event_names_the_folder_the_evaluation_wrote():
    """The VICARIUS process log is credited with the artifact that was made.

    A foreign evaluation writes into the module-wide evaluation folder, not
    into the open project, so its closed event carries that folder and the
    sentence saying whose run it was. The project's own evaluation keeps the
    behaviour it always had: the event carries the project's own recorded
    output pointers.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        _r, cmd = _evaluate(client, {"run_dir": fx["lab_run"], "imgsz": 640})
        foreign_eval = _argv_value(cmd, "--out_dir") or ""
        check(foreign_eval, f"the foreign evaluation chose a folder: {cmd}")
        _poll_step(client, 7)

        events = _step7_end_events()
        check(len(events) == 1, f"exactly one event was closed: {events}")
        check(events[0]["status"] == "success",
              f"the process really ran and really succeeded: {events[0]}")
        check(events[0]["outputs"] == [foreign_eval],
              f"the event names the folder the evaluation wrote, not this "
              f"project's outputs: {events[0]['outputs']!r}")
        check("lab_run" in events[0]["notes"],
              f"and its note names the run that was scored: {events[0]['notes']!r}")

        _FakeVic.ends.clear()
        _r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        own_eval = _argv_value(own_cmd, "--out_dir") or ""
        _poll_step(client, 7)
        owned_events = _step7_end_events()
        check(len(owned_events) == 1, f"one event for the owned run: {owned_events}")
        check(owned_events[0]["status"] == "success",
              f"closed as a success, as it always was: {owned_events[0]}")
        check(own_eval in owned_events[0]["outputs"],
              f"the project's own evaluation folder is in the event: "
              f"{owned_events[0]['outputs']!r}")
        check(foreign_eval not in owned_events[0]["outputs"],
              f"and the lab's folder is not: {owned_events[0]['outputs']!r}")
    finally:
        _close(fx)


def test_a_launch_is_still_readable_when_its_own_project_comes_back():
    """A poll taken for another project does not destroy this one's record.

    The operator starts an evaluation of the open project's own run, then
    opens a second project whose Evaluate step still reads "running" from an
    earlier crash. The second project's poller sees the first project's
    process finish. It must record nothing for the second project, and it
    must leave the first project's record where the first project's own poll
    can still read it, or that genuine evaluation is never recorded and the
    operator has to press Evaluate again.
    """
    fx = _make_fixture()
    try:
        client = _client(fx)
        appmod.current_project["steps"]["8"]["status"] = "locked"
        r, own_cmd = _evaluate(client, {"run_dir": fx["own_run"], "imgsz": 512})
        check(r.status_code == 200, f"the project's own run launches: {r.get_json()}")
        own_eval = _argv_value(own_cmd, "--out_dir") or ""

        appmod.current_project = fx["other_state"]
        appmod.current_project["steps"]["7"]["status"] = "running"
        other = _poll_step(client, 7)
        check(other.get("completed") is not True,
              f"the second project is not credited with the first one's work: {other}")
        check(_open_record()["status"] != "completed",
              f"its Evaluate step is not completed: {_open_record()}")

        appmod.current_project = fx["state"]
        owned = _poll_step(client, 7)
        check(owned.get("completed") is True,
              f"the first project's own evaluation is still recorded when it "
              f"comes back: {owned}")
        record = _open_record()
        check(record["status"] == "completed" and record["completed_at"],
              f"the step is completed and stamped: {record}")
        check(record["outputs"].get("out_dir") == own_eval,
              f"pointing at the evaluation that was really run: {record['outputs']}")
    finally:
        _close(fx)


if __name__ == "__main__":
    print("TASKS 9 + 11 -- Evaluate follows the run you picked, and only the "
          "project's own evaluation is recorded as its work\n")
    print("OWN RUN (regression guard):")
    run(test_own_run_command_and_pointers_are_unchanged)
    run(test_own_run_gets_no_foreign_warning)
    print("LAB RUN:")
    run(test_lab_run_is_scored_on_its_own_dataset_and_ledger)
    run(test_lab_run_leaves_the_projects_step7_pointers_alone)
    run(test_lab_run_warning_names_the_dataset_and_the_ledger)
    run(test_a_run_of_another_project_is_treated_as_foreign)
    print("REFUSALS:")
    run(test_a_lab_without_a_dataset_is_refused_by_name)
    run(test_a_run_outside_the_inprocess_root_is_refused)
    run(test_a_traversal_path_out_of_the_project_is_refused)
    run(test_a_symlink_into_the_lab_from_outside_is_refused)
    run(test_empty_selection_keeps_the_existing_wording)
    run(test_own_run_without_a_dataset_keeps_the_existing_message)
    print("POINTERS:")
    run(test_report_without_a_run_dir_names_the_projects_own_evaluation)
    run(test_a_refused_foreign_run_is_not_left_in_the_saved_selection)
    run(test_a_refused_own_run_leaves_the_saved_pair_describing_one_evaluation)
    run(test_a_busy_refusal_keeps_the_projects_evaluation_pointers)
    print("COMPLETION:")
    run(test_an_owned_evaluation_still_completes_the_step)
    run(test_a_foreign_evaluation_leaves_the_projects_step_exactly_as_it_was)
    run(test_a_foreign_evaluation_reports_success_and_names_the_run)
    run(test_an_owned_evaluation_after_a_foreign_one_still_completes)
    run(test_a_failed_foreign_evaluation_does_not_complete_the_step)
    run(test_other_steps_are_untouched_by_the_evaluation_rule)
    run(test_a_forgotten_launch_does_not_complete_the_step)
    run(test_a_launch_remembered_for_another_project_never_completes_this_one)
    run(test_a_launch_is_still_readable_when_its_own_project_comes_back)
    run(test_a_foreign_evaluation_over_a_running_step_does_not_re_enter)
    run(test_the_process_event_names_the_folder_the_evaluation_wrote)
    print("KNOBS:")
    run(test_every_knob_reaches_the_command_for_both_kinds_of_run)
    run(test_pdf_export_dir_is_forwarded_for_both_kinds_of_run)
    run(test_blank_image_size_follows_the_picked_runs_args_yaml)
    run(test_detector_family_note_still_appears_for_a_lab_run)
    print("END TO END:")
    run(test_the_report_card_finds_a_lab_runs_evaluation_where_it_was_written)

    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    if leftovers:
        print(f"\nFIXTURES NOT REMOVED: {leftovers}")
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed; "
          f"fixtures removed: {not leftovers}")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
    sys.exit(1 if failed or leftovers else 0)
