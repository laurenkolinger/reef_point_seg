"""One champion per scope, and every reader says which one it is showing.

Two champion records live on disk and they answer two different questions:

  module   <inprocess>/_models/champion.json: the promoted model. The weights
           Inference pre-selects, the base the fine-tune recipe continues
           from, and the run the Trained runs card ranks every other run
           against. One for the whole module.
  project  <step6_dir>/champion.json beside a project's own rounds ledger:
           what evaluate_run.py measured that ledger's gate columns against.

Before 2026-09-16 the Trained runs card read the first, the Rounds block read
the second, and both called what they found "the champion", so one window
could show two champions and the Evaluate panel could say no champion was
promoted while Train and Inference were both honouring one. That is the live
state this suite pins shut.

Covered here:
  SCOPE       /api/step/6/all_runs, /api/step/7/rounds and
              /api/step/6/list_runs all show the module champion and all say
              so; the project's own record rides beside it as gate_baseline.
  PROMISE     promoting a run from a model lab really does change what
              Inference pre-selects and what GET /api/step/6/list_runs
              reports as the champion for the Train panel's Continue from a
              VEGA model select, which is what the button's own copy
              promises. A champion of the other task is still reported (so
              the client can hide it by task) and is never auto-offered as a
              launch's weights.
  GATE        a run whose ledger row records a failing or incomparable gate
              is refused until the caller confirms, and the refusal states
              the verdict; a passing run promotes with no extra step.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_champion_scope.py

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level. Every fixture points app._inprocess_root at its own
temporary folder, because the promote route writes the module-wide record
under that root; the real module tree is fingerprinted before and after and
must not move. Fixtures are removed at the end and the removal is verified.
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


# Standing rule: no orchestrator test may reach a real subprocess.
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402


class _FakeVic:
    """The VICARIUS bridge with every call recorded and nothing written."""
    notes = []

    @staticmethod
    def process_start(*_a, **_k):
        return None

    @staticmethod
    def process_end(*_a, **_k):
        return None

    @classmethod
    def note(cls, message, *_a, **_k):
        cls.notes.append(message)


appmod.vic = _FakeVic()

_REAL_INPROCESS_ROOT = appmod._inprocess_root
_MODULE_INPROCESS = os.path.abspath(_REAL_INPROCESS_ROOT())
_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()

# The two scopes, spelled as the payloads spell them.
MODULE = "module"
PROJECT = "project"
# The fixture project's folder name. It starts with "run_" because
# labruns.list_all_runs only scans inprocess children with that prefix.
FIXTURE_PROJECT_NAME = "run_26-09-16_LO_champion_scope"
CHAMPION_FILE_NAME = "champion.json"
ROUNDS_CSV_HEADER = [
    "round", "at", "run_name", "run_dir", "base_model", "split_pinned",
    "n_train", "n_valid", "n_test",
    "map50_M", "map50_95_M", "recall_M", "precision_M",
    "map50_B", "map50_95_B", "recall_B", "precision_B",
    "gate_map", "gate_class", "gate_recall",
    "per_class_json", "metric_family", "metrics_schema",
    "conf", "iou", "imgsz",
]


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

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def _write_run(run_dir, task, with_best=True):
    """A run folder shaped the way ultralytics leaves one: args.yaml naming
    the task, and weights/best.pt when the run finished."""
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "args.yaml"),
           f"task: {task}\nimgsz: 512\nmodel: yolo11m{'-seg' if task == 'segment' else ''}.pt\n")
    if with_best:
        _write(os.path.join(run_dir, "weights", "best.pt"), "x")
    return run_dir


def _write_rounds(ledger_dir, rows):
    """rounds.csv in the ledger folder, in the writer's own column order."""
    import csv
    os.makedirs(ledger_dir, exist_ok=True)
    path = os.path.join(ledger_dir, "rounds.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROUNDS_CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ROUNDS_CSV_HEADER})
    return path


def _row(run_name, run_dir, family="box", score=0.21, gates=("pass", "pass", "pass")):
    """One ledger row for a run, scored in its own family."""
    key = "map50_95_B" if family == "box" else "map50_95_M"
    return {"round": 1, "at": "2026-09-16T09:00:00-04:00", "run_name": run_name,
            "run_dir": run_dir, "base_model": "yolo11m.pt", "split_pinned": "true",
            key: score, "metric_family": family, "metrics_schema": "2",
            "gate_map": gates[0], "gate_class": gates[1], "gate_recall": gates[2],
            "conf": 0.001, "iou": 0.6, "imgsz": 512}


def _make_project_state(project_dir):
    """Project state with every step folder created under project_dir."""
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    return {"id": os.path.basename(project_dir), "name": "scope fixture",
            "project_dir": project_dir, "created_at": "x", "updated_at": "x",
            "current_step": 6, "steps": steps}


def _fixture():
    """A module tree with one model lab and one project, each holding one
    evaluated run, and no champion promoted anywhere yet.

    Returns a dict of the paths every test needs. The caller closes with
    _close(fx).
    """
    tmp = tempfile.mkdtemp(prefix="champ_scope_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    lab_dir = os.path.join(root, "_models", "lab_det")
    lab_run = _write_run(os.path.join(lab_dir, "runs", "det_run"), "detect")
    _write(os.path.join(lab_dir, "lab.json"), json.dumps({"task": "detect", "source": "/x"}))
    _write(os.path.join(lab_dir, "dataset", "data.yaml"), "names:\n  0: Alpha\n")
    _write(os.path.join(lab_dir, "dataset", "split_manifest.json"),
           json.dumps({"split_fingerprint": "fp-box", "split_fingerprint_scheme": "v1"}))
    _write_rounds(lab_dir, [_row("det_run", lab_run, "box", 0.21)])

    project_dir = os.path.join(root, FIXTURE_PROJECT_NAME)
    state = _make_project_state(project_dir)
    step6 = state["steps"]["6"]["dir"]
    proj_run = _write_run(os.path.join(step6, "runs", "seg_run"), "segment")
    _write(os.path.join(step6, "dataset", "data.yaml"), "names:\n  0: PA\n")
    _write(os.path.join(step6, "dataset", "split_manifest.json"),
           json.dumps({"split_fingerprint": "fp-mask", "split_fingerprint_scheme": "v1"}))
    _write_rounds(step6, [_row("seg_run", proj_run, "mask", 0.42)])
    return {"tmp": tmp, "root": root, "lab_dir": lab_dir, "lab_run": lab_run,
            "project_dir": project_dir, "state": state, "step6": step6,
            "proj_run": proj_run}


def _client(fx, open_project=True):
    appmod._inprocess_root = lambda: fx["root"]
    appmod.current_project = fx["state"] if open_project else None
    return appmod.create_app().test_client()


def _close(fx):
    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    shutil.rmtree(fx["tmp"], ignore_errors=True)


def _module_champion_path(fx):
    return os.path.join(fx["root"], "_models", CHAMPION_FILE_NAME)


def _real_tree_champions():
    """Every champion record under the real module inprocess folder, as
    {path: contents}. Read only. The promote route writes champion.json and
    champion.json.tmp and nothing else, so watching those two names keeps the
    guard precise while a training job writes its own files into the tree."""
    watched = {}
    for dirpath, dirnames, filenames in os.walk(_MODULE_INPROCESS):
        for name in dirnames + filenames:
            if name not in (CHAMPION_FILE_NAME, CHAMPION_FILE_NAME + ".tmp"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "rb") as fh:
                    watched[path] = fh.read()
            except OSError:
                watched[path] = None
    return watched


# ---------------------------------------------------------------------------
# SCOPE
# ---------------------------------------------------------------------------

def test_every_reader_shows_the_module_champion_and_says_so():
    """The window's three champion readers agree on one run and each labels
    the scope it is showing."""
    fx = _fixture()
    try:
        client = _client(fx)
        r = client.post("/api/step/7/promote", json={"run_dir": fx["lab_run"]})
        check(r.status_code == 200, f"promote: {r.status_code} {r.get_json()}")

        card = client.get("/api/step/6/all_runs").get_json() or {}
        champ = card.get("champion") or {}
        check(champ.get("run_dir") == fx["lab_run"], f"the runs card shows the lab run: {champ}")
        check(champ.get("scope") == MODULE and card.get("champion_scope") == MODULE,
              f"and says it is the module champion: {champ} {card.get('champion_scope')}")

        rounds = client.get("/api/step/7/rounds").get_json() or {}
        check((rounds.get("champion") or {}).get("run_dir") == fx["lab_run"],
              f"the Rounds block shows the same run: {rounds.get('champion')}")
        check(rounds.get("champion_scope") == MODULE,
              f"and says which scope that is: {rounds.get('champion_scope')}")

        picker = client.get("/api/step/6/list_runs").get_json() or {}
        check(picker.get("champion_run_dir") == fx["lab_run"],
              f"the run picker follows the same champion: {picker}")
        check(picker.get("champion_scope") == MODULE, f"and says so: {picker}")
        check(picker.get("champion_task") == "detect",
              f"the champion's own task rides with it: {picker}")
    finally:
        _close(fx)


def test_the_rounds_block_never_says_no_champion_while_the_card_shows_one():
    """The live defect, in one assertion: a run promoted from a model lab
    while a project that has never promoted anything is open. The Evaluate
    panel used to read that project's own champion.json, find none and say
    "No champion promoted yet" in the same window where Train and Inference
    were both honouring the promoted detector."""
    fx = _fixture()
    try:
        client = _client(fx)
        check(client.post("/api/step/7/promote", json={"run_dir": fx["lab_run"]}).status_code == 200,
              "promote failed")
        check(not os.path.isfile(os.path.join(fx["step6"], CHAMPION_FILE_NAME)),
              "the open project still has no champion record of its own")

        rounds = client.get("/api/step/7/rounds").get_json() or {}
        check(rounds.get("champion") is not None,
              f"the Rounds block must not report 'no champion' here: {rounds}")
        check(rounds.get("gate_baseline") is None,
              f"the project's own gate baseline is separately null: {rounds}")
        check(rounds.get("gate_baseline_scope") == PROJECT,
              f"and is labelled with its own scope: {rounds}")
    finally:
        _close(fx)


def test_the_project_gate_baseline_rides_beside_the_module_champion():
    """Promoting one of the project's own runs fills both records, and each
    is reported under its own name and scope."""
    fx = _fixture()
    try:
        client = _client(fx)
        check(client.post("/api/step/7/promote", json={"run_dir": fx["proj_run"]}).status_code == 200,
              "promote failed")
        rounds = client.get("/api/step/7/rounds").get_json() or {}
        champion = rounds.get("champion") or {}
        baseline = rounds.get("gate_baseline") or {}
        check(champion.get("run_dir") == fx["proj_run"], f"module champion: {champion}")
        check(baseline.get("run_dir") == fx["proj_run"], f"project gate baseline: {baseline}")
        check(rounds.get("champion_scope") == MODULE and rounds.get("gate_baseline_scope") == PROJECT,
              f"each record says its own scope: {rounds.get('champion_scope')} "
              f"{rounds.get('gate_baseline_scope')}")
        check(baseline.get("metric_family") == "mask" and abs(baseline.get("score") - 0.42) < 1e-9,
              f"the baseline is the normalised record, not the raw file: {baseline}")
    finally:
        _close(fx)


def test_a_later_promotion_elsewhere_moves_every_reader_together():
    """Promote a project run, then a lab run. Both readers follow the second
    promotion; neither is left showing the first."""
    fx = _fixture()
    try:
        client = _client(fx)
        check(client.post("/api/step/7/promote", json={"run_dir": fx["proj_run"]}).status_code == 200,
              "first promote failed")
        check(client.post("/api/step/7/promote", json={"run_dir": fx["lab_run"]}).status_code == 200,
              "second promote failed")
        card = (client.get("/api/step/6/all_runs").get_json() or {}).get("champion") or {}
        rounds = (client.get("/api/step/7/rounds").get_json() or {}).get("champion") or {}
        check(card.get("run_dir") == fx["lab_run"] and rounds.get("run_dir") == fx["lab_run"],
              f"both readers moved to the newer promotion: {card} {rounds}")
        stale = (client.get("/api/step/7/rounds").get_json() or {}).get("gate_baseline") or {}
        check(stale.get("run_dir") == fx["proj_run"],
              f"the project's ledger baseline is its own history and does not move: {stale}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# PROMISE
# ---------------------------------------------------------------------------

def test_promoting_a_lab_run_makes_it_the_champion_list_runs_reports():
    """The Promote button promises the run becomes the default weights for
    Inference and for Continue from a VEGA model (design 2026-09-21-vega-
    starting-model-and-training-data-design.md section 3.2). The built-in
    Fine-tune preset that used to read this off /api/step/6/presets/finetune
    is gone; the panel now feeds its Continue-from select from
    GET /api/step/6/list_runs, and this is what promoting a run outside the
    open project changes."""
    fx = _fixture()
    try:
        client = _client(fx)
        # The project trains a detector, so a detector champion suits it.
        fx["state"]["training_source"] = {"task_default": "detect"}
        before = client.get("/api/step/6/list_runs").get_json() or {}
        check(before.get("champion_run_dir") is None,
              f"nothing is promoted yet, so list_runs names no champion: {before}")

        check(client.post("/api/step/7/promote", json={"run_dir": fx["lab_run"]}).status_code == 200,
              "promote failed")
        after = client.get("/api/step/6/list_runs").get_json() or {}
        check(after.get("champion_run_dir") == fx["lab_run"],
              f"list_runs now names the promoted lab run as the champion: {after}")
        check(after.get("champion_task") == "detect",
              f"and carries the run's own task, 'detect': {after}")
        check(after.get("champion_scope") == MODULE,
              f"promotion is always module-scoped: {after}")
    finally:
        _close(fx)


def test_a_champion_of_the_other_task_is_reported_by_list_runs_and_never_auto_offered():
    """Ultralytics takes the head from the checkpoint and ignores the task it
    is handed, so seeding an outline project from a detector's weights would
    quietly train a detector into this project's run folder. list_runs still
    reports the mismatched champion (champion_task carries its own task, not
    the panel's), which is how the client's Continue from a VEGA model select
    hides a run of the wrong task (design section 3.2) rather than the server
    silently withholding it. No route substitutes the champion for a launch's
    weights on its own: Start Training with no model_path chosen trains from
    the catalog default, and the old auto-offer route is gone entirely."""
    fx = _fixture()
    try:
        client = _client(fx)
        check(client.post("/api/step/7/promote", json={"run_dir": fx["lab_run"]}).status_code == 200,
              "promote failed")

        picker = client.get("/api/step/6/list_runs").get_json() or {}
        check(picker.get("champion_run_dir") == fx["lab_run"],
              f"list_runs still names the module champion, whatever task it trained: {picker}")
        check(picker.get("champion_task") == "detect",
              f"and carries its own task, so the client can filter it out: {picker}")
        check(picker.get("champion_scope") == MODULE, f"{picker}")

        r = client.get("/api/step/6/presets/finetune")
        check(r.status_code == 404,
              f"the route that used to auto-offer a champion's weights is gone: "
              f"{r.status_code} {r.get_json()}")

        # A launch with no model_path chosen must not pick up the mismatched
        # champion by itself: the project's own step4test export is enough to
        # launch, and the captured cmd's model positional is the catalog
        # default, never the detector champion's weights.
        export_dir = os.path.join(fx["project_dir"], pm.STEP_DIRS["step4test"])
        os.makedirs(os.path.join(export_dir, "all_images"), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "all_labels"), exist_ok=True)
        _write(os.path.join(export_dir, "data.yaml"),
              "path: x\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: PA\n")
        captured = {}
        orig = appmod.runner.run_cli_stage
        appmod.runner.run_cli_stage = (
            lambda step, cmd, cwd=None, env_extra=None: captured.update(cmd=cmd) or {"pid": 1})
        try:
            r = client.post("/api/step/6/run")
            check(r.status_code == 200, f"launch: {r.status_code} {r.get_json()}")
        finally:
            appmod.runner.run_cli_stage = orig
        cmd = captured.get("cmd") or []
        champion_weights = os.path.join(fx["lab_run"], "weights", "best.pt")
        check(len(cmd) >= 8 and cmd[7] == "yolo11m-seg.pt",
              f"the launch uses the catalog default, not the champion: {cmd}")
        check(champion_weights not in cmd,
              f"the mismatched champion's weights must never be offered by default: {cmd}")
    finally:
        _close(fx)


def test_promote_says_which_records_it_wrote():
    """Both records are written and the answer names both scopes, so nothing
    has to be inferred from the file system."""
    fx = _fixture()
    real_before = _real_tree_champions()
    try:
        client = _client(fx)
        data = client.post("/api/step/7/promote", json={"run_dir": fx["lab_run"]}).get_json() or {}
        check(data.get("scopes_written") == [PROJECT, MODULE],
              f"the answer names both records: {data.get('scopes_written')}")
        check(os.path.isfile(os.path.join(fx["lab_dir"], CHAMPION_FILE_NAME)),
              "the run's own ledger record is written")
        check(os.path.isfile(_module_champion_path(fx)), "the module-wide record is written")
    finally:
        _close(fx)
    real_after = _real_tree_champions()
    check(real_before == real_after,
          "the promote route moved a champion record in the real module tree: "
          f"{sorted(set(real_before) ^ set(real_after))}")


# ---------------------------------------------------------------------------
# GATE
# ---------------------------------------------------------------------------

def _promote(client, run_dir, **body):
    return client.post("/api/step/7/promote", json={"run_dir": run_dir, **body})


def test_a_failing_gate_is_refused_until_it_is_confirmed():
    fx = _fixture()
    try:
        _write_rounds(fx["lab_dir"], [_row("det_run", fx["lab_run"], "box", 0.21,
                                           gates=("fail", "pass", "pass"))])
        client = _client(fx)
        r = _promote(client, fx["lab_run"])
        body = r.get_json() or {}
        check(r.status_code == 409, f"a failing gate is refused: {r.status_code} {body}")
        check(body.get("needs_confirmation") is True, f"and says a confirmation is what is missing: {body}")
        err = str(body.get("error") or "")
        check("says fail" in err, f"the refusal states the verdict: {err!r}")
        check("confirm_gate" in err, f"and names what to send: {err!r}")
        check(body.get("gate", {}).get("gate_map") == "fail", f"the verdicts ride with it: {body}")
        check(not os.path.isfile(_module_champion_path(fx)),
              "nothing is promoted while the confirmation is missing")
        check(not os.path.isfile(os.path.join(fx["lab_dir"], CHAMPION_FILE_NAME)),
              "and no ledger record is written either")
    finally:
        _close(fx)


def test_a_confirmed_failing_gate_promotes_and_the_record_keeps_the_verdict():
    fx = _fixture()
    try:
        _write_rounds(fx["lab_dir"], [_row("det_run", fx["lab_run"], "box", 0.21,
                                           gates=("fail", "pass", "pass"))])
        client = _client(fx)
        r = _promote(client, fx["lab_run"], confirm_gate=True)
        body = r.get_json() or {}
        check(r.status_code == 200, f"a confirmed promotion goes through: {r.status_code} {body}")
        check(body.get("gate", {}).get("gate_map") == "fail", f"the answer repeats the verdict: {body}")
        with open(_module_champion_path(fx)) as fh:
            record = json.load(fh)
        check(record.get("gate", {}).get("gate_map") == "fail",
              f"the record keeps what the gates said at the moment of promotion: {record}")
    finally:
        _close(fx)


def test_an_incomparable_gate_is_refused_the_same_way():
    fx = _fixture()
    try:
        _write_rounds(fx["lab_dir"], [_row("det_run", fx["lab_run"], "box", 0.21,
                                           gates=("incomparable",) * 3)])
        client = _client(fx)
        r = _promote(client, fx["lab_run"])
        body = r.get_json() or {}
        check(r.status_code == 409, f"an incomparable gate is refused: {r.status_code} {body}")
        check("incomparable" in str(body.get("error") or ""),
              f"the refusal states the verdict: {body}")
    finally:
        _close(fx)


def test_a_flagged_class_is_refused_with_the_codes_it_named():
    fx = _fixture()
    try:
        _write_rounds(fx["lab_dir"], [_row("det_run", fx["lab_run"], "box", 0.21,
                                           gates=("pass", "flag:MC,PA", "pass"))])
        client = _client(fx)
        r = _promote(client, fx["lab_run"])
        body = r.get_json() or {}
        check(r.status_code == 409, f"a flagged class gate is refused: {r.status_code} {body}")
        check("flag:mc,pa" in str(body.get("error") or "").lower(),
              f"the refusal carries the codes the gate named: {body}")
    finally:
        _close(fx)


def test_a_passing_gate_needs_no_confirmation():
    fx = _fixture()
    try:
        client = _client(fx)
        r = _promote(client, fx["lab_run"])
        check(r.status_code == 200, f"a passing run promotes in one press: {r.status_code} {r.get_json()}")
    finally:
        _close(fx)


def test_an_unpinned_split_needs_no_confirmation():
    """"unpinned" says the split was loose so the gates never ran. That is
    the state of almost every ledger written before splits were pinned, and
    it says nothing about the run itself, so it must not block promotion."""
    fx = _fixture()
    try:
        _write_rounds(fx["lab_dir"], [_row("det_run", fx["lab_run"], "box", 0.21,
                                           gates=("unpinned",) * 3)])
        client = _client(fx)
        r = _promote(client, fx["lab_run"])
        check(r.status_code == 200, f"an unpinned ledger still promotes: {r.status_code} {r.get_json()}")
    finally:
        _close(fx)


def test_a_ledger_with_no_gate_columns_needs_no_confirmation():
    """A run evaluated before the gates existed records no verdict at all.
    Silence is not a stated failure."""
    fx = _fixture()
    try:
        row = _row("det_run", fx["lab_run"], "box", 0.21, gates=("", "", ""))
        _write_rounds(fx["lab_dir"], [row])
        client = _client(fx)
        r = _promote(client, fx["lab_run"])
        check(r.status_code == 200, f"an ungated ledger still promotes: {r.status_code} {r.get_json()}")
    finally:
        _close(fx)


if __name__ == "__main__":
    print("CHAMPION SCOPE -- one champion per scope, every reader says which\n")
    print("SCOPE:")
    run(test_every_reader_shows_the_module_champion_and_says_so)
    run(test_the_rounds_block_never_says_no_champion_while_the_card_shows_one)
    run(test_the_project_gate_baseline_rides_beside_the_module_champion)
    run(test_a_later_promotion_elsewhere_moves_every_reader_together)
    print("PROMISE:")
    run(test_promoting_a_lab_run_makes_it_the_champion_list_runs_reports)
    run(test_a_champion_of_the_other_task_is_reported_by_list_runs_and_never_auto_offered)
    run(test_promote_says_which_records_it_wrote)
    print("GATE:")
    run(test_a_failing_gate_is_refused_until_it_is_confirmed)
    run(test_a_confirmed_failing_gate_promotes_and_the_record_keeps_the_verdict)
    run(test_an_incomparable_gate_is_refused_the_same_way)
    run(test_a_flagged_class_is_refused_with_the_codes_it_named)
    run(test_a_passing_gate_needs_no_confirmation)
    run(test_an_unpinned_split_needs_no_confirmation)
    run(test_a_ledger_with_no_gate_columns_needs_no_confirmation)

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
