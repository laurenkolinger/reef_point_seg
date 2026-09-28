"""Task T9, Step 5 (internal step 6): the cross-project runs table, the
widened champion promotion, and the Step 6 launch hand-offs.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_all_runs.py

Routes under test (plan docs/superpowers/plans/2026-09-04-vega-training-foldin.md,
"API contract for wave 2"):
  GET  /api/step/6/all_runs   every run from the lab root (_models/<lab>/runs)
                              and every project (run_*/step6_trainModel/runs),
                              newest first, with the module champion and a
                              comparability verdict per row
  GET  /api/step/6/list_runs  the open project's runs, now with task and family
  POST /api/step/7/promote    accepts a run outside the open project, writes a
                              family-aware champion.json beside that run's
                              ledger and the module-wide copy the runs table
                              reads
  GET  /api/step/7/rounds     recall_B and precision_B cast to float
  POST /api/step/6/run        the split seed is the normalised dataset name,
                              the panel's fractions reach the builder, a
                              project retrained from a different source is
                              refused with 409, and a staged box dataset can
                              be chosen as the training source
  GET  /api/composites        staged box datasets merged into the source list,
                              each row saying its kind

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, the VICARIUS bridge is replaced by a recorder, and
the stage runner's launch is replaced by a capture. Every fixture lives under
one temp folder that is removed at the end, and the removal is verified.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

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
from _modellab import labruns  # noqa: E402
from _reefreview.training_projects import mint_training_project  # noqa: E402


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

_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()
ISO_OFFSET = "-04:00"
# score_source joined the row on 2026-09-05: 'ledger' when the numbers come
# from the run's rounds.csv row, 'evaluation' when no ledger row names the run
# and they were read from its newest evaluation folder, '' when it has neither.
# Only a run with a ledger row may be promoted, so the card needs the two apart.
# measured_at joined the row on 2026-09-16: the confidence, overlap and image
# size the ledger recorded for this run's score. Two runs measured at
# different settings are not a comparison, and the card needs the numbers to
# say so.
ALL_RUNS_KEYS = {"run_dir", "run_name", "origin", "project_name", "task", "family",
                 "map50", "map50_95", "evaluated_at", "score_source", "has_weights", "champion",
                 "imgsz", "split_fingerprint", "comparable_to_champion", "comparability_reason",
    # 2026-09-17: what the run was trained on, and its import provenance (Lauren: filter by the training data subset)
    "trained_on", "trained_on_kind", "origin_kind", "imported_from", "imported_at",
                 "measured_at"}
# The champion view: the five contract keys, "missing" (added 2026-09-05 so
# the card can say the promoted run's folder is no longer on disk instead of
# measuring every other run against a guess) and "scope" (added 2026-09-16 so
# a reader is never handed a champion without being told which of the two it
# is: the module-wide promoted model, or one project's ledger baseline).
CHAMPION_KEYS = {"run_dir", "run_name", "promoted_at", "score", "family", "missing", "scope"}
MEASUREMENT_KEYS = {"conf", "iou", "imgsz"}
ROUNDS_HEADER = [
    "round", "at", "run_name", "run_dir", "base_model", "split_pinned",
    "n_train", "n_valid", "n_test",
    "map50_M", "map50_95_M", "recall_M", "precision_M",
    "map50_B", "map50_95_B", "recall_B", "precision_B",
    "gate_map", "gate_class", "gate_recall",
    "per_class_json", "metric_family", "metrics_schema",
    # What a score was measured with, added to the writer's header on
    # 2026-09-15. A row that leaves these blank is a ledger written before
    # that date, and blank means "not recorded", never a default.
    "conf", "iou", "imgsz",
]
FINGERPRINT_SCHEME = "sha256-of-sorted-basename-split-pairs-v1"
HOSTILE_RUN_NAME = "<img src=x onerror=alert(1)>"


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

def _write(path, text):
    """Write text to path, creating the folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _write_run(run_dir, task, imgsz=512, with_weights=True):
    """A training run folder: args.yaml with task and imgsz lines, best.pt."""
    _write(os.path.join(run_dir, "args.yaml"),
           f"task: {task}\nmode: train\nmodel: yolo11m{'-seg' if task == 'segment' else ''}.pt\n"
           f"imgsz: {imgsz}\nepochs: 5\n")
    if with_weights:
        _write(os.path.join(run_dir, "weights", "best.pt"), "weights")
    return run_dir


def _write_split_manifest(dataset_dir, fingerprint):
    """dataset/ beside runs/: data.yaml plus a split manifest with a fingerprint."""
    _write(os.path.join(dataset_dir, "data.yaml"), "names:\n  0: PA\n")
    _write(os.path.join(dataset_dir, "split_manifest.json"), json.dumps({
        "schema_version": 2, "holdout_mode": "lab-random", "seed": "fx",
        "valid_frac": 0.1, "test_frac": 0.1,
        "split_fingerprint": fingerprint,
        "split_fingerprint_scheme": FINGERPRINT_SCHEME,
        "pins": {"a.jpg": "train"},
    }))


def _write_rounds(ledger_dir, rows):
    """rounds.csv in the current header order."""
    os.makedirs(ledger_dir, exist_ok=True)
    with open(os.path.join(ledger_dir, "rounds.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROUNDS_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ROUNDS_HEADER})


def _make_project_state(project_dir, name="fx"):
    """A loaded-project dict of the shape app.py keeps, with every step dir made."""
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    return {"id": os.path.basename(project_dir), "name": name, "project_dir": project_dir,
            "created_at": "x", "updated_at": "x", "current_step": 6, "steps": steps}


def _make_root():
    """A fake inprocess root with one lab detector run and one project segmenter run.

    Returns a dict with root, the two run dirs, their ledger dirs and the
    project dir. Folder times: the lab run is older than the project run.
    """
    tmp = tempfile.mkdtemp(prefix="t9_all_runs_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    lab_dir = os.path.join(root, "_models", "lab_det")
    lab_run = _write_run(os.path.join(lab_dir, "runs", "det_run"), "detect", imgsz=256)
    _write(os.path.join(lab_dir, "lab.json"), json.dumps({"task": "detect", "source": "/x"}))
    _write_split_manifest(os.path.join(lab_dir, "dataset"), "fp-box-1")
    _write_rounds(lab_dir, [{
        "round": 1, "at": "2026-09-05T09:00:00-04:00", "run_name": "det_run",
        "run_dir": lab_run, "base_model": "yolo11m.pt", "split_pinned": "false",
        "map50_B": 0.31, "map50_95_B": 0.21, "recall_B": 0.4, "precision_B": 0.5,
        "gate_map": "baseline", "metric_family": "box", "metrics_schema": "2",
    }])
    project_dir = os.path.join(root, "run_26-09-05_LO_train_fx")
    state = _make_project_state(project_dir)
    step6 = state["steps"]["6"]["dir"]
    proj_run = _write_run(os.path.join(step6, "runs", "seg_run"), "segment", imgsz=640)
    _write_split_manifest(os.path.join(step6, "dataset"), "fp-mask-1")
    _write_rounds(step6, [{
        "round": 1, "at": "2026-09-05 10:00:00", "run_name": "seg_run",
        "run_dir": proj_run, "base_model": "yolo11m-seg.pt", "split_pinned": "true",
        "map50_M": 0.55, "map50_95_M": 0.42, "recall_M": 0.6, "precision_M": 0.7,
        "map50_B": 0.6, "map50_95_B": 0.45,
        "gate_map": "baseline", "metric_family": "mask", "metrics_schema": "2",
    }])
    now = time.time()
    os.utime(lab_run, (now - 3600, now - 3600))
    os.utime(proj_run, (now, now))
    return {"tmp": tmp, "root": root, "lab_dir": lab_dir, "lab_run": lab_run,
            "project_dir": project_dir, "state": state, "step6": step6, "proj_run": proj_run}


def _client(fx, open_project=True):
    """A test client with the inprocess root pointed at the fixture."""
    appmod._inprocess_root = lambda: fx["root"]
    appmod.current_project = fx["state"] if open_project else None
    return appmod.create_app().test_client()


def _close(fx):
    """Restore the module state and remove the fixture."""
    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    appmod._bbox_root = _REAL_BBOX_ROOT
    appmod._composite_root = _REAL_COMPOSITE_ROOT
    shutil.rmtree(fx["tmp"], ignore_errors=True)


_REAL_INPROCESS_ROOT = appmod._inprocess_root
_REAL_BBOX_ROOT = appmod._bbox_root
_REAL_COMPOSITE_ROOT = appmod._composite_root


# ---------------------------------------------------------------------------
# GET /api/step/6/all_runs
# ---------------------------------------------------------------------------

def test_all_runs_rows_carry_every_contract_key():
    fx = _make_root()
    try:
        r = _client(fx).get("/api/step/6/all_runs")
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"status {r.status_code}: {data}")
        runs = data.get("runs") or []
        check(len(runs) == 2, f"one lab run and one project run expected: {runs}")
        for row in runs:
            check(set(row) == ALL_RUNS_KEYS,
                  f"row keys must be exactly the contract keys: {sorted(set(row) ^ ALL_RUNS_KEYS)}")
        by_name = {row["run_name"]: row for row in runs}
        det = by_name["det_run"]
        seg = by_name["seg_run"]
        check(det["origin"] == "lab" and det["project_name"] == "lab_det",
              f"a lab run says origin lab and names its lab folder as its home: {det}")
        check(seg["origin"] == "project" and seg["project_name"] == "run_26-09-05_LO_train_fx",
              f"a project run names its project folder: {seg}")
        check(det["task"] == "detect" and det["family"] == "box", f"detector family: {det}")
        check(seg["task"] == "seg" and seg["family"] == "mask", f"segmenter family: {seg}")
        check(abs(det["map50"] - 0.31) < 1e-9 and abs(det["map50_95"] - 0.21) < 1e-9,
              f"a detector's scores come from the box columns: {det}")
        check(abs(seg["map50"] - 0.55) < 1e-9 and abs(seg["map50_95"] - 0.42) < 1e-9,
              f"a segmenter's scores come from the mask columns: {seg}")
        check(det["evaluated_at"] == "2026-09-05T09:00:00-04:00",
              f"evaluated_at is the ledger time as ISO AST: {det['evaluated_at']!r}")
        check(seg["evaluated_at"] == "2026-09-05T10:00:00-04:00",
              f"a space-separated ledger time is normalised to ISO AST: {seg['evaluated_at']!r}")
        check(det["imgsz"] == 256 and seg["imgsz"] == 640, f"imgsz from args.yaml: {det} {seg}")
        check(det["split_fingerprint"] == "fp-box-1" and seg["split_fingerprint"] == "fp-mask-1",
              f"split fingerprints from the split manifests: {det} {seg}")
        check(det["has_weights"] is True and seg["has_weights"] is True, "both runs have best.pt")
        check(data.get("champion") is None, f"no champion promoted yet: {data.get('champion')}")
    finally:
        _close(fx)


def test_all_runs_newest_first():
    fx = _make_root()
    try:
        runs = (_client(fx).get("/api/step/6/all_runs").get_json() or {}).get("runs") or []
        names = [row["run_name"] for row in runs]
        check(names == ["seg_run", "det_run"], f"newest run folder first: {names}")
    finally:
        _close(fx)


def test_all_runs_comparable_is_null_without_a_champion():
    fx = _make_root()
    try:
        runs = (_client(fx).get("/api/step/6/all_runs").get_json() or {}).get("runs") or []
        check(all(row["comparable_to_champion"] is None and row["comparability_reason"] == "" for row in runs),
              f"null comparability and no reason without a champion: {runs}")
        check(not any(row["champion"] for row in runs), f"no row is the champion: {runs}")
    finally:
        _close(fx)


def test_all_runs_marks_the_champion_once_and_compares_family_and_split():
    fx = _make_root()
    try:
        # A second detector lab whose split has a different fingerprint, and a
        # third whose dataset recorded no fingerprint at all.
        other_dir = os.path.join(fx["root"], "_models", "lab_det_other")
        other_run = _write_run(os.path.join(other_dir, "runs", "det_other"), "detect")
        _write_split_manifest(os.path.join(other_dir, "dataset"), "fp-box-2")
        old_dir = os.path.join(fx["root"], "_models", "lab_det_old")
        _write_run(os.path.join(old_dir, "runs", "det_old"), "detect")
        _write(os.path.join(old_dir, "dataset", "data.yaml"), "names:\n  0: PA\n")
        _write(os.path.join(fx["root"], "_models", "champion.json"), json.dumps({
            "run_dir": fx["lab_run"], "promoted_at": "2026-09-05T11:00:00-04:00",
            "metric_family": "box", "map50_95_B": 0.21, "note": "",
        }))
        data = _client(fx).get("/api/step/6/all_runs").get_json() or {}
        champ = data.get("champion") or {}
        check(set(champ) == CHAMPION_KEYS, f"champion keys: {sorted(set(champ) ^ CHAMPION_KEYS)}")
        check(champ["run_dir"] == fx["lab_run"] and champ["run_name"] == "det_run", f"champion: {champ}")
        check(champ["family"] == "box" and abs(champ["score"] - 0.21) < 1e-9, f"champion score: {champ}")
        check(champ["promoted_at"] == "2026-09-05T11:00:00-04:00", f"promoted_at ISO AST: {champ}")
        check(champ["scope"] == "module" and data.get("champion_scope") == "module",
              f"the card is told which champion it is showing: {champ} {data.get('champion_scope')}")
        check(set(data.get("champion_measured_at") or {}) == MEASUREMENT_KEYS,
              f"and what that champion's score was measured with: {data.get('champion_measured_at')}")
        by_name = {row["run_name"]: row for row in data.get("runs") or []}
        check([n for n, row in by_name.items() if row["champion"]] == ["det_run"],
              f"exactly one row is the champion: {by_name}")
        check(by_name["seg_run"]["comparable_to_champion"] is False
              and by_name["seg_run"]["comparability_reason"] == "different metric family",
              f"a mask run is not comparable to a box champion, and says why: {by_name['seg_run']}")
        check(by_name["det_other"]["comparable_to_champion"] is False
              and by_name["det_other"]["comparability_reason"] == "different held-out images",
              f"a box run on a different split is not comparable, and says why: {by_name['det_other']}")
        check(by_name["det_old"]["comparable_to_champion"] is False
              and by_name["det_old"]["comparability_reason"] == "split not recorded",
              f"a run whose split was never fingerprinted is not comparable, and says why: {by_name['det_old']}")
        check(by_name["det_run"]["comparable_to_champion"] is True
              and by_name["det_run"]["comparability_reason"] == "",
              f"the champion row is comparable to itself with no reason: {by_name['det_run']}")
        check(other_run and os.path.isdir(other_run), "fixture run exists")
    finally:
        _close(fx)


def _champion_and_challenger(fx, champion_measured, challenger_measured):
    """Two detector runs on the SAME split, one promoted, measured as asked.

    Everything except the measurement settings is held equal: same metric
    family, same split fingerprint, same weights on disk. Whatever the runs
    card then says about comparability is about the measurement and nothing
    else.

    Parameters:
        fx: the fixture from _make_root().
        champion_measured, challenger_measured: dicts of conf, iou and imgsz
            to write into each ledger row; a key left out is written blank,
            which is how a pre-2026-09-15 ledger reads.
    Returns:
        the challenger's row from GET /api/step/6/all_runs.
    """
    rival_dir = os.path.join(fx["root"], "_models", "lab_det_rival")
    rival_run = _write_run(os.path.join(rival_dir, "runs", "det_rival"), "detect")
    _write(os.path.join(rival_dir, "lab.json"), json.dumps({"task": "detect", "source": "/x"}))
    _write_split_manifest(os.path.join(rival_dir, "dataset"), "fp-box-1")
    _write_rounds(rival_dir, [dict({
        "round": 1, "at": "2026-09-16T09:00:00-04:00", "run_name": "det_rival",
        "run_dir": rival_run, "map50_B": 0.33, "map50_95_B": 0.22,
        "gate_map": "pass", "metric_family": "box", "metrics_schema": "2",
    }, **challenger_measured)])
    _write_rounds(fx["lab_dir"], [dict({
        "round": 1, "at": "2026-09-16T09:00:00-04:00", "run_name": "det_run",
        "run_dir": fx["lab_run"], "map50_B": 0.31, "map50_95_B": 0.21,
        "gate_map": "pass", "metric_family": "box", "metrics_schema": "2",
    }, **champion_measured)])
    _write(os.path.join(fx["root"], "_models", "champion.json"), json.dumps({
        "run_dir": fx["lab_run"], "promoted_at": "2026-09-16T09:30:00-04:00",
        "metric_family": "box", "map50_95_B": 0.21, "note": "",
    }))
    rows = {row["run_name"]: row for row in
            (_client(fx).get("/api/step/6/all_runs").get_json() or {}).get("runs") or []}
    return rows["det_rival"]


def test_the_same_measurement_is_comparable():
    """Same family, same split, same confidence, overlap and image size: the
    two scores are a comparison."""
    fx = _make_root()
    try:
        measured = {"conf": 0.001, "iou": 0.6, "imgsz": 512}
        row = _champion_and_challenger(fx, measured, dict(measured))
        check(row["comparable_to_champion"] is True and row["comparability_reason"] == "",
              f"identical measurement settings rank against each other: {row}")
        check(row["measured_at"] == {"conf": 0.001, "iou": 0.6, "imgsz": 512},
              f"and the row carries what it was measured with: {row['measured_at']}")
    finally:
        _close(fx)


def test_a_different_overlap_is_not_comparable():
    """The ledger records the non-maximum-suppression overlap with every
    score. Two runs measured at different overlaps were never a comparison,
    and gating on the confidence alone ranked them head to head."""
    fx = _make_root()
    try:
        row = _champion_and_challenger(fx,
                                       {"conf": 0.001, "iou": 0.6, "imgsz": 512},
                                       {"conf": 0.001, "iou": 0.7, "imgsz": 512})
        check(row["comparable_to_champion"] is False,
              f"a different overlap is not comparable: {row}")
        check(row["comparability_reason"] == "different measurement settings",
              f"and the card is told why: {row}")
    finally:
        _close(fx)


def test_a_different_image_size_is_not_comparable():
    fx = _make_root()
    try:
        row = _champion_and_challenger(fx,
                                       {"conf": 0.001, "iou": 0.6, "imgsz": 512},
                                       {"conf": 0.001, "iou": 0.6, "imgsz": 1024})
        check(row["comparable_to_champion"] is False
              and row["comparability_reason"] == "different measurement settings",
              f"a different image size is not comparable: {row}")
    finally:
        _close(fx)


def test_a_different_confidence_is_not_comparable():
    fx = _make_root()
    try:
        row = _champion_and_challenger(fx,
                                       {"conf": 0.001, "iou": 0.6, "imgsz": 512},
                                       {"conf": 0.25, "iou": 0.6, "imgsz": 512})
        check(row["comparable_to_champion"] is False
              and row["comparability_reason"] == "different measurement settings",
              f"a different confidence is not comparable: {row}")
    finally:
        _close(fx)


def test_one_side_recording_nothing_is_not_comparable():
    """A run measured before the ledger recorded settings, against one
    measured after: nobody can tell whether the two numbers came from the
    same measurement, and the card says exactly that."""
    fx = _make_root()
    try:
        row = _champion_and_challenger(fx,
                                       {"conf": 0.001, "iou": 0.6, "imgsz": 512},
                                       {})
        check(row["comparable_to_champion"] is False
              and row["comparability_reason"] == "measurement settings not recorded",
              f"an unrecorded measurement is not a comparison: {row}")
        check(row["measured_at"] == {"conf": None, "iou": None, "imgsz": None},
              f"and reads as not recorded rather than as a default: {row['measured_at']}")
    finally:
        _close(fx)


def test_two_ledgers_that_record_nothing_stay_comparable():
    """Every ledger written before 2026-09-15 carries three blank cells.
    Those rows were all measured the same way, and treating them as
    incomparable would silence the card on almost every run on disk."""
    fx = _make_root()
    try:
        row = _champion_and_challenger(fx, {}, {})
        check(row["comparable_to_champion"] is True and row["comparability_reason"] == "",
              f"two unrecorded ledgers still rank against each other: {row}")
    finally:
        _close(fx)


def test_all_runs_hostile_run_name_is_a_plain_string():
    fx = _make_root()
    try:
        _write_run(os.path.join(fx["lab_dir"], "runs", HOSTILE_RUN_NAME), "detect")
        runs = (_client(fx).get("/api/step/6/all_runs").get_json() or {}).get("runs") or []
        names = [row["run_name"] for row in runs]
        check(HOSTILE_RUN_NAME in names, f"the hostile folder is listed unchanged: {names}")
        row = next(r for r in runs if r["run_name"] == HOSTILE_RUN_NAME)
        check(isinstance(row["run_name"], str) and isinstance(row["run_dir"], str),
              "names are plain strings; escaping is the client's job")
    finally:
        _close(fx)


def test_champion_row_is_never_marked_not_comparable():
    """A champion promoted from a dataset with no split fingerprint would fail
    the library's own comparison with itself; the runs card must still show
    it as the baseline."""
    fx = _make_root()
    try:
        os.remove(os.path.join(fx["lab_dir"], "dataset", "split_manifest.json"))
        _write(os.path.join(fx["root"], "_models", "champion.json"), json.dumps({
            "run_dir": fx["lab_run"], "promoted_at": "2026-09-05T11:00:00-04:00",
            "metric_family": "box", "map50_95_B": 0.21, "note": "",
        }))
        by_name = {row["run_name"]: row for row in
                   (_client(fx).get("/api/step/6/all_runs").get_json() or {}).get("runs") or []}
        check(by_name["det_run"]["champion"] is True and by_name["det_run"]["comparable_to_champion"] is True
              and by_name["det_run"]["comparability_reason"] == "" and by_name["det_run"]["split_fingerprint"] == "",
              f"the champion row is comparable to itself even with no fingerprint: {by_name['det_run']}")
        check(by_name["seg_run"]["comparability_reason"] == "different metric family",
              f"the family check comes before the split check: {by_name['seg_run']}")
    finally:
        _close(fx)


def test_train_and_infer_page_is_gone():
    """The /models mount is retired: its runs table, champion pick and
    inference live in the main window now."""
    fx = _make_root()
    try:
        client = _client(fx)
        check(client.get("/models/").status_code == 404, "the Train + Infer page no longer answers")
        check(client.get("/models/api/runs").status_code == 404, "its API is unmounted too")
        check(client.get("/api/step/6/all_runs").status_code == 200, "the runs table answers here instead")
    finally:
        _close(fx)


def test_all_runs_works_without_an_open_project():
    fx = _make_root()
    try:
        r = _client(fx, open_project=False).get("/api/step/6/all_runs")
        data = r.get_json() or {}
        check(r.status_code == 200 and len(data.get("runs") or []) == 2,
              f"the runs table is project-free: {r.status_code} {data}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# GET /api/step/6/list_runs
# ---------------------------------------------------------------------------

def test_list_runs_rows_carry_task_and_family():
    fx = _make_root()
    try:
        r = _client(fx).get("/api/step/6/list_runs")
        data = r.get_json() or {}
        check(r.status_code == 200, f"list_runs: {r.status_code} {data}")
        rows = {row["name"]: row for row in data.get("runs") or []}
        check(rows["seg_run"]["task"] == "seg" and rows["seg_run"]["family"] == "mask",
              f"task and family on the project run list: {rows}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# POST /api/step/7/promote
# ---------------------------------------------------------------------------

def test_promote_accepts_a_foreign_run_and_writes_its_own_ledger():
    fx = _make_root()
    try:
        client = _client(fx)
        r = client.post("/api/step/7/promote", json={"run_dir": fx["lab_run"]})
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"promote: {r.status_code} {data}")
        champ = data.get("champion") or {}
        check(set(champ) == CHAMPION_KEYS, f"champion keys: {sorted(set(champ) ^ CHAMPION_KEYS)}")
        check(champ["run_dir"] == fx["lab_run"] and champ["family"] == "box"
              and abs(champ["score"] - 0.21) < 1e-9, f"champion view: {champ}")
        check(champ["promoted_at"].endswith(ISO_OFFSET), f"promoted_at carries the AST offset: {champ}")

        ledger_file = os.path.join(fx["lab_dir"], "champion.json")
        check(os.path.isfile(ledger_file), "champion.json is written beside the run's own ledger")
        with open(ledger_file) as fh:
            on_disk = json.load(fh)
        check(set(on_disk) == {"run_dir", "promoted_at", "metric_family", "map50_95_B", "note", "gate"},
              f"a box champion carries the box score key, no mask key, and the gate it was "
              f"promoted over: {on_disk}")
        check(set(on_disk["gate"]) == {"gate_map", "gate_class", "gate_recall"},
              f"the record keeps what the gates said at the moment of promotion: {on_disk['gate']}")
        check(data.get("scopes_written") == ["project", "module"],
              f"the answer says which champion records were written: {data.get('scopes_written')}")
        check(on_disk["metric_family"] == "box" and abs(on_disk["map50_95_B"] - 0.21) < 1e-9,
              f"family-aware record: {on_disk}")
        check(not os.path.isfile(os.path.join(fx["step6"], "champion.json")),
              "the open project's own champion file is untouched by a foreign promotion")

        module_file = os.path.join(fx["root"], "_models", "champion.json")
        check(os.path.isfile(module_file), "the module-wide champion.json the runs table reads is written")
        record = labruns.read_champion_file(module_file)
        check(record["run_dir"] == fx["lab_run"] and record["score"] == on_disk["map50_95_B"],
              f"labruns reads the record back: {record}")

        rows = (client.get("/api/step/6/all_runs").get_json() or {}).get("runs") or []
        check([row["run_name"] for row in rows if row["champion"]] == ["det_run"],
              f"the runs table now marks the promoted run: {rows}")
    finally:
        _close(fx)


def test_promote_writes_the_mask_score_for_a_segmenter():
    fx = _make_root()
    try:
        r = _client(fx).post("/api/step/7/promote", json={"run_dir": fx["proj_run"], "note": "round 1"})
        data = r.get_json() or {}
        check(r.status_code == 200, f"promote: {r.status_code} {data}")
        with open(os.path.join(fx["step6"], "champion.json")) as fh:
            on_disk = json.load(fh)
        check(on_disk["metric_family"] == "mask" and abs(on_disk["map50_95_M"] - 0.42) < 1e-9
              and "map50_95_B" not in on_disk and on_disk["note"] == "round 1",
              f"a mask champion carries the mask score key: {on_disk}")
        check(data["champion"]["family"] == "mask" and abs(data["champion"]["score"] - 0.42) < 1e-9,
              f"champion view: {data['champion']}")
    finally:
        _close(fx)


def test_promote_refuses_a_run_outside_both_roots():
    fx = _make_root()
    outside = tempfile.mkdtemp(prefix="t9_outside_", dir=_TMP_ROOT)
    _FIXTURES.append(outside)
    try:
        run_dir = _write_run(os.path.join(outside, "runs", "stray"), "detect")
        r = _client(fx).post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json() or {}
        check(r.status_code == 403, f"outside both roots answers 403: {r.status_code} {data}")
        check("outside" in str(data.get("error", "")).lower(), f"the error says why: {data}")
        check(not os.path.isfile(os.path.join(fx["root"], "_models", "champion.json")),
              "nothing is written for a refused promotion")
    finally:
        _close(fx)
        shutil.rmtree(outside, ignore_errors=True)


def test_promote_refuses_a_run_without_a_ledger_row():
    fx = _make_root()
    try:
        run_dir = _write_run(os.path.join(fx["lab_dir"], "runs", "never_scored"), "detect")
        r = _client(fx).post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json() or {}
        check(r.status_code == 400, f"no ledger row answers 400: {r.status_code} {data}")
        check("evaluat" in str(data.get("error", "")).lower(), f"the error says to evaluate first: {data}")
    finally:
        _close(fx)


def test_promote_refuses_a_run_without_weights():
    fx = _make_root()
    try:
        run_dir = _write_run(os.path.join(fx["lab_dir"], "runs", "no_weights"), "detect", with_weights=False)
        r = _client(fx).post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json() or {}
        check(r.status_code == 400 and "best.pt" in str(data.get("error", "")),
              f"missing weights answers 400 naming best.pt: {r.status_code} {data}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# GET /api/step/7/rounds
# ---------------------------------------------------------------------------

def test_rounds_route_casts_box_recall_and_precision():
    fx = _make_root()
    try:
        r = _client(fx).get("/api/step/7/rounds")
        data = r.get_json() or {}
        check(r.status_code == 200, f"rounds: {r.status_code} {data}")
        row = (data.get("rounds") or [{}])[0]
        _write_rounds(fx["step6"], [{"round": 1, "run_name": "seg_run", "run_dir": fx["proj_run"],
                                     "recall_B": "0.25", "precision_B": "0.75", "metric_family": "box"}])
        row = (_client(fx).get("/api/step/7/rounds").get_json() or {}).get("rounds")[0]
        check(isinstance(row["recall_B"], float) and abs(row["recall_B"] - 0.25) < 1e-9,
              f"recall_B cast to float: {row}")
        check(isinstance(row["precision_B"], float) and abs(row["precision_B"] - 0.75) < 1e-9,
              f"precision_B cast to float: {row}")
        check(row["metric_family"] == "box", f"text columns pass through: {row}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# POST /api/step/6/run: seed, fractions, same-source guard, box source
# ---------------------------------------------------------------------------

def _stage_box_dataset(root, name="coral_intel", pairs=2):
    """A staged detection dataset the way the Bounding Boxes page writes it."""
    d = os.path.join(root, "bbox_datasets", name)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    for i in range(pairs):
        _write(os.path.join(d, "images", f"img{i}.jpg"), "x")
        _write(os.path.join(d, "labels", f"img{i}.txt"), "0 0.5 0.5 0.2 0.2\n")
    _write(os.path.join(d, "classes.txt"), "Alpha\n")
    _write(os.path.join(d, "data.yaml"), f"path: {d}\ntrain: images\nval: images\nnc: 1\nnames:\n  0: Alpha\n")
    _write(os.path.join(d, "ingest_manifest.json"),
           json.dumps({"n_images": pairs, "n_boxes": pairs, "classes": ["Alpha"]}))
    return d


def _open_bbox_project(pairs=2, trailing_slash=False):
    """(fx, client, dataset) with a box-dataset training project opened."""
    tmp = tempfile.mkdtemp(prefix="t9_bbox_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    os.makedirs(root, exist_ok=True)
    dataset = _stage_box_dataset(tmp, pairs=pairs)
    res = mint_training_project(root, "bbox", "coral_intel", dataset, "LO",
                                extra={"task_default": "detect"})
    if trailing_slash:
        project_json = os.path.join(res["run_dir"], "project.json")
        with open(project_json) as fh:
            state = json.load(fh)
        state["training_source"]["path"] = dataset + os.sep
        with open(project_json, "w") as fh:
            json.dump(state, fh)
    appmod._inprocess_root = lambda: root
    appmod._bbox_root = lambda: os.path.join(tmp, "bbox_datasets")
    client = appmod.create_app().test_client()
    r = client.post("/api/project/open", json={"project_dir": res["run_dir"]})
    check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
    fx = {"tmp": tmp, "root": root, "project_dir": res["run_dir"]}
    return fx, client, dataset


def _launch_step6(client, config, build=None):
    """POST /api/step/6/run with the runner and (optionally) the builder stubbed.
    Returns (response, argv, env, build calls as (source, dataset, seed, kwargs))."""
    captured = {}
    builds = []
    real_run = appmod.runner.run_cli_stage
    real_build = appmod._build_detect_dataset

    def _fake_run(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env_extra or {})
        return {"pid": 99}

    def _fake_build(source_dir, dataset_dir, seed_name, **kwargs):
        builds.append((source_dir, dataset_dir, seed_name, dict(kwargs)))
        os.makedirs(dataset_dir, exist_ok=True)
        _write(os.path.join(dataset_dir, "data.yaml"), f"path: {dataset_dir}\n")
        return {"n_pairs": 2, "n_unlabelled": 0, "counts": {"train": 2, "valid": 0, "test": 0}}

    appmod.runner.run_cli_stage = _fake_run
    if build != "real":
        appmod._build_detect_dataset = _fake_build
    try:
        r = client.put("/api/project/config/6", json=config)
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        r = client.post("/api/step/6/run")
        return r, captured.get("cmd") or [], captured.get("env") or {}, builds
    finally:
        appmod.runner.run_cli_stage = real_run
        appmod._build_detect_dataset = real_build


def test_step6_seed_is_the_normalised_dataset_name():
    fx, client, dataset = _open_bbox_project(trailing_slash=True)
    try:
        r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                                     "device": "0", "batch": 8,
                                                     "valid_ratio": 0.2, "test_ratio": 0.1})
        body = r.get_json() or {}
        check(r.status_code == 200, f"launch refused: {r.status_code} {body}")
        check(len(builds) == 1, f"the dataset is built once: {builds}")
        source, _dataset_dir, seed, kwargs = builds[0]
        check(seed == "coral_intel", f"a trailing slash on the source no longer empties the seed: {seed!r}")
        check(source == dataset, f"the source path is normalised: {source!r}")
        check(abs(kwargs.get("valid_frac") - 0.2) < 1e-9 and abs(kwargs.get("test_frac") - 0.1) < 1e-9,
              f"the panel's fractions reach the builder as keywords: {kwargs}")
        check(cmd[1] == "train", f"a prepared dataset drives train mode: {cmd}")
    finally:
        _close(fx)


def test_step6_forwards_the_panel_fractions_to_the_real_split():
    fx, client, dataset = _open_bbox_project(pairs=8)
    try:
        r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                                     "device": "0", "batch": 8,
                                                     "valid_ratio": 0.25, "test_ratio": 0.25},
                                            build="real")
        body = r.get_json() or {}
        check(r.status_code == 200, f"launch refused: {r.status_code} {body}")
        manifest_path = os.path.join(fx["project_dir"], pm.STEP_DIRS["6"], "dataset", "split_manifest.json")
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        check(manifest["seed"] == "coral_intel", f"seeded on the dataset name: {manifest['seed']!r}")
        check(abs(manifest["valid_frac"] - 0.25) < 1e-9 and abs(manifest["test_frac"] - 0.25) < 1e-9,
              f"the split manifest records the panel's fractions: {manifest}")
        counts = manifest["counts"]
        check(counts["valid"] == 2 and counts["test"] == 2 and counts["train"] == 4,
              f"8 images at 0.25 and 0.25 give 4, 2, 2: {counts}")
    finally:
        _close(fx)


def test_step6_refuses_a_non_numeric_fraction():
    fx, client, dataset = _open_bbox_project()
    try:
        r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                                     "device": "0", "batch": 8,
                                                     "valid_ratio": "lots", "test_ratio": 0.1})
        body = r.get_json() or {}
        check(r.status_code == 400, f"a non-numeric fraction is refused before any build: {r.status_code} {body}")
        check("valid" in str(body.get("error", "")).lower(), f"the error names the field: {body}")
        check(not builds and not cmd, "nothing was built or launched")
    finally:
        _close(fx)


def _make_composite(root, name):
    """A minimal composite export under <root>/_composite/<name>."""
    d = os.path.join(root, "_composite", name)
    os.makedirs(os.path.join(d, "all_images"), exist_ok=True)
    os.makedirs(os.path.join(d, "all_labels"), exist_ok=True)
    _write(os.path.join(d, "all_images", "a.jpg"), "x")
    _write(os.path.join(d, "all_labels", "a.txt"), "0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    _write(os.path.join(d, "data.yaml"), "names:\n  0: PA\n")
    _write(os.path.join(d, "composite.json"), json.dumps({"created_at": "2026-09-05 08:00:00"}))
    return d


def _open_composite_project():
    """(fx, client, set_a, set_b) with a composite training project on set_a opened."""
    tmp = tempfile.mkdtemp(prefix="t9_composite_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    set_a = _make_composite(root, "set_a")
    set_b = _make_composite(root, "set_b")
    res = mint_training_project(root, "composite", "set_a", set_a, "LO")
    appmod._inprocess_root = lambda: root
    appmod._composite_root = lambda: os.path.join(root, "_composite")
    client = appmod.create_app().test_client()
    r = client.post("/api/project/open", json={"project_dir": res["run_dir"]})
    check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
    return {"tmp": tmp, "root": root, "project_dir": res["run_dir"]}, client, set_a, set_b


def test_step6_same_source_guard_answers_409():
    fx, client, set_a, set_b = _open_composite_project()
    try:
        r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 512,
                                                     "device": "0", "batch": 8, "source_dir": set_b})
        body = r.get_json() or {}
        check(r.status_code == 409, f"a different source than the recorded one is refused: {r.status_code} {body}")
        err = str(body.get("error", ""))
        check("already trains from a different source" in err and "set_a" in err,
              f"the refusal uses the lab guard's wording and names the recorded source: {err!r}")
        check(not cmd, "nothing was launched")
    finally:
        _close(fx)


def test_step6_same_source_is_not_refused():
    fx, client, set_a, set_b = _open_composite_project()
    try:
        r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 512,
                                                     "device": "0", "batch": 8, "source_dir": set_a})
        body = r.get_json() or {}
        check(r.status_code == 200, f"the recorded source trains again: {r.status_code} {body}")
        check(cmd and cmd[1] == "all", f"a composite drives split-then-train mode: {cmd}")
    finally:
        _close(fx)


def _open_ordinary_project_with_box_root():
    """(fx, client, dataset): an ordinary project plus a staged box dataset root."""
    fx = _make_root()
    dataset = _stage_box_dataset(fx["tmp"])
    appmod._bbox_root = lambda: os.path.join(fx["tmp"], "bbox_datasets")
    client = _client(fx)
    return fx, client, dataset


def test_step6_refuses_a_box_dataset_on_an_outline_project():
    """An outline project pointed at a staged box dataset is refused before
    anything is written. Building that split replaces <step6>/dataset, which
    is the split this project's own runs were scored on, and its ledger would
    then rank box scores against mask scores."""
    fx, client, dataset = _open_ordinary_project_with_box_root()
    try:
        before = sorted(os.listdir(os.path.join(fx["step6"], "dataset")))
        r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                                     "device": "0", "batch": 8, "source_dir": dataset,
                                                     "valid_ratio": 0.2, "test_ratio": 0.1})
        body = r.get_json() or {}
        check(r.status_code == 409, f"the mix is refused: {r.status_code} {body}")
        err = str(body.get("error", ""))
        check("coral_intel" in err, f"the refusal names the dataset: {err!r}")
        check("Bounding Boxes" in err, f"and says where to take it instead: {err!r}")
        check(not builds and not cmd, "nothing was built or launched")
        check(sorted(os.listdir(os.path.join(fx["step6"], "dataset"))) == before,
              "the project's own split folder is untouched")
    finally:
        _close(fx)


def test_step6_refuses_a_box_source_whatever_the_task():
    """The refusal does not depend on the task the panel asked for: a box
    dataset belongs to a box training project either way."""
    fx, client, dataset = _open_ordinary_project_with_box_root()
    try:
        for task in ("segment", "detect"):
            r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                                         "device": "0", "batch": 8,
                                                         "source_dir": dataset, "task": task})
            body = r.get_json() or {}
            check(r.status_code == 409, f"task {task}: {r.status_code} {body}")
            check(not builds and not cmd, f"task {task}: nothing was built or launched")
    finally:
        _close(fx)


def test_step6_refuses_a_half_staged_box_dataset():
    fx, client, dataset = _open_ordinary_project_with_box_root()
    try:
        os.remove(os.path.join(dataset, "ingest_manifest.json"))
        r, cmd, env, builds = _launch_step6(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                                     "device": "0", "batch": 8, "source_dir": dataset})
        body = r.get_json() or {}
        check(r.status_code == 400 and "ingest_manifest.json" in str(body.get("error", "")),
              f"a folder without the ingest manifest is not a staged dataset: {r.status_code} {body}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# GET /api/composites
# ---------------------------------------------------------------------------

def test_composites_route_merges_box_datasets_with_their_kind():
    fx, client, dataset = _open_ordinary_project_with_box_root()
    try:
        appmod._composite_root = lambda: os.path.join(fx["root"], "_composite")
        _make_composite(fx["root"], "set_a")
        r = client.get("/api/composites")
        data = r.get_json() or {}
        check(r.status_code == 200, f"composites: {r.status_code} {data}")
        rows = data.get("composites") or []
        kinds = {row["name"]: (row.get("kind"), row.get("task")) for row in rows}
        check(kinds.get("set_a") == ("composite", "seg"), f"a composite says its kind and task: {kinds}")
        check(kinds.get("coral_intel") == ("bbox", "detect"), f"a box dataset says its kind and task: {kinds}")
        box = next(row for row in rows if row["name"] == "coral_intel")
        check(box["path"] == dataset and box["n_images"] == 2 and box["labels"] == ["Alpha"],
              f"the box row carries path, count and labels: {box}")
        check(data.get("bbox_root") == appmod._bbox_root(), f"the box root is echoed: {data}")
    finally:
        _close(fx)


if __name__ == "__main__":
    print("T9 -- cross-project runs table, champion promotion, Step 6 hand-offs\n")
    print("GET /api/step/6/all_runs:")
    run(test_all_runs_rows_carry_every_contract_key)
    run(test_all_runs_newest_first)
    run(test_all_runs_comparable_is_null_without_a_champion)
    run(test_all_runs_marks_the_champion_once_and_compares_family_and_split)
    run(test_the_same_measurement_is_comparable)
    run(test_a_different_overlap_is_not_comparable)
    run(test_a_different_image_size_is_not_comparable)
    run(test_a_different_confidence_is_not_comparable)
    run(test_one_side_recording_nothing_is_not_comparable)
    run(test_two_ledgers_that_record_nothing_stay_comparable)
    run(test_all_runs_hostile_run_name_is_a_plain_string)
    run(test_champion_row_is_never_marked_not_comparable)
    run(test_train_and_infer_page_is_gone)
    run(test_all_runs_works_without_an_open_project)
    print("GET /api/step/6/list_runs:")
    run(test_list_runs_rows_carry_task_and_family)
    print("POST /api/step/7/promote:")
    run(test_promote_accepts_a_foreign_run_and_writes_its_own_ledger)
    run(test_promote_writes_the_mask_score_for_a_segmenter)
    run(test_promote_refuses_a_run_outside_both_roots)
    run(test_promote_refuses_a_run_without_a_ledger_row)
    run(test_promote_refuses_a_run_without_weights)
    print("GET /api/step/7/rounds:")
    run(test_rounds_route_casts_box_recall_and_precision)
    print("POST /api/step/6/run:")
    run(test_step6_seed_is_the_normalised_dataset_name)
    run(test_step6_forwards_the_panel_fractions_to_the_real_split)
    run(test_step6_refuses_a_non_numeric_fraction)
    run(test_step6_same_source_guard_answers_409)
    run(test_step6_same_source_is_not_refused)
    run(test_step6_refuses_a_box_dataset_on_an_outline_project)
    run(test_step6_refuses_a_box_source_whatever_the_task)
    run(test_step6_refuses_a_half_staged_box_dataset)
    print("GET /api/composites:")
    run(test_composites_route_merges_box_datasets_with_their_kind)

    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    if leftovers:
        print(f"\nFIXTURES NOT REMOVED: {leftovers}")
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed; fixtures removed: {not leftovers}")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
    sys.exit(1 if failed or leftovers else 0)
