"""A project trains from one family of source, and nothing mixes the two.

The Train panel's source list offers the staged box datasets beside the
outline training sets, so an outline project could be pointed at a box
dataset. Building that split replaces <step6>/dataset whole, which is the
split that project's own outline runs were trained and scored on: those runs
lose the dataset the Evaluate step reads for them, and the ledger beside them
starts ranking box scores against mask scores. Until 2026-09-16 that happened
with no warning at all.

Covered here:
  REFUSAL   an outline project pointed at a staged box dataset is refused
            before anything is written, with one sentence saying why and
            where to take the dataset instead; the refusal does not depend on
            the task the panel asked for, on whether the project already has
            runs, or on how the path was spelled.
  ALLOWED   the legitimate paths are untouched: a box training project trains
            from its own staged dataset, and an outline project trains from
            an outline training set.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_source_family_guard.py

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, the stage runner's launch is replaced by a capture,
and the dataset builder is replaced by a recorder so no split is ever built.
Fixtures are removed at the end and the removal is verified.
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


subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402
from _reefreview.training_projects import mint_training_project  # noqa: E402


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
_REAL_INPROCESS_ROOT = appmod._inprocess_root
_REAL_BBOX_ROOT = appmod._bbox_root
BOX_DATASET_NAME = "coral_intel"
EXISTING_RUN_NAME = "seg_run"


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


def _stage_box_dataset(root, name=BOX_DATASET_NAME, pairs=2):
    """A finished staged detection dataset, the way the Bounding Boxes page
    leaves one: images/ beside labels/, classes.txt, data.yaml and the ingest
    manifest."""
    d = os.path.join(root, "bbox_datasets", name)
    for i in range(pairs):
        _write(os.path.join(d, "images", f"img{i}.jpg"), "x")
        _write(os.path.join(d, "labels", f"img{i}.txt"), "0 0.5 0.5 0.2 0.2\n")
    _write(os.path.join(d, "classes.txt"), "Alpha\n")
    _write(os.path.join(d, "data.yaml"),
           f"path: {d}\ntrain: images\nval: images\nnc: 1\nnames:\n  0: Alpha\n")
    _write(os.path.join(d, "ingest_manifest.json"),
           json.dumps({"n_images": pairs, "n_boxes": pairs, "classes": ["Alpha"]}))
    return d


def _outline_project(root):
    """An ordinary annotation project that has already trained once: a Step 4
    export to train from, a split dataset, and one finished run scored
    against it."""
    project_dir = os.path.join(root, "run_26-09-16_LO_outline")
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    export = os.path.join(project_dir, pm.STEP_DIRS["step4test"])
    _write(os.path.join(export, "all_images", "a.jpg"), "x")
    _write(os.path.join(export, "all_labels", "a.txt"), "0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    _write(os.path.join(export, "data.yaml"), "names:\n  0: PA\n")
    step6 = steps["6"]["dir"]
    _write(os.path.join(step6, "dataset", "data.yaml"), "names:\n  0: PA\n")
    _write(os.path.join(step6, "dataset", "split_manifest.json"),
           json.dumps({"split_fingerprint": "fp-mask", "split_fingerprint_scheme": "v1"}))
    _write(os.path.join(step6, "runs", EXISTING_RUN_NAME, "weights", "best.pt"), "x")
    _write(os.path.join(step6, "runs", EXISTING_RUN_NAME, "args.yaml"),
           "task: segment\nimgsz: 512\nmodel: yolo11m-seg.pt\n")
    state = {"id": os.path.basename(project_dir), "name": "outline fixture",
             "project_dir": project_dir, "created_at": "x", "updated_at": "x",
             "current_step": 6, "steps": steps}
    pm.save_project(state)
    return state


def _fixture():
    tmp = tempfile.mkdtemp(prefix="source_family_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    os.makedirs(root, exist_ok=True)
    dataset = _stage_box_dataset(tmp)
    state = _outline_project(root)
    appmod._inprocess_root = lambda: root
    appmod._bbox_root = lambda: os.path.join(tmp, "bbox_datasets")
    return {"tmp": tmp, "root": root, "dataset": dataset, "state": state,
            "step6": state["steps"]["6"]["dir"]}


def _client(fx):
    appmod.current_project = fx["state"]
    return appmod.create_app().test_client()


def _close(fx):
    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    appmod._bbox_root = _REAL_BBOX_ROOT
    shutil.rmtree(fx["tmp"], ignore_errors=True)


def _launch(client, config):
    """Save the Train config and press Start Training, with the launcher and
    the dataset builder replaced. Returns (response, argv, builds)."""
    captured = {}
    builds = []
    real_run = appmod.runner.run_cli_stage
    real_build = appmod._build_detect_dataset

    def _fake_run(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env_extra or {})
        return {"pid": 99}

    def _fake_build(source_dir, dataset_dir, seed_name, **kwargs):
        builds.append((source_dir, dataset_dir, seed_name))
        os.makedirs(dataset_dir, exist_ok=True)
        _write(os.path.join(dataset_dir, "data.yaml"), f"path: {dataset_dir}\n")
        return {"n_pairs": 2, "n_unlabelled": 0, "counts": {"train": 2, "valid": 0, "test": 0}}

    appmod.runner.run_cli_stage = _fake_run
    appmod._build_detect_dataset = _fake_build
    try:
        r = client.put("/api/project/config/6", json=config)
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        return client.post("/api/step/6/run"), captured.get("cmd") or [], builds
    finally:
        appmod.runner.run_cli_stage = real_run
        appmod._build_detect_dataset = real_build


# ---------------------------------------------------------------------------
# REFUSAL
# ---------------------------------------------------------------------------

def test_an_outline_project_is_refused_a_box_dataset():
    fx = _fixture()
    try:
        client = _client(fx)
        r, cmd, builds = _launch(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                          "device": "0", "batch": 8,
                                          "source_dir": fx["dataset"]})
        body = r.get_json() or {}
        check(r.status_code == 409, f"the mix is refused: {r.status_code} {body}")
        err = str(body.get("error") or "")
        check(BOX_DATASET_NAME in err, f"the refusal names the dataset: {err!r}")
        check("Bounding Boxes" in err, f"and says where to take it instead: {err!r}")
        check(err.count(".") <= 3, f"the reason is short enough to read: {err!r}")
        check(not builds and not cmd, "nothing was built and nothing was launched")
    finally:
        _close(fx)


def test_the_refusal_leaves_the_project_exactly_as_it_was():
    """The whole point of the guard: the split the project's own runs were
    scored on, and the run folder itself, are still there."""
    fx = _fixture()
    try:
        client = _client(fx)
        dataset_dir = os.path.join(fx["step6"], "dataset")
        run_dir = os.path.join(fx["step6"], "runs", EXISTING_RUN_NAME)
        before = sorted(os.listdir(dataset_dir))
        _launch(client, {"run_name": "r1", "device": "0", "source_dir": fx["dataset"]})
        check(sorted(os.listdir(dataset_dir)) == before,
              f"the split folder is untouched: {sorted(os.listdir(dataset_dir))}")
        check(os.path.isfile(os.path.join(dataset_dir, "split_manifest.json")),
              "including the split manifest the runs are matched against")
        check(os.path.isfile(os.path.join(run_dir, "weights", "best.pt")),
              "and the existing run still has its weights")
    finally:
        _close(fx)


def test_the_refusal_does_not_depend_on_the_task():
    fx = _fixture()
    try:
        client = _client(fx)
        for task in ("segment", "detect", ""):
            config = {"run_name": "r1", "device": "0", "source_dir": fx["dataset"]}
            if task:
                config["task"] = task
            r, cmd, builds = _launch(client, config)
            check(r.status_code == 409, f"task {task!r}: {r.status_code} {r.get_json()}")
            check(not builds and not cmd, f"task {task!r}: nothing built or launched")
    finally:
        _close(fx)


def test_a_trailing_slash_does_not_slip_past_the_guard():
    fx = _fixture()
    try:
        client = _client(fx)
        r, cmd, builds = _launch(client, {"run_name": "r1", "device": "0",
                                          "source_dir": fx["dataset"] + "/"})
        check(r.status_code == 409, f"a trailing slash is the same path: {r.status_code} {r.get_json()}")
        check(not builds and not cmd, "nothing built or launched")
    finally:
        _close(fx)


def test_a_half_staged_dataset_is_still_reported_as_half_staged():
    """The guard does not swallow the more specific refusal: a folder that is
    not a finished staged dataset says so by name."""
    fx = _fixture()
    try:
        os.remove(os.path.join(fx["dataset"], "ingest_manifest.json"))
        client = _client(fx)
        r, cmd, builds = _launch(client, {"run_name": "r1", "device": "0",
                                          "source_dir": fx["dataset"]})
        body = r.get_json() or {}
        check(r.status_code == 400, f"an unfinished dataset is a 400: {r.status_code} {body}")
        check("ingest_manifest.json" in str(body.get("error") or ""),
              f"and names the file that is missing: {body}")
    finally:
        _close(fx)


def test_the_source_list_says_a_box_dataset_is_not_usable_here():
    """The panel must not offer a choice the launch always refuses. Every row
    says whether the open project may train from it, and a row it may not
    carries the reason."""
    fx = _fixture()
    try:
        client = _client(fx)
        rows = (client.get("/api/composites").get_json() or {}).get("composites") or []
        box = next((r for r in rows if r["name"] == BOX_DATASET_NAME), None)
        check(box is not None, f"the staged dataset is still listed: {rows}")
        check(box.get("usable_here") is False,
              f"and is marked unusable in an outline project: {box}")
        check("Bounding Boxes" in str(box.get("refusal") or ""),
              f"with the reason the launch would give: {box}")
    finally:
        _close(fx)


def test_the_source_list_marks_a_box_dataset_usable_in_a_box_project():
    fx = _fixture()
    try:
        res = mint_training_project(fx["root"], "bbox", BOX_DATASET_NAME, fx["dataset"], "LO",
                                    extra={"task_default": "detect"})
        client = appmod.create_app().test_client()
        check(client.post("/api/project/open", json={"project_dir": res["run_dir"]}).status_code == 200,
              "open failed")
        rows = (client.get("/api/composites").get_json() or {}).get("composites") or []
        box = next((r for r in rows if r["name"] == BOX_DATASET_NAME), None)
        check(box is not None and box.get("usable_here") is True and box.get("refusal") == "",
              f"a box training project may train from it: {box}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# ALLOWED
# ---------------------------------------------------------------------------

def test_a_box_training_project_still_trains_from_its_own_dataset():
    """The legitimate path: a project minted for that dataset trains from it
    exactly as before."""
    fx = _fixture()
    try:
        res = mint_training_project(fx["root"], "bbox", BOX_DATASET_NAME, fx["dataset"], "LO",
                                    extra={"task_default": "detect"})
        client = appmod.create_app().test_client()
        r = client.post("/api/project/open", json={"project_dir": res["run_dir"]})
        check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
        r, cmd, builds = _launch(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                          "device": "0", "batch": 8})
        body = r.get_json() or {}
        check(r.status_code == 200, f"a box project trains from its own dataset: {r.status_code} {body}")
        check(len(builds) == 1 and builds[0][0] == fx["dataset"],
              f"the split is built from the staged dataset: {builds}")
        check(cmd and cmd[1] == "train", f"and the engine is driven over it: {cmd}")
    finally:
        _close(fx)


def test_an_outline_project_still_trains_from_its_own_export():
    fx = _fixture()
    try:
        client = _client(fx)
        r, cmd, builds = _launch(client, {"run_name": "r1", "epochs": 2, "imgsz": 256,
                                          "device": "0", "batch": 8})
        body = r.get_json() or {}
        check(r.status_code == 200, f"the ordinary path is untouched: {r.status_code} {body}")
        check(not builds, "an outline export is split by the engine, not by the box builder")
        check(cmd and cmd[1] == "all", f"and drives split-then-train mode: {cmd}")
    finally:
        _close(fx)


if __name__ == "__main__":
    print("SOURCE FAMILY GUARD -- no project trains from both families\n")
    print("REFUSAL:")
    run(test_an_outline_project_is_refused_a_box_dataset)
    run(test_the_refusal_leaves_the_project_exactly_as_it_was)
    run(test_the_refusal_does_not_depend_on_the_task)
    run(test_a_trailing_slash_does_not_slip_past_the_guard)
    run(test_a_half_staged_dataset_is_still_reported_as_half_staged)
    run(test_the_source_list_says_a_box_dataset_is_not_usable_here)
    run(test_the_source_list_marks_a_box_dataset_usable_in_a_box_project)
    print("ALLOWED:")
    run(test_a_box_training_project_still_trains_from_its_own_dataset)
    run(test_an_outline_project_still_trains_from_its_own_export)

    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    appmod._bbox_root = _REAL_BBOX_ROOT
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
