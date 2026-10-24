"""Task N2: a bbox training project opened in the orchestrator.

A run dir minted from a staged box dataset (project_type "bbox_training")
holds no annotation work: steps 1 to 4 are skipped, it opens on Train, and
Step 6 builds its own train/valid/test split from the dataset where it is
staged before driving the engine in "train" mode. No imagery of any kind
enters the run dir.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_bbox_training_project.py

Covers:
  STATE     /api/project/state marks the project as a training project
            generically (is_training_project + a source label the SPA banner
            reads) alongside the raw training_source block.
  GUARDS    the annotation routes refuse a bbox project exactly as they refuse
            a composite one.
  LAUNCH    Step 6 calls build_detect_dataset(source, <step6>/dataset, seed)
            and then drives run_step6.sh 'train' over the step 6 dir with
            TCRMP_STEP6_TASK=detect and the detection base weights.
  REFUSALS  a bbox project asked to train a segmentation model is refused with
            a plain message; a dataset with no image/label pair is refused
            with the builder's own message and nothing is launched.
  UI        the SPA's disabled-tile and banner logic keys off the generic
            flag, so one code path serves both kinds; and a box-dataset
            project hides the two Train-panel cards it has nothing to
            choose in (training source, class picker), replacing them
            with one line naming the dataset.

Nothing here reaches a real process or a real dataset build: subprocess is
stubbed module wide and the dataset builder is replaced per test.
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


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
from _reefreview.training_projects import mint_training_project  # noqa: E402

# Test isolation (2026-09-28 review): _kill_all_annotators' flush-before-kill
# (POST /api/flush, pipeline_orchestrator/app.py) reaches the network even
# when subprocess.run/Popen are stubbed, since it goes through urllib, not
# subprocess. This file's step4test/step4loop/editmasks route calls can
# drive it, so appmod._flush_annotator is stubbed here (never a real save,
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

REFUSAL = ("Not available for a training project. "
           "This project trains from the training set or box dataset named "
           "in its README.")

_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_TPL = os.path.join(_ORCH, "templates", "index.html")


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

def _stage_dataset(root, name="coral_intel", pairs=2):
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


def _open_bbox_project(pairs=2):
    """(client, tmp_root, run_dir, dataset_dir) with a bbox project opened."""
    tmp = tempfile.mkdtemp(prefix="bbox_train_proj_")
    inprocess = os.path.join(tmp, "inprocess")
    os.makedirs(inprocess, exist_ok=True)
    dataset = _stage_dataset(tmp, pairs=pairs)
    res = mint_training_project(inprocess, "bbox", "coral_intel", dataset, "LO",
                                extra={"task_default": "detect"})
    client = appmod.create_app().test_client()
    r = client.post("/api/project/open", json={"project_dir": res["run_dir"]})
    check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
    return client, tmp, res["run_dir"], dataset


def _close(tmp):
    appmod.current_project = None
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def test_state_marks_a_training_project_generically():
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        data = client.get("/api/project/state").get_json()
        st = data["state"]
        check(st.get("project_type") == "bbox_training",
              f"project_type must survive the load (got {st.get('project_type')!r})")
        check(st.get("is_training_project") is True,
              f"the state must carry the generic flag (got {st.get('is_training_project')!r})")
        check(st.get("training_source_label") == "box dataset",
              f"the banner label for a bbox project (got {st.get('training_source_label')!r})")
        check(st.get("training_source_name") == "coral_intel",
              f"the banner names the dataset (got {st.get('training_source_name')!r})")
        src = st.get("training_source") or {}
        check(src.get("kind") == "bbox" and src.get("path") == dataset,
              f"the raw training_source block is preserved (got {src})")
        check(src.get("task_default") == "detect",
              f"a bbox project defaults to the detector (got {src})")
        check(st.get("current_step") == 6, f"the project opens on Train (got {st.get('current_step')})")
        check(st["steps"]["6"]["status"] != "locked",
              f"Train is unlocked from the start (got {st['steps']['6']['status']!r})")
        # No imagery, no symlink: a bbox project reads the dataset in place.
        check(not os.path.exists(os.path.join(run_dir, "step4test_combinedAnnotate")),
              "a bbox project must not carry a step4test link")
    finally:
        _close(tmp)


def test_composite_state_still_carries_the_generic_flag():
    """One flag serves both kinds, so the SPA needs one code path."""
    from _matrix.composite import create_training_project
    tmp = tempfile.mkdtemp(prefix="bbox_train_comp_")
    try:
        out = os.path.join(tmp, "_composite", "set_one")
        os.makedirs(os.path.join(out, "all_images"))
        os.makedirs(os.path.join(out, "all_labels"))
        with open(os.path.join(out, "all_images", "i.jpg"), "w") as fh:
            fh.write("x")
        with open(os.path.join(out, "data.yaml"), "w") as fh:
            fh.write("path: x\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: MC\n")
        res = create_training_project(tmp, "set_one", "LO")
        client = appmod.create_app().test_client()
        client.post("/api/project/open", json={"project_dir": res["run_dir"]})
        st = client.get("/api/project/state").get_json()["state"]
        check(st.get("is_training_project") is True, "a composite project carries the flag too")
        check(st.get("training_source_label") == "training set",
              f"the composite banner label (got {st.get('training_source_label')!r})")
        check(st.get("training_source_name") == "set_one",
              f"the composite banner names the set (got {st.get('training_source_name')!r})")
    finally:
        _close(tmp)


def test_ordinary_project_is_not_marked():
    import project_manager as pm
    tmp = tempfile.mkdtemp(prefix="bbox_train_plain_")
    try:
        for sd in pm.STEP_DIRS.values():
            os.makedirs(os.path.join(tmp, sd), exist_ok=True)
        with open(os.path.join(tmp, "project.json"), "w") as fh:
            json.dump({"id": "plain", "name": "plain", "project_dir": tmp,
                       "created_at": "x", "updated_at": "x", "current_step": 1,
                       "steps": {}}, fh)
        client = appmod.create_app().test_client()
        client.post("/api/project/open", json={"project_dir": tmp})
        st = client.get("/api/project/state").get_json()["state"]
        check(st.get("is_training_project") is False,
              f"an ordinary project is not a training project (got {st.get('is_training_project')!r})")
        check(not st.get("training_source_label"),
              f"and carries no source label (got {st.get('training_source_label')!r})")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_annotation_routes_refuse_a_bbox_project():
    client, tmp, _run_dir, _ds = _open_bbox_project()
    try:
        for step in (1, 2, 3):
            r = client.post(f"/api/step/{step}/run", json={})
            check(r.status_code == 400, f"step {step} run must be refused (got {r.status_code})")
            check((r.get_json() or {}).get("error") == REFUSAL,
                  f"step {step} refusal message (got {r.get_json()})")
        for route in ("/api/step/step4test/run", "/api/step/step4loop/run",
                      "/api/step/editmasks/run", "/api/step/step4test/reset"):
            r = client.post(route, json={})
            body = r.get_json() or {}
            check(r.status_code == 400, f"{route} must be refused (got {r.status_code} {body})")
            check(body.get("error") == REFUSAL, f"{route}: wrong message {body.get('error')!r}")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# step 6 launch
# ---------------------------------------------------------------------------

def _run_step6(client, config, build=None):
    """Run step 6 with the dataset builder and the stage runner stubbed.
    Returns (response, cmd, env, build calls)."""
    captured = {}
    builds = []
    real_run = appmod.runner.run_cli_stage
    real_build = appmod._build_detect_dataset

    def _fake_run(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env_extra or {})
        return {"pid": 99}

    def _fake_build(source_dir, dataset_dir, seed_name, **fractions):
        builds.append((source_dir, dataset_dir, seed_name))
        if build is not None:
            return build(source_dir, dataset_dir, seed_name, **fractions)
        os.makedirs(dataset_dir, exist_ok=True)
        with open(os.path.join(dataset_dir, "data.yaml"), "w") as fh:
            fh.write("path: %s\n" % dataset_dir)
        return {"n_pairs": 2, "n_unlabelled": 0,
                "counts": {"train": 2, "valid": 0, "test": 0}}

    appmod.runner.run_cli_stage = _fake_run
    appmod._build_detect_dataset = _fake_build
    try:
        r = client.put("/api/project/config/6", json=config)
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        r = client.post("/api/step/6/run")
        return r, captured.get("cmd") or [], captured.get("env") or {}, builds
    finally:
        appmod.runner.run_cli_stage = real_run
        appmod._build_detect_dataset = real_build


def test_step6_builds_the_dataset_then_drives_train_mode():
    client, tmp, run_dir, dataset = _open_bbox_project()
    try:
        r, cmd, env, builds = _run_step6(client, {"run_name": "bbox_run", "epochs": 20,
                                                  "imgsz": 256, "device": "0", "batch": 8})
        body = r.get_json() or {}
        check(r.status_code == 200, f"the launch was refused: {body}")

        step6_dir = appmod.current_project["steps"]["6"]["dir"]
        check(len(builds) == 1, f"the dataset is built exactly once (got {len(builds)})")
        src, dst, seed = builds[0]
        check(src == dataset, f"the build reads the dataset where it is staged (got {src!r})")
        check(dst == os.path.join(step6_dir, "dataset"),
              f"the build writes <step6>/dataset (got {dst!r})")
        check(seed == "coral_intel",
              f"the split is seeded on the dataset name so it is reproducible (got {seed!r})")

        check(cmd[1] == "train",
              f"a prepared dataset drives the engine's train mode (got {cmd!r})")
        check(cmd[2] == step6_dir, f"train mode takes the step 6 dir (got {cmd[2]!r})")
        check(cmd[3] == "bbox_run", f"the run name reaches the engine (got {cmd[3]!r})")
        check(cmd[4] == "20" and cmd[5] == "256",
              f"epochs and image size pass through (got {cmd!r})")
        check(cmd[6] == "yolo11m.pt",
              f"the base weights follow the detection task (got {cmd[6]!r})")
        check(env.get("TCRMP_STEP6_TASK") == "detect",
              f"the task env is detect (got {env.get('TCRMP_STEP6_TASK')!r})")
        check(env.get("TCRMP_STEP6_BATCH") == "8" and env.get("TCRMP_STEP6_DEVICE") == "0",
              f"the ordinary knobs still pass through (got {env})")
        split = body.get("split") or {}
        check(split.get("n_pairs") == 2, f"the response reports the split (got {body})")

        # Nothing was copied into the run dir except the built split.
        entries = set(os.listdir(run_dir))
        check("step4test_combinedAnnotate" not in entries,
              f"no export link is invented (got {sorted(entries)})")
    finally:
        _close(tmp)


def test_step6_defaults_to_detect_without_an_explicit_task():
    client, tmp, _run_dir, _ds = _open_bbox_project()
    try:
        r, cmd, env, builds = _run_step6(client, {"run_name": "auto", "device": "0"})
        check(r.status_code == 200, f"the launch was refused: {r.get_json()}")
        check(env.get("TCRMP_STEP6_TASK") == "detect",
              f"a bbox project trains a detector by default (got {env.get('TCRMP_STEP6_TASK')!r})")
        check(len(builds) == 1, "the dataset is built")
    finally:
        _close(tmp)


def test_step6_refuses_a_segmentation_task_on_a_box_dataset():
    client, tmp, _run_dir, _ds = _open_bbox_project()
    try:
        r, cmd, env, builds = _run_step6(client, {"run_name": "wrong", "task": "segment",
                                                  "device": "0"})
        check(r.status_code == 400,
              f"a box dataset cannot train a segmentation model (got {r.status_code})")
        check(not builds, "nothing is built for a refused pairing")
        check(not cmd, "nothing is launched for a refused pairing")
        err = str((r.get_json() or {}).get("error") or "")
        check("box" in err.lower() and "outline" in err.lower(),
              f"the refusal says why in plain language (got {err!r})")
    finally:
        _close(tmp)


def test_step6_surfaces_an_unbuildable_dataset():
    client, tmp, _run_dir, _ds = _open_bbox_project()

    def _raise(source_dir, dataset_dir, seed_name, **_fractions):
        raise ValueError(f"no image and label pairs found in {source_dir}")

    try:
        r, cmd, env, builds = _run_step6(client, {"run_name": "empty", "device": "0"},
                                         build=_raise)
        check(r.status_code == 400, f"an unbuildable dataset is a 400 (got {r.status_code})")
        check(not cmd, "nothing is launched when the build fails")
        err = str((r.get_json() or {}).get("error") or "")
        check("no image and label pairs" in err,
              f"the builder's own message reaches the panel (got {err!r})")
    finally:
        _close(tmp)


def test_step6_refuses_a_missing_dataset_folder():
    client, tmp, _run_dir, dataset = _open_bbox_project()
    try:
        shutil.rmtree(dataset)
        r, cmd, env, builds = _run_step6(client, {"run_name": "gone", "device": "0"})
        check(r.status_code == 400, f"a vanished dataset is a 400 (got {r.status_code})")
        check(not builds and not cmd, "nothing is built or launched for a vanished dataset")
        err = str((r.get_json() or {}).get("error") or "")
        check("coral_intel" in err, f"the refusal names the dataset (got {err!r})")
    finally:
        _close(tmp)


def test_step6_really_builds_the_split():
    """The real builder, not the stub: only the launcher is replaced, so this
    proves the wiring reaches TCRMPtrain_oceankindCV/src/detect_dataset.py and
    writes the layout the engine and the evaluation step both read."""
    client, tmp, _run_dir, dataset = _open_bbox_project(pairs=12)
    captured = {}
    real_run = appmod.runner.run_cli_stage

    def _fake_run(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        return {"pid": 99}

    appmod.runner.run_cli_stage = _fake_run
    try:
        client.put("/api/project/config/6", json={"run_name": "real", "epochs": 1,
                                                  "imgsz": 256, "device": "0"})
        r = client.post("/api/step/6/run")
        check(r.status_code == 200, f"the launch was refused: {r.get_json()}")

        step6_dir = appmod.current_project["steps"]["6"]["dir"]
        ds = os.path.join(step6_dir, "dataset")
        check(os.path.isfile(os.path.join(ds, "data.yaml")),
              "the built dataset carries the data.yaml the engine trains from")
        check(os.path.isfile(os.path.join(ds, "test.yaml")),
              "and the test.yaml the evaluation step reads")
        manifest_path = os.path.join(ds, "split_manifest.json")
        check(os.path.isfile(manifest_path), "and a split manifest")
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        check(manifest.get("holdout_mode") == "lab-random",
              f"the holdout is recorded honestly (got {manifest.get('holdout_mode')!r})")
        check(manifest.get("seed") == "coral_intel",
              f"the split is seeded on the dataset name (got {manifest.get('seed')!r})")
        check(manifest.get("n_pairs") == 12, f"every pair is split (got {manifest.get('n_pairs')})")
        total = 0
        for split in ("train", "valid", "test"):
            imgs = os.path.join(ds, split, "images")
            lbls = os.path.join(ds, split, "labels")
            check(os.path.isdir(imgs) and os.path.isdir(lbls),
                  f"the {split} split has images/ and labels/")
            total += len(os.listdir(imgs))
        check(total == 12, f"every image reaches a split exactly once (got {total})")

        # The staged dataset is untouched: nothing was moved out of it.
        check(len(os.listdir(os.path.join(dataset, "images"))) == 12,
              "the staged dataset is left exactly as it was")

        # Re-running replays the pins rather than reshuffling.
        first = dict(manifest["pins"])
        client.post("/api/step/6/run")
        with open(manifest_path) as fh:
            again = json.load(fh)
        check(again["pins"] == first,
              "retraining replays the same split rather than moving images between them")
    finally:
        appmod.runner.run_cli_stage = real_run
        _close(tmp)


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------

def test_box_dataset_hides_the_source_and_class_cards():
    """A box-dataset project has nothing to choose in either card: the server
    ignores cfg.source_dir on that path and forces include_classes to None, and
    /api/step/6/list_classes answers 400 naming steps the project cannot run.
    Both cards are hidden and one line names the dataset instead."""
    tpl = open(_TPL).read()
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    for cid in ("s6-source-card", "s6-classes-card", "s6-bbox-source-note"):
        check(f'id="{cid}"' in panel, f"the Train panel must carry {cid}")
    note_tag = panel[panel.index('id="s6-bbox-source-note"') - 200:]
    note_tag = note_tag[:note_tag.index(">", note_tag.index('id="s6-bbox-source-note"')) + 1]
    check("display:none" in note_tag,
          "the box-dataset note is hidden until a box-dataset project opens")
    check("title=" in note_tag, "the box-dataset note carries a tooltip")

    js = open(_JS).read()
    check("function isBoxDatasetProject" in js, "orchestrator.js must define isBoxDatasetProject")
    fn = js[js.index("function isBoxDatasetProject"):]
    fn = fn[:fn.index("\n}") + 2]
    check("'box dataset'" in fn,
          "the box-dataset mode keys off the server's source label")
    check("training_source_label" in fn,
          "and reads it from the generic state key, not the project type")

    check("function applyStep6BoxDatasetMode" in js,
          "orchestrator.js must define applyStep6BoxDatasetMode")
    mode = js[js.index("function applyStep6BoxDatasetMode"):]
    mode = mode[:mode.index("\n}\n") + 3]
    for cid in ("s6-source-card", "s6-classes-card"):
        check(cid in mode, f"the mode must hide {cid}")
    check("'none'" in mode, "the two cards are hidden rather than left rendered")
    check("s6-bbox-source-note" in mode and "training_source_name" in mode,
          "the replacement line must name the dataset")
    check("escapeHtml(" in mode, "the dataset name must be escaped")

    i = js.index("const COMPOSITE_DISABLED_TILES")
    body = js[i:js.index("function enterApp")]
    check("applyStep6BoxDatasetMode()" in body,
          "the box-dataset mode must run when the training-project mode is applied")

    # The two requests those cards drive are skipped as well, so a box-dataset
    # project never fires a call whose only outcome is a 400 in the console.
    for name in ("refreshStep6Sources", "refreshStep6Classes"):
        fn = js[js.index(f"async function {name}("):]
        fn = fn[:fn.index("\n}") + 2]
        check("isBoxDatasetProject()" in fn,
              f"{name} must return early for a box-dataset project")

    # Composite and ordinary projects are untouched: nothing hides those cards
    # outside the box-dataset branch.
    check(js.count("s6-source-card") == 1 and js.count("s6-classes-card") == 1,
          "only the box-dataset branch touches the two cards")


def test_spa_keys_off_the_generic_flag():
    js = open(_JS).read()
    i = js.index("const COMPOSITE_DISABLED_TILES")
    body = js[i:js.index("function enterApp")]
    check("is_training_project" in body,
          "the disabled-tile mode must key off the generic state flag")
    check("training_source_label" in body,
          "the Train banner must read the generic source label")
    check("training_source_name" in body,
          "the Train banner must read the generic source name")
    check("Training from " in body,
          "the banner text must read 'Training from <label> <name>'")
    check("escapeHtml(" in body, "the banner text must be escaped")
    for key in ("'1'", "'2'", "'3'", "'step4test'", "'step4loop'", "'editmasks'"):
        check(key in body, f"the disabled-tile list must still cover {key}")


if __name__ == "__main__":
    print("TASK N2: bbox training projects\n")
    print("STATE:")
    run(test_state_marks_a_training_project_generically)
    run(test_composite_state_still_carries_the_generic_flag)
    run(test_ordinary_project_is_not_marked)
    print("GUARDS:")
    run(test_annotation_routes_refuse_a_bbox_project)
    print("STEP 6 LAUNCH:")
    run(test_step6_builds_the_dataset_then_drives_train_mode)
    run(test_step6_defaults_to_detect_without_an_explicit_task)
    run(test_step6_refuses_a_segmentation_task_on_a_box_dataset)
    run(test_step6_surfaces_an_unbuildable_dataset)
    run(test_step6_refuses_a_missing_dataset_folder)
    run(test_step6_really_builds_the_split)
    print("UI:")
    run(test_box_dataset_hides_the_source_and_class_cards)
    run(test_spa_keys_off_the_generic_flag)

    failed = [n for n, ok, _ in _RESULTS if not ok]
    print()
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAILED {name}:\n{detail}")
    print("PASS" if not failed else f"{len(failed)} FAILED")
    sys.exit(1 if failed else 0)
