"""GET /api/step/6/source_summary: the Train panel's training-data preview.

Spec: docs/superpowers/specs/2026-09-21-vega-starting-model-and-training-
data-design.md section 7. The route is read-only (it never builds a dataset)
and answers how many images and labels a source holds and what its train,
validation and test split would look like.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_source_summary.py

Covers:
  OUTLINE   a segmentation export's Pictures and Split lines: the transect
            rule (parse_transect), the random fallback when no transect 5 or
            6 image is present, pin_split=0 forcing random without a
            fallback warning, include_classes filtering, an image with no
            label excluded from the count, hostile basenames (spaces,
            quotes, a control character), and a 1,500-image timing check.
  BBOX      a staged box dataset's counts (every image counts; n_background
            is the count with no box) and split: the pinned rule
            (split_pins.json), an unreadable pins file falling back to
            random with a warning, and no pins file at all.
  ERRORS    a missing folder, an empty all_images/, a path outside the
            allowed roots, no project loaded, an unknown task value, and a
            box source asked to preview a segmentation split.
  DEFAULT   a blank source_dir resolves to the open project's own export, or
            (on a bbox_training project) its recorded box dataset.

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, as every orchestrator suite does.
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []


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
# fixtures
# ---------------------------------------------------------------------------

def _make_project_state(tmp):
    """An ordinary (non-training) project state, every step dir on disk,
    opened on Train."""
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    return {"id": os.path.basename(tmp), "name": "t", "project_dir": tmp,
            "created_at": "x", "updated_at": "x", "current_step": 6, "steps": steps}


@contextlib.contextmanager
def _project_client():
    """(client) with an ordinary project open; the project dir is removed and
    current_project cleared on exit."""
    tmp = tempfile.mkdtemp(prefix="step6_summary_proj_")
    appmod.current_project = _make_project_state(tmp)
    try:
        yield appmod.create_app().test_client()
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


@contextlib.contextmanager
def _no_project_client():
    """(client) with no project open."""
    appmod.current_project = None
    try:
        yield appmod.create_app().test_client()
    finally:
        appmod.current_project = None


@contextlib.contextmanager
def _bbox_project_client(dataset_dir):
    """(client) with a bbox_training project open, recorded as training from
    dataset_dir (the way _reefreview/training_projects.py mints one)."""
    tmp = tempfile.mkdtemp(prefix="step6_summary_bboxproj_")
    state = _make_project_state(tmp)
    state["project_type"] = appmod.BBOX_PROJECT_TYPE
    state["training_source"] = {"kind": "bbox", "name": os.path.basename(dataset_dir),
                                "path": dataset_dir, "task_default": "detect"}
    appmod.current_project = state
    try:
        yield appmod.create_app().test_client()
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


class _Roots:
    """Points appmod's composite/standalone/bbox root helpers at one tmp
    tree, the same monkeypatch pattern test_step6_source.py and
    test_step6_all_runs.py use, and restores them on exit."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="step6_summary_roots_")
        self.composite = os.path.join(self.tmp, "_composite")
        self.standalone = os.path.join(self.tmp, "_standalone")
        self.bbox = os.path.join(self.tmp, "bbox_datasets")
        for d in (self.composite, self.standalone, self.bbox):
            os.makedirs(d, exist_ok=True)
        self._orig = (appmod._composite_root, appmod._standalone_root, appmod._bbox_root)

    def __enter__(self):
        appmod._composite_root = lambda: self.composite
        appmod._standalone_root = lambda: self.standalone
        appmod._bbox_root = lambda: self.bbox
        return self

    def __exit__(self, *exc):
        appmod._composite_root, appmod._standalone_root, appmod._bbox_root = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False


def _write_outline_export(d, images_labels, names):
    """Populate an existing directory d as an outline export: all_images/,
    all_labels/, data.yaml.

    images_labels: {basename: [class_id, ...]}, one label line per class id;
    an empty list writes the image with no label file at all.
    names: {class_id: name} for data.yaml.
    """
    os.makedirs(os.path.join(d, "all_images"), exist_ok=True)
    os.makedirs(os.path.join(d, "all_labels"), exist_ok=True)
    lines = ["path: x", "train: all_images", "val: all_images", f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in names.items()]
    with open(os.path.join(d, "data.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for bn, classes in images_labels.items():
        with open(os.path.join(d, "all_images", bn), "w", encoding="utf-8") as f:
            f.write("x")
        if classes:
            stem = os.path.splitext(bn)[0]
            with open(os.path.join(d, "all_labels", stem + ".txt"), "w") as f:
                for cid in classes:
                    f.write(f"{cid} 0.1 0.1 0.2 0.2 0.3 0.3\n")


def _make_outline_source(root, name, images_labels, names=None):
    """A composite/standalone-shaped outline export under root/name.
    Returns the export directory."""
    d = os.path.join(root, name)
    _write_outline_export(d, images_labels, names or {0: "MC"})
    return d


def _make_bbox_source(root, name, n_images, *, pins=None, pins_text=None,
                      background_indices=()):
    """A staged detection dataset under root/name: images/, labels/,
    classes.txt, data.yaml, ingest_manifest.json (required by
    _validate_bbox_source), and split_pins.json when pins or pins_text is
    given.

    pins: {basename: split} written as split_pins.json's pins object.
    pins_text: raw text written verbatim to split_pins.json (a hostile or
        malformed file); mutually exclusive with pins.
    background_indices: image indices that get an empty label file (a
        background frame with no box) instead of one box line.
    Returns the dataset directory.
    """
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    for i in range(n_images):
        bn = f"{name}_{i:03d}.jpg"
        with open(os.path.join(d, "images", bn), "w") as f:
            f.write("x")
        label_path = os.path.join(d, "labels", os.path.splitext(bn)[0] + ".txt")
        with open(label_path, "w") as f:
            if i not in background_indices:
                f.write("0 0.5 0.5 0.2 0.2\n")
    with open(os.path.join(d, "classes.txt"), "w") as f:
        f.write("Alpha\n")
    with open(os.path.join(d, "data.yaml"), "w") as f:
        f.write(f"path: {d}\ntrain: images\nval: images\nnc: 1\nnames:\n  0: Alpha\n")
    with open(os.path.join(d, "ingest_manifest.json"), "w") as f:
        json.dump({"n_images": n_images, "n_boxes": n_images, "classes": ["Alpha"]}, f)
    if pins is not None:
        with open(os.path.join(d, "split_pins.json"), "w") as f:
            json.dump({"schema_version": 1, "source": "the original experiment", "pins": pins}, f)
    elif pins_text is not None:
        with open(os.path.join(d, "split_pins.json"), "w") as f:
            f.write(pins_text)
    return d


def _get_summary(client, **query):
    r = client.get("/api/step/6/source_summary", query_string=query)
    return r.status_code, (r.get_json() or {})


# ---------------------------------------------------------------------------
# OUTLINE
# ---------------------------------------------------------------------------

def test_outline_transect_rule():
    """4 transects (1-4) train, transect 5 validation, transect 6 test: the
    real names of the query params, no image whose transect cannot be
    parsed."""
    with _Roots() as roots:
        images = {
            "clip_T105.jpg": [0], "clip_T205.jpg": [0], "clip_T310.jpg": [0], "clip_T410.jpg": [0],
            "clip_T512.jpg": [0], "clip_T534.jpg": [0],
            "clip_T601.jpg": [0], "clip_T650.jpg": [0],
        }
        src = _make_outline_source(roots.composite, "set_a", images)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check(data["ok"] is True, f"ok must be true: {data}")
            check(data["kind"] == "composite", f"kind must be composite: {data}")
            check(os.path.basename(data["path"]) == "set_a", f"path: {data}")
            check(data["n_images"] == 8, f"n_images: {data}")
            check(data["n_labels"] == 8, f"n_labels: {data}")
            check(data["n_classes"] == 1, f"n_classes: {data}")
            check(data["n_background"] == 0, f"n_background is always 0 for an outline source: {data}")
            check(data["label_noun"] == "outlines", f"label_noun: {data}")
            check(data["pictures_sentence"] == "8 images, 8 outlines, 1 class",
                  f"pictures_sentence: {data}")
            split = data["split"]
            check(split["rule"] == "transect", f"rule: {split}")
            check((split["n_train"], split["n_valid"], split["n_test"]) == (4, 2, 2), f"{split}")
            check(split["approximate"] is True, f"transect counts are approximate: {split}")
            check(split["sentence"] == (
                "Split by transect: transects 1 to 4 train, transect 5 validation, "
                "transect 6 test, about 4 / 2 / 2 images."), f"sentence: {split['sentence']!r}")
            check(data["warnings"] == [], f"no warning on a clean transect split: {data}")


def test_outline_random_fallback_when_no_transect_5_or_6():
    """Basenames that carry no _T<3 digits> at all: parse_transect returns
    None for every one, so the pinned split degrades to random, exactly as
    pinned_split.py's own main() degrades, and the summary says so."""
    with _Roots() as roots:
        images = {f"photo_{i:03d}.jpg": [0] for i in range(10)}
        src = _make_outline_source(roots.composite, "set_b", images)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check(data["n_images"] == 10, f"n_images: {data}")
            split = data["split"]
            check(split["rule"] == "random", f"rule must degrade to random: {split}")
            check((split["n_train"], split["n_valid"], split["n_test"]) == (7, 2, 1), f"{split}")
            check(split["approximate"] is True, f"{split}")
            check(data["warnings"] and "transect 5" in data["warnings"][0]
                  and "transect 6" in data["warnings"][0],
                  f"the fallback must be named: {data['warnings']}")


def test_outline_pin_split_zero_is_random_with_no_warning():
    """Holdout split unticked (pin_split=0): random by the operator's own
    choice, not a degrade, so no fallback warning, even over the same
    transect-shaped basenames test_outline_transect_rule uses."""
    with _Roots() as roots:
        images = {
            "clip_T105.jpg": [0], "clip_T205.jpg": [0], "clip_T310.jpg": [0], "clip_T410.jpg": [0],
            "clip_T512.jpg": [0], "clip_T534.jpg": [0],
            "clip_T601.jpg": [0], "clip_T650.jpg": [0],
        }
        src = _make_outline_source(roots.composite, "set_c", images)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src, pin_split="0",
                                        valid_ratio="0.25", test_ratio="0.125")
            check(status == 200, f"status {status}: {data}")
            split = data["split"]
            check(split["rule"] == "random", f"pin_split=0 must read as random: {split}")
            check((split["n_train"], split["n_valid"], split["n_test"]) == (5, 2, 1), f"{split}")
            check(data["warnings"] == [], f"an operator's own random choice carries no warning: {data}")
            check(split["sentence"] == (
                "Random split: 63% train, 25% validation, 12% test, about 5 / 2 / 1 images."),
                  f"sentence: {split['sentence']!r}")


def test_outline_include_classes_filtering():
    """An image counts only when it has at least one line of an included
    class; classes.txt/data.yaml's own name count is restricted the same
    way."""
    with _Roots() as roots:
        images = {
            "img_a.jpg": [0],       # class 0 only
            "img_b.jpg": [1],       # class 1 only
            "img_c.jpg": [0, 1],    # both
        }
        src = _make_outline_source(roots.composite, "set_d", images, names={0: "MC", 1: "PA"})
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check(data["n_images"] == 3, f"every class counted: {data}")
            check(data["n_labels"] == 4, f"1+1+2 lines: {data}")
            check(data["n_classes"] == 2, f"{data}")

            status, data = _get_summary(client, source_dir=src, include_classes="0")
            check(status == 200, f"status {status}: {data}")
            check(data["n_images"] == 2, f"only img_a and img_c carry class 0: {data}")
            check(data["n_labels"] == 2, f"one class-0 line each: {data}")
            check(data["n_classes"] == 1, f"only the included class counts: {data}")


def test_outline_image_with_no_label_is_excluded():
    with _Roots() as roots:
        images = {"has_a.jpg": [0], "has_b.jpg": [0], "no_label.jpg": []}
        src = _make_outline_source(roots.composite, "set_e", images)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check(data["n_images"] == 2,
                  f"an image with no label file must not count: {data}")
            check(data["n_labels"] == 2, f"{data}")


def test_outline_hostile_basenames():
    """Spaces, a quote, an apostrophe and a control character in the file
    name must not crash the route."""
    with _Roots() as roots:
        images = {
            "img with spaces.jpg": [0],
            'img"quote.jpg': [0],
            "img'apostrophe.jpg": [0],
            "img\x01ctrl.jpg": [0],
        }
        src = _make_outline_source(roots.composite, "set_f", images)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"hostile basenames must not crash the route: {status} {data}")
            check(data["n_images"] == 4, f"every hostile basename still counts: {data}")
            check(data["n_labels"] == 4, f"{data}")


def test_outline_1500_image_timing_sanity():
    """A 1,500-image source answers in a reasonable time: the route reads
    every basename and its label file once, never a dataset build."""
    with _Roots() as roots:
        images = {f"img_{i:04d}.jpg": [0] for i in range(1500)}
        src = _make_outline_source(roots.composite, "set_big", images)
        with _project_client() as client:
            start = time.monotonic()
            status, data = _get_summary(client, source_dir=src)
            elapsed = time.monotonic() - start
            check(status == 200, f"status {status}: {str(data)[:400]}")
            check(data["n_images"] == 1500, f"n_images: {data.get('n_images')}")
            check(elapsed < 8.0, f"1,500 images took {elapsed:.2f}s, expected well under 8s")


def test_outline_default_source_is_the_project_export():
    """A blank source_dir on an ordinary project resolves to its own
    step4test export, kind 'export'."""
    tmp = tempfile.mkdtemp(prefix="step6_summary_export_")
    try:
        state = _make_project_state(tmp)
        export_dir = os.path.join(tmp, pm.STEP_DIRS["step4test"])
        _write_outline_export(export_dir, {"a.jpg": [0], "b.jpg": [0]}, {0: "MC"})
        appmod.current_project = state
        client = appmod.create_app().test_client()
        status, data = _get_summary(client)
        check(status == 200, f"status {status}: {data}")
        check(data["kind"] == "export", f"kind: {data}")
        check(data["name"] == "t", f"name falls back to the project name: {data}")
        check(data["n_images"] == 2, f"{data}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_outline_singular_pictures_sentence():
    """One image, one label line, one named class: every count in the
    Pictures sentence reads singular ('1 image, 1 outline, 1 class'), not
    '1 images, 1 outlines, 1 classes'."""
    with _Roots() as roots:
        src = _make_outline_source(roots.composite, "set_one", {"only.jpg": [0]}, names={0: "MC"})
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check((data["n_images"], data["n_labels"], data["n_classes"]) == (1, 1, 1), f"{data}")
            check(data["pictures_sentence"] == "1 image, 1 outline, 1 class",
                  f"pictures_sentence: {data}")


# ---------------------------------------------------------------------------
# BBOX
# ---------------------------------------------------------------------------

def test_bbox_pinned_rule():
    """split_pins.json pins every image: the pinned rule counts it exactly,
    approximate False, no warning."""
    with _Roots() as roots:
        pins = {"set_g_000.jpg": "train", "set_g_001.jpg": "train", "set_g_002.jpg": "train",
                "set_g_003.jpg": "valid", "set_g_004.jpg": "test"}
        src = _make_bbox_source(roots.bbox, "set_g", 5, pins=pins)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check(data["kind"] == "bbox", f"kind: {data}")
            check(data["label_noun"] == "boxes", f"{data}")
            check(data["n_images"] == 5, f"{data}")
            check(data["n_labels"] == 5, f"one box per image: {data}")
            check(data["n_background"] == 0, f"{data}")
            check(data["pictures_sentence"] == "5 images, 5 boxes, 1 class",
                  f"pictures_sentence: {data}")
            split = data["split"]
            check(split["rule"] == "pinned", f"rule: {split}")
            check((split["n_train"], split["n_valid"], split["n_test"]) == (3, 1, 1), f"{split}")
            check(split["approximate"] is False, f"the pinned rule is exact: {split}")
            check(split["sentence"] == "Split pinned by the dataset: 3 train, 1 validation, 1 test.",
                  f"sentence: {split['sentence']!r}")
            check(data["warnings"] == [], f"{data}")


def test_bbox_pinned_rule_keeps_the_trailing_noun_when_no_count_is_one():
    """With every count above 1 the pinned sentence keeps its trailing
    'images' noun, matching the design's own example ('5,300 train, 1,514
    validation, 758 test images.'); the noun drops only when a count is 1
    (test_bbox_pinned_rule)."""
    with _Roots() as roots:
        pins = {f"set_m_{i:03d}.jpg": split for i, split in enumerate(
            ["train"] * 4 + ["valid"] * 4 + ["test"] * 2)}
        src = _make_bbox_source(roots.bbox, "set_m", 10, pins=pins)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            split = data["split"]
            check((split["n_train"], split["n_valid"], split["n_test"]) == (4, 4, 2), f"{split}")
            check(split["sentence"] == "Split pinned by the dataset: 4 train, 4 validation, 2 test images.",
                  f"sentence: {split['sentence']!r}")


def test_bbox_singular_pictures_sentence_one_box_no_background():
    """One image, one box, one class, zero background: 'with no box' is left
    out entirely (0 is below the clause's threshold of 1)."""
    with _Roots() as roots:
        src = _make_bbox_source(roots.bbox, "set_n", 1)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check((data["n_images"], data["n_labels"], data["n_classes"], data["n_background"])
                  == (1, 1, 1, 0), f"{data}")
            check(data["pictures_sentence"] == "1 image, 1 box, 1 class",
                  f"pictures_sentence: {data}")


def test_bbox_one_background_image_reads_singular():
    """Exactly one background image: the clause reads '1 with no box', not
    '1 with no boxs' or the plural count word."""
    with _Roots() as roots:
        src = _make_bbox_source(roots.bbox, "set_o", 2, background_indices=(1,))
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check(data["n_background"] == 1, f"{data}")
            check(data["pictures_sentence"] == "2 images, 1 box, 1 class, 1 with no box",
                  f"pictures_sentence: {data}")


def test_bbox_unreadable_pins_falls_back_to_random():
    with _Roots() as roots:
        src = _make_bbox_source(roots.bbox, "set_h", 5, pins_text="{not json at all")
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            split = data["split"]
            check(split["rule"] == "random", f"an unreadable pins file must fall back: {split}")
            check(split["approximate"] is True, f"{split}")
            check(data["warnings"] and "could not be read" in data["warnings"][0],
                  f"the fallback must be explained: {data['warnings']}")


def test_bbox_no_pins_file_is_random():
    with _Roots() as roots:
        src = _make_bbox_source(roots.bbox, "set_i", 8)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src,
                                        valid_ratio="0.25", test_ratio="0.125")
            check(status == 200, f"status {status}: {data}")
            split = data["split"]
            check(split["rule"] == "random", f"no pins file must read as random: {split}")
            check((split["n_train"], split["n_valid"], split["n_test"]) == (5, 2, 1), f"{split}")
            check(data["warnings"] == [], f"no pins file is the ordinary case, not a failure: {data}")


def test_bbox_background_images_counted():
    """Every image counts, whether or not it has a box; n_background is how
    many carry none (an empty label file, or no label file at all)."""
    with _Roots() as roots:
        src = _make_bbox_source(roots.bbox, "set_j", 4, background_indices=(2,))
        os.remove(os.path.join(src, "labels", "set_j_003.txt"))
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"status {status}: {data}")
            check(data["n_images"] == 4, f"every image counts on a box source: {data}")
            check(data["n_background"] == 2,
                  f"the empty label file and the missing one are both background: {data}")
            check(data["n_labels"] == 2, f"two images still carry one box each: {data}")


def test_bbox_project_blank_source_dir():
    """A bbox_training project's blank source_dir resolves to its own
    recorded box dataset, bypassing the staged-root validation the same way
    _recorded_training_source does for a launch."""
    with _Roots() as roots:
        src = _make_bbox_source(roots.bbox, "set_k", 3)
        with _bbox_project_client(src) as client:
            status, data = _get_summary(client)
            check(status == 200, f"status {status}: {data}")
            check(data["kind"] == "bbox", f"kind: {data}")
            check(os.path.basename(data["path"]) == "set_k", f"path: {data}")
            check(data["n_images"] == 3, f"{data}")


def test_bbox_source_refuses_a_segmentation_preview():
    with _Roots() as roots:
        src = _make_bbox_source(roots.bbox, "set_l", 3)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src, task="segment")
            check(status == 400, f"a box source must refuse a segmentation preview: {status} {data}")
            check("can only preview a detector" in data.get("error", ""), f"{data}")


# ---------------------------------------------------------------------------
# ERRORS
# ---------------------------------------------------------------------------

def test_missing_folder_is_400():
    with _Roots() as roots:
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=os.path.join(roots.composite, "nope"))
            check(status == 400, f"a missing folder must answer 400: {status} {data}")
            check("error" in data, f"{data}")


def test_empty_all_images_does_not_crash():
    with _Roots() as roots:
        src = _make_outline_source(roots.composite, "set_empty", {})
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src)
            check(status == 200, f"an empty source must not crash the route: {status} {data}")
            check(data["n_images"] == 0 and data["n_labels"] == 0, f"{data}")
            split = data["split"]
            check((split["n_train"], split["n_valid"], split["n_test"]) == (0, 0, 0), f"{split}")


def test_path_outside_the_allowed_roots_is_400():
    with _Roots() as roots:
        outside = os.path.join(roots.tmp, "elsewhere")
        os.makedirs(outside, exist_ok=True)
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=outside)
            check(status == 400, f"a path outside every root must answer 400: {status} {data}")
            check(outside in data.get("error", ""), f"the error must name the path: {data}")


def test_no_project_loaded_is_400():
    with _no_project_client() as client:
        status, data = _get_summary(client)
        check(status == 400, f"no project must answer 400: {status} {data}")
        check(data.get("error") == "No project loaded", f"{data}")


def test_unknown_task_is_400():
    with _Roots() as roots:
        src = _make_outline_source(roots.composite, "set_task", {"a.jpg": [0]})
        with _project_client() as client:
            status, data = _get_summary(client, source_dir=src, task="pose")
            check(status == 400, f"an unknown task must answer 400: {status} {data}")
            check("Unknown training task" in data.get("error", ""), f"{data}")


if __name__ == "__main__":
    print("Step 6 training-data summary (/api/step/6/source_summary)\n")
    print("OUTLINE:")
    run(test_outline_transect_rule)
    run(test_outline_random_fallback_when_no_transect_5_or_6)
    run(test_outline_pin_split_zero_is_random_with_no_warning)
    run(test_outline_include_classes_filtering)
    run(test_outline_image_with_no_label_is_excluded)
    run(test_outline_hostile_basenames)
    run(test_outline_1500_image_timing_sanity)
    run(test_outline_default_source_is_the_project_export)
    run(test_outline_singular_pictures_sentence)
    print("BBOX:")
    run(test_bbox_pinned_rule)
    run(test_bbox_pinned_rule_keeps_the_trailing_noun_when_no_count_is_one)
    run(test_bbox_singular_pictures_sentence_one_box_no_background)
    run(test_bbox_one_background_image_reads_singular)
    run(test_bbox_unreadable_pins_falls_back_to_random)
    run(test_bbox_no_pins_file_is_random)
    run(test_bbox_background_images_counted)
    run(test_bbox_project_blank_source_dir)
    run(test_bbox_source_refuses_a_segmentation_preview)
    print("ERRORS:")
    run(test_missing_folder_is_400)
    run(test_empty_all_images_does_not_crash)
    run(test_path_outside_the_allowed_roots_is_400)
    run(test_no_project_loaded_is_400)
    run(test_unknown_task_is_400)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
