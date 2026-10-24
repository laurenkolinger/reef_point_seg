#!/usr/bin/env python
"""Task N2: the matrix minter delegates to the shared one, and every scan of
annotation work skips BOTH kinds of training project.

House harness, no pytest. Run with:
  env/bin/python scripts/_matrix/tests/test_training_project_delegation.py

Covers:
  DELEGATION  composite.create_training_project calls
              _reefreview.training_projects.mint_training_project, and the
              project.json it writes is identical (modulo the three timestamp
              fields and the absolute fixture paths) to the contract captured
              from the pre-delegation implementation. The four caller-side
              responsibilities stay with the caller: the name identity and
              traversal check, the empty-set refusal, the source-project
              display-name enrichment, and the linked_projects back link.
  EXCEPTION   RunDirExhausted has ONE definition. composite.RunDirExhausted is
              training_projects.RunDirExhausted, so the blueprint's 409 mapping
              catches a mint raised from either module.
  WIDENING    builder.scan_projects (and therefore stats.compute_stats, which
              reads it) skips a bbox_training run dir the same way it skips a
              composite_training one, and both modules agree on the shared
              TRAINING_PROJECT_TYPES list.
  LISTING     composite.list_composites hides the .rebuild-tmp / .rebuild-backup
              siblings a rebuild leaves behind mid-swap.
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
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix import builder  # noqa: E402
from _matrix import composite  # noqa: E402
from _matrix.stats import compute_stats  # noqa: E402
from _reefreview import training_projects as TP  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


# ---------------------------------------------------------------------------
# The project.json contract, captured from the implementation that existed
# BEFORE composite.create_training_project delegated. Timestamps and the run
# dir's own absolute paths are the only fields allowed to differ.
# ---------------------------------------------------------------------------
_TIMESTAMP_KEYS = ("created_at", "updated_at", "ts")

_SKIPPED_STEPS = {
    "1": ("Make All Points", "step1_makeAllPoints"),
    "2": ("Recode Species", "step2_recodeSpecies"),
    "3": ("Choose Images", "step3_chooseImages"),
    "4": ("Place Points", "step4_routeChosenImages"),
}


def _normalize(doc, run_dir, repo_dir):
    """The document with every timestamp blanked and every absolute path made
    relative to the run dir, so two mints made a second apart compare equal."""
    def _walk(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in _TIMESTAMP_KEYS and isinstance(v, str):
                    out[k] = "<stamp>"
                else:
                    out[k] = _walk(v)
            return out
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            if node == run_dir:
                return "<run_dir>"
            if node.startswith(run_dir + os.sep):
                return "<run_dir>/" + os.path.relpath(node, run_dir)
            if node == repo_dir:
                return "<repo_dir>"
        return node
    return _walk(doc)


def _expected(project_id, set_name, sources):
    steps = {}
    for key, (name, dirname) in _SKIPPED_STEPS.items():
        steps[key] = {
            "status": "skipped",
            "name": name,
            "dir": "<run_dir>/" + dirname,
            "config": {},
            "outputs": {},
            "note": f"from training set {set_name}",
        }
    return {
        "id": project_id,
        "name": f"TRAIN_{set_name}",
        "created_at": "<stamp>",
        "updated_at": "<stamp>",
        "project_dir": "<run_dir>",
        "repo_dir": "<repo_dir>",
        "current_step": 6,
        "steps": steps,
        "access_log": [{
            "ts": "<stamp>",
            "initials": "LO",
            "action": "create",
            "purpose": f"training project for the training set {set_name}",
        }],
        "initials": "LO",
        "project_type": "composite_training",
        "composite": {
            "name": set_name,
            "path": os.path.join("inprocess", "_composite", set_name),
            "source_projects": sources,
            "created_at": "<stamp>",
        },
    }


def _make_composite(root, name, sources=None, images=("i.jpg",)):
    out = os.path.join(root, "_composite", name)
    os.makedirs(os.path.join(out, "all_images"), exist_ok=True)
    os.makedirs(os.path.join(out, "all_labels"), exist_ok=True)
    for fn in images:
        with open(os.path.join(out, "all_images", fn), "w") as fh:
            fh.write("x")
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write("path: x\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: MC\n")
    with open(os.path.join(out, "composite.json"), "w") as fh:
        json.dump({"name": name, "created_at": "2026-09-02 09:00:00",
                   "source_projects": list(sources or [{"project_id": "projA", "n_masks": 3}]),
                   "linked_projects": []}, fh, indent=2)
    return out


def _write_project(root, dirname, project_type, extra=None, planted_label=""):
    """A run dir carrying a project.json. ``planted_label`` puts one accepted,
    labeled mask inside it, in the segmentations.json shape STATS derives its
    records from, so the exclusion assertions are answered by real data rather
    than by an empty tree: remove the skip and that label shows up in STATS."""
    d = os.path.join(root, dirname)
    os.makedirs(d, exist_ok=True)
    doc = {"id": dirname[len("run_"):], "name": dirname, "project_type": project_type}
    doc.update(extra or {})
    with open(os.path.join(d, "project.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    if planted_label:
        image = "TCRMP20140930_clip_SCP_T%03d.jpeg" % (abs(hash(dirname)) % 900 + 1)
        seg_dir = os.path.join(d, "step5_segmentImages", "segmentations")
        os.makedirs(seg_dir, exist_ok=True)
        with open(os.path.join(seg_dir, "segmentations.json"), "w") as fh:
            json.dump({image: {
                "image_path_abs": os.path.join(d, image),
                "image_width": 100, "image_height": 100,
                "reviewed": True,
                "masks": [{"species": planted_label, "status": "accepted",
                           "source_type": "manual_click", "reviewer": "ZZ",
                           "polygon_norm": [[0.1, 0.1, 0.4, 0.1, 0.4, 0.4]]}],
            }}, fh, indent=1)
    return d


# ---------------------------------------------------------------------------
# 1. delegation
# ---------------------------------------------------------------------------

def test_delegates_to_the_shared_minter(root):
    print("\nDELEGATION:")
    calls = []
    real = TP.mint_training_project

    def _spy(*a, **kw):
        calls.append((a, kw))
        return real(*a, **kw)

    _make_composite(root, "snapset")
    composite.mint_training_project = _spy
    try:
        res = composite.create_training_project(root, "snapset", "lo")
    finally:
        composite.mint_training_project = real

    check(len(calls) == 1,
          f"create_training_project calls the shared minter exactly once (got {len(calls)})")
    if calls:
        args, kwargs = calls[0]
        passed = list(args) + [kwargs.get(k) for k in ("inprocess_root", "source_kind")]
        check("composite" in passed or kwargs.get("source_kind") == "composite",
              f"the mint is asked for a composite source (got {args!r} {kwargs!r})")

    run_dir = res["run_dir"]
    with open(os.path.join(run_dir, "project.json")) as fh:
        doc = json.load(fh)
    got = _normalize(doc, run_dir, TP._REPO_DIR)
    want = _expected(res["project_id"], "snapset", [{"project_id": "projA", "n_masks": 3}])
    check(got == want,
          "the delegated mint writes the pre-delegation project.json contract")
    if got != want:
        for key in sorted(set(list(got) + list(want))):
            if got.get(key) != want.get(key):
                print(f"      key {key!r}:\n        got  {got.get(key)!r}\n        want {want.get(key)!r}")
    check(list(got.keys()) == list(want.keys()),
          f"the key ORDER is unchanged too (got {list(got.keys())})")


def test_caller_side_responsibilities_stay(root):
    print("\nCALLER-SIDE GUARDS:")
    # 1. name identity / traversal: a name holding a separator is unknown,
    #    and the refusal happens before any run dir is minted.
    before = set(os.listdir(root))
    err = ""
    try:
        composite.create_training_project(root, os.path.join("..", "snapset"), "LO")
    except FileNotFoundError as e:
        err = str(e)
    check(err, f"a traversing set name is refused (got {err!r})")
    check(set(os.listdir(root)) == before, "the refused traversal minted nothing")

    # 2. empty-set refusal
    empty = _make_composite(root, "empty_set", images=())
    check(os.path.isdir(empty), "the empty fixture set exists")
    err = ""
    try:
        composite.create_training_project(root, "empty_set", "LO")
    except ValueError as e:
        err = str(e)
    check("no images" in err, f"an image-free set is refused caller-side (got {err!r})")
    check(set(os.listdir(root)) == before, "the refused empty set minted nothing")

    # 3. display-name enrichment: a source project still on disk contributes
    #    its display name, which the shared minter never looks up.
    live = os.path.join(root, "run_26-01-01_ZZ_live")
    os.makedirs(os.path.join(live, "step4test_combinedAnnotate"), exist_ok=True)
    with open(os.path.join(live, "project.json"), "w") as fh:
        json.dump({"id": "projLive", "name": "Live Project"}, fh)
    _make_composite(root, "named_set", sources=[{"project_id": "projLive", "n_masks": 7}])
    res = composite.create_training_project(root, "named_set", "LO")
    with open(os.path.join(res["run_dir"], "project.json")) as fh:
        doc = json.load(fh)
    srcs = (doc.get("composite") or {}).get("source_projects") or []
    check(srcs and srcs[0].get("name") == "Live Project",
          f"a live source project keeps its display name (got {srcs})")

    # 4. linked_projects back link, written by the composite module under its
    #    own lock.
    meta_path = os.path.join(composite.composite_root(root), "named_set", "composite.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    check(meta.get("linked_projects") == [os.path.basename(res["run_dir"])],
          f"the composite records the training project (got {meta.get('linked_projects')})")


def test_one_run_dir_exhausted_class():
    print("\nEXCEPTION:")
    check(composite.RunDirExhausted is TP.RunDirExhausted,
          "composite.RunDirExhausted IS training_projects.RunDirExhausted (one class)")
    src = open(os.path.join(_SCRIPTS, "_matrix", "composite.py")).read()
    check("class RunDirExhausted" not in src,
          "composite.py no longer defines a second RunDirExhausted")
    check("def _mint_run_dir" not in src,
          "composite.py no longer carries its own copy of the run-dir minter")


# ---------------------------------------------------------------------------
# 2. builder / stats widening
# ---------------------------------------------------------------------------

def test_scan_skips_both_training_kinds(root):
    print("\nWIDENING:")
    check(set(builder.TRAINING_PROJECT_TYPES) == {"composite_training", "bbox_training"},
          f"the builder skips every training project type (got {builder.TRAINING_PROJECT_TYPES})")
    check(tuple(builder.TRAINING_PROJECT_TYPES) == tuple(TP.TRAINING_PROJECT_TYPES),
          "the builder and the shared minter agree on the list, letter for letter")

    # Each planted project holds one accepted, labeled mask, so every
    # assertion below has something to find if the skip ever stops working.
    _write_project(root, "run_26-09-02_LO_train_boxes", "bbox_training",
                   {"training_source": {"kind": "bbox", "name": "coral_intel",
                                        "path": "/nowhere", "task_default": "detect"}},
                   planted_label="BOXPLANT")
    _write_project(root, "run_26-09-02_LO_train_comp", "composite_training",
                   planted_label="COMPPLANT")
    _write_project(root, "run_26-09-02_LO_ordinary", None, planted_label="REALPLANT")

    ids = {p["project_id"] for p in builder.scan_projects(root)}
    check("26-09-02_LO_train_boxes" not in ids,
          f"scan_projects skips a bbox_training run dir (got {sorted(ids)})")
    check("26-09-02_LO_train_comp" not in ids,
          f"scan_projects skips a composite_training run dir (got {sorted(ids)})")
    check("26-09-02_LO_ordinary" in ids,
          f"an ordinary project still scans (got {sorted(ids)})")

    check(builder.is_training_project({"project_type": "bbox_training"}),
          "is_training_project answers True for a bbox project")
    check(not builder.is_training_project({"project_type": "annotation"}),
          "is_training_project answers False for an ordinary project")

    stats = compute_stats(root)
    counted = set(stats.get("legacy_projects") or [])
    codes = {s["code"] for s in (stats.get("species") or [])}
    # The control: the ordinary project's planted mask DOES reach STATS, which
    # is what makes the two absences below evidence rather than an empty tree.
    check("REALPLANT" in codes,
          f"the ordinary project's planted mask reaches STATS (got {sorted(codes)})")
    check("26-09-02_LO_ordinary" in counted,
          f"and its project is counted (got {sorted(counted)})")
    check("BOXPLANT" not in codes,
          f"STATS counts no mask from a bbox training project (got {sorted(codes)})")
    check("COMPPLANT" not in codes,
          f"STATS counts no mask from a composite training project (got {sorted(codes)})")
    check("26-09-02_LO_train_boxes" not in counted,
          f"STATS derives nothing from a bbox training project (got {sorted(counted)})")
    check("26-09-02_LO_train_comp" not in counted,
          f"STATS derives nothing from a composite training project (got {sorted(counted)})")

    matrix = builder.build_matrix(root)
    labels = set(matrix.get("labels") or [])
    check("REALPLANT" in labels, f"the control label reaches the matrix (got {sorted(labels)})")
    check("BOXPLANT" not in labels and "COMPPLANT" not in labels,
          f"neither training project contributes a label to the matrix (got {sorted(labels)})")


# ---------------------------------------------------------------------------
# 3. list_composites hides rebuild scratch dirs
# ---------------------------------------------------------------------------

def test_list_composites_hides_rebuild_dirs(root):
    print("\nLISTING:")
    _make_composite(root, "live_set")
    _make_composite(root, "live_set.rebuild-tmp")
    _make_composite(root, "live_set.rebuild-backup")
    names = [c["name"] for c in composite.list_composites(root)]
    check("live_set" in names, f"the real set is listed (got {names})")
    check("live_set.rebuild-tmp" not in names,
          f"the mid-rebuild temp dir is not offered as a training set (got {names})")
    check("live_set.rebuild-backup" not in names,
          f"the rebuild backup dir is not offered as a training set (got {names})")

    # Hidden from the listing is not enough: a hand-typed or stale name still
    # reaches the minter, and either scratch dir would be gone minutes later.
    for scratch in ("live_set.rebuild-tmp", "live_set.rebuild-backup"):
        before = set(os.listdir(root))
        err = ""
        try:
            composite.create_training_project(root, scratch, "LO")
        except FileNotFoundError as e:
            err = str(e)
        check(scratch in err,
              f"the minter itself refuses {scratch} by name (got {err!r})")
        check(set(os.listdir(root)) == before, f"the refused {scratch} minted nothing")
    # The real set of the same stem still mints, so the refusal is a suffix
    # rule and not a prefix match that would swallow the live set too.
    res = composite.create_training_project(root, "live_set", "LO")
    check(os.path.isdir(res["run_dir"]), "the real set still mints a project")


def main():
    root = tempfile.mkdtemp(prefix="matrix_delegation_test_")
    try:
        test_delegates_to_the_shared_minter(root)
        test_caller_side_responsibilities_stay(root)
        test_one_run_dir_exhausted_class()
        test_scan_skips_both_training_kinds(root)
        test_list_composites_hides_rebuild_dirs(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check(not os.path.exists(root), "temp fixture removed")

    print()
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
