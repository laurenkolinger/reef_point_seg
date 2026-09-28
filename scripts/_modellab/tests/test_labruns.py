#!/usr/bin/env python
"""Pure helpers of the shared run library (scripts/_modellab/labruns.py).

No-pytest house harness. Nothing here spawns a process or trains anything:
every path is a temp fixture removed in the finally block.

Covers:
  source discovery   composite exports under inprocess/_composite plus staged
                     detection datasets under supporting_data/bbox_datasets
  name validation    lab names accept [A-Za-z0-9_-]{1,60} and reject traversal
  lab paths          every lab dir stays under inprocess/_models
  run scanning       lab runs and project step6 runs merged with origin labels,
                     mAP from the adjacent rounds.csv, champion flag
  guards             infer source refused inside a project step dir, run dir
                     without weights refused, run dir outside the given root
                     refused (.. and symlink escapes)
  library use        labruns.py loads and runs with Flask and ultralytics
                     blocked, so the VEGA main window can import it directly
  split defaults     the detection wrappers default to the builder's
                     DEFAULT_VALID_FRAC and DEFAULT_TEST_FRAC, never a
                     restated literal
  champion mirror    mirror_champion_to_lab writes only inside a lab under
                     _models: a sibling folder whose name extends _models, a
                     lab symlinked out of it, the _models root itself and a
                     project run are all refused
  task               a run's task comes from the task: line ultralytics wrote
                     into args.yaml; the model-name guess is only a fallback
  champion           all three champion.json shapes on disk read into one
                     record with a metric family and a single family score
  comparability      two runs rank against each other only with the same
                     metric family and the same split fingerprint
  split record       the split a run trained on is recorded into the run
                     folder when it is trained (or on the first read of a run
                     that predates records) and read from there afterwards, so
                     a rebuilt dataset never moves an old run's claim and two
                     runs of one project can report two different splits
  adversarial        empty and malformed args.yaml, a missing run folder,
                     corrupt champion JSON, a damaged or missing split record,
                     a run folder nobody may write to, eight writers stamping
                     one run at once, a ledger with a missing column or a
                     hostile cell, an unreadable folder in the scan

Run with:
  env/bin/python scripts/_modellab/tests/test_labruns.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _modellab import labruns  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


ROUNDS_HEADER = ("round,at,run_name,run_dir,base_model,split_pinned,"
                 "n_train,n_valid,n_test,map50_M,map50_95_M,recall_M,precision_M,"
                 "map50_B,map50_95_B,gate_map,gate_class,gate_recall,per_class_json\n")


def build_inprocess(root):
    """A miniature inprocess tree: one composite, one lab with a trained run,
    one project with a step6 run, one champion."""
    comp = os.path.join(root, "_composite", "comp_A")
    write(os.path.join(comp, "data.yaml"), "path: .\ntrain: train\nnames:\n  0: PA\n  1: MC\n")
    write(os.path.join(comp, "all_images", "SITE-20260101-T101.jpeg"), "x")
    write(os.path.join(comp, "all_images", "SITE-20260101-T102.jpeg"), "x")
    write(os.path.join(comp, "all_labels", "SITE-20260101-T101.txt"), "0 0.1 0.1 0.2 0.2 0.1 0.2\n")
    write(os.path.join(comp, "composite.json"), json.dumps({"created_at": "2026-09-01 10:00:00"}))
    # Half-built composite (no data.yaml): never offered as a training source.
    os.makedirs(os.path.join(root, "_composite", "half_built", "all_images"), exist_ok=True)

    lab = os.path.join(root, "_models", "labA")
    write(os.path.join(lab, "lab.json"), json.dumps({"task": "seg", "source": comp}))
    write(os.path.join(lab, "runs", "r1", "weights", "best.pt"), "weights")
    write(os.path.join(lab, "runs", "r1", "args.yaml"), "model: yolo11m-seg.pt\nimgsz: 512\n")
    write(os.path.join(lab, "rounds.csv"),
          ROUNDS_HEADER + f"1,2026-09-01,r1,{os.path.join(lab, 'runs', 'r1')},yolo11m-seg.pt,"
          "yes,10,2,2,0.62,0.41,0.55,0.7,,,pass,pass,pass,\n")
    write(os.path.join(lab, "dataset", "data.yaml"), "names:\n  0: PA\n")

    proj = os.path.join(root, "run_26-08-26_LO", "step6_trainModel")
    os.makedirs(os.path.join(proj, "runs", "p1"), exist_ok=True)
    write(os.path.join(proj, "dataset", "data.yaml"), "names:\n  0: PA\n")
    write(os.path.join(root, "run_26-08-26_LO", "project.json"), json.dumps({"id": "p", "name": "P"}))

    write(labruns.champion_path(root), json.dumps(
        {"run_dir": os.path.join(lab, "runs", "r1"), "at": "2026-09-01 11:00:00"}))
    return {"comp": comp, "lab": lab, "proj": proj}


def build_bbox(bbox_root, n_images=20):
    """The pinned staged-detection layout the Bounding Boxes task writes:
    images/, labels/ (optionally labels_localized/), data.yaml, classes.txt,
    ingest_manifest.json. Deliberately NOT the all_images/all_labels shape
    pinned_split requires, which is why the lab builds this split itself."""
    ds = os.path.join(bbox_root, "coral_intel")
    write(os.path.join(ds, "data.yaml"), "names:\n  0: AAGA\n  1: ACER\n")
    write(os.path.join(ds, "classes.txt"), "AAGA\nACER\n")
    write(os.path.join(ds, "ingest_manifest.json"), json.dumps(
        {"n_images": 42, "n_boxes": 88, "classes": ["AAGA", "ACER"],
         "ingested_at": "2026-09-01 09:00:00"}))
    for i in range(n_images):
        write(os.path.join(ds, "images", f"img_{i:03d}.jpg"), "x")
        write(os.path.join(ds, "labels", f"img_{i:03d}.txt"), f"{i % 2} 0.5 0.5 0.2 0.3\n")
    # An image with no label file is background, not a training pair.
    write(os.path.join(ds, "images", "unlabelled.jpg"), "x")
    # A staged dir without ingest_manifest.json is not a finished dataset.
    write(os.path.join(bbox_root, "partial", "data.yaml"), "names:\n  0: X\n")
    return ds


# ── T5 fixtures ────────────────────────────────────────────────────────

def _write_run(run_dir, args_yaml=None, weights=True):
    """A run folder, optionally with an args.yaml text and weights/best.pt."""
    os.makedirs(run_dir, exist_ok=True)
    if args_yaml is not None:
        write(os.path.join(run_dir, "args.yaml"), args_yaml)
    if weights:
        write(os.path.join(run_dir, "weights", "best.pt"), "weights")
    return run_dir


SCHEME = "sha256-16/netstring-utf8-sorted-v1"


def _write_lab_run(root, lab, run, args_yaml=None, lab_task="", fingerprint=None, weights=True,
                   scheme=SCHEME):
    """A lab run at _models/<lab>/runs/<run> with a dataset beside it.
    ``fingerprint`` None writes no split manifest, '' writes one without the
    fingerprint fields, anything else writes one carrying it with ``scheme``."""
    lab_dir = os.path.join(root, "_models", lab)
    if lab_task:
        write(os.path.join(lab_dir, "lab.json"), json.dumps({"task": lab_task}))
    write(os.path.join(lab_dir, "dataset", "data.yaml"), "names:\n  0: PA\n")
    if fingerprint is not None:
        manifest = {"seed": lab, "holdout_mode": "lab-random"}
        if fingerprint != "":
            manifest[labruns.SPLIT_FINGERPRINT_FIELD] = fingerprint
            manifest[labruns.SPLIT_FINGERPRINT_SCHEME_FIELD] = scheme
        write(os.path.join(lab_dir, "dataset", labruns.SPLIT_MANIFEST_NAME), json.dumps(manifest))
    return _write_run(os.path.join(lab_dir, "runs", run), args_yaml, weights)


def _project_step6(root, folder):
    """A project run folder with its project.json; returns the step6 dir."""
    write(os.path.join(root, folder, "project.json"), json.dumps({"id": folder, "name": folder}))
    return os.path.join(root, folder, "step6_trainModel")


def _write_split_manifest(dataset_dir, fingerprint, scheme=SCHEME):
    """The data.yaml and split manifest the dataset builder writes beside a
    training set. Calling it a second time with another fingerprint is what a
    dataset rebuild does to the folder an already trained run sits beside."""
    write(os.path.join(dataset_dir, "data.yaml"), "names:\n  0: PA\n")
    write(os.path.join(dataset_dir, labruns.SPLIT_MANIFEST_NAME), json.dumps({
        "seed": "fx", "holdout_mode": "lab-random",
        labruns.SPLIT_FINGERPRINT_FIELD: fingerprint,
        labruns.SPLIT_FINGERPRINT_SCHEME_FIELD: scheme,
    }))
    return dataset_dir


# ── T5 cases ──────────────────────────────────────────────────────────

def test_validate_run_dir_refuses_a_path_outside_the_root(root, outside):
    print("test_validate_run_dir_refuses_a_path_outside_the_root")
    good = os.path.join(root, "_models", "labA", "runs", "r1")
    path, err = labruns.validate_run_dir(good, root)
    check(not err and path == good, f"a run inside the root is accepted when the root is given (got {err!r})")
    foreign = _write_run(os.path.join(outside, "foreign_run"), "task: segment\n")
    _p, err = labruns.validate_run_dir(foreign, root)
    check(bool(err) and "outside" in err, f"a run outside the root is refused (got {err!r})")
    dotdot = os.path.join(root, "_models", "..", "..", os.path.basename(outside), "foreign_run")
    check(os.path.isdir(dotdot), "fixture: the .. path really resolves to the foreign run")
    _p, err = labruns.validate_run_dir(dotdot, root)
    check(bool(err) and "outside" in err, f"a .. escape from the root is refused (got {err!r})")
    _p, err = labruns.validate_run_dir(root, root)
    check(bool(err), f"the root itself is not a run folder (got {err!r})")
    narrow = os.path.join(outside, "rootA")
    sibling = _write_run(os.path.join(outside, "rootAx", "runs", "s"), "task: detect\n")
    os.makedirs(narrow, exist_ok=True)
    _p, err = labruns.validate_run_dir(sibling, narrow)
    check(bool(err) and "outside" in err, f"a sibling folder sharing the root's name prefix is refused (got {err!r})")
    path, err = labruns.validate_run_dir(foreign)
    check(not err and path == foreign,
          "with no root given the legacy behaviour holds (a leftover of the retired Train + Infer page; no live caller)")
    _p, err = labruns.validate_run_dir(good, os.path.join(root, "does_not_exist"))
    check(bool(err), f"a root that does not exist refuses every run (got {err!r})")
    _p, err = labruns.validate_run_dir(good, None)
    check(bool(err) and "not set" in err, f"a root of None is refused, not treated as the legacy default (got {err!r})")
    _p, err = labruns.validate_run_dir(good, os.path.join(root, "run_26-08-26_LO", "project.json"))
    check(bool(err) and "not a folder" in err, f"a root that is a file is named as such (got {err!r})")


def test_validate_run_dir_refuses_a_symlink_escape(root, outside):
    print("test_validate_run_dir_refuses_a_symlink_escape")
    good = os.path.join(root, "_models", "labA", "runs", "r1")
    foreign = _write_run(os.path.join(outside, "foreign_link_target"), "task: segment\n")
    link_out = os.path.join(root, "_models", "labA", "runs", "linked_out")
    link_in = os.path.join(root, "_models", "labA", "runs", "linked_in")
    root_link = os.path.join(outside, "root_link")
    os.symlink(foreign, link_out)
    os.symlink(good, link_in)
    os.symlink(root, root_link)
    try:
        check(os.path.isfile(os.path.join(link_out, "weights", "best.pt")),
              "fixture: the escaping link points at a run with weights")
        _p, err = labruns.validate_run_dir(link_out, root)
        check(bool(err) and "outside" in err, f"a symlink inside the root pointing outside it is refused (got {err!r})")
        path, err = labruns.validate_run_dir(link_in, root)
        check(not err, f"a symlink that stays inside the root is accepted (got {err!r})")
        path, err = labruns.validate_run_dir(good, root_link)
        check(not err and path == good, f"a root given through a symlink still contains its own runs (got {err!r})")
        _p, err = labruns.validate_run_dir(link_out, root_link)
        check(bool(err), f"the escape is refused through a linked root too (got {err!r})")
        names = {r["run_name"] for r in labruns.list_all_runs(root)}
        check("linked_out" not in names and "linked_in" in names,
              f"the runs table skips a run that is a symlink out of the root and keeps one that stays inside (got {sorted(names)})")
    finally:
        for link in (link_out, link_in, root_link):
            os.remove(link)


_BARE_CHILD = r"""
import importlib.util, json, os, sys
sys.modules["flask"] = None
sys.modules["ultralytics"] = None
spec = importlib.util.spec_from_file_location("labruns_bare", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
root = sys.argv[2]
runs = mod.list_all_runs(root)
lab_run = next(r["run_dir"] for r in runs if r["origin"].startswith("lab:") and r["has_weights"])
ledger = mod.rounds_dir_for_run(lab_run)
out = {
    "origins": sorted(set(r["origin"].split(":")[0] for r in runs)),
    "n_runs": len(runs),
    "champion": mod.load_champion(root),
    "newest": mod.newest_weights(root, "labA", "seg"),
    "task": mod.task_for_run_dir(lab_run),
    "imgsz": mod.run_imgsz(lab_run),
    "dataset": mod.dataset_for_run(lab_run),
    "rounds_dir": ledger,
    "rounds": len(mod.read_rounds(os.path.join(ledger, "rounds.csv"))),
    "validated": list(mod.validate_run_dir(lab_run, root)),
    "comparable": list(mod.runs_comparable(lab_run, lab_run)),
    "saved": mod.save_champion(root, lab_run),
    "mirrored": mod.mirror_champion_to_lab(root, lab_run, {"run_dir": lab_run, "at": "2026-09-04 15:00:00"}),
    "flask_loaded": sys.modules.get("flask") is not None,
    "ultralytics_loaded": sys.modules.get("ultralytics") is not None,
}
print(json.dumps(out))
"""


def test_library_imports_without_flask():
    print("test_library_imports_without_flask")
    sub = tempfile.mkdtemp(prefix="modellab_labruns_bare_")
    try:
        fx = build_inprocess(sub)
        proc = subprocess.run([sys.executable, "-c", _BARE_CHILD, labruns.__file__, sub],
                              capture_output=True, text=True, timeout=120)
        check(proc.returncode == 0,
              f"labruns.py loads by file path with flask and ultralytics blocked (stderr: {proc.stderr.strip()[-600:]})")
        out = {}
        if proc.returncode == 0:
            try:
                out = json.loads(proc.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                check(False, f"child printed no JSON (stdout: {proc.stdout[-300:]!r})")
        check(out.get("flask_loaded") is False and out.get("ultralytics_loaded") is False,
              "neither flask nor ultralytics was imported by the library")
        check(out.get("origins") == ["lab", "project"] and out.get("n_runs") == 2,
              f"list_all_runs returns both lab and project rows without the blueprint (got {out.get('origins')})")
        r1 = os.path.join(fx["lab"], "runs", "r1")
        check((out.get("champion") or {}).get("run_dir") == r1, "load_champion works in the bare interpreter")
        check(out.get("newest") == os.path.join(r1, "weights", "best.pt"), "newest_weights works in the bare interpreter")
        check(out.get("task") == "seg" and out.get("imgsz") == 512, "task_for_run_dir and run_imgsz work in the bare interpreter")
        check(out.get("dataset") == os.path.join(fx["lab"], "dataset") and out.get("rounds_dir") == fx["lab"]
              and out.get("rounds") == 1, "dataset_for_run, rounds_dir_for_run and read_rounds work in the bare interpreter")
        check(out.get("validated") == [r1, ""], f"validate_run_dir with a root works in the bare interpreter (got {out.get('validated')})")
        comparable = out.get("comparable") or [None, ""]
        check(comparable[0] is False and "unknown" in comparable[1],
              f"runs_comparable works and reports an unknown split (got {comparable})")
        check((out.get("saved") or {}).get("run_dir") == r1 and out.get("mirrored") == os.path.join(fx["lab"], "champion.json"),
              "save_champion and mirror_champion_to_lab work in the bare interpreter")
    finally:
        shutil.rmtree(sub, ignore_errors=True)
    check(not os.path.exists(sub), "bare-interpreter fixture removed")


def test_split_fraction_defaults_come_from_the_builder(root):
    """The wrappers' default fractions are the builder's constants, read
    through labruns rather than restated, and the wrapper with no fractions
    given splits exactly as the builder does with its own defaults."""
    print("test_split_fraction_defaults_come_from_the_builder")
    import importlib.util
    import inspect
    spec = importlib.util.spec_from_file_location("_test_detect_dataset", labruns._DETECT_DATASET_PATH)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    check(labruns.DEFAULT_VALID_FRAC == builder.DEFAULT_VALID_FRAC
          and labruns.DEFAULT_TEST_FRAC == builder.DEFAULT_TEST_FRAC,
          f"labruns re-exports the builder's fractions ({labruns.DEFAULT_VALID_FRAC}, {labruns.DEFAULT_TEST_FRAC})")
    for fn in (labruns.assign_split, labruns.build_detect_dataset):
        params = inspect.signature(fn).parameters
        check(params["valid_frac"].default == builder.DEFAULT_VALID_FRAC
              and params["test_frac"].default == builder.DEFAULT_TEST_FRAC,
              f"{fn.__name__} defaults to the builder's fractions")
    names = [f"img_{i:03d}.jpg" for i in range(40)]
    check(labruns.assign_split(names, "seedA")
          == builder.assign_split(names, "seedA", None, builder.DEFAULT_VALID_FRAC, builder.DEFAULT_TEST_FRAC),
          "assign_split with no fractions given splits exactly as the builder does with its defaults")
    with open(labruns.__file__) as fh:
        text = fh.read()
    check("valid_frac=0.1" not in text and "test_frac=0.1" not in text,
          "labruns.py restates no split-fraction literal in a signature")


def test_mirror_champion_writes_only_inside_a_models_lab(root, outside, fx):
    """mirror_champion_to_lab decides containment with the commonpath check
    on resolved paths: a real lab under _models gets the file; a sibling
    whose name merely extends _models, a lab symlinked out of _models, the
    _models root itself and a project run all get nothing written."""
    print("test_mirror_champion_writes_only_inside_a_models_lab")
    record = {"run_dir": "", "at": "2026-09-05 10:00:00"}

    sibling_lab = os.path.join(root, "_models_backup", "labS")
    sibling_run = os.path.join(sibling_lab, "runs", "r1")
    os.makedirs(sibling_run, exist_ok=True)
    check(labruns.mirror_champion_to_lab(root, sibling_run, dict(record, run_dir=sibling_run)) == ""
          and not os.path.exists(os.path.join(sibling_lab, "champion.json")),
          "a lab under a sibling folder whose name extends _models is not mirrored")

    ext_lab = os.path.join(outside, "ext_lab")
    ext_run = os.path.join(ext_lab, "runs", "r1")
    os.makedirs(ext_run, exist_ok=True)
    link = os.path.join(root, "_models", "linked_lab")
    os.symlink(ext_lab, link)
    linked_run = os.path.join(link, "runs", "r1")
    check(labruns.mirror_champion_to_lab(root, linked_run, dict(record, run_dir=linked_run)) == ""
          and not os.path.exists(os.path.join(ext_lab, "champion.json")),
          "a lab symlinked out of _models is refused")

    module_champion = labruns.champion_path(root)
    with open(module_champion) as fh:
        before = fh.read()
    direct_run = os.path.join(root, "_models", "runs", "r1")
    os.makedirs(direct_run, exist_ok=True)
    with open(module_champion) as fh:
        after = fh.read()
    check(labruns.mirror_champion_to_lab(root, direct_run, dict(record, run_dir=direct_run)) == ""
          and before == after,
          "a run whose ledger would be the _models root itself is refused and the module champion is untouched")

    project_run = os.path.join(fx["proj"], "runs", "p1")
    check(labruns.mirror_champion_to_lab(root, project_run, dict(record, run_dir=project_run)) == ""
          and not os.path.exists(os.path.join(fx["proj"], "champion.json")),
          "a project run is left alone")

    lab = os.path.join(root, "_models", "labMirror")
    lab_run = os.path.join(lab, "runs", "r1")
    os.makedirs(lab_run, exist_ok=True)
    written = labruns.mirror_champion_to_lab(root, lab_run, dict(record, run_dir=lab_run))
    check(written == os.path.join(lab, "champion.json") and os.path.isfile(written),
          "a lab run under _models gets its ledger champion.json")
    with open(written) as fh:
        check(json.load(fh).get("run_dir") == lab_run, "the mirrored record carries the run dir")

    check(labruns.mirror_champion_to_lab(root, "", dict(record)) == "",
          "an empty run dir mirrors nothing")

    # Leave the shared fixture as this test found it: later tests count runs.
    os.unlink(link)
    for added in (os.path.join(root, "_models_backup"), os.path.join(root, "_models", "runs"), lab):
        shutil.rmtree(added, ignore_errors=True)
    check(not os.path.lexists(link) and not os.path.isdir(lab), "the mirror fixtures were removed")


def test_task_comes_from_args_yaml_not_the_filename(root):
    print("test_task_comes_from_args_yaml_not_the_filename")
    proj = _project_step6(root, "run_26-09-04_LO_continued")
    cont = _write_run(os.path.join(proj, "runs", "cont1"),
                      "task: segment\nmodel: /elsewhere/runs/prev/weights/best.pt\nimgsz: 640\n")
    check(labruns.task_for_run_dir(cont) == "seg",
          "a run continued from best.pt whose args.yaml says segment is a segmenter (the exact bug)")
    row = {r["run_dir"]: r for r in labruns.list_all_runs(root)}[cont]
    check(row["task"] == "seg", f"list_all_runs files that run as seg too (got {row['task']!r})")
    check(labruns.run_imgsz(cont) == 640, "the image size still comes from the same file")
    contra = _write_run(os.path.join(proj, "runs", "contra"), "task: detect\nmodel: yolo11m-seg.pt\n")
    check(labruns.task_for_run_dir(contra) == "detect", "the task line outranks a contradicting model name")
    lab_run = _write_lab_run(root, "labArgsWins", "r", "task: detect\nmodel: yolo11n.pt\n", lab_task="seg")
    check(labruns.task_for_run_dir(lab_run) == "detect", "the task line outranks the lab's own lab.json")
    spelled = _write_run(os.path.join(proj, "runs", "spelled"), "task: 'Segment'\nmodel: yolo11n.pt\n")
    check(labruns.task_for_run_dir(spelled) == "seg", "a quoted or capitalised task value is still read")


def test_task_falls_back_to_the_filename_when_absent(root):
    print("test_task_falls_back_to_the_filename_when_absent")
    proj = _project_step6(root, "run_26-09-04_LO_legacy")
    seg = _write_run(os.path.join(proj, "runs", "legacy_seg"), "model: yolo11m-seg.pt\nimgsz: 512\n")
    det = _write_run(os.path.join(proj, "runs", "legacy_det"), "model: yolo11n.pt\nimgsz: 512\n")
    bare = _write_run(os.path.join(proj, "runs", "bare"), None)
    check(labruns.task_for_run_dir(seg) == "seg" and labruns.task_for_run_dir(det) == "detect",
          "with no task line the model-name guess decides")
    check(labruns.task_for_run_dir(bare) == "seg", "with no args.yaml at all the default is seg")
    lab_run = _write_lab_run(root, "labFallback", "r", "model: yolo11m-seg.pt\n", lab_task="detect")
    check(labruns.task_for_run_dir(lab_run) == "detect", "with no task line lab.json outranks the model-name guess")


def test_both_champion_shapes_are_readable(root, fx):
    print("test_both_champion_shapes_are_readable")
    lab_path = labruns.champion_path(root)
    lab_record = {"run_dir": os.path.join(fx["lab"], "runs", "r1"), "at": "2026-09-01 11:00:00"}
    write(lab_path, json.dumps(lab_record))
    orch_path = os.path.join(fx["proj"], "champion.json")
    orch_record = {"run_dir": os.path.join(fx["proj"], "runs", "p1"),
                   "promoted_at": "2026-09-03T15:05:11-04:00", "map50_95_M": 0.41, "note": "best so far"}
    write(orch_path, json.dumps(orch_record))
    before = {p: (open(p).read(), os.path.getmtime(p)) for p in (lab_path, orch_path)}
    a = labruns.read_champion_file(lab_path)
    b = labruns.read_champion_file(orch_path)
    check(a is not None and set(a) == set(labruns.CHAMPION_FIELDS),
          f"the lab shape {{run_dir, at}} reads into the normalised record (got {a})")
    check(b is not None and set(b) == set(labruns.CHAMPION_FIELDS),
          f"the orchestrator shape {{run_dir, promoted_at, map50_95_M, note}} reads into the same record (got {b})")
    check(a["run_dir"] == lab_record["run_dir"] and a["promoted_at"] == "2026-09-01 11:00:00"
          and a["map50_95_M"] is None and a["map50_95_B"] is None and a["score"] is None and a["note"] == "",
          f"the lab shape maps at to promoted_at and leaves the scores and note empty (got {a})")
    check(a["metric_family"] == "mask", f"the lab shape derives its family from the run's task on disk (got {a['metric_family']!r})")
    check(b["run_dir"] == orch_record["run_dir"] and b["promoted_at"] == "2026-09-03T15:05:11-04:00"
          and b["map50_95_M"] == 0.41 and b["score"] == 0.41 and b["map50_95_B"] is None and b["note"] == "best so far",
          f"the older orchestrator shape keeps its mask score as the score (got {b})")
    check(labruns.load_champion(root) == a, "load_champion returns the same normalised record for the module champion")
    after = {p: (open(p).read(), os.path.getmtime(p)) for p in (lab_path, orch_path)}
    check(before == after, "reading rewrites neither file")
    check(labruns.read_champion_file(os.path.join(root, "nope.json")) is None, "a missing champion file is None")
    runs = {r["run_name"]: r for r in labruns.list_all_runs(root)}
    check(runs["r1"]["champion"] is True and runs["p1"]["champion"] is False,
          "the champion flag on the runs table still follows the module champion")
    os.remove(orch_path)


def test_new_champion_shape_carries_its_family_score(root, fx):
    print("test_new_champion_shape_carries_its_family_score")
    det = _write_lab_run(root, "labChampDet", "d", "task: detect\nmodel: yolo11n.pt\n")
    seg = os.path.join(fx["lab"], "runs", "r1")
    path = os.path.join(root, "_models", "labChampDet", "champion.json")
    write(path, json.dumps({"run_dir": det, "promoted_at": "2026-09-04T19:30:00-04:00",
                            "metric_family": "box", "map50_95_B": 0.29, "note": "detector"}))
    rec = labruns.read_champion_file(path)
    check(rec is not None and set(rec) == set(labruns.CHAMPION_FIELDS), f"the new box shape reads into the normalised record (got {rec})")
    check(rec["metric_family"] == "box" and rec["map50_95_B"] == 0.29 and rec["map50_95_M"] is None
          and rec["score"] == 0.29 and rec["note"] == "detector",
          f"a box champion's score is its box mAP50-95 (got {rec})")
    write(path, json.dumps({"run_dir": seg, "promoted_at": "2026-09-04T19:31:00-04:00",
                            "metric_family": "mask", "map50_95_M": 0.44, "note": ""}))
    rec = labruns.read_champion_file(path)
    check(rec["metric_family"] == "mask" and rec["map50_95_M"] == 0.44 and rec["map50_95_B"] is None
          and rec["score"] == 0.44, f"a mask champion's score is its mask mAP50-95 (got {rec})")
    write(path, json.dumps({"run_dir": det, "promoted_at": "t", "metric_family": "box",
                            "map50_95_M": 0.9, "map50_95_B": 0.3, "note": ""}))
    rec = labruns.read_champion_file(path)
    check(rec["map50_95_M"] == 0.9 and rec["map50_95_B"] == 0.3 and rec["score"] == 0.3,
          f"a file carrying both keys keeps both and scores by its family (got {rec})")
    write(path, json.dumps({"run_dir": seg, "promoted_at": "t", "metric_family": "mask",
                            "map50_95_M": 0.9, "map50_95_B": 0.3, "note": ""}))
    rec = labruns.read_champion_file(path)
    check(rec["score"] == 0.9, f"the same two keys under the mask family score by the mask key (got {rec})")
    write(path, json.dumps({"run_dir": det, "promoted_at": "t", "map50_95_M": None, "note": ""}))
    rec = labruns.read_champion_file(path)
    check(rec["metric_family"] == "box" and rec["score"] is None,
          f"the older orchestrator shape on a detector derives box and has no score (the old defect, now visible) (got {rec})")
    write(path, json.dumps({"run_dir": det, "promoted_at": "t", "metric_family": "box", "map50_95_M": 0.5, "note": ""}))
    rec = labruns.read_champion_file(path)
    check(rec["metric_family"] == "box" and rec["map50_95_M"] == 0.5 and rec["score"] is None,
          f"a box champion whose only score is a mask score reads as score None (got {rec})")
    write(path, json.dumps({"run_dir": det, "promoted_at": "t", "metric_family": "polygon", "map50_95_B": 0.2, "note": ""}))
    rec = labruns.read_champion_file(path)
    check(rec["metric_family"] == "box" and rec["score"] == 0.2,
          f"an unknown metric_family is replaced by the run's own family (got {rec})")
    write(path, json.dumps({"run_dir": det, "promoted_at": "t", "metric_family": 7, "map50_95_B": "nan", "note": ""}))
    rec = labruns.read_champion_file(path)
    check(rec["metric_family"] == "box" and rec["score"] is None and rec["map50_95_B"] is None,
          f"a non-string family and a NaN score both fall back safely (got {rec})")
    os.remove(path)


def test_comparability_requires_same_family_and_same_split(root):
    print("test_comparability_requires_same_family_and_same_split")
    a = _write_lab_run(root, "cmpA", "a", "task: segment\nmodel: yolo11m-seg.pt\n", fingerprint="fp-1")
    b = _write_lab_run(root, "cmpB", "b", "task: segment\nmodel: /x/runs/a/weights/best.pt\n", fingerprint="fp-1")
    c = _write_lab_run(root, "cmpC", "c", "task: detect\nmodel: yolo11n.pt\n", fingerprint="fp-1")
    d = _write_lab_run(root, "cmpD", "d", "task: segment\nmodel: yolo11m-seg.pt\n", fingerprint="fp-2")
    e = _write_lab_run(root, "cmpE", "e", "task: segment\nmodel: yolo11m-seg.pt\n", fingerprint=None)
    f = _write_lab_run(root, "cmpF", "f", "task: segment\nmodel: yolo11m-seg.pt\n", fingerprint="")
    check(labruns.metric_family_for_task("seg") == labruns.METRIC_FAMILY_MASK
          and labruns.metric_family_for_task("detect") == labruns.METRIC_FAMILY_BOX
          and labruns.metric_family_for_task("") == "",
          "seg scores in the mask family, detect in the box family, unknown in none")
    g = _write_lab_run(root, "cmpG", "g", "task: segment\nmodel: yolo11m-seg.pt\n", fingerprint="fp-1",
                       scheme="other-scheme-v9")
    h = _write_lab_run(root, "cmpH", "h", "task: segment\nmodel: yolo11m-seg.pt\n", fingerprint={"a": 1})
    check(labruns.split_fingerprint_for_run(a) == ("fp-1", SCHEME)
          and labruns.split_fingerprint_for_run(e) == ("", "")
          and labruns.split_fingerprint_for_run(f) == ("", "")
          and labruns.split_fingerprint_for_run(h) == ("", ""),
          "the fingerprint and its scheme are read from the split manifest; absent or non-string is unknown")
    ok, why = labruns.runs_comparable(a, g)
    check(ok is False and "scheme" in why, f"the same fingerprint under a different hashing scheme is not comparable ({why})")
    ok, why = labruns.runs_comparable(a, h)
    check(ok is False and "unknown" in why, f"a fingerprint that is not a string is unknown ({why})")
    ok, why = labruns.runs_comparable(a, b)
    check(ok is True, f"same family and same split: comparable ({why})")
    ok, why = labruns.runs_comparable(a, a)
    check(ok is True, f"a run is comparable with itself when its split is known ({why})")
    ok, why = labruns.runs_comparable(a, c)
    check(ok is False and "famil" in why, f"mask against box is not comparable ({why})")
    ok, why = labruns.runs_comparable(a, d)
    check(ok is False and "split" in why, f"two different split fingerprints are not comparable ({why})")
    ok, why = labruns.runs_comparable(a, e)
    check(ok is False and "unknown" in why, f"a run with no split manifest is unknown, so not comparable ({why})")
    ok, why = labruns.runs_comparable(a, f)
    check(ok is False and "unknown" in why, f"a manifest without the fingerprint field is unknown too ({why})")
    ok, why = labruns.runs_comparable(a, os.path.join(root, "_models", "nope", "runs", "x"))
    check(ok is False and "not found" in why, f"a run folder that does not exist is refused by name ({why})")
    check(labruns.runs_comparable(a, c)[0] == labruns.runs_comparable(c, a)[0]
          and labruns.runs_comparable(a, d)[0] == labruns.runs_comparable(d, a)[0],
          "the verdict is symmetric")


def test_split_fingerprint_is_recorded_on_the_run(root):
    """The split a run was scored on is recorded into the run's own folder and
    read from there for the rest of that run's life.

    The dataset folder beside a run is rebuilt whenever its training set
    changes, so reading it live made every run of one project report the same
    fingerprint and let an old run claim a split it was never scored on. The
    runs card decides comparability on that fingerprint, so the claim was
    false. These checks pin the recorded value: stamped at training time, kept
    across a rebuild, different for two runs of one project that really
    trained on different splits, and honestly absent when nobody recorded it.
    """
    print("test_split_fingerprint_is_recorded_on_the_run")

    # -- recorded when the run is trained ------------------------------
    step6 = _project_step6(root, "run_26-09-16_LO_split")
    dataset = _write_split_manifest(os.path.join(step6, "dataset"), "fp-first")
    first = _write_run(os.path.join(step6, "runs", "first"), "task: segment\nmodel: yolo11m-seg.pt\n")
    stamped = labruns.record_split_fingerprint(first, source=labruns.SPLIT_RECORD_SOURCE_TRAINING)
    check(stamped == ("fp-first", SCHEME),
          f"training records the split the run was trained on (got {stamped})")
    record_path = labruns.run_split_record_path(first)
    check(record_path == os.path.join(first, labruns.RUN_SPLIT_RECORD_NAME) and os.path.isfile(record_path),
          f"the record is written into the run folder itself (got {record_path})")
    with open(record_path) as fh:
        record = json.load(fh)
    check(set(record) == set(labruns.RUN_SPLIT_RECORD_FIELDS),
          f"the record carries exactly its published fields (got {sorted(record)})")
    check(record["recorded_source"] == labruns.SPLIT_RECORD_SOURCE_TRAINING
          and record["dataset_dir"] == dataset
          and len(record["recorded_at"]) == len("2026-09-16 14:00:00"),
          f"the record says which occasion wrote it, from which dataset, and when (got {record})")
    check(labruns.split_fingerprint_for_run(first) == ("fp-first", SCHEME),
          "the recorded pair is what the run reports")

    # -- the dataset is rebuilt; the recorded value does not move -------
    _write_split_manifest(dataset, "fp-second")
    check(labruns.split_fingerprint_for_run(first) == ("fp-first", SCHEME),
          "rebuilding the dataset does not move a run already trained on the old split (the defect)")

    # -- two runs of one project, two real splits ----------------------
    second = _write_run(os.path.join(step6, "runs", "second"), "task: segment\nmodel: yolo11m-seg.pt\n")
    check(labruns.split_fingerprint_for_run(second) == ("fp-second", SCHEME),
          "a run trained after the rebuild records the split it really trained on")
    ok, why = labruns.runs_comparable(first, second)
    check(ok is False and "split" in why,
          f"two runs of one project on different splits are not comparable (got {ok}, {why})")
    check(labruns.runs_comparable(first, first)[0] is True
          and labruns.runs_comparable(second, second)[0] is True,
          "each of them is still comparable with itself")

    # -- the dataset is never read again -------------------------------
    shutil.rmtree(dataset)
    check(labruns.split_fingerprint_for_run(first) == ("fp-first", SCHEME)
          and labruns.split_fingerprint_for_run(second) == ("fp-second", SCHEME),
          "both runs keep their recorded split with the dataset folder gone entirely")

    # -- a run trained before records existed, filled in on first read --
    legacy = _write_lab_run(root, "labLegacySplit", "r", "task: segment\nmodel: yolo11m-seg.pt\n",
                            fingerprint="fp-legacy")
    check(not os.path.exists(labruns.run_split_record_path(legacy)),
          "fixture: the legacy run starts with no record of its own")
    check(labruns.split_fingerprint_for_run(legacy) == ("fp-legacy", SCHEME),
          "a run that predates records reads its dataset once")
    with open(labruns.run_split_record_path(legacy)) as fh:
        filled = json.load(fh)
    check(filled["recorded_source"] == labruns.SPLIT_RECORD_SOURCE_FIRST_READ,
          f"the record says it was filled in on that first read (got {filled.get('recorded_source')!r})")
    _write_split_manifest(os.path.join(root, "_models", "labLegacySplit", "dataset"), "fp-legacy-rebuilt")
    check(labruns.split_fingerprint_for_run(legacy) == ("fp-legacy", SCHEME),
          "the filled-in value is frozen too: a later rebuild does not move it")

    # -- no record and nothing to record from --------------------------
    bare = _project_step6(root, "run_26-09-16_LO_nosplit")
    write(os.path.join(bare, "dataset", "data.yaml"), "names:\n  0: PA\n")
    unrecorded = _write_run(os.path.join(bare, "runs", "unrecorded"), "task: segment\nmodel: yolo11m-seg.pt\n")
    check(labruns.split_fingerprint_for_run(unrecorded) == ("", ""),
          "a run whose dataset never carried a fingerprint reports an unrecorded split")
    check(not os.path.exists(labruns.run_split_record_path(unrecorded)),
          "no record is invented for a run whose split nobody can recover")
    ok, why = labruns.runs_comparable(unrecorded, first)
    check(ok is False and "unknown" in why, f"an unrecorded split is never comparable (got {why})")

    # -- a damaged record is reported, not repaired over ----------------
    damaged = ("{not json", "[1, 2]", "null", "{}", "",
               '{"split_fingerprint": 5, "split_fingerprint_scheme": "s"}',
               '{"split_fingerprint": "   ", "split_fingerprint_scheme": "s"}',
               '{"split_fingerprint": "fp-x"}')
    for i, bad in enumerate(damaged):
        run = _write_lab_run(root, "labDamagedSplit", f"c{i}", "task: segment\nmodel: yolo11m-seg.pt\n",
                             fingerprint="fp-good")
        write(labruns.run_split_record_path(run), bad)
        check(labruns.split_fingerprint_for_run(run) == ("", ""),
              f"a damaged record {bad[:30]!r} reports an unrecorded split")
        with open(labruns.run_split_record_path(run)) as fh:
            check(fh.read() == bad,
                  f"the damaged record {bad[:30]!r} stays on disk for repair, not silently replaced")

    # -- adversarial arguments ------------------------------------------
    check(labruns.run_split_record_path("") == "" and labruns.run_split_record_path(None) == "",
          "an empty run folder yields no record path at all")
    check(labruns.split_fingerprint_for_run("") == ("", "")
          and labruns.split_fingerprint_for_run(None) == ("", ""),
          "an empty run folder reports an unrecorded split without an exception")
    check(not os.path.exists(labruns.RUN_SPLIT_RECORD_NAME),
          "no record is written into the folder the process happens to stand in")
    ghost = os.path.join(root, "_models", "labGhostSplit", "runs", "g")
    check(labruns.split_fingerprint_for_run(ghost) == ("", "") and not os.path.exists(ghost),
          "a run folder that does not exist reports nothing and is not created")
    try:
        labruns.record_split_fingerprint(first, source="whenever")
        check(False, "record_split_fingerprint refuses a source it does not know")
    except ValueError as exc:
        check("source" in str(exc),
              f"record_split_fingerprint refuses a source it does not know ({exc})")

    # -- eight writers stamping one run at once --------------------------
    busy = _write_lab_run(root, "labBusySplit", "r", "task: segment\nmodel: yolo11m-seg.pt\n",
                          fingerprint="fp-busy")
    stamps = []
    lock = threading.Lock()

    def stamp():
        pair = labruns.record_split_fingerprint(busy, source=labruns.SPLIT_RECORD_SOURCE_TRAINING)
        with lock:
            stamps.append(pair)

    workers = [threading.Thread(target=stamp) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    check(stamps == [("fp-busy", SCHEME)] * 8,
          f"eight writers at once all record the same pair (got {sorted(set(stamps))})")
    with open(labruns.run_split_record_path(busy)) as fh:
        check(json.load(fh)[labruns.SPLIT_FINGERPRINT_FIELD] == "fp-busy",
              "the record they leave behind is one whole file")
    leftovers = sorted(n for n in os.listdir(busy) if n.endswith(".tmp"))
    check(leftovers == [], f"no half-written temporary record is left behind (got {leftovers})")

    # -- a run folder nobody may write to --------------------------------
    locked = _write_lab_run(root, "labReadOnlySplit", "r", "task: segment\nmodel: yolo11m-seg.pt\n",
                            fingerprint="fp-locked")
    os.chmod(locked, 0o555)
    try:
        if os.geteuid() == 0:
            print("  NOTE running as root: a read-only folder is still writable, the check is not meaningful")
        else:
            check(labruns.split_fingerprint_for_run(locked) == ("fp-locked", SCHEME),
                  "a run folder nobody may write to still reports the split it can read")
            check(not os.path.exists(labruns.run_split_record_path(locked)),
                  "and nothing is written into it")
    finally:
        os.chmod(locked, 0o755)


def test_adversarial_inputs(root, fx):
    print("test_adversarial_inputs")
    proj = _project_step6(root, "run_26-09-04_LO_hostile")
    empty = _write_run(os.path.join(proj, "runs", "empty_args"), "")
    check(labruns.task_for_run_dir(empty) == "seg" and labruns.run_imgsz(empty) == 512,
          "an empty args.yaml gives the default task and image size without an exception")
    malformed = (("task:\nmodel: yolo11m-seg.pt\n", "seg"),
                 ("task: 42\nmodel: yolo11m-seg.pt\n", "seg"),
                 ("task: [seg\nmodel: yolo11n.pt\n", "detect"),
                 ("  task: detect\nmodel: yolo11m-seg.pt\n", "seg"))
    for i, (text, want) in enumerate(malformed):
        rd = _write_run(os.path.join(proj, "runs", f"malformed_{i}"), text)
        check(labruns.task_for_run_dir(rd) == want,
              f"malformed task line {text.splitlines()[0]!r} falls back to the model-name guess ({want})")
    for i, bad_size in enumerate(("abc", "inf", "1e400", "-5", "0", "nan")):
        rd = _write_run(os.path.join(proj, "runs", f"badsize_{i}"), f"task: detect\nimgsz: {bad_size}\n")
        check(labruns.run_imgsz(rd) == 512, f"an unusable imgsz {bad_size!r} gives the default without an exception")
    binary = os.path.join(proj, "runs", "binary")
    os.makedirs(binary, exist_ok=True)
    with open(os.path.join(binary, "args.yaml"), "wb") as fh:
        fh.write(b"\xff\xfe\x00task: detect\n")
    check(labruns.task_for_run_dir(binary) == "seg" and labruns.run_imgsz(binary) == 512,
          "an args.yaml that is not text gives the defaults without an exception")

    ghost = os.path.join(root, "_models", "ghost", "runs", "g")
    check(labruns.task_for_run_dir(ghost) == "seg" and labruns.run_imgsz(ghost) == 512
          and labruns.dataset_for_run(ghost) == "" and labruns.split_fingerprint_for_run(ghost) == ("", ""),
          "a run folder that does not exist gives defaults, never an exception")
    _p, err = labruns.validate_run_dir(ghost, root)
    check("not found" in err, f"validate_run_dir names a missing folder (got {err!r})")

    champ_path = labruns.champion_path(root)
    keep = open(champ_path).read()
    for bad in ("{not json", "[1, 2]", '{"at": "x"}', '{"run_dir": 5}', "", "null"):
        write(champ_path, bad)
        check(labruns.load_champion(root) is None, f"corrupt champion {bad!r} reads as None")
    write(champ_path, "{not json")
    check(all(r["champion"] is False for r in labruns.list_all_runs(root)),
          "no run is champion while the file is corrupt, and the scan still runs")
    write(champ_path, '{"run_dir": "  /x  ", "promoted_at": "t", "map50_95_M": "n/a", "note": 7}')
    rec = labruns.load_champion(root)
    check(rec is not None and rec["map50_95_M"] is None and rec["note"] == "7" and rec["run_dir"] == "/x",
          f"a non-numeric score reads as None, a non-string note is stringified, run_dir is stripped (got {rec})")
    for odd in ("true", "NaN", "Infinity"):
        write(champ_path, '{"run_dir": "/x", "map50_95_M": ' + odd + "}")
        rec = labruns.load_champion(root)
        check(rec is not None and rec["map50_95_M"] is None, f"a score of {odd} reads as None")
    write(champ_path, keep)

    lab_dir = os.path.join(root, "_models", "labMissingCol")
    m1 = _write_lab_run(root, "labMissingCol", "m1", "task: detect\nmodel: yolo11n.pt\n")
    write(os.path.join(lab_dir, "rounds.csv"), f"round,run_name,run_dir,map50_B\n1,m1,{m1},0.33\n2,short\n")
    rows = labruns.read_rounds(os.path.join(lab_dir, "rounds.csv"))
    check(len(rows) == 2 and rows[1].get("run_dir") is None,
          f"a ledger missing columns reads, with None for the missing cells (got {rows})")
    check({r["run_dir"]: r for r in labruns.list_all_runs(root)}[m1]["map50"] == 0.33,
          "the runs table still finds the box mAP without the mask column")

    lab_dir = os.path.join(root, "_models", "labHostile")
    h1 = _write_lab_run(root, "labHostile", "h1", "task: segment\nmodel: yolo11m-seg.pt\n")
    hostile = '=HYPERLINK("http://evil")\n"quoted"\r\nsecond line'
    with open(os.path.join(lab_dir, "rounds.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "run_name", "run_dir", "map50_M"])
        w.writerow([1, hostile, h1, "0.5"])
    rows = labruns.read_rounds(os.path.join(lab_dir, "rounds.csv"))
    check(len(rows) == 1 and rows[0]["run_name"] == hostile,
          "a cell with quotes, newlines and a formula reads back verbatim as text")
    check({r["run_dir"]: r for r in labruns.list_all_runs(root)}[h1]["map50"] == 0.5,
          "the hostile row still scores its run by run_dir")
    with open(os.path.join(lab_dir, "rounds.csv"), "wb") as fh:
        fh.write(b"\xff\xfe\x00\x00")
    check(labruns.read_rounds(os.path.join(lab_dir, "rounds.csv")) == [],
          "a ledger that is not text reads as [] without an exception")

    unreadable_runs = os.path.join(root, "_models", "labUnreadable", "runs")
    _write_lab_run(root, "labUnreadable", "u1", "task: detect\nmodel: yolo11n.pt\n")
    unreadable_proj_runs = os.path.join(root, "run_26-09-04_LO_unreadable", "step6_trainModel", "runs")
    os.makedirs(os.path.join(unreadable_proj_runs, "x"), exist_ok=True)
    os.chmod(unreadable_runs, 0)
    os.chmod(unreadable_proj_runs, 0)
    try:
        runs = labruns.list_all_runs(root)
        if os.geteuid() == 0:
            print("  NOTE running as root: permission bits do not apply, the skip check is not meaningful")
        else:
            check(not any(r["origin"] in ("lab:labUnreadable", "project:run_26-09-04_LO_unreadable") for r in runs),
                  "an unreadable lab runs folder and an unreadable project runs folder are both skipped")
        check(any(r["run_name"] == "r1" for r in runs), "the readable runs are still listed")
    except PermissionError as exc:
        check(False, f"list_all_runs raised on an unreadable folder: {exc}")
    finally:
        os.chmod(unreadable_runs, 0o755)
        os.chmod(unreadable_proj_runs, 0o755)


def main():
    root = tempfile.mkdtemp(prefix="modellab_labruns_inprocess_")
    bbox_root = tempfile.mkdtemp(prefix="modellab_labruns_bbox_")
    outside = tempfile.mkdtemp(prefix="modellab_labruns_outside_")
    try:
        fx = build_inprocess(root)
        ds = build_bbox(bbox_root)

        # ---- roots -------------------------------------------------------
        check(labruns.models_root(root) == os.path.join(root, "_models"), "models_root is inprocess/_models")
        check(labruns.composite_root(root) == os.path.join(root, "_composite"),
              "composite_root is inprocess/_composite")
        check(labruns.inference_root(root) == os.path.join(root, "_models", "_inference"),
              "inference_root is inprocess/_models/_inference")
        check(labruns.champion_path(root) == os.path.join(root, "_models", "champion.json"),
              "champion.json sits at the _models root")

        # ---- data.yaml labels -------------------------------------------
        check(labruns.data_yaml_labels(os.path.join(fx["comp"], "data.yaml")) == ["PA", "MC"],
              "data_yaml_labels reads the {id: name} mapping form")
        listy = os.path.join(outside, "list.yaml")
        write(listy, "names: [PA, MC, DLAB]\n")
        check(labruns.data_yaml_labels(listy) == ["PA", "MC", "DLAB"],
              "data_yaml_labels reads the inline list form")
        check(labruns.data_yaml_labels(os.path.join(outside, "nope.yaml")) == [],
              "data_yaml_labels on a missing file is []")

        # ---- sources -----------------------------------------------------
        sources = labruns.list_sources(root, bbox_root)
        comps = [s for s in sources if s["kind"] == "composite"]
        boxes = [s for s in sources if s["kind"] == "bbox"]
        check(len(comps) == 1 and comps[0]["name"] == "comp_A" and comps[0]["path"] == fx["comp"],
              f"one composite source listed (got {[s['name'] for s in comps]})")
        check(comps[0]["labels"] == ["PA", "MC"] and comps[0]["n_images"] == 2,
              f"composite carries labels + image count (got {comps[0]})")
        check(comps[0]["built_at"] == "2026-09-01 10:00:00", "composite built_at comes from composite.json")
        check(comps[0]["task"] == "seg", "composite trains a segmentation model")
        check(len(boxes) == 1 and boxes[0]["name"] == "coral_intel" and boxes[0]["path"] == ds,
              f"one bbox source listed (got {[s['name'] for s in boxes]})")
        check(boxes[0]["task"] == "detect" and boxes[0]["n_images"] == 42
              and boxes[0]["labels"] == ["AAGA", "ACER"],
              f"bbox source is a detection dataset with manifest counts (got {boxes[0]})")
        check(boxes[0]["built_at"] == "2026-09-01 09:00:00", "bbox built_at comes from ingest_manifest.json")
        check(labruns.list_sources(os.path.join(root, "nope"), os.path.join(bbox_root, "nope")) == [],
              "missing roots list no sources at all (empty groups are fine)")

        # ---- find_source -------------------------------------------------
        check(labruns.find_source(sources, fx["comp"])["name"] == "comp_A",
              "find_source resolves a listed path")
        check(labruns.find_source(sources, os.path.join(root, "_composite", "half_built")) is None,
              "find_source refuses an unlisted path")

        # ---- lab name validation ----------------------------------------
        for good in ("labA", "reef_2026-09", "x", "a" * 60):
            name, err = labruns.validate_lab_name(good)
            check(not err and name == good, f"lab name {good!r} accepted")
        for bad in ("", "   ", "..", "../evil", "a/b", "a b", "a" * 61, "lab.name", "/abs"):
            _name, err = labruns.validate_lab_name(bad)
            check(bool(err), f"lab name {bad!r} rejected ({err!r})")

        # ---- lab paths ---------------------------------------------------
        p = labruns.lab_paths(root, "labB")
        check(p["lab_dir"] == os.path.join(root, "_models", "labB"), "lab_dir under _models")
        check(p["dataset_dir"] == os.path.join(p["lab_dir"], "dataset")
              and p["runs_dir"] == os.path.join(p["lab_dir"], "runs")
              and p["rounds_csv"] == os.path.join(p["lab_dir"], "rounds.csv"),
              f"lab paths shape (got {p})")
        try:
            labruns.lab_paths(root, "../evil")
            check(False, "lab_paths refuses a traversal name")
        except ValueError:
            check(True, "lab_paths refuses a traversal name")

        # ---- run scanning ------------------------------------------------
        runs = labruns.list_all_runs(root)
        by_name = {r["run_name"]: r for r in runs}
        check(set(by_name) == {"r1", "p1"}, f"both roots scanned (got {sorted(by_name)})")
        check(by_name["r1"]["origin"] == "lab:labA" and by_name["p1"]["origin"] == "project:run_26-08-26_LO",
              f"origin labels (got {[(r['run_name'], r['origin']) for r in runs]})")
        check(by_name["r1"]["map50"] == 0.62 and by_name["p1"]["map50"] is None,
              f"mAP from the adjacent rounds.csv, None without one (got {by_name['r1']['map50']!r})")
        check(by_name["r1"]["has_weights"] is True and by_name["p1"]["has_weights"] is False,
              "has_weights reflects weights/best.pt")
        check(by_name["r1"]["champion"] is True and by_name["p1"]["champion"] is False,
              "champion flag comes from champion.json")
        check(by_name["r1"]["task"] == "seg" and by_name["p1"]["task"] == "seg",
              f"task labels (got {[(r['run_name'], r['task']) for r in runs]})")
        check(by_name["r1"]["run_dir"] == os.path.join(fx["lab"], "runs", "r1"), "run_dir is absolute")
        check(labruns.list_all_runs(os.path.join(root, "nope")) == [], "missing inprocess root scans to []")

        # A lab whose lab.json says detect labels its runs detect.
        write(os.path.join(root, "_models", "labD", "lab.json"), json.dumps({"task": "detect"}))
        os.makedirs(os.path.join(root, "_models", "labD", "runs", "d1"), exist_ok=True)
        d1 = {r["run_name"]: r for r in labruns.list_all_runs(root)}["d1"]
        check(d1["task"] == "detect" and d1["origin"] == "lab:labD", f"detect lab run labeled (got {d1})")
        check(not any(r["run_name"] == "_inference" for r in labruns.list_all_runs(root)),
              "the _inference bucket is never scanned as a lab")

        # ---- champion write ----------------------------------------------
        saved = labruns.save_champion(root, os.path.join(fx["lab"], "runs", "r1"))
        check(saved["run_dir"] == os.path.join(fx["lab"], "runs", "r1") and saved["at"],
              f"save_champion writes run_dir + at (got {saved})")
        check(labruns.load_champion(root)["run_dir"] == saved["run_dir"], "load_champion reads it back")
        check(labruns.newest_weights(root, "labA", "seg")
              == os.path.join(fx["lab"], "runs", "r1", "weights", "best.pt"),
              "newest_weights continues from the champion when its task matches")
        check(labruns.newest_weights(root, "labD", "detect") == "",
              "newest_weights never hands a segmentation champion to a detection run")

        # ---- rounds ------------------------------------------------------
        rows = labruns.read_rounds(os.path.join(fx["lab"], "rounds.csv"))
        check(len(rows) == 1 and rows[0]["run_name"] == "r1" and rows[0]["map50_M"] == "0.62",
              f"read_rounds returns the csv rows (got {rows})")
        check(labruns.read_rounds(os.path.join(fx["lab"], "nope.csv")) == [], "read_rounds on a missing file is []")

        # ---- run dir + dataset guards ------------------------------------
        good_run = os.path.join(fx["lab"], "runs", "r1")
        path, err = labruns.validate_run_dir(good_run)
        check(not err and path == good_run, f"run dir with weights accepted (got {err!r})")
        _p, err = labruns.validate_run_dir(os.path.join(fx["proj"], "runs", "p1"))
        check("best.pt" in err, f"run dir without weights refused (got {err!r})")
        _p, err = labruns.validate_run_dir("")
        check(bool(err), "empty run dir refused")
        check(labruns.dataset_for_run(good_run) == os.path.join(fx["lab"], "dataset"),
              "dataset_for_run finds the lab dataset next to runs/")
        check(labruns.dataset_for_run(os.path.join(root, "_models", "labD", "runs", "d1")) == "",
              "dataset_for_run is empty when the run has no adjacent dataset")
        check(labruns.task_for_run_dir(good_run) == "seg"
              and labruns.task_for_run_dir(os.path.join(root, "_models", "labD", "runs", "d1")) == "detect",
              "task_for_run_dir reads one run's task without scanning every root")

        # ---- inference source guard ---------------------------------------
        ok_dir = os.path.join(outside, "photos")
        os.makedirs(ok_dir, exist_ok=True)
        path, err = labruns.validate_infer_source(root, ok_dir)
        check(not err and path == ok_dir, f"a plain folder is an acceptable inference source (got {err!r})")
        _p, err = labruns.validate_infer_source(root, os.path.join(outside, "missing"))
        check(bool(err), "a missing inference source is refused")
        _p, err = labruns.validate_infer_source(root, os.path.join(fx["proj"], "runs"))
        check("project" in err.lower(), f"a project step dir is refused (got {err!r})")
        _p, err = labruns.validate_infer_source(
            root, os.path.join(root, "run_26-08-26_LO", "step4test_combinedAnnotate"))
        check(bool(err), "any step* dir of a project run is refused")
        path, err = labruns.validate_infer_source(root, os.path.join(fx["comp"], "all_images"))
        check(not err, f"a composite image folder stays usable (got {err!r})")

        # ---- base model choice ---------------------------------------------
        train_dir = os.path.join(outside, "train")
        write(os.path.join(train_dir, "yolo11m-seg.pt"), "w")
        check(labruns.base_model_for("seg", "m", train_dir) == os.path.join(train_dir, "yolo11m-seg.pt"),
              "a weights file already in the training dir is used by absolute path")
        check(labruns.base_model_for("seg", "x", train_dir) == "yolo11x-seg.pt",
              "a weights file not on disk is passed by name for ultralytics to fetch")
        check(labruns.base_model_for("detect", "n", train_dir) == "yolo11n.pt",
              "a detection source trains a detection model, not a seg one")

        # ---- detection dataset build -----------------------------------
        # A staged bbox dataset has no all_images/all_labels, so pinned_split
        # cannot read it: the lab builds the YOLO split itself, in the same
        # on-disk shape pinned_split produces (train/valid/test with a
        # data.yaml whose val points at valid/images).
        check(labruns.read_classes_txt(os.path.join(ds, "classes.txt")) == ["AAGA", "ACER"],
              "read_classes_txt reads the pinned class list")
        check(labruns.bbox_label_dir(ds) == os.path.join(ds, "labels"),
              "bbox_label_dir finds labels/")
        os.makedirs(os.path.join(ds, "labels_localized"), exist_ok=True)
        write(os.path.join(ds, "labels_localized", "img_000.txt"), "0 0.5 0.5 0.2 0.3\n")
        check(labruns.bbox_label_dir(ds) == os.path.join(ds, "labels_localized"),
              "labels_localized wins over labels when it is present")
        shutil.rmtree(os.path.join(ds, "labels_localized"))

        out = os.path.join(root, "_models", "boxlab", "dataset")
        summary = labruns.build_detect_dataset(ds, out, "boxlab")
        check(summary["n_pairs"] == 20 and summary["n_unlabelled"] == 1,
              f"every labelled image is a pair, the unlabelled one is skipped (got {summary})")
        counts = {s: len(os.listdir(os.path.join(out, s, "images"))) for s in ("train", "valid", "test")}
        check(sum(counts.values()) == 20 and counts["valid"] == 2 and counts["test"] == 2,
              f"80/10/10 across train, valid and test (got {counts})")
        for s in ("train", "valid", "test"):
            imgs = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, s, "images")))
            lbls = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, s, "labels")))
            check(imgs == lbls, f"{s}/ carries one label per image")
        import yaml as _yaml
        doc = _yaml.safe_load(open(os.path.join(out, "data.yaml")))
        check(doc["train"] == "train/images" and doc["val"] == "valid/images"
              and doc["test"] == "test/images" and doc["path"] == os.path.abspath(out),
              f"data.yaml uses the same split paths pinned_split writes (got {doc})")
        check(doc["names"] == {0: "AAGA", 1: "ACER"}, f"data.yaml names come from classes.txt (got {doc['names']})")
        check(os.path.isfile(os.path.join(out, "test.yaml")), "test.yaml written for parity with pinned_split")
        manifest = json.load(open(os.path.join(out, "split_manifest.json")))
        check(manifest["holdout_mode"] == "lab-random" and len(manifest["pins"]) == 20,
              f"split_manifest records every pin and is honest about the holdout (got {manifest.get('holdout_mode')})")

        # Deterministic on the lab name, and existing pins never move.
        out2 = os.path.join(root, "_models", "boxlab2", "dataset")
        labruns.build_detect_dataset(ds, out2, "boxlab")
        m2 = json.load(open(os.path.join(out2, "split_manifest.json")))
        check(m2["pins"] == manifest["pins"], "the same lab name splits the same images the same way")
        out3 = os.path.join(root, "_models", "boxlab3", "dataset")
        labruns.build_detect_dataset(ds, out3, "otherlab")
        m3 = json.load(open(os.path.join(out3, "split_manifest.json")))
        check(m3["pins"] != manifest["pins"], "a different lab name gets its own split")
        write(os.path.join(ds, "images", "img_999.jpg"), "x")
        write(os.path.join(ds, "labels", "img_999.txt"), "1 0.5 0.5 0.2 0.3\n")
        labruns.build_detect_dataset(ds, out, "boxlab")
        m4 = json.load(open(os.path.join(out, "split_manifest.json")))
        check(all(m4["pins"][k] == v for k, v in manifest["pins"].items()) and "img_999.jpg" in m4["pins"],
              "a rebuild replays every existing pin and only assigns the new image")
        os.remove(os.path.join(ds, "images", "img_999.jpg"))
        os.remove(os.path.join(ds, "labels", "img_999.txt"))
        try:
            labruns.build_detect_dataset(os.path.join(bbox_root, "partial"),
                                         os.path.join(root, "_models", "nope", "dataset"), "nope")
            check(False, "a source with no image/label pairs is refused")
        except ValueError as e:
            check("pair" in str(e).lower() or "image" in str(e).lower(),
                  f"a source with no image/label pairs is refused ({e})")

        # ---- detection mAP column ----------------------------------------
        # A detection run has no mask mAP: the ledger fills map50_B instead.
        det_lab = os.path.join(root, "_models", "labD")
        write(os.path.join(det_lab, "rounds.csv"),
              ROUNDS_HEADER + f"1,2026-09-01,d1,{os.path.join(det_lab, 'runs', 'd1')},yolo11n.pt,"
              "no,10,2,2,,,,,0.48,0.29,unpinned,unpinned,unpinned,\n")
        d1row = {r["run_name"]: r for r in labruns.list_all_runs(root)}["d1"]
        check(d1row["map50"] == 0.48,
              f"a detection run scores from the box mAP column (got {d1row['map50']!r})")

        # ---- note text -------------------------------------------------
        check(labruns.safe_note("a b/../c") == "a_b_.._c", f"safe_note flattens separators (got {labruns.safe_note('a b/../c')!r})")
        check(labruns.safe_note("") == "infer", "an empty note falls back to a fixed leaf")
        out = labruns.inference_out_dir(root, "reef check", ts="20260901_120000")
        check(out == os.path.join(root, "_models", "_inference", "20260901_120000_reef_check"),
              f"inference output folder is timestamped under _models/_inference (got {out})")

        # ---- T5: the shared library the VEGA main window imports ----------
        test_validate_run_dir_refuses_a_path_outside_the_root(root, outside)
        test_validate_run_dir_refuses_a_symlink_escape(root, outside)
        test_library_imports_without_flask()
        test_split_fraction_defaults_come_from_the_builder(root)
        test_mirror_champion_writes_only_inside_a_models_lab(root, outside, fx)
        test_task_comes_from_args_yaml_not_the_filename(root)
        test_task_falls_back_to_the_filename_when_absent(root)
        test_both_champion_shapes_are_readable(root, fx)
        test_new_champion_shape_carries_its_family_score(root, fx)
        test_comparability_requires_same_family_and_same_split(root)
        test_split_fingerprint_is_recorded_on_the_run(root)
        test_adversarial_inputs(root, fx)
    finally:
        for d in (root, bbox_root, outside):
            shutil.rmtree(d, ignore_errors=True)
    check(not os.path.exists(root) and not os.path.exists(bbox_root) and not os.path.exists(outside),
          "temp fixtures removed")

    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
