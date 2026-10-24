"""The Trained on identity on every /api/step/6/all_runs row (2026-09-17, LO).

Lauren: "maybe just make a table that includes the training data subset (so we
can filter by those)". The runs card renders trained_on and filters by it, so
the route must name the data each run learned from, and must never guess it
from a folder name. This suite pins the derivation order the card's contract
states, and the provenance fields the imported marker draws from.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_all_runs_trained_on.py

Derivation order, first hit wins:
  lab run      lab.json "source" (last folder) with "source_kind"
  project run  project.json training_source.name (kind bbox)
               project.json composite.name (kind composite)
               step6 dataset/split_manifest.json label_dir under
                 bbox_datasets/<name>/ (kind bbox)
               a step4test_combinedAnnotate export: the project's name
                 (kind export)
  otherwise    "" and "" (the card shows "not recorded")
Provenance: origin_kind, imported_from, imported_at from the run's
provenance.json, else from lab.json when the run is listed in imported_runs.

Every fixture lives under one temp folder removed at the end, and the removal
is verified. subprocess.run and subprocess.Popen are stubbed at module level.
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

_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()
_REAL_INPROCESS_ROOT = appmod._inprocess_root


def check(cond, msg):
    """Record one assertion outcome."""
    _RESULTS.append((msg, bool(cond), msg))
    print(("PASS " if cond else "FAIL ") + msg)


def run(fn):
    """Run one test function, recording an exception as a failure."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the suite reports, never hides
        import traceback
        _RESULTS.append((fn.__name__, False, traceback.format_exc()))
        print(f"FAIL {fn.__name__}: {exc}")


def _write(path, data):
    """Write data as JSON (or raw text when data is a str), creating folders."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(data, str):
            fh.write(data)
        else:
            json.dump(data, fh)


def _root():
    """A fresh fake inprocess root, registered for removal."""
    tmp = tempfile.mkdtemp(prefix="trained_on_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    os.makedirs(os.path.join(root, "_models"), exist_ok=True)
    return root


def _run_folder(parent_runs, name):
    """A run folder with weights, returning its path."""
    run_dir = os.path.join(parent_runs, name)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    _write(os.path.join(run_dir, "weights", "best.pt"), "x")
    return run_dir


def _project(root, folder, name, **project_fields):
    """A project with a step6 runs folder; returns (project_dir, runs_dir)."""
    project_dir = os.path.join(root, folder)
    step6 = os.path.join(project_dir, "step6_trainModel")
    runs = os.path.join(step6, "runs")
    os.makedirs(runs, exist_ok=True)
    state = {"name": name, "project_dir": project_dir, "current_step": 6,
             "steps": {"6": {"name": "trainModel", "status": "completed", "dir": step6}}}
    state.update(project_fields)
    _write(os.path.join(project_dir, "project.json"), state)
    return project_dir, runs


def _entry(run_dir, origin):
    """A labruns.list_all_runs-shaped entry."""
    return {"run_dir": run_dir, "run_name": os.path.basename(run_dir), "origin": origin,
            "task": "detect", "has_weights": True, "champion": False}


def _row(run_dir, origin):
    return appmod._all_runs_row(_entry(run_dir, origin), "", {})


# ---- project runs -----------------------------------------------------------

def test_bbox_project_names_its_staged_dataset():
    root = _root()
    _, runs = _project(root, "run_26-09-15_LO_train_coral_intel_subset", "TRAIN_coral_intel_subset",
                       training_source={"kind": "bbox", "name": "coral_intel_subset",
                                        "path": "/x/bbox_datasets/coral_intel_subset"})
    r = _row(_run_folder(runs, "TRAIN_coral_intel_subset_20260915"), "project:run_26-09-15_LO_train_coral_intel_subset")
    check(r["trained_on"] == "coral_intel_subset", f"bbox project: trained_on is the dataset name, got {r['trained_on']!r}")
    check(r["trained_on_kind"] == "bbox", f"bbox project: kind bbox, got {r['trained_on_kind']!r}")


def test_composite_project_names_its_training_set():
    root = _root()
    _, runs = _project(root, "run_26-09-02_LO_train_qs", "TRAIN_qs",
                       composite={"name": "qs20260902_pa_ss", "path": "inprocess/_composite/qs20260902_pa_ss"})
    r = _row(_run_folder(runs, "TRAIN_qs_1"), "project:run_26-09-02_LO_train_qs")
    check(r["trained_on"] == "qs20260902_pa_ss", f"composite project: training set name, got {r['trained_on']!r}")
    check(r["trained_on_kind"] == "composite", f"composite project: kind composite, got {r['trained_on_kind']!r}")


def test_project_with_no_source_field_reads_the_split_manifest():
    root = _root()
    project_dir, runs = _project(root, "run_26-09-16_LO_fish_ablation_backend", "TRAIN_fish_ablation",
                                 training_source=None)
    _write(os.path.join(project_dir, "step6_trainModel", "dataset", "split_manifest.json"),
           {"label_dir": "supporting_data/bbox_datasets/fish_photo_28feb25/labels", "pins": {}})
    r = _row(_run_folder(runs, "TRAIN_fish_A1"), "project:run_26-09-16_LO_fish_ablation_backend")
    check(r["trained_on"] == "fish_photo_28feb25", f"manifest label_dir under bbox_datasets names the dataset, got {r['trained_on']!r}")
    check(r["trained_on_kind"] == "bbox", "manifest-derived kind is bbox")


def test_manifest_label_dir_outside_bbox_datasets_is_not_a_guess():
    root = _root()
    project_dir, runs = _project(root, "run_x", "X", training_source=None)
    _write(os.path.join(project_dir, "step6_trainModel", "dataset", "split_manifest.json"),
           {"label_dir": "/somewhere/else/labels"})
    r = _row(_run_folder(runs, "R"), "project:run_x")
    check(r["trained_on"] == "" and r["trained_on_kind"] == "",
          f"a label_dir not under bbox_datasets yields no name, got {r['trained_on']!r}/{r['trained_on_kind']!r}")


def test_ordinary_export_project_names_itself():
    root = _root()
    project_dir, runs = _project(root, "run_18aprilpm_PA_20260418", "18aprilpm_PA", training_source=None)
    os.makedirs(os.path.join(project_dir, "step4test_combinedAnnotate"), exist_ok=True)
    r = _row(_run_folder(runs, "18apr_1"), "project:run_18aprilpm_PA_20260418")
    check(r["trained_on"] == "18aprilpm_PA", f"an annotation project trains on its own export, named by the project, got {r['trained_on']!r}")
    check(r["trained_on_kind"] == "export", "export kind")


def test_project_with_nothing_reads_not_recorded():
    root = _root()
    _, runs = _project(root, "run_bare", "BARE", training_source=None)
    r = _row(_run_folder(runs, "R"), "project:run_bare")
    check(r["trained_on"] == "" and r["trained_on_kind"] == "", "no source of any kind: empty, never a guess from the folder name")


def test_training_source_wins_over_manifest_and_export():
    root = _root()
    project_dir, runs = _project(root, "run_both", "BOTH",
                                 training_source={"kind": "bbox", "name": "tutorial_boxes", "path": "/x"},
                                 composite={"name": "should_lose"})
    _write(os.path.join(project_dir, "step6_trainModel", "dataset", "split_manifest.json"),
           {"label_dir": "supporting_data/bbox_datasets/other/labels"})
    os.makedirs(os.path.join(project_dir, "step4test_combinedAnnotate"), exist_ok=True)
    r = _row(_run_folder(runs, "R"), "project:run_both")
    check(r["trained_on"] == "tutorial_boxes" and r["trained_on_kind"] == "bbox", "training_source is read first")


# ---- lab runs ---------------------------------------------------------------

def _lab(root, name, **lab_fields):
    lab_dir = os.path.join(root, "_models", name)
    runs = os.path.join(lab_dir, "runs")
    os.makedirs(runs, exist_ok=True)
    lab = {"name": name, "lab_name": name, "task": "detect"}
    lab.update(lab_fields)
    _write(os.path.join(lab_dir, "lab.json"), lab)
    return lab_dir, runs


def test_lab_run_names_its_source_dataset():
    root = _root()
    _, runs = _lab(root, "fish_photo_28feb25_paper",
                   source="/mnt/x/supporting_data/bbox_datasets/fish_photo_28feb25", source_kind="bbox")
    r = _row(_run_folder(runs, "run3_heavy_augment"), "lab:fish_photo_28feb25_paper")
    check(r["trained_on"] == "fish_photo_28feb25", f"lab run: last folder of lab.json source, got {r['trained_on']!r}")
    check(r["trained_on_kind"] == "bbox", "lab run: source_kind")


def test_lab_run_source_with_trailing_slash_and_unknown_kind():
    root = _root()
    _, runs = _lab(root, "L", source="/a/b/coral_intel_det_260902/", source_kind="")
    r = _row(_run_folder(runs, "R"), "lab:L")
    check(r["trained_on"] == "coral_intel_det_260902", f"trailing slash tolerated, got {r['trained_on']!r}")
    check(r["trained_on_kind"] == "", "an unrecorded source_kind stays empty rather than invented")


def test_lab_without_source_reads_not_recorded():
    root = _root()
    _, runs = _lab(root, "L")
    r = _row(_run_folder(runs, "R"), "lab:L")
    check(r["trained_on"] == "" and r["trained_on_kind"] == "", "a lab.json with no source names nothing")


# ---- provenance -------------------------------------------------------------

def test_imported_run_forwards_its_provenance():
    root = _root()
    _, runs = _lab(root, "L", source="/x/fish", source_kind="bbox", origin_kind="imported",
                   imported_runs=["run3"])
    run_dir = _run_folder(runs, "run3")
    _write(os.path.join(run_dir, "provenance.json"),
           {"imported_from": "/home/bizon/Desktop/ai/oceankind_CV/run3", "imported_at": "2026-09-15 02:37:02"})
    r = _row(run_dir, "lab:L")
    check(r["origin_kind"] == "imported", f"origin_kind from provenance, got {r['origin_kind']!r}")
    check(r["imported_from"] == "/home/bizon/Desktop/ai/oceankind_CV/run3", "imported_from forwarded")
    check(r["imported_at"] == "2026-09-15 02:37:02", "imported_at forwarded")


def test_imported_run_without_provenance_file_falls_back_to_lab_json():
    root = _root()
    _, runs = _lab(root, "L", source="/x/fish", origin_kind="imported", imported_runs=["run4"],
                   created_at="2026-09-15 02:37:02")
    r = _row(_run_folder(runs, "run4"), "lab:L")
    check(r["origin_kind"] == "imported", "listed in lab.json imported_runs counts as imported")
    check(r["imported_from"] == "" and r["imported_at"] == "2026-09-15 02:37:02",
          f"no provenance.json: from empty, at from the lab record, got {r['imported_from']!r}/{r['imported_at']!r}")


def test_native_run_has_empty_provenance():
    root = _root()
    _, runs = _project(root, "run_p", "P", training_source={"kind": "bbox", "name": "d", "path": "/x"})
    r = _row(_run_folder(runs, "R"), "project:run_p")
    check(r["origin_kind"] == "" and r["imported_from"] == "" and r["imported_at"] == "", "a native run carries empty provenance fields")


# ---- adversarial ------------------------------------------------------------

def test_corrupt_json_yields_not_recorded_and_no_traceback():
    root = _root()
    project_dir, runs = _project(root, "run_c", "C")
    _write(os.path.join(project_dir, "project.json"), "{not json")
    _write(os.path.join(project_dir, "step6_trainModel", "dataset", "split_manifest.json"), "\x00\x01garbage")
    run_dir = _run_folder(runs, "R")
    _write(os.path.join(run_dir, "provenance.json"), "[1,2")
    r = _row(run_dir, "project:run_c")
    check(r["trained_on"] == "" and r["trained_on_kind"] == "" and r["origin_kind"] == "",
          "corrupt project.json, manifest and provenance all read as not recorded")


def test_wrong_types_in_the_records_are_tolerated():
    root = _root()
    _, runs = _project(root, "run_t", "T", training_source="a string, not a dict", composite=42)
    r = _row(_run_folder(runs, "R"), "project:run_t")
    check(r["trained_on"] == "" and r["trained_on_kind"] == "", "training_source and composite of the wrong type are ignored")
    _, lruns = _lab(root, "L", source=123, source_kind=["bbox"], imported_runs="run")
    r2 = _row(_run_folder(lruns, "run"), "lab:L")
    check(r2["trained_on"] == "" and r2["origin_kind"] == "", "a non-string source and a non-list imported_runs are ignored")


def test_hostile_names_are_passed_through_verbatim_as_data():
    root = _root()
    hostile = '<img src=x onerror=alert(1)>"; rm -rf /'
    _, runs = _project(root, "run_h", "H", training_source={"kind": "bbox", "name": hostile, "path": "/x"})
    r = _row(_run_folder(runs, "R"), "project:run_h")
    check(r["trained_on"] == hostile, "the route returns the recorded name unchanged; escaping is the renderer's job")


def test_huge_name_is_not_truncated_by_the_route():
    root = _root()
    big = "d" * 5000
    _, runs = _project(root, "run_big", "B", training_source={"kind": "bbox", "name": big, "path": "/x"})
    r = _row(_run_folder(runs, "R"), "project:run_big")
    check(r["trained_on"] == big, "the route never truncates; the card decides how to show a long name")


def test_the_route_carries_the_new_keys_on_every_row():
    root = _root()
    _, runs = _project(root, "run_r", "R", training_source={"kind": "bbox", "name": "coral_intel", "path": "/x"})
    _run_folder(runs, "R1")
    _, lruns = _lab(root, "L", source="/x/fish", source_kind="bbox")
    _run_folder(lruns, "L1")
    appmod._inprocess_root = lambda: root
    try:
        client = appmod.create_app().test_client()
        rows = (client.get("/api/step/6/all_runs").get_json() or {}).get("runs") or []
    finally:
        appmod._inprocess_root = _REAL_INPROCESS_ROOT
    keys = {"trained_on", "trained_on_kind", "origin_kind", "imported_from", "imported_at"}
    check(len(rows) == 2, f"two rows listed, got {len(rows)}")
    check(all(keys <= set(r) for r in rows), "every row carries the five new keys")
    by = {r["run_name"]: r for r in rows}
    check(by.get("R1", {}).get("trained_on") == "coral_intel" and by.get("L1", {}).get("trained_on") == "fish",
          f"the route resolves both a project and a lab run: {[(r['run_name'], r['trained_on']) for r in rows]}")


def test_a_deeply_nested_record_cannot_take_the_table_down():
    root = _root()
    project_dir, runs = _project(root, "run_deep", "D")
    _write(os.path.join(project_dir, "project.json"), "[" * 200000 + "]" * 200000)
    r = _row(_run_folder(runs, "R"), "project:run_deep")
    check(r["trained_on"] == "" and r["trained_on_kind"] == "", "a hostile deeply nested project.json reads as not recorded rather than raising")


def test_export_folder_with_no_recorded_name_is_not_named_from_the_folder():
    root = _root()
    project_dir, runs = _project(root, "run_folder_only", "")
    os.makedirs(os.path.join(project_dir, "step4test_combinedAnnotate"), exist_ok=True)
    r = _row(_run_folder(runs, "R"), "project:run_folder_only")
    check(r["trained_on"] == "" and r["trained_on_kind"] == "",
          f"an export project whose record carries no name is never named from its folder, got {r['trained_on']!r}")


def test_the_innermost_bbox_datasets_segment_wins():
    check(appmod._dataset_from_bbox_path("/clone/bbox_datasets/backup/supporting_data/bbox_datasets/fish/labels") == "fish",
          "a clone path that itself contains bbox_datasets cannot misname the run")
    check(appmod._dataset_from_bbox_path("/x/old_bbox_datasets_copy/fish/labels") == "", "only a whole bbox_datasets segment counts")


def test_an_unexpected_origin_is_refused_at_entry():
    root = _root()
    _, runs = _lab(root, "L", source="/x/fish", source_kind="bbox")
    run_dir = _run_folder(runs, "R")
    refused = 0
    for fn in (appmod._trained_on_for_run, appmod._provenance_for_run):
        try:
            fn(run_dir, "weird")
        except ValueError as exc:
            refused += 1 if "origin" in str(exc) else 0
    check(refused == 2, "both helpers refuse an origin that is neither lab nor project, naming the field")


def test_provenance_file_without_import_keys_reads_native():
    root = _root()
    _, runs = _lab(root, "L", source="/x/fish", source_kind="bbox", imported_runs=[])
    run_dir = _run_folder(runs, "R")
    _write(os.path.join(run_dir, "provenance.json"), {"note": "trained here", "best_pt_sha256": "abc"})
    r = _row(run_dir, "lab:L")
    check(r["origin_kind"] == "" and r["imported_from"] == "" and r["imported_at"] == "",
          "a provenance.json with no import fields, on a run the lab does not list as imported, is native")


def test_a_run_trained_from_a_chosen_session_or_training_set_names_it():
    root = _root()
    standalone = os.path.join(root, "_standalone"); composite = os.path.join(root, "_composite")
    session = os.path.join(standalone, "26-09-17_LO_reef"); os.makedirs(session)
    tset = os.path.join(composite, "qs_pa_ss"); os.makedirs(tset)
    saved = (appmod._standalone_root, appmod._composite_root)
    appmod._standalone_root = lambda: standalone; appmod._composite_root = lambda: composite
    try:
        _, runs = _project(root, "run_a", "A", training_source=None,
                           steps={"6": {"name": "trainModel", "status": "completed", "config": {"source_dir": session}}})
        os.makedirs(os.path.join(root, "run_a", "step4test_combinedAnnotate"))
        ra = _row(_run_folder(runs, "R"), "project:run_a")
        _, runs2 = _project(root, "run_b", "B", training_source=None,
                            steps={"6": {"name": "trainModel", "status": "completed", "config": {"source_dir": tset}}})
        rb = _row(_run_folder(runs2, "R"), "project:run_b")
    finally:
        appmod._standalone_root, appmod._composite_root = saved
    check(ra["trained_on"] == "26-09-17_LO_reef" and ra["trained_on_kind"] == "standalone",
          f"a run trained from a chosen session names the session, not the project's own export: {ra['trained_on']!r}/{ra['trained_on_kind']!r}")
    check(rb["trained_on"] == "qs_pa_ss" and rb["trained_on_kind"] == "composite",
          f"a run trained from a chosen training set names it: {rb['trained_on']!r}/{rb['trained_on_kind']!r}")


if __name__ == "__main__":
    for fn in [
        test_bbox_project_names_its_staged_dataset,
        test_composite_project_names_its_training_set,
        test_project_with_no_source_field_reads_the_split_manifest,
        test_manifest_label_dir_outside_bbox_datasets_is_not_a_guess,
        test_ordinary_export_project_names_itself,
        test_project_with_nothing_reads_not_recorded,
        test_training_source_wins_over_manifest_and_export,
        test_lab_run_names_its_source_dataset,
        test_lab_run_source_with_trailing_slash_and_unknown_kind,
        test_lab_without_source_reads_not_recorded,
        test_imported_run_forwards_its_provenance,
        test_imported_run_without_provenance_file_falls_back_to_lab_json,
        test_native_run_has_empty_provenance,
        test_corrupt_json_yields_not_recorded_and_no_traceback,
        test_wrong_types_in_the_records_are_tolerated,
        test_hostile_names_are_passed_through_verbatim_as_data,
        test_huge_name_is_not_truncated_by_the_route,
        test_the_route_carries_the_new_keys_on_every_row,
        test_a_deeply_nested_record_cannot_take_the_table_down,
        test_export_folder_with_no_recorded_name_is_not_named_from_the_folder,
        test_the_innermost_bbox_datasets_segment_wins,
        test_an_unexpected_origin_is_refused_at_entry,
        test_provenance_file_without_import_keys_reads_native,
        test_a_run_trained_from_a_chosen_session_or_training_set_names_it,
    ]:
        run(fn)
    for p in _FIXTURES:
        shutil.rmtree(os.path.dirname(p) if p.endswith("inprocess") else p, ignore_errors=True)
    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    if leftovers:
        print(f"\nFIXTURES NOT REMOVED: {leftovers}")
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed; fixtures removed: {not leftovers}")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{str(detail)[:300]}")
    sys.exit(1 if failed or leftovers else 0)
