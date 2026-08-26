"""
Task 4: dual-pattern segmentation readers find flat segmentations.json.

NO-pytest harness (mirrors test_step4test_relabel.py). Run with:
    env/bin/python scripts/_expertids/tests/test_flat_segmentations.py

Covers:
  UNIT  _iter_segmentation_files returns the flat segmentations/segmentations.json
        when present (step4test flat layout).
  UNIT  _iter_segmentation_files returns the year-nested file when only that
        exists (legacy step5 layout).
"""

import os
import sys
import json
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)            # scripts/_expertids
_SCRIPTS = os.path.dirname(_PKG)         # scripts/
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, 'pipeline_orchestrator'))

from _expertids import importer as imp  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print(f"  FAIL: {msg}")


def test_iter_finds_flat():
    """Flat layout: segmentations/segmentations.json is returned directly."""
    with tempfile.TemporaryDirectory() as t:
        d = os.path.join(t, "step4test_combinedAnnotate")
        seg_dir = os.path.join(d, "segmentations")
        os.makedirs(seg_dir, exist_ok=True)
        flat_path = os.path.join(seg_dir, "segmentations.json")
        with open(flat_path, "w") as f:
            json.dump({"x.jpeg": {}}, f)
        files = imp._iter_segmentation_files(d)
        check(files == [flat_path],
              f"flat: expected [{flat_path!r}], got {files!r}")
    print("  PASS test_iter_finds_flat")


def test_iter_finds_year_nested():
    """Year-nested layout: segmentations/<year>/segmentations.json is found."""
    with tempfile.TemporaryDirectory() as t:
        d = os.path.join(t, "step5")
        nested_dir = os.path.join(d, "segmentations", "2014")
        os.makedirs(nested_dir, exist_ok=True)
        nested_path = os.path.join(nested_dir, "segmentations.json")
        with open(nested_path, "w") as f:
            json.dump({"y.jpeg": {}}, f)
        files = imp._iter_segmentation_files(d)
        check(bool(files), "year-nested: no files returned")
        check(files and files[0].endswith(os.path.join("2014", "segmentations.json")),
              f"year-nested: expected path ending in 2014/segmentations.json, got {files!r}")
    print("  PASS test_iter_finds_year_nested")


def test_iter_flat_takes_precedence_over_nested():
    """When BOTH layouts are present, flat is returned (step4test migration scenario)."""
    with tempfile.TemporaryDirectory() as t:
        d = os.path.join(t, "step4test_combinedAnnotate")
        seg_dir = os.path.join(d, "segmentations")
        os.makedirs(seg_dir, exist_ok=True)
        flat_path = os.path.join(seg_dir, "segmentations.json")
        with open(flat_path, "w") as f:
            json.dump({"x.jpeg": {}}, f)
        nested_dir = os.path.join(seg_dir, "2020")
        os.makedirs(nested_dir, exist_ok=True)
        with open(os.path.join(nested_dir, "segmentations.json"), "w") as f:
            json.dump({"old.jpeg": {}}, f)
        files = imp._iter_segmentation_files(d)
        check(files == [flat_path],
              f"flat-priority: expected only [{flat_path!r}], got {files!r}")
    print("  PASS test_iter_flat_takes_precedence_over_nested")


if __name__ == "__main__":
    print("TASK 4 — dual-pattern segmentation readers\n")
    print("UNIT:")
    test_iter_finds_flat()
    test_iter_finds_year_nested()
    test_iter_flat_takes_precedence_over_nested()
    print()
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
