"""A moved-aside export is never read as live data.

Images are never deleted: resets and re-routes move folders to
<project>/_superseded/<stamp>/. A recursive reader that walks a whole project
would then count the moved copy beside the live one. Each reader here is run on
a project holding a live export and a moved-aside export, and must see only the
live one: _matrix.builder._ledger_files, the builder's legacy segmentation
discovery, _matrix.cellinfo.segmentation_files and load_project_segmentations
(which the composite builder and the screening ledger use), and
check_project_integrity.stray_tmp_files.

Run: env/bin/python scripts/_matrix/tests/test_superseded_readers.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _SCRIPTS)
sys.dont_write_bytecode = True

from _matrix import builder, cellinfo  # noqa: E402
import check_project_integrity as cic  # noqa: E402

_R = []
_TMP = []
EXPORT = "step4test_combinedAnnotate"


def check(c, m):
    if not c:
        raise AssertionError(m)


def run(fn):
    try:
        fn()
        _R.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001
        _R.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _export(base, frame):
    seg = os.path.join(base, EXPORT, "segmentations")
    os.makedirs(seg)
    with open(os.path.join(seg, "segmentations.json"), "w") as fh:
        json.dump({frame: {"masks": [], "reviewed": True}}, fh)
    with open(os.path.join(base, EXPORT, "label_provenance.csv"), "w") as fh:
        fh.write("basename,label,outcome,source,reviewer,at,project_id\n")
    nested = os.path.join(seg, "2024")
    os.makedirs(nested)
    with open(os.path.join(nested, "segmentations.json"), "w") as fh:
        json.dump({frame + "_n": {"masks": []}}, fh)
    old = os.path.join(base, EXPORT, "old.tmp")
    with open(old, "w") as fh:
        fh.write("x")
    os.utime(old, (time.time() - 100000,) * 2)


def _project():
    d = tempfile.mkdtemp(prefix="sup_rd_")
    _TMP.append(d)
    _export(d, "live.jpg")
    _export(os.path.join(d, "_superseded", "20261015-101010"), "aside.jpg")
    return d


def _under_aside(paths):
    return [p for p in paths if "_superseded" in p.split(os.sep)]


def test_ledger_files():
    got = builder._ledger_files(_project())
    check(len(got) == 1 and not _under_aside(got), f"ledger files: {got}")


def test_segmentation_discovery_in_the_builder():
    rows = builder._legacy_rows(_project(), ["PA"], set())
    names = {r["basename"] for r in rows}
    check(any("live" in n for n in names), f"live frames are derived: {names}")
    check(not any("aside" in n for n in names), f"a moved-aside frame was derived: {names}")


def test_cellinfo_segmentation_files():
    got = cellinfo.segmentation_files(_project())
    check(len(got) == 2 and not _under_aside(got), f"segmentation files: {got}")


def test_load_project_segmentations_sees_only_live_frames():
    segs = cellinfo.load_project_segmentations(_project())
    text = json.dumps(segs, default=str)
    check("live.jpg" in text and "aside.jpg" not in text, "only the live frames are loaded")


def test_integrity_stray_tmp_walk():
    d = _project()
    got = cic.stray_tmp_files(d, time.time())
    check(len(got) == 1 and not _under_aside(got), f"stray tmp files: {got}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            run(fn)
    for t in _TMP:
        shutil.rmtree(t, ignore_errors=True)
    left = [t for t in _TMP if os.path.exists(t)]
    print("fixtures removed" if not left else f"fixtures NOT removed: {left}")
    ok = sum(1 for _, p, _ in _R if p)
    for n, p, det in _R:
        if not p:
            print(f"\n--- {n}\n{det}")
    print(f"\n{ok}/{len(_R)} passed")
    sys.exit(0 if ok == len(_R) and not left else 1)
