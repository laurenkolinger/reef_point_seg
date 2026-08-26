# CA/tests/test_export_manifest.py
import os, sys, csv, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import app as A
_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1
d = tempfile.mkdtemp(prefix="man_")
A.write_export_manifest(d, [
    {"basename": "A", "n_masks": 2, "label_file": "A.txt", "image_file": "A.jpg", "outcome": "exported_with_masks"},
    {"basename": "B", "n_masks": 0, "label_file": "B.txt", "image_file": "B.jpg", "outcome": "exported_empty"},
    {"basename": "C", "n_masks": 1, "label_file": "", "image_file": "", "outcome": "review_only"},
])
with open(os.path.join(d, "export_manifest.csv")) as f:
    rows = list(csv.DictReader(f))
check(len(rows) == 3, f"3 manifest rows, got {len(rows)}")
by = {r["basename"]: r for r in rows}
check(by["B"]["outcome"] == "exported_empty", "negative recorded")
check(by["C"]["outcome"] == "review_only", "review-only recorded")
check(set(rows[0].keys()) >= {"basename","n_masks","label_file","image_file","outcome"}, "header complete")
# ── reconcile_export tests ────────────────────────────────────────
def _recon_case():
    import tempfile, os
    d = tempfile.mkdtemp(prefix="recon_")
    os.makedirs(os.path.join(d, "all_images")); os.makedirs(os.path.join(d, "all_labels"))
    # one positive on disk, one negative on disk
    for stem in ("A", "B"):
        open(os.path.join(d, "all_images", stem + ".jpg"), "w").close()
        open(os.path.join(d, "all_labels", stem + ".txt"), "w").close()
    entries = [
        {"basename": "A", "n_masks": 1, "label_file": "A.txt", "image_file": "A.jpg", "outcome": "exported_with_masks"},
        {"basename": "B", "n_masks": 0, "label_file": "B.txt", "image_file": "B.jpg", "outcome": "exported_empty"},
        {"basename": "C", "n_masks": 1, "label_file": "", "image_file": "", "outcome": "review_only"},
    ]
    return d, entries

_d, _e = _recon_case()
_res = A.reconcile_export(_d, _e)
check(_res["on_disk_images"] == 2 and _res["exported"] == 2, "reconcile counts match")
# missing-file case must raise
import os as _os
_os.remove(_os.path.join(_d, "all_labels", "B.txt"))
_raised = False
try:
    A.reconcile_export(_d, _e)
except AssertionError:
    _raised = True
check(_raised, "missing label file for a routed frame raises")

# ── image_missing reconcile test ─────────────────────────────────
# An entries list that includes an image_missing row must still reconcile:
# image_missing is excluded from the on-disk-exported count, so all_images
# on disk (2) must equal exported-outcome rows (2), not total entries (3).
def _recon_case_with_missing():
    import tempfile, os
    d = tempfile.mkdtemp(prefix="recon_miss_")
    os.makedirs(os.path.join(d, "all_images")); os.makedirs(os.path.join(d, "all_labels"))
    for stem in ("A", "B"):
        open(os.path.join(d, "all_images", stem + ".jpg"), "w").close()
        open(os.path.join(d, "all_labels", stem + ".txt"), "w").close()
    entries = [
        {"basename": "A", "n_masks": 1, "label_file": "A.txt", "image_file": "A.jpg", "outcome": "exported_with_masks"},
        {"basename": "B", "n_masks": 0, "label_file": "B.txt", "image_file": "B.jpg", "outcome": "exported_empty"},
        {"basename": "X", "n_masks": 0, "label_file": "", "image_file": "", "outcome": "image_missing"},
    ]
    return d, entries

_dm, _de = _recon_case_with_missing()
_res2 = A.reconcile_export(_dm, _de)
check(_res2["on_disk_images"] == 2 and _res2["exported"] == 2,
      f"image_missing row excluded from reconcile count (got {_res2})")

# preview_file is part of the manifest schema (Task A2).
check('preview_file' in A.EXPORT_MANIFEST_FIELDS, "preview_file in manifest fields")
import tempfile as _tf, csv as _csv2, os as _os2
_d2 = _tf.mkdtemp(prefix="man_prev_")
A.write_export_manifest(_d2, [
    {"basename": "A", "n_masks": 2, "label_file": "A.txt", "image_file": "A.jpg",
     "outcome": "exported_with_masks", "preview_file": "previews/A_seg.jpg"},
])
with open(_os2.path.join(_d2, "export_manifest.csv")) as _f2:
    _rows2 = list(_csv2.DictReader(_f2))
check(_rows2[0]["preview_file"] == "previews/A_seg.jpg", "preview_file persisted in manifest")

# ── batch_name + image_path + species columns ────────────────────
_d3 = tempfile.mkdtemp(prefix="man_batch_")
A.write_export_manifest(_d3, [{
    'basename': 'TCRMP2014_a', 'n_masks': 2, 'label_file': 'TCRMP2014_a.txt',
    'image_file': 'TCRMP2014_a.jpeg', 'outcome': 'exported_with_masks',
    'preview_file': 'previews/TCRMP2014_a_seg.jpg',
    'batch_name': 'batch_001_20260630-120000', 'image_path': '/abs/TCRMP2014_a.jpeg',
    'OFRA': 2}])
_r3 = A._read_export_manifest(_d3)
check('TCRMP2014_a' in _r3, "batch+species row present after write")
check(_r3['TCRMP2014_a'].get('batch_name') == 'batch_001_20260630-120000',
      f"batch_name round-trips (got {_r3['TCRMP2014_a'].get('batch_name')!r})")
check(_r3['TCRMP2014_a'].get('image_path') == '/abs/TCRMP2014_a.jpeg',
      f"image_path round-trips (got {_r3['TCRMP2014_a'].get('image_path')!r})")
check(_r3['TCRMP2014_a'].get('OFRA') == '2',
      f"species count OFRA round-trips as string (got {_r3['TCRMP2014_a'].get('OFRA')!r})")

# ── multi-batch species backfill ─────────────────────────────────
# batch1 writes species OFRA; batch2 introduces species PA
# after batch2: b1 must have PA='0', b2 must have OFRA='0'
_d4 = tempfile.mkdtemp(prefix="man_backfill_")
A.write_export_manifest(_d4, [{'basename': 'b1', 'n_masks': 1, 'label_file': 'b1.txt',
    'image_file': 'b1.jpeg', 'outcome': 'exported_with_masks', 'preview_file': '',
    'batch_name': 'batch_001_x', 'image_path': '/a/b1.jpeg', 'OFRA': 1}])
A.write_export_manifest(_d4, [{'basename': 'b2', 'n_masks': 1, 'label_file': 'b2.txt',
    'image_file': 'b2.jpeg', 'outcome': 'exported_with_masks', 'preview_file': '',
    'batch_name': 'batch_002_x', 'image_path': '/a/b2.jpeg', 'PA': 1}])
_r4 = A._read_export_manifest(_d4)
check(set(_r4) == {'b1', 'b2'}, f"both basenames present after 2 batches (got {set(_r4)})")
check(_r4['b1'].get('PA') == '0',
      f"b1 PA backfilled to '0' (got {_r4['b1'].get('PA')!r})")
check(_r4['b1'].get('OFRA') == '1',
      f"b1 OFRA preserved as '1' (got {_r4['b1'].get('OFRA')!r})")
check(_r4['b2'].get('OFRA') == '0',
      f"b2 OFRA backfilled to '0' (got {_r4['b2'].get('OFRA')!r})")
check(_r4['b2'].get('PA') == '1',
      f"b2 PA preserved as '1' (got {_r4['b2'].get('PA')!r})")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
