# CA/tests/test_export_previews.py
"""_publish_preview copies the reviewed overlay into a flat previews/ folder
and returns its export-root-relative path. Run with env/bin/python.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import app as A

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

d = tempfile.mkdtemp(prefix="exp_prev_")
stem = "TCRMP20220601_clip_FLC_T307"
# Seed a fake reviewed overlay where render_segmentation_overlays would write it.
# Flat path: segmentations/overlays_reviewed/<stem>_seg.jpg (no year subdirectory).
ov_dir = os.path.join(d, "segmentations", "overlays_reviewed")
os.makedirs(ov_dir)
with open(os.path.join(ov_dir, stem + "_seg.jpg"), "wb") as f:
    f.write(b"\xff\xd8\xff\xe0JFIFfake")  # bytes stand in for a JPEG

rel = A._publish_preview(d, stem)
check(rel == os.path.join("previews", stem + "_seg.jpg"), f"relpath wrong: {rel}")
check(os.path.exists(os.path.join(d, "previews", stem + "_seg.jpg")), "preview copied into previews/")

# Missing source overlay -> empty string, no crash.
rel2 = A._publish_preview(d, "TCRMP_NOPE")
check(rel2 == "", f"missing overlay should yield empty string, got {rel2!r}")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
