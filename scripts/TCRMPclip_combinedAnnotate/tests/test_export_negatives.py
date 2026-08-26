# CA/tests/test_export_negatives.py
"""Zero-mask frames export as negatives (image + empty label); review-only
frames are NOT training negatives. Run with env/bin/python.
"""
import os, sys, tempfile
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import export_yolo as EY

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

def _img(d, name):
    p = os.path.join(d, name); Image.new("RGB", (64, 48), (1, 2, 3)).save(p); return p

d = tempfile.mkdtemp(prefix="exp_neg_")
img_empty = _img(d, "EMPTY.jpg")
img_rev = _img(d, "REV.jpg")
img_pos = _img(d, "POS.jpg")
export_dir = os.path.join(d, "out")
segs = {
    "EMPTY.jpg": {"image_path_abs": img_empty, "image_width": 64, "image_height": 48, "masks": []},
    "REV.jpg": {"image_path_abs": img_rev, "image_width": 64, "image_height": 48,
                "masks": [{"status": "accepted", "review": True, "species": "REVIEW",
                           "polygon_norm": [[0.1,0.1,0.2,0.1,0.2,0.2]]}]},
    "POS.jpg": {"image_path_abs": img_pos, "image_width": 64, "image_height": 48,
                "masks": [{"status": "accepted", "species": "PA",
                           "polygon_norm": [[0.1,0.1,0.3,0.1,0.3,0.3]]}]},
}
stats = EY.export_batch(segs, export_dir, {}, symlink=False)

imgs = set(os.listdir(os.path.join(export_dir, "all_images")))
check("EMPTY.jpg" in imgs, "genuine-empty image is exported as a negative")
check("POS.jpg" in imgs, "positive image exported")
check("REV.jpg" not in imgs, "review-only image NOT a training negative")

empty_lbl = os.path.join(export_dir, "all_labels", "EMPTY.txt")
check(os.path.exists(empty_lbl), "empty label file written for negative")
check(open(empty_lbl).read().strip() == "", "negative label file is empty")
check(stats.get("exported_empty") == 1, f"exported_empty=1, got {stats.get('exported_empty')}")
check(stats.get("review_only_skipped") == 1, f"review_only_skipped=1, got {stats.get('review_only_skipped')}")
check(stats["exported_images"] == 2, f"EMPTY+POS exported, got {stats['exported_images']}")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
