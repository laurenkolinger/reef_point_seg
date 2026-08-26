# CA/tests/test_at_test_export_complete.py
"""Export completeness integration test.

Synthesizes a routed set with:
  - 3 positive frames (accepted masks, not review-flagged)
  - 2 empty frames (zero masks -> genuine negatives)
  - 1 review-only frame (REVIEW-flagged, must be skipped from training export)

Runs export_yolo.export_batch and asserts:
  - len(all_images) == len(all_labels) == positives + negatives (5, not 6)
  - exported_empty count == 2
  - review_only_skipped count == 1
  - stats["exported_images"] == positives + negatives

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_at_test_export_complete.py
"""
import os
import sys
import tempfile
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

import export_yolo as EY

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _img(d, name):
    p = os.path.join(d, name)
    Image.new("RGB", (64, 48), (10, 20, 30)).save(p)
    return p


# ── build the synthetic routed set ───────────────────────────────────────────
d = tempfile.mkdtemp(prefix="exp_complete_")
export_dir = os.path.join(d, "out")

# 3 positives
pos1 = _img(d, "POS1.jpg")
pos2 = _img(d, "POS2.jpg")
pos3 = _img(d, "POS3.jpg")
# 2 genuine empties
emp1 = _img(d, "EMP1.jpg")
emp2 = _img(d, "EMP2.jpg")
# 1 review-only (must be skipped)
rev1 = _img(d, "REV1.jpg")

POSITIVES = 3
NEGATIVES = 2
EXPECTED_EXPORTED = POSITIVES + NEGATIVES  # 5; review-only is excluded

segs = {
    "POS1.jpg": {
        "image_path_abs": pos1, "image_width": 64, "image_height": 48,
        "masks": [{"status": "accepted", "species": "PA",
                   "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]]}],
    },
    "POS2.jpg": {
        "image_path_abs": pos2, "image_width": 64, "image_height": 48,
        "masks": [{"status": "accepted", "species": "OFRA",
                   "polygon_norm": [[0.2, 0.2, 0.4, 0.2, 0.4, 0.4]]}],
    },
    "POS3.jpg": {
        "image_path_abs": pos3, "image_width": 64, "image_height": 48,
        "masks": [{"status": "accepted", "species": "PA",
                   "polygon_norm": [[0.05, 0.05, 0.15, 0.05, 0.15, 0.15]]}],
    },
    "EMP1.jpg": {
        "image_path_abs": emp1, "image_width": 64, "image_height": 48,
        "masks": [],
    },
    "EMP2.jpg": {
        "image_path_abs": emp2, "image_width": 64, "image_height": 48,
        "masks": [],
    },
    "REV1.jpg": {
        "image_path_abs": rev1, "image_width": 64, "image_height": 48,
        "masks": [{"status": "accepted", "review": True, "species": "REVIEW",
                   "polygon_norm": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]]}],
    },
}

# ── run export ────────────────────────────────────────────────────────────────
stats = EY.export_batch(segs, export_dir, {}, symlink=False)

# ── assertions ────────────────────────────────────────────────────────────────
all_images_dir = os.path.join(export_dir, "all_images")
all_labels_dir = os.path.join(export_dir, "all_labels")

all_images = sorted(os.listdir(all_images_dir))
all_labels = sorted(os.listdir(all_labels_dir))

check(
    len(all_images) == EXPECTED_EXPORTED,
    f"all_images count == {EXPECTED_EXPORTED} (positives+negatives, no review), "
    f"got {len(all_images)}: {all_images}",
)
check(
    len(all_labels) == EXPECTED_EXPORTED,
    f"all_labels count == {EXPECTED_EXPORTED}, got {len(all_labels)}: {all_labels}",
)
check(
    len(all_images) == len(all_labels),
    f"all_images ({len(all_images)}) == all_labels ({len(all_labels)})",
)
check(
    stats.get("exported_empty") == NEGATIVES,
    f"exported_empty == {NEGATIVES}, got {stats.get('exported_empty')}",
)
check(
    stats.get("review_only_skipped") == 1,
    f"review_only_skipped == 1, got {stats.get('review_only_skipped')}",
)
check(
    stats.get("exported_images") == EXPECTED_EXPORTED,
    f"exported_images == {EXPECTED_EXPORTED}, got {stats.get('exported_images')}",
)

# Review-only must NOT appear on disk.
check(
    "REV1.jpg" not in all_images,
    "REV1.jpg (review-only) must NOT appear in all_images",
)

# Empty-frame label files must be empty.
for emp_name in ("EMP1.txt", "EMP2.txt"):
    lp = os.path.join(all_labels_dir, emp_name)
    check(os.path.exists(lp), f"empty-frame label file exists: {emp_name}")
    if os.path.exists(lp):
        check(open(lp).read().strip() == "", f"empty-frame label is blank: {emp_name}")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
