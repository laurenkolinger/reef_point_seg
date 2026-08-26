# CA/tests/test_draw_clip.py
"""A freehand draw edit clips to free space: it never overlaps another mask.
Run with env/bin/python.
"""
import os, sys
import numpy as np
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
import app as A
from mask_ops import rle_encode

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

H = W = 10
# Neighbor B occupies the left half (accepted, id=1).
b = np.zeros((H, W), dtype=bool); b[:, :5] = True
neighbor = {"id": 1, "status": "accepted", "rle": rle_encode(b)}

# A draw edit that would cover the WHOLE frame must keep only the right half.
new_binary = np.ones((H, W), dtype=bool)
blocking = A._blocking_masks([neighbor], rejected_ids=[])
clipped = A._clip_to_neighbors(new_binary, blocking)
check(not (clipped & b).any(), "clipped edit does not overlap neighbor B")
check(clipped[:, 5:].all(), "clipped edit keeps the free (right) half")
check(int(clipped.sum()) == H * 5, f"exactly the free area survives, got {int(clipped.sum())}")

# A REJECTED neighbor does not block (it's "deleted" in the user's model).
rej = {"id": 2, "status": "rejected", "rle": rle_encode(b)}
blocking2 = A._blocking_masks([rej], rejected_ids=[])
clipped2 = A._clip_to_neighbors(np.ones((H, W), dtype=bool), blocking2)
check(clipped2.all(), "rejected neighbor does not clip the edit")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
