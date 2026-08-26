"""Delete a mask, then add a new mask in its freed region: the just-deleted
mask (reported via rejected_ids) must NOT clip the new one, but an unrelated
LIVE neighbor still must. Exercises the /add clip path directly (no SAM engine).
Run with env/bin/python.
"""
import os, sys
import numpy as np
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
import app as A
from mask_ops import rle_encode, rle_decode

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

H = W = 20
left = np.zeros((H, W), bool); left[:, :10] = True     # deleted mask (left half)
right = np.zeros((H, W), bool); right[:, 15:] = True    # live neighbor (right strip)
masks = [
    {"id": 1, "status": "pending", "species": "PA", "rle": rle_encode(left)},
    {"id": 2, "status": "pending", "species": "SS", "rle": rle_encode(right)},
]

# Operator deleted mask #1 (client sends rejected_ids=[1]); a new full-frame SAM
# mask must keep the freed left half and only lose the right strip to mask #2.
blocking = A._blocking_masks(masks, rejected_ids=[1])
check([m["id"] for m in blocking] == [2], "deleted mask #1 excluded; live #2 still blocks")

new_full = np.ones((H, W), bool)
clipped = A._clip_to_neighbors(new_full, blocking)
check(bool(clipped[:, :10].all()), "freed left half fully available for the remake")
check(not clipped[:, 15:].any(), "live neighbor #2 still protected (no overlap)")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
