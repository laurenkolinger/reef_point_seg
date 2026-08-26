"""When a new/edited mask is fully eaten by live neighbors, the API names the
blocker(s) via `blocked_by` instead of a generic error. Run with env/bin/python.
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

# 10x10 frame; a live SS mask fills the whole frame.
H = W = 10
full = np.ones((H, W), dtype=bool)
masks = [{"id": 7, "status": "accepted", "species": "SS", "label": "B",
          "rle": rle_encode(full)}]

# A new mask covering the frame is fully subtracted by mask #7.
blocking = A._blocking_masks(masks, rejected_ids=[])
blockers = A._overlapping_blockers(full, blocking)
check([b["id"] for b in blockers] == [7], "blocker id reported")
check(blockers[0]["species"] == "SS", "blocker species reported")

# A rejected neighbor is NOT reported (it never blocks).
masks_rej = [{"id": 7, "status": "rejected", "species": "SS", "label": "B",
              "rle": rle_encode(full)}]
b2 = A._overlapping_blockers(full, A._blocking_masks(masks_rej, rejected_ids=[]))
check(b2 == [], "rejected neighbor not reported as blocker")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
