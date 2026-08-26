"""clean_mask must bridge two nearby fragments of one colony at 4K (where a
fixed 15px kernel is too small), while leaving 1920px behavior unchanged.
Run with env/bin/python.
"""
import os, sys
import numpy as np
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
from mask_ops import clean_mask
import cv2

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

def two_blobs(H, W, gap):
    """Two filled squares separated by `gap` px of background."""
    m = np.zeros((H, W), dtype=np.uint8)
    s = 200
    cy = H // 2
    x0 = W // 2 - s - gap // 2
    m[cy - s//2:cy + s//2, x0:x0 + s] = 1
    x1 = x0 + s + gap
    m[cy - s//2:cy + s//2, x1:x1 + s] = 1
    return m

# 4K frame, 24px gap: proportional kernel (~30px) bridges -> ONE component kept
# at (near) full area; the old fixed 15px kernel would keep only one blob.
m4k = two_blobs(2160, 3840, 24)
out4k = clean_mask(m4k.copy(), min_fragment=500)
n4k, _ = cv2.connectedComponents(out4k)
check(n4k == 2, "4K: fragments bridged into a single component")  # 2 = bg + 1 fg
check(int(out4k.sum()) > int(m4k.sum()) * 0.9, "4K: bridged mask keeps both blobs' area")

# 1920 frame, 24px gap: kernel stays ~15px, does NOT bridge a 24px gap ->
# largest-component-only keeps a single blob (unchanged legacy behavior).
m2k = two_blobs(1080, 1920, 24)
out2k = clean_mask(m2k.copy(), min_fragment=500)
check(int(out2k.sum()) < int(m2k.sum()) * 0.75, "1920: 24px gap not bridged (legacy behavior kept)")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
