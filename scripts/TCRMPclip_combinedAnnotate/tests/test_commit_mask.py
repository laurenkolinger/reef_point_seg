"""commit_mask appends an accepted mask to seg['masks'] with a fresh id."""
# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import os, sys, importlib.util
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
import app as A
_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1
# Inject a seg via the app's test seam, then call the commit helper directly.
seg = {"image_width": 64, "image_height": 48, "masks": [{"id": 0, "status": "accepted"}]}
cand = {"species": "PA", "polygon_norm": [[0.1,0.1,0.2,0.1,0.2,0.2]], "rle": None,
        "area": 10, "score": 0.9, "source_type": "exemplar"}
mask = A._commit_candidate_to_seg(seg, cand)
check(mask["id"] == 1, f"fresh id assigned, got {mask['id']}")
check(mask["status"] == "accepted", "committed as accepted")
check(len(seg["masks"]) == 2, "appended to seg['masks']")
print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
