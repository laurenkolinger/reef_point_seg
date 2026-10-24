"""POST /api/mask_size sets the engine variant; status reports it; bad values 400.
Also covers the box-aware re-prompt family: /refine with rerun: true (the
re-prompt the mask-size slider fires), /undo_refine keeping the box, and
/reset_mask re-running the stored box and clipping to live neighbours.
Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_mask_size.py
"""
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

import os, sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, SRC)
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA'); os.environ.setdefault('TCRMP_MANUAL_ANNOTATE', '1')
import app as A
import sam_engine as S
_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1
class E:
    mask_size = "generous"
    def set_mask_size(self, s):
        if s not in S.MASK_SIZES: raise ValueError(s)
        self.mask_size = s
A.session['sam_engine'] = E()
c = A.app.test_client()
r = c.post('/api/mask_size', json={"size": "tight"}); check(r.status_code == 200 and A.session['sam_engine'].mask_size == "tight", "sets tight")
check(c.get('/api/status').get_json().get("mask_size") == "tight", "status reports mask_size")
r = c.post('/api/mask_size', json={"size": "huge"}); check(r.status_code == 400, "unknown size is 400")

# ── /refine rerun: re-prompt the stored clicks, append nothing ──────────────
import numpy as np
from mask_ops import rle_encode
H, W = 120, 160
seen = {}
class RerunEngine(E):
    _current_image_path = "x"
    def set_image(self, p): return (W, H)
    def refine_mask(self, clicks, labels, box=None):
        seen['clicks'] = list(clicks); seen['labels'] = list(labels); seen['box'] = box
        m = np.zeros((H, W), dtype=bool); m[20:70, 20:80] = True
        return {'mask': m, 'score': 0.8, 'bbox': None}
A.session['sam_engine'] = RerunEngine()
A._ensure_image_embedded = lambda fn: True
base = np.zeros((H, W), dtype=bool); base[20:60, 20:60] = True
mask = {"id": 3, "status": "pending", "species": "PA", "rle": rle_encode(base),
        "polygon_norm": [[0, 0, 0, 0, 0, 0]], "source_x": 30, "source_y": 30,
        "source_type": "manual_click", "refinement_clicks": [{"x": 44, "y": 45, "label": 0}],
        "source_box": [20, 20, 60, 60]}
A.session['segmentations'] = {"f.jpg": {"image_width": W, "image_height": H,
                                        "image_path_abs": "x", "masks": [mask]}}
r = c.post('/api/image/f.jpg/refine', json={"mask_id": 3, "rerun": True})
check(r.status_code == 200, f"rerun refine ok, got {r.get_json()}")
check(seen['clicks'] == [(30, 30), (44, 45)] and seen['labels'] == [1, 0], f"rerun re-prompts the stored clicks only, got {seen.get('clicks')}")
check(seen['box'] == [20, 20, 60, 60], "box-made mask keeps its box on refine")
check(mask['refinement_clicks'] == [{"x": 44, "y": 45, "label": 0}], "rerun appends nothing to refinement_clicks")

# ── /undo_refine re-prompts with the box too ────────────────────────────────
seen.clear()
r = c.post('/api/image/f.jpg/undo_refine', json={"mask_id": 3})
check(r.status_code == 200, f"undo ok, got {r.get_json()}")
check(seen['box'] == [20, 20, 60, 60], f"undo_refine keeps the mask's box, got {seen.get('box')}")
check(seen['clicks'] == [(30, 30)] and seen['labels'] == [1], f"undo drops the last click, got {seen.get('clicks')}")
check(mask['refinement_clicks'] == [], "undo popped the stored click")

# ── /reset_mask re-runs the stored box and clips to live neighbours ─────────
from mask_ops import rle_encode as _enc
calls = {}


class ResetEngine(E):
    _current_image_path = "x"

    def set_image(self, p):
        return (W, H)

    def segment_box(self, *box):
        calls['box'] = [int(v) for v in box]
        m = np.zeros((H, W), dtype=bool); m[20:80, 20:100] = True
        return {'mask': m, 'score': 0.7, 'bbox': None}

    def segment_point(self, x, y, label=1):
        calls['point'] = (x, y)
        m = np.zeros((H, W), dtype=bool); m[20:80, 20:100] = True
        return {'mask': m, 'score': 0.7, 'bbox': None}


A.session['sam_engine'] = ResetEngine()


def frame(target_extra, neighbour_rows=None):
    """One frame: mask #3 is the reset target, mask #4 an optional live neighbour."""
    calls.clear()
    t = {"id": 3, "status": "pending", "species": "PA", "rle": _enc(np.zeros((H, W), dtype=bool)),
         "polygon_norm": [[0, 0, 0, 0, 0, 0]], "source_x": 60, "source_y": 50,
         "refinement_clicks": [{"x": 44, "y": 45, "label": 0}], "source_type": "manual_box"}
    t.update(target_extra)
    masks = [t]
    if neighbour_rows is not None:
        nb = np.zeros((H, W), dtype=bool); nb[neighbour_rows[0]:neighbour_rows[1], neighbour_rows[2]:neighbour_rows[3]] = True
        masks.append({"id": 4, "status": "accepted", "species": "MC", "label": "N",
                      "rle": _enc(nb), "polygon_norm": [[0, 0, 0, 0, 0, 0]]})
    A.session['segmentations'] = {"f.jpg": {"image_width": W, "image_height": H,
                                            "image_path_abs": "x", "masks": masks}}
    return t


# Box branch, clipped against a live neighbour covering the right half.
t = frame({"source_box": [20, 20, 100, 80]}, neighbour_rows=(20, 80, 60, 100))
r = c.post('/api/image/f.jpg/reset_mask', json={"mask_id": 3}); j = r.get_json()
check(r.status_code == 200 and calls.get('box') == [20, 20, 100, 80] and 'point' not in calls,
      f"reset re-runs the stored box, not the point, got {calls}")
check(j["mask"]["area"] == 60 * 40, f"reset result clipped against the live neighbour, area {j['mask'].get('area')}")
check(j["mask"]["refinement_clicks"] == [] and j["mask"]["source_type"] == "manual_box", "reset clears clicks, keeps manual_box")

# Fully overlapped by the neighbour -> 400 with blocked_by.
t = frame({"source_box": [20, 20, 100, 80]}, neighbour_rows=(0, H, 0, W))
r = c.post('/api/image/f.jpg/reset_mask', json={"mask_id": 3}); j = r.get_json()
check(r.status_code == 400 and j.get("blocked_by") and j["blocked_by"][0]["id"] == 4,
      f"reset fully overlapped -> 400 with blocked_by, got {r.status_code} {j}")
check(t["refinement_clicks"] == [{"x": 44, "y": 45, "label": 0}] and "area" not in t,
      "refused reset leaves the mask untouched (clicks kept, no new geometry)")

# A rejected neighbour does not block the reset.
t = frame({"source_box": [20, 20, 100, 80]}, neighbour_rows=(0, H, 0, W))
r = c.post('/api/image/f.jpg/reset_mask', json={"mask_id": 3, "rejected_ids": [4]}); j = r.get_json()
check(r.status_code == 200 and j["mask"]["area"] == 60 * 80, f"rejected_ids frees the reset, got {j}")

# No source_box -> the point branch, clipped the same way.
t = frame({}, neighbour_rows=(20, 80, 60, 100))
r = c.post('/api/image/f.jpg/reset_mask', json={"mask_id": 3}); j = r.get_json()
check(r.status_code == 200 and calls.get('point') == (60, 50) and 'box' not in calls,
      f"no source_box falls back to the point prompt, got {calls}")
check(j["mask"]["area"] == 60 * 40 and j["mask"]["source_type"] == "auto", "point reset is clipped too and stamps auto")

# A merged mask is refused before any prompt runs.
t = frame({"source_type": "merged", "merged_from": [3, 5]})
r = c.post('/api/image/f.jpg/reset_mask', json={"mask_id": 3}); j = r.get_json()
check(r.status_code == 400 and "Draw" in (j.get("error") or "") and not calls,
      f"reset on a merged mask is refused with no prompt, got {r.status_code} {j} {calls}")

print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
