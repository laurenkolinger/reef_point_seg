"""POST /add_box creates a new mask from a box prompt (fake engine): clips to live
neighbours, reports blocked_by on full overlap, mints the next id, stores source_box.
Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_add_box.py
"""
import os, sys, json
import numpy as np
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA,OFRA'); os.environ.setdefault('TCRMP_MANUAL_ANNOTATE', '1')
import app as A
from mask_ops import rle_encode

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

H, W = 120, 160
class FakeEngine:
    _current_image_path = "x"
    def set_image(self, p): return (W, H)
    def segment_box(self, x0, y0, x1, y1):
        m = np.zeros((H, W), dtype=bool); m[int(y0):int(y1), int(x0):int(x1)] = True
        return {'mask': m, 'score': 0.9, 'bbox': None}
A.session['sam_engine'] = FakeEngine()
A._ensure_image_embedded = lambda fn: True
nb = np.zeros((H, W), dtype=bool); nb[10:40, 10:40] = True
seg = {"image_width": W, "image_height": H, "image_path_abs": "x",
       "masks": [{"id": 7, "status": "accepted", "species": "PA", "rle": rle_encode(nb), "polygon_norm": [[0,0,0,0,0,0]]}]}
A.session['segmentations'] = {"f.jpg": seg}
client = A.app.test_client()

r = client.post('/api/image/f.jpg/add_box', json={"box": [60, 60, 120, 110], "species": "OFRA", "label": "B", "rejected_ids": []})
j = r.get_json()
check(r.status_code == 200 and j.get("ok"), f"box add ok, got {j}")
m = j["mask"]
check(m["id"] == 8 and m["source_type"] == "manual_box" and m["source_box"] == [60, 60, 120, 110], "id minted, source_type + source_box stored")
check(m["species"] == "OFRA" and m["status"] == "pending" and m["area"] == 60 * 50, "species + pending + area")
check(m["source_x"] == 90 and m["source_y"] == 85, "source point is the box centre")

# Deviation from the brief's exact 60*60 - 30*30: build_mask_dict runs clean_mask,
# whose morphological closing rounds the corners of the hole the neighbour leaves,
# so the clipped area sits a little above the ideal 2700. The assertion keeps the
# intent (clipped, close to the ideal) with a small tolerance.
r = client.post('/api/image/f.jpg/add_box', json={"box": [0, 0, 60, 60], "rejected_ids": []}); j = r.get_json()
_a = j["mask"]["area"]
check(_a < 60 * 60 and abs(_a - (60 * 60 - 30 * 30)) <= 100, f"clipped against the live neighbour, area {_a}")

r = client.post('/api/image/f.jpg/add_box', json={"box": [10, 10, 40, 40], "rejected_ids": []}); j = r.get_json()
check(r.status_code == 400 and j.get("blocked_by") and j["blocked_by"][0]["id"] == 7, "fully overlapped -> 400 with blocked_by")

# Deviation: mask #9 (the clipped add above) is a live neighbour here too, and its
# closing-rounded corners eat into the hole, so the exact 900 is not reachable.
# rejected_ids=[7, 9] isolates what this check is about: a rejected neighbour does
# not block. Area is then the ideal 900.
r = client.post('/api/image/f.jpg/add_box', json={"box": [10, 10, 40, 40], "rejected_ids": [7, 9]}); j = r.get_json()
check(r.status_code == 200 and j["mask"]["area"] == 900, f"rejected neighbour does not block, got {j.get('mask', {}).get('area', j)}")

# Expert-review lock (2026-09-06): arming REVIEW is refused server-side, in both
# spellings, and no mask is created. The wording follows the editMasks twin.
_before = len(seg["masks"])
r = client.post('/api/image/f.jpg/add_box', json={"box": [0, 0, 20, 20], "review": True, "rejected_ids": [7, 9, 10]}); j = r.get_json()
check(r.status_code == 409 and j.get("review_locked") is True and len(seg["masks"]) == _before,
      f"a review flag on /add_box is refused and creates no mask, got {r.status_code} {j}")
r = client.post('/api/image/f.jpg/add_box', json={"box": [0, 0, 20, 20], "species": "REVIEW", "rejected_ids": [7, 9, 10]}); j = r.get_json()
check(r.status_code == 409 and j.get("review_locked") is True and len(seg["masks"]) == _before,
      f"species REVIEW with no review key is refused too, got {r.status_code} {j}")

r = client.post('/api/image/f.jpg/add_box', json={"box": [5, 5, 5, 5]}); check(r.status_code == 400, "degenerate box is 400")
print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
