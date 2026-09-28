"""POST /add_box creates a new mask from a box prompt (fake engine): clips to live
neighbours, reports blocked_by on full overlap, mints the next id, stores source_box.

editMasks fork of combinedAnnotate's test of the same name. Two differences:
this app never arms REVIEW (since 2026-09-06 a body asking for an expert-review
tag is refused with HTTP 409 and creates nothing), and /reset_mask refuses an
expert-locked mask outright.

Run: env/bin/python scripts/TCRMPclip_editMasks/tests/test_add_box.py
"""
import copy
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

# Deviation from combinedAnnotate's exact 60*60 - 30*30: build_mask_dict runs
# clean_mask, whose morphological closing rounds the corners of the hole the
# neighbour leaves, so the clipped area sits a little above the ideal 2700. The
# assertion keeps the intent (clipped, close to the ideal) with a small tolerance.
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

# Expert-review flagging is locked (Lauren Olinger, 2026-09-06): a review flag
# on /add_box is now REFUSED outright rather than quietly ignored, and nothing
# is created. This assertion was inverted with that change: the old one asserted
# the 2026-07 half-pin, which forced the boolean to False and let the species
# string through, so a body of {"species": "REVIEW"} still made a flagged mask.
# Both spellings are checked here, the second of which nothing tested before.
_before = len(seg["masks"])
for _body, _why in (({"review": True}, "the review boolean"),
                    ({"species": "REVIEW"}, "the species string alone")):
    _b = {"box": [0, 0, 20, 20], "rejected_ids": [7, 9, 10]}; _b.update(_body)
    r = client.post('/api/image/f.jpg/add_box', json=_b); j = r.get_json()
    check(r.status_code == 409 and j.get("review_locked") is True and j.get("ok") is False,
          f"/add_box refuses {_why}, got {r.status_code} {j}")
check(len(seg["masks"]) == _before, "a refused /add_box created no mask")

# Control: the ordinary mask this app makes is still not expert-locked.
r = client.post('/api/image/f.jpg/add_box', json={"box": [0, 0, 20, 20], "rejected_ids": [7, 9, 10]}); j = r.get_json()
check(not A._is_expert_locked(j["mask"]), "the mask this app just made is not expert-locked")

r = client.post('/api/image/f.jpg/add_box', json={"box": [5, 5, 5, 5]}); check(r.status_code == 400, "degenerate box is 400")

# ── EDITMASKS LOCK: /reset_mask refuses an expert-locked mask ────────────────
locked = np.zeros((H, W), dtype=bool); locked[70:100, 10:50] = True
seg["masks"].append({"id": 40, "status": "accepted", "species": "PA", "rle": rle_encode(locked),
                     "polygon_norm": [[0, 0, 0, 0, 0, 0]], "refinement_clicks": [],
                     "source_x": 30, "source_y": 85, "source_type": "manual_click",
                     "expert_id": {"mode": "EXPERT", "code": "SS"}})
# deepcopy, not dict(): a shallow copy shares the rle/polygon objects, so a
# refusal that mutated them in place would still compare equal.
before = copy.deepcopy(seg["masks"][-1])
r = client.post('/api/image/f.jpg/reset_mask', json={"mask_id": 40}); j = r.get_json()
check(r.status_code == 403 and j.get("locked") is True,
      f"/reset_mask refuses an expert-locked mask, got {r.status_code} {j}")
check(seg["masks"][-1] == before, "the refused reset left the locked mask exactly as it was")

# Control: an ordinary box mask resets through its stored box, not a point.
r = client.post('/api/image/f.jpg/reset_mask', json={"mask_id": 8}); j = r.get_json()
check(r.status_code == 200 and j["mask"]["source_type"] == "manual_box",
      f"control: an unlocked box mask resets through its box, got {r.status_code} {j}")

print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
