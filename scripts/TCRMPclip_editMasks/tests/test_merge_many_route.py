"""POST /merge_many unions N masks into the lowest id, keeps disjoint parts, removes
the others; /refine on a merged mask is refused with guidance.

editMasks fork of combinedAnnotate's test of the same name. The merge machinery
is identical; what differs is the expert-review lock. combinedAnnotate MERGES
masks that carry review state (the survivor inherits it); editMasks refuses to
touch them at all, so the inheritance cases there are refusal cases here.

Run: env/bin/python scripts/TCRMPclip_editMasks/tests/test_merge_many_route.py
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
import numpy as np
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, SRC)
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA'); os.environ.setdefault('TCRMP_MANUAL_ANNOTATE', '1')
# Merging masks into an accepted survivor is an accept, refused without
# initials (test_reviewer_required covers that). The orchestrator always
# launches with the opener's initials, so seed them the same way.
os.environ['TCRMP_REVIEWER'] = 'LO'
import app as A
from mask_ops import rle_encode, rle_decode
_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1
H, W = 300, 300
def blob(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool); m[y0:y1, x0:x1] = True; return m
masks = []
for i, (b, st) in enumerate([(blob(10, 60, 10, 60), "pending"), (blob(100, 160, 100, 160), "accepted"), (blob(200, 280, 200, 280), "pending")]):
    masks.append({"id": i + 1, "status": st, "species": "PA", "label": chr(65 + i), "rle": rle_encode(b),
                  "polygon_norm": [[0, 0, 0, 0, 0, 0]], "refinement_clicks": [{"x": 1, "y": 1, "label": 1}],
                  "source_x": 1, "source_y": 1, "source_type": "manual_click"})
seg = {"image_width": W, "image_height": H, "image_path_abs": "x", "masks": masks}
A.session['segmentations'] = {"f.jpg": seg}
client = A.app.test_client()
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [3, 1, 2]}); j = r.get_json()
check(r.status_code == 200 and j.get("ok"), f"merge_many ok, got {j}")
m = j["mask"]
check(m["id"] == 1 and sorted(j["removed_ids"]) == [2, 3], "survivor is lowest id, others removed")
check(m["area"] == 2500 + 3600 + 6400, f"all three parts kept, area {m['area']}")
check(len(m["polygon_px"]) == 3, "three polygons")
# The wire copy drops the two fields no client reads (see app.wire_mask); the
# stored survivor still carries both, which the rle check below proves.
check("polygon_norm" not in m and "rle" not in m,
      f"merge_many answers with a wire mask (no rle/polygon_norm), got {sorted(m)}")
check(m["status"] == "accepted", "accepted if any input accepted")
check(m["source_type"] == "merged" and m["merged_from"] == [1, 2, 3] and m["refinement_clicks"] == [], "merged provenance")
check([x["id"] for x in seg["masks"]] == [1], "server list holds only the survivor")
dec = rle_decode(seg["masks"][0]["rle"], (H, W))
check(int(dec.sum()) == m["area"], "stored rle matches the reported area")
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1]}); check(r.status_code == 400, "fewer than 2 ids is 400")
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 99]}); check(r.status_code == 404, "unknown id is 404")
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": ["a", "b"]}); j = r.get_json()
check(r.status_code == 400 and "integers" in (j.get("error") or ""), "non-integer mask_ids is 400")
A.session['sam_engine'] = object(); A._ensure_image_embedded = lambda fn: True
r = client.post('/api/image/f.jpg/refine', json={"mask_id": 1, "click_x": 5, "click_y": 5, "click_label": 1}); j = r.get_json()
check(r.status_code == 400 and "Draw" in (j.get("error") or ""), "refine on merged mask refused with Draw guidance")
r = client.post('/api/image/f.jpg/reset_mask', json={"mask_id": 1}); j = r.get_json()
check(r.status_code == 400 and "Draw" in (j.get("error") or ""), "reset on merged mask refused with the same guidance")


def fresh(masks):
    """Install a frame carrying these mask dicts and hand back the seg."""
    seg = {"image_width": W, "image_height": H, "image_path_abs": "x", "masks": masks}
    A.session['segmentations'] = {"f.jpg": seg}
    return seg


def mk(mid, y0, y1, x0, x1, **extra):
    m = {"id": mid, "status": "pending", "species": "PA", "rle": rle_encode(blob(y0, y1, x0, x1)),
         "polygon_norm": [[0, 0, 0, 0, 0, 0]], "refinement_clicks": [], "source_x": 1, "source_y": 1,
         "source_type": "manual_click"}
    m.update(extra)
    return m


# ── Pairwise /merge: the LOWER id survives whichever way round it is sent ────
seg = fresh([mk(2, 10, 60, 10, 60), mk(5, 200, 280, 200, 280)])
r = client.post('/api/image/f.jpg/merge', json={"mask_id_a": 5, "mask_id_b": 2}); j = r.get_json()
check(r.status_code == 200 and j["mask"]["id"] == 2 and j["removed_id"] == 5,
      f"pairwise merge with the higher id first keeps the lower id, got {j.get('mask', {}).get('id')}/{j.get('removed_id')}")
check(j["mask"]["area"] == 2500 + 6400 and len(j["mask"]["polygon_px"]) == 2, "both parts kept")
check(j["mask"]["source_type"] == "merged" and [x["id"] for x in seg["masks"]] == [2], "merged provenance, only the survivor remains")

seg = fresh([mk(2, 10, 60, 10, 60), mk(5, 200, 280, 200, 280)])
r = client.post('/api/image/f.jpg/merge', json={"mask_id_a": 2, "mask_id_b": 2}); j = r.get_json()
check(r.status_code == 400 and "different" in (j.get("error") or ""), f"pairwise merge of a mask with itself is 400, got {r.status_code} {j}")
r = client.post('/api/image/f.jpg/merge', json={"mask_id_a": 2, "mask_id_b": 77}); j = r.get_json()
check(r.status_code == 404 and j.get("error") == "one or both masks not found", "pairwise unknown id keeps its 404 wording")
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [2, 5]})
check(r.status_code == 200, "control: those two do merge")

# ── A rejected mask is never absorbed ────────────────────────────────────────
seg = fresh([mk(1, 10, 60, 10, 60), mk(2, 200, 280, 200, 280, status="rejected")])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 2]}); j = r.get_json()
check(r.status_code == 400 and "rejected" in (j.get("error") or ""), f"merging a rejected mask is 400, got {j}")

# ── EDITMASKS LOCK: an expert-review mask is never merged, absorbed, or moved ─
# combinedAnnotate merges these and hands the survivor their review state.
# editMasks refuses before any geometry is touched, so the frame is untouched.
LOCKED_FORMS = [
    ("review flag", {"review": True}),
    ("REVIEW species", {"species": "REVIEW"}),
    ("expert_id dict", {"expert_id": {"mode": "EXPERT", "code": "SS"}}),
    ("pending expert_id", {"expert_id": {"mode": "PENDING", "code": "SS"}}),
    ("review_uid", {"review_uid": "uid-c"}),
    ("review history", {"reviews": [{"reviewer": "LO", "code": "PA", "at": "t0"}]}),
]
for name, extra in LOCKED_FORMS:
    seg = fresh([mk(1, 10, 60, 10, 60), mk(3, 200, 280, 200, 280, **extra)])
    r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]}); j = r.get_json()
    check(r.status_code == 403 and j.get("locked") is True and j.get("locked_ids") == [3],
          f"/merge_many refuses a locked mask ({name}), got {r.status_code} {j}")
    check([x["id"] for x in seg["masks"]] == [1, 3] and "merged_from" not in seg["masks"][0],
          f"refused merge leaves the frame untouched ({name})")

seg = fresh([mk(1, 10, 60, 10, 60, review_uid="uid-a"), mk(3, 200, 280, 200, 280)])
r = client.post('/api/image/f.jpg/merge', json={"mask_id_a": 3, "mask_id_b": 1}); j = r.get_json()
check(r.status_code == 403 and j.get("locked") is True and j.get("locked_ids") == [1],
      f"pairwise /merge refuses a locked mask too, got {r.status_code} {j}")
check([x["id"] for x in seg["masks"]] == [1, 3], "refused pairwise merge leaves both masks in place")

# A locked id that is not on the frame keeps the not-found answer, not the lock one.
seg = fresh([mk(1, 10, 60, 10, 60, review=True)])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 42]})
check(r.status_code == 404, f"an unknown id still answers 404 even alongside a locked one, got {r.status_code}")

# Control: two ordinary masks with no review state merge normally.
seg = fresh([mk(1, 10, 60, 10, 60), mk(3, 200, 280, 200, 280)])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]})
check(r.status_code == 200, "control: two unlocked masks still merge")

# ── merged_from extends an earlier lineage instead of overwriting it ─────────
seg = fresh([mk(1, 10, 60, 10, 60), mk(2, 100, 160, 100, 160), mk(4, 200, 280, 200, 280)])
client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 2]})
seg["masks"].append(mk(9, 20, 40, 200, 260))
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 4, 9]}); j = r.get_json()
check(j["mask"]["merged_from"] == [1, 2, 4, 9], f"re-merge extends the lineage, got {j['mask'].get('merged_from')}")

# ── Draw edit on a merged mask keeps every part ──────────────────────────────
seg = fresh([mk(1, 10, 60, 10, 60), mk(2, 200, 280, 200, 280)])
client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 2]})
before = seg["masks"][0]["area"]
chunk = [[120, 120], [159, 120], [159, 159], [120, 159]]
r = client.post('/api/image/f.jpg/draw_edit', json={"mask_id": 1, "path": chunk}); j = r.get_json()
check(r.status_code == 200 and j.get("op") == "add", f"draw edit adds a chunk, got {j}")
check(j["mask"]["area"] == before + 40 * 40, f"both merged parts survive the draw edit, area {j['mask']['area']} (was {before})")
check(len(j["mask"]["polygon_px"]) == 3, f"three polygons after the edit, got {len(j['mask']['polygon_px'])}")

# A single-prompt mask still collapses to one component (the rule is unchanged).
seg = fresh([mk(1, 10, 60, 10, 60)])
r = client.post('/api/image/f.jpg/draw_edit', json={"mask_id": 1, "path": chunk}); j = r.get_json()
check(len(j["mask"]["polygon_px"]) == 1 and j["mask"]["area"] == 2500,
      f"non-merged mask keeps the single-component rule, got {j['mask']['area']}")


# ── A merged survivor drops the box it inherited (fix wave) ──────────────────
# /rebox re-prompts from a mask's source_box and keeps only that one result, so
# a survivor that inherited one would collapse to a single part. The mask-size
# slider re-boxes any mask that carries a source_box, which is how it surfaced.
seg = fresh([mk(1, 10, 60, 10, 60, source_box=[10, 10, 60, 60], source_type="manual_box"),
             mk(2, 200, 280, 200, 280, source_box=[200, 200, 280, 280], source_type="manual_box")])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 2]}); j = r.get_json()
check(r.status_code == 200 and "source_box" not in j["mask"],
      f"merged survivor carries no source_box, got {j['mask'].get('source_box')}")
check("source_box" not in seg["masks"][0], "the stored survivor drops the inherited box too")
seg = fresh([mk(2, 10, 60, 10, 60, source_box=[10, 10, 60, 60], source_type="manual_box"),
             mk(5, 200, 280, 200, 280)])
r = client.post('/api/image/f.jpg/merge', json={"mask_id_a": 5, "mask_id_b": 2}); j = r.get_json()
check(r.status_code == 200 and "source_box" not in j["mask"], "pairwise /merge drops it as well")

# ── An unhashable id answers 400, not 500 (the set() pre-guard is gone) ──────
seg = fresh([mk(1, 10, 60, 10, 60), mk(2, 200, 280, 200, 280)])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [[1], [2]]}); j = r.get_json()
check(r.status_code == 400 and "integers" in (j.get("error") or ""),
      f"a list-valued mask id is 400, not 500, got {r.status_code} {j}")
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 1]}); j = r.get_json()
check(r.status_code == 400 and "different" in (j.get("error") or ""),
      f"the same id twice is still 400, got {r.status_code} {j}")
check([x["id"] for x in seg["masks"]] == [1, 2], "both refusals leave the frame untouched")


# ── Two masks each awaiting their own expert ID refuse the merge ─────────────
# _merge_ids carries this guard in both forks so the merge helper stays in
# parity. In editMasks the expert lock answers 403 before it is ever reached
# (LOCKED_FORMS above proves that), so the lock is lifted here to exercise the
# guard itself on both routes.
_orig_locked = A._is_expert_locked
A._is_expert_locked = lambda m: False
try:
    seg = fresh([mk(1, 10, 60, 10, 60, species="REVIEW", review_uid="uid-1"),
                 mk(3, 200, 280, 200, 280, species="REVIEW", review_uid="uid-2")])
    r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]}); j = r.get_json()
    err = j.get("error") or ""
    check(r.status_code == 400 and "#1" in err and "#3" in err and "awaiting their own expert ID" in err,
          f"two open review round trips refuse /merge_many, got {r.status_code} {j}")
    check([x["id"] for x in seg["masks"]] == [1, 3] and "merged_from" not in seg["masks"][0]
          and "area" not in seg["masks"][0], "refused merge leaves the frame untouched")

    seg = fresh([mk(1, 10, 60, 10, 60, species="REVIEW", review_uid="uid-1"),
                 mk(3, 200, 280, 200, 280, species="REVIEW", review_uid="uid-2")])
    r = client.post('/api/image/f.jpg/merge', json={"mask_id_a": 3, "mask_id_b": 1}); j = r.get_json()
    check(r.status_code == 400 and "awaiting their own expert ID" in (j.get("error") or ""),
          f"pairwise /merge refuses the same pair, got {r.status_code} {j}")
    check([x["id"] for x in seg["masks"]] == [1, 3], "pairwise refusal leaves both masks in place")

    seg = fresh([mk(1, 10, 60, 10, 60, species="REVIEW", review_uid="uid-1"),
                 mk(3, 200, 280, 200, 280, species="REVIEW", review_uid="uid-1")])
    r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]})
    check(r.status_code == 200, "the same review_uid on both is one round trip, so it merges")

    seg = fresh([mk(1, 10, 60, 10, 60, species="REVIEW", review_uid="uid-1"),
                 mk(3, 200, 280, 200, 280, species="REVIEW")])
    r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]})
    check(r.status_code == 200, "one review_uid and one REVIEW mask without one still merges")
finally:
    A._is_expert_locked = _orig_locked

print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
