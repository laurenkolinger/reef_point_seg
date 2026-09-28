"""POST /merge_many unions N masks into the lowest id, keeps disjoint parts, removes
the others; /refine on a merged mask is refused with guidance.
Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_merge_many_route.py
"""
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

# ── The survivor inherits the absorbed masks' expert-review state ────────────
seg = fresh([
    mk(1, 10, 60, 10, 60, reviews=[{"reviewer": "LO", "code": "PA", "confidence": "high", "at": "t0"}]),
    mk(2, 100, 160, 100, 160, review_uid="uid-b", review=True,
       reviews=[{"reviewer": "AB", "code": "MC", "confidence": "low", "at": "t1"}]),
    mk(3, 200, 280, 200, 280, expert_id={"mode": "EXPERT", "code": "SS"}, review_uid="uid-c"),
])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 2, 3]}); j = r.get_json()
m = j["mask"]
check(m["review_uid"] == "uid-b", f"review_uid carried from the first absorbed mask that has one, got {m.get('review_uid')}")
check(m["expert_id"] == {"mode": "EXPERT", "code": "SS"}, "expert_id carried forward")
check(not m.get("review"), f"the survivor gains no pending-review flag it lacked (expert-review lock, 2026-09-06), got {m.get('review')!r}")
check([rv["reviewer"] for rv in m["reviews"]] == ["LO", "AB"], f"reviews concatenated, survivor first, got {m.get('reviews')}")

# ── Two DIFFERENT accepted expert IDs refuse the merge ───────────────────────
def eid(code, mode="EXPERT"):
    return {"mode": mode, "code": code}

seg = fresh([
    mk(1, 10, 60, 10, 60, expert_id=eid("PA")),
    mk(3, 200, 280, 200, 280, expert_id=eid("SS")),
])
before_areas = [m["area"] if "area" in m else None for m in seg["masks"]]
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]}); j = r.get_json()
err = j.get("error") or ""
check(r.status_code == 400 and "#1" in err and "#3" in err and "PA" in err and "SS" in err
      and "Edit Masks" in err, f"conflicting expert IDs refuse the merge, got {r.status_code} {j}")
check([x["id"] for x in seg["masks"]] == [1, 3] and seg["masks"][0].get("expert_id") == eid("PA"),
      "refused merge leaves both masks and their expert IDs untouched")
r = client.post('/api/image/f.jpg/merge', json={"mask_id_a": 3, "mask_id_b": 1}); j = r.get_json()
check(r.status_code == 400 and "different expert IDs" in (j.get("error") or ""),
      f"pairwise /merge refuses the same conflict, got {r.status_code} {j}")

# One accepted ID and no rival: merges, and the survivor adopts it.
seg = fresh([mk(1, 10, 60, 10, 60), mk(3, 200, 280, 200, 280, expert_id=eid("SS"))])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]}); j = r.get_json()
check(r.status_code == 200 and j["mask"]["expert_id"] == eid("SS"), f"one accepted ID merges and carries, got {j}")

# The SAME accepted code on both is not a conflict.
seg = fresh([mk(1, 10, 60, 10, 60, expert_id=eid("SS")), mk(3, 200, 280, 200, 280, expert_id=eid("SS"))])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]})
check(r.status_code == 200, "the same expert code on both masks is not a conflict")

# Neither mask carries an ACCEPTED id: two different PENDING codes must not refuse
# (only mode EXPERT counts, matching mask_ops._merge_review_state).
seg = fresh([mk(1, 10, 60, 10, 60, expert_id=eid("PA", mode="PENDING")),
             mk(3, 200, 280, 200, 280, expert_id=eid("SS", mode="PENDING"))])
r = client.post('/api/image/f.jpg/merge_many', json={"mask_ids": [1, 3]})
check(r.status_code == 200, "pending (not accepted) expert IDs never conflict")

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
# A survivor carries a single review_uid, so merging two placeholder-species
# masks that are each out for their own expert ID would drop one round trip.
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

print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
