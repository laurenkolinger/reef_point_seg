"""Merge must keep every part: clean_mask(keep_largest=False), update_mask_geometry
(keep_all_components=True), union_many, and export joins polygons into one line.
Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_merge_components.py
"""
import os, sys
import numpy as np
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
import mask_ops as M
import export_yolo as E

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

def two_blobs():
    a = np.zeros((400, 400), dtype=bool); a[20:120, 20:120] = True      # 10000 px
    b = np.zeros((400, 400), dtype=bool); b[260:340, 260:340] = True    # 6400 px
    return a, b

a, b = two_blobs()
u = M.union_many([a, b])
check(u.dtype == bool and int(u.sum()) == 10000 + 6400, "union_many unions disjoint blobs")

kept = M.clean_mask(u, min_fragment=500, keep_largest=True)
check(int(kept.sum()) == 10000, "default clean_mask still keeps only the largest component")

allc = M.clean_mask(u, min_fragment=500, keep_largest=False)
check(int(allc.sum()) == 10000 + 6400, "keep_largest=False keeps both components")

speck = u.copy(); speck[390:395, 390:395] = True   # 25 px fragment
allc2 = M.clean_mask(speck, min_fragment=500, keep_largest=False)
check(int(allc2.sum()) == 10000 + 6400, "keep_largest=False still drops fragments below min_fragment")

d = {"id": 3, "rle": None}
out = M.update_mask_geometry(d, u, 0.001, keep_all_components=True)
check(out is not None and out["area"] == 10000 + 6400, "update_mask_geometry keep_all_components keeps area")
check(len(out["polygon_norm"]) == 2, f"two polygons emitted, got {len(out['polygon_norm'])}")
dec = M.rle_decode(out["rle"], (400, 400)).astype(bool)
check(int(dec.sum()) == 10000 + 6400, "RLE round-trips both components")

d2 = {"id": 4, "rle": None}
out2 = M.update_mask_geometry(d2, u, 0.001)
check(out2["area"] == 10000, "default update_mask_geometry unchanged (largest only)")

# export: two polygons -> ONE label line, covering both parts
joined = E.join_polygons(out["polygon_norm"])
check(isinstance(joined, list) and len(joined) >= 12 and len(joined) % 2 == 0,
      "join_polygons returns one flat even-length list")
xs = joined[0::2]; ys = joined[1::2]
check(min(xs) < 0.31 and max(xs) > 0.64, "joined polygon spans both blobs in x")
line = E.mask_to_yolo_line(joined, 0)
check(line.startswith("0 ") and line.count(" ") == len(joined), "YOLO line from joined polygon")
check(E.join_polygons([out2["polygon_norm"][0]]) == out2["polygon_norm"][0], "single polygon passes through unchanged")

# union_many contract: empty input is an error, not a silent None
try:
    M.union_many([])
    check(False, "union_many([]) raises ValueError")
except ValueError:
    check(True, "union_many([]) raises ValueError")

# clean_mask(keep_largest=False) fallback: every component below min_fragment
# keeps exactly the larger one (never vanishes to nothing)
small_a = np.zeros((400, 400), dtype=bool); small_a[10:20, 10:20] = True       # 100 px
small_b = np.zeros((400, 400), dtype=bool); small_b[200:205, 200:205] = True   # 25 px
u_small = M.union_many([small_a, small_b])
fallback = M.clean_mask(u_small, min_fragment=500, keep_largest=False)
check(int(fallback.sum()) == 100, f"keep_largest=False fallback keeps the larger of two sub-min_fragment components, got {int(fallback.sum())}")

# join_polygons edge cases
check(E.join_polygons([]) == [], "join_polygons([]) returns an empty list")

real_poly = [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]
degenerate = [0.5, 0.5, 0.6, 0.6]  # 2 points, filtered out as degenerate
check(E.join_polygons([degenerate, real_poly]) == real_poly, "degenerate polygon is filtered out; the real polygon passes through unchanged")

shared_a = [0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1]
shared_b = [0.0, 0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1]   # shares vertex (0.0, 0.0) with shared_a
joined_shared = E.join_polygons([shared_a, shared_b])
check(isinstance(joined_shared, list) and len(joined_shared) % 2 == 0,
      "shared-vertex join returns one flat even-length list")
sx, sy = joined_shared[0::2], joined_shared[1::2]
def _has_vertex(vx, vy):
    return any(abs(sx[k] - vx) < 1e-9 and abs(sy[k] - vy) < 1e-9 for k in range(len(sx)))
all_a_present = all(_has_vertex(shared_a[2 * k], shared_a[2 * k + 1]) for k in range(len(shared_a) // 2))
all_b_present = all(_has_vertex(shared_b[2 * k], shared_b[2 * k + 1]) for k in range(len(shared_b) // 2))
check(all_a_present and all_b_present, "shared-vertex join contains every vertex of both input polygons at least once")

# merge keeps a sub-500 px part: each input mask already passed its own 500 px
# gate when it was created, so the merge must never delete one of them.
big = np.zeros((400, 400), dtype=bool); big[20:120, 20:120] = True        # 10000 px
tiny = np.zeros((400, 400), dtype=bool); tiny[300:315, 300:315] = True    # 225 px
u_mix = M.union_many([big, tiny])
d3 = {"id": 5, "rle": None}
out3 = M.update_mask_geometry(d3, u_mix, 0.001, keep_all_components=True)
check(out3 is not None and out3["area"] == 10225,
      f"merge keeps the 225 px part, area {out3 and out3['area']}")
check(len(out3["polygon_norm"]) == 2, f"two polygons emitted, got {len(out3['polygon_norm'])}")
d4 = {"id": 6, "rle": None}
out4 = M.update_mask_geometry(d4, u_mix, 0.001)
check(out4["area"] == 10000, "a single-prompt mask still keeps the largest component only")

# min_hole is independent of min_fragment: a merge lowers the component floor to
# 1 px, and its interior holes must still be filled. mask_to_polygon traces
# external contours only, so an unfilled hole leaves the reported area smaller
# than the polygon shipped with it, by the size of the hole.
holed = np.zeros((400, 400), dtype=bool)
holed[100:300, 100:300] = True     # 40000 px block
holed[150:172, 150:172] = False    # 484 px interior hole
d5 = {"id": 7, "rle": None}
out5 = M.update_mask_geometry(d5, holed, 0.001, keep_all_components=True, min_fragment=1)
poly_area5 = int(M.polygon_to_mask(out5["polygon_px"][0], 400, 400).sum())
check(out5["area"] == 40000, f"a merge fills a 484 px interior hole, area {out5['area']}")
check(out5["area"] == poly_area5,
      f"merged area agrees with the polygon it reports, {out5['area']} vs {poly_area5}")
check(len(out5["polygon_norm"]) == 1, f"one polygon, got {len(out5['polygon_norm'])}")

# The same union with hole filling tied to the component floor is what the bug
# looked like: area and polygon disagree by the hole.
unfilled = M.clean_mask(holed, min_fragment=1, keep_largest=False, min_hole=1).astype(bool)
poly_unfilled, _ = M.mask_to_polygon(unfilled, 0.001)
check(int(unfilled.sum()) < int(M.polygon_to_mask(poly_unfilled[0], 400, 400).sum()),
      "control: min_hole=1 leaves the hole open and the two disagree")

# draw_edit on a merged mask: keep_all_components=True with min_fragment=20, so a
# sliver left by a subtract stroke or the neighbour clip goes and real parts stay.
de_big = np.zeros((400, 400), dtype=bool); de_big[20:120, 20:120] = True      # 10000 px
de_part = np.zeros((400, 400), dtype=bool); de_part[300:310, 300:310] = True  # 100 px
de_sliver = np.zeros((400, 400), dtype=bool); de_sliver[380:383, 380:384] = True  # 12 px
de_u = M.union_many([de_big, de_part, de_sliver])
d6 = {"id": 8, "rle": None}
out6 = M.update_mask_geometry(d6, de_u, 0.001, keep_all_components=True, min_fragment=20)
check(out6["area"] == 10100, f"draw edit drops the 12 px sliver, keeps the 100 px part, area {out6['area']}")
check(len(out6["polygon_norm"]) == 2, f"two polygons after the sliver goes, got {len(out6['polygon_norm'])}")
d7 = {"id": 9, "rle": None}
out7 = M.update_mask_geometry(d7, de_u, 0.001, keep_all_components=True)
check(out7["area"] == 10112 and len(out7["polygon_norm"]) == 3,
      f"the merge default (min_fragment=1) still keeps all three, area {out7['area']}")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
