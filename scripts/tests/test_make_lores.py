"""make_lores_variants: downscales >1920 frames into a parallel mirror with a
manifest, is idempotent, and skips already-small frames. Run with env/bin/python.
"""
import os, sys, csv, tempfile
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # scripts/
import make_lores_variants as M

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

with tempfile.TemporaryDirectory() as d:
    clip = os.path.join(d, "TCRMP_clip", "TCRMP2018_clip", "Annual", "T105")
    os.makedirs(clip)
    big = os.path.join(clip, "TCRMP20181207_clip_MRS_T105.jpg")
    small = os.path.join(clip, "TCRMP20141012_clip_CBS_T306.jpg")
    pts = os.path.join(clip, "TCRMP20181207_clip_MRS_T105_pts.jpg")  # 4K overlay, must be skipped
    Image.new("RGB", (3840, 2160), (10, 20, 30)).save(big)
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(small)
    Image.new("RGB", (3840, 2160), (40, 50, 60)).save(pts)
    lores_root = os.path.join(d, "TCRMP_clip_lores")

    stats = M.generate(os.path.join(d, "TCRMP_clip"), lores_root, log=lambda *_: None)
    check(stats["generated"] == 1, "one lores generated (the 4K raw frame; _pts overlay skipped)")
    check(stats["skipped_small"] == 1, "the 1920 frame skipped")
    check(not os.path.exists(os.path.join(lores_root, "TCRMP2018_clip", "Annual", "T105",
          "TCRMP20181207_clip_MRS_T105_pts_lores.jpg")), "_pts overlay produced no lores copy")

    lp = os.path.join(lores_root, "TCRMP2018_clip", "Annual", "T105",
                      "TCRMP20181207_clip_MRS_T105_lores.jpg")
    check(os.path.isfile(lp), "lores file at mirrored path")
    w, h = Image.open(lp).size
    check(max(w, h) == 1920, "lores long edge is 1920")

    man = os.path.join(lores_root, "lores_manifest.csv")
    rows = list(csv.DictReader(open(man)))
    check(len(rows) == 1, "manifest has one row")
    r = rows[0]
    check(r["basename"] == "TCRMP20181207_clip_MRS_T105", "manifest basename")
    check(abs(float(r["scale"]) - 1920/3840) < 1e-6, "manifest scale = 0.5")

    stats2 = M.generate(os.path.join(d, "TCRMP_clip"), lores_root, log=lambda *_: None)
    check(stats2["up_to_date"] == 1 and stats2["generated"] == 0, "idempotent re-run")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
