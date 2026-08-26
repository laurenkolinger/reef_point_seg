"""placePoints lores delivery: every >1920 frame is delivered via its lores twin
(keyed off the resolved raw path, no name guessing) with points scaled to match;
already-small frames and frames with no twin fall back to the original (never
skipped). Run with env/bin/python.
"""
import os, sys, csv, json, tempfile
from PIL import Image
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
import export as E

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1


def _mk(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


with tempfile.TemporaryDirectory() as d:
    clip = os.path.join(d, "TCRMP_clip", "TCRMP2017_clip", "TCRMP20171202_clip_CKR")
    lores = os.path.join(d, "TCRMP_clip_lores", "TCRMP2017_clip", "TCRMP20171202_clip_CKR")
    big = os.path.join(clip, "TCRMP20171202_clip_CKR_T403.jpg")      # 4K, has twin
    big_twin = os.path.join(lores, "TCRMP20171202_clip_CKR_T403_lores.jpg")
    small = os.path.join(clip, "TCRMP20171202_clip_CKR_T114.jpg")    # already HD, no twin
    orphan = os.path.join(clip, "TCRMP20171202_clip_CKR_T999.jpg")   # 4K but twin missing
    _mk(big, (3840, 2160)); _mk(big_twin, (1920, 1080))
    _mk(small, (1920, 1080)); _mk(orphan, (3840, 2160))

    # ── _lores_twin: deterministic path mirror ──
    check(E._lores_twin(big) == big_twin, "twin path mirrors TCRMP_clip -> TCRMP_clip_lores + _lores.jpg")
    check(E._lores_twin("/nope/x.jpg") is None, "non-clip path -> no twin")

    # ── lores_delivery decisions ──
    check(E.lores_delivery(big, False) == (big, 1.0), "lores_mode off -> original, scale 1.0")
    check(E.lores_delivery(small, True) == (small, 1.0), "already-small frame -> original, scale 1.0")
    p, s = E.lores_delivery(big, True)
    check(p == big_twin and abs(s - 0.5) < 1e-9, "4K frame with twin -> twin, scale 0.5")
    check(E.lores_delivery(orphan, True) == (orphan, 1.0), "4K frame, twin missing -> original (NEVER skipped)")

    # ── export_batch end-to-end: lores_mode copies the twin (1920px) + halves pts ──
    det = {
        "filename": "TCRMP20171202_clip_CKR_T403_pts.jpg",
        "raw_path": big, "year": 2017, "date_str": "20171202",
        "site": "CKR", "t_id": "403",
        "points": [{"label": "A", "x": 1871, "y": 778,
                    "species_code": "PA", "species_name": "", "category": ""}],
    }
    out = os.path.join(d, "out")
    stats = E.export_batch([det], out, log_fn=lambda *_: None, lores_mode=True)
    sam = json.load(open(os.path.join(out, "2017", "ids", "sam_click_prompts.json")))
    pt = sam["TCRMP20171202_clip_CKR_T403.jpeg"]["points"][0]
    check(pt["x"] == 936 and pt["y"] == 389, f"points halved to lores space (got {pt['x']},{pt['y']})")
    copied = os.path.join(out, "2017", "raw", "TCRMP20171202_clip_CKR_T403.jpeg")
    check(Image.open(copied).size == (1920, 1080), "copied raw is the 1920px lores image")
    check(stats["lores_delivered"] == 1, "stats count the lores-delivered frame")

    # ── lores_mode OFF: original 4K copied, points untouched ──
    out2 = os.path.join(d, "out2")
    E.export_batch([det], out2, log_fn=lambda *_: None, lores_mode=False)
    sam2 = json.load(open(os.path.join(out2, "2017", "ids", "sam_click_prompts.json")))
    pt2 = sam2["TCRMP20171202_clip_CKR_T403.jpeg"]["points"][0]
    check(pt2["x"] == 1871 and pt2["y"] == 778, "lores off -> points unchanged")
    copied2 = os.path.join(out2, "2017", "raw", "TCRMP20171202_clip_CKR_T403.jpeg")
    check(Image.open(copied2).size == (3840, 2160), "lores off -> original 4K copied")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
