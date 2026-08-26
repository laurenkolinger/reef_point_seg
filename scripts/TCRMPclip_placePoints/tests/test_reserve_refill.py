"""On ocr_failed, the next valid reserve frame is pulled in and recorded.

Mirrors test_routing_report.py's fixture + test_client pattern. Run with the
unified env python (no pytest), standalone:

    env/bin/python scripts/TCRMPclip_placePoints/tests/test_reserve_refill.py

The fixture is GPU-free: the OCR frame's _pts image is a blank solid-color
image, so detect_annotations finds no red letter annotations -> run_detection
returns [] -> ocr_failed -> the reserve refill fires. A second blank reserve
_pts also "fails" OCR, but it is still pulled in and *recorded* with refill
provenance, which is what this test asserts (the swap + the report row).
"""
import os
import sys
import csv
import json
import tempfile
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
PP_SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

passed = failed = 0


def ok(c, m):
    global passed, failed
    if c:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", m)


tmp = tempfile.mkdtemp(prefix="pp_refill_")
selected = os.path.join(tmp, "selected_frames.csv")
reserve = os.path.join(tmp, "reserve_frames.csv")
export_dir = os.path.join(tmp, "export")
clip_dir = os.path.join(tmp, "clip")
os.makedirs(clip_dir)
TARGET = "PA"

try:
    from PIL import Image as PILImage
except Exception as e:  # noqa: BLE001
    print("  FAIL: Pillow unavailable for fixture:", repr(e))
    print("test_reserve_refill: 0 passed, 1 failed")
    sys.exit(1)


def _img(p):
    # Solid dark frame: no red letter blobs -> OCR detects nothing.
    PILImage.new("RGB", (64, 48), (9, 9, 9)).save(p)


# Selected: one OCR frame whose _pts EXISTS but OCR will detect nothing.
fail_base = "TCRMP20220601_clip_FLC_T307"
_img(os.path.join(clip_dir, fail_base + ".jpg"))
_img(os.path.join(clip_dir, fail_base + "_pts.jpg"))  # blank _pts -> OCR finds nothing
with open(selected, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["basename", "year", "route", "date", "site",
                "transect", "frame", "species_present"])
    w.writerow([fail_base, 2022, "ocr_needed", "2022-06-01", "FLC", 3, 7, TARGET])

# Reserve: a CPC row (rank 0, NO _pts) placed BEFORE the ocr_needed row to
# verify it is not wastefully consumed first (M2 fix).
cpc_base = "TCRMP20180601_clip_FLC_T301"
# CPC rows have no _pts image — do NOT create one, that is the point.
_img(os.path.join(clip_dir, cpc_base + ".jpg"))
# ocr_needed reserve frame with a _pts that exists (resolvable).
res_base = "TCRMP20220601_clip_FLC_T308"
_img(os.path.join(clip_dir, res_base + ".jpg"))
_img(os.path.join(clip_dir, res_base + "_pts.jpg"))
with open(reserve, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_id", "basename", "date", "year", "site", "transect",
                "frame", "species", "reserve_rank", "route",
                "source_image", "pts_image"])
    # CPC row first (lower rank = higher priority under old code) — must be skipped.
    w.writerow([cpc_base, cpc_base, "2018-06-01", 2018, "FLC", 3, 1, TARGET, 0,
                "cpc",
                os.path.join(clip_dir, cpc_base + ".jpg"),
                ""])
    w.writerow([res_base, res_base, "2022-06-01", 2022, "FLC", 3, 8, TARGET, 1,
                "ocr_needed",
                os.path.join(clip_dir, res_base + ".jpg"),
                os.path.join(clip_dir, res_base + "_pts.jpg")])

os.environ.update({
    "TCRMP_SELECTED_FRAMES": selected,
    "TCRMP_EXPORT_DIR": export_dir,
    "TCRMP_CLIP_DIR": clip_dir,
    "TCRMP_TARGET_SPECIES": TARGET,
    "TCRMP_REMAP_LOG": os.path.join(tmp, "_no_remap.json"),
    "TCRMP_RESERVE_FRAMES": reserve,
})

prev = os.getcwd()
try:
    os.chdir(PP_SRC)
    sys.path.insert(0, PP_SRC)
    spec = importlib.util.spec_from_file_location(
        "pp_refill_app", os.path.join(PP_SRC, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Headless: never touch the GPU for OCR in this test.
    mod.USE_GPU = False
    client = mod.app.test_client()

    r = client.post("/api/configure", json={
        "selected_frames": selected,
        "export_dir": export_dir,
        "species_filter": [],
        "review_batch_size": 999999,
    })
    ok(r.status_code == 200,
       f"/configure status {r.status_code}: {r.get_data(as_text=True)[:200]}")

    # Helper the app exposes to run the OCR queue headlessly (test seam).
    res = mod.run_ocr_queue_for_test()
    ok(res.get("refilled", 0) >= 1, f"a reserve frame was pulled in: {res}")

    rep_path = os.path.join(export_dir, "routing_report.json")
    ok(os.path.exists(rep_path), f"routing_report.json missing: {rep_path}")
    rep = json.load(open(rep_path))
    refilled = [e for e in rep if e.get("refilled_from")]
    ok(any(e.get("basename") == res_base for e in rep),
       "reserve frame appears in report")
    ok(len(refilled) >= 1 and refilled[0]["refill_reason"] == "ocr_failed",
       f"refill recorded with reason: {refilled}")
    # The refill must point back at the frame it replaced.
    ok(any(e.get("refilled_from") == fail_base for e in refilled),
       f"refill provenance names the failed frame: {refilled}")
    # M2: CPC reserve row must NOT have been consumed — the ocr_needed row
    # should have been selected directly without burning through cpc_base first.
    ok(not any(e.get("basename") == cpc_base for e in rep),
       f"CPC reserve row must not appear in report (should be skipped by M2 filter): "
       f"{[e.get('basename') for e in rep]}")
    # routing_report.csv must carry the new columns too.
    csv_path = os.path.join(export_dir, "routing_report.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        ok(rows and "refilled_from" in rows[0] and "refill_reason" in rows[0],
           f"routing_report.csv header lacks refill columns: "
           f"{list(rows[0].keys()) if rows else None}")
except Exception as e:  # noqa: BLE001
    failed += 1
    print("  FAIL: refill flow:", repr(e))
finally:
    os.chdir(prev)

print(f"test_reserve_refill: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
