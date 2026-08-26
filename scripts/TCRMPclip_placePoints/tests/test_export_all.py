#!/usr/bin/env python3
"""Headless /api/export_all verification for TCRMPclip_placePoints.

Run with the unified env python:
    env/bin/python scripts/TCRMPclip_placePoints/tests/test_export_all.py

Exercises the new POST /api/export_all route that flushes EVERY routed frame
across ALL years in one call (not a review-batch slice), used by the
orchestrator's background routing driver so 4.test can fully replace Step 4.

CPC-only fixture so NO GPU / OCR is needed:
  * a tmp selected_frames.csv with 2 cpc rows (two different years)
  * a tmp cpc_all/{year}/ids/point_coords.csv with matching points
  * tmp raw images under clip_dir/TCRMP{year}_clip/

Asserts {export_dir}/{year}/ids/sam_click_prompts.json exists for EVERY year
and that every routed frame appears in it.
"""
import os
import sys
import csv
import json
import tempfile
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
PP_SRC = os.path.normpath(os.path.join(HERE, '..', 'src'))

passed = 0
failed = 0


def ok(cond, msg):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print('  FAIL:', msg)


# ── Build a CPC-only fixture (2 frames across 2 years) ──────────────────
# Frames: BWR T101 (2014) and BWR T102 (2016). raw_image stems must match the
# selected_frames basenames so load_cpc_points pairs coords to frames.
FRAMES = [
    # (year, basename, transect, frame, site, date)
    (2014, 'TCRMP20140926_clip_BWR_T101', 1, 1, 'BWR', '2014-09-26'),
    (2016, 'TCRMP20160815_clip_BWR_T102', 1, 2, 'BWR', '2016-08-15'),
]
TARGET = 'DUMMY'  # unlikely to collide with any real species remap

tmp = tempfile.mkdtemp(prefix='pp_export_all_')
selected_frames = os.path.join(tmp, 'selected_frames.csv')
export_dir = os.path.join(tmp, 'export')
cpc_dir = os.path.join(tmp, 'cpc_all')
clip_dir = os.path.join(tmp, 'clip')

# 1) selected_frames.csv
with open(selected_frames, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['basename', 'year', 'route', 'date', 'site',
                'transect', 'frame', 'species_present'])
    for (year, basename, transect, frame, site, date) in FRAMES:
        w.writerow([basename, year, 'cpc', date, site,
                    transect, frame, TARGET])

# 2) cpc_all/{year}/ids/point_coords.csv + 3) raw images
try:
    from PIL import Image as PILImage
except Exception as e:  # noqa: BLE001
    print('  FAIL: Pillow unavailable for fixture:', repr(e))
    print('test_export_all: 0 passed, 1 failed')
    sys.exit(1)

for (year, basename, transect, frame, site, date) in FRAMES:
    ids_dir = os.path.join(cpc_dir, str(year), 'ids')
    os.makedirs(ids_dir, exist_ok=True)
    pc = os.path.join(ids_dir, 'point_coords.csv')
    with open(pc, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'raw_image', 'label', 'species_code', 'species_name',
            'category', 'x', 'y', 'source'])
        w.writeheader()
        # two points per frame
        for i, (lbl, x, y) in enumerate([('A', 100.0, 120.0), ('B', 200.0, 220.0)]):
            w.writerow({
                'raw_image': basename + '.jpeg',
                'label': lbl,
                'species_code': TARGET,
                'species_name': 'Dummy species',
                'category': 'coral',
                'x': x, 'y': y, 'source': 'cpc',
            })
    # raw image at clip_dir/TCRMP{year}_clip/<basename>.jpeg
    year_clip = os.path.join(clip_dir, f'TCRMP{year}_clip')
    os.makedirs(year_clip, exist_ok=True)
    PILImage.new('RGB', (640, 480), (10, 20, 30)).save(
        os.path.join(year_clip, basename + '.jpeg'))

# ── Env BEFORE importing app.py (config.py + ALL_TARGET_SPECIES read at import) ──
os.environ['TCRMP_SELECTED_FRAMES'] = selected_frames
os.environ['TCRMP_EXPORT_DIR'] = export_dir
os.environ['TCRMP_CPC_DIR'] = cpc_dir
os.environ['TCRMP_CLIP_DIR'] = clip_dir
os.environ['TCRMP_TARGET_SPECIES'] = TARGET
os.environ['TCRMP_REMAP_LOG'] = os.path.join(tmp, '_no_remap.json')  # absent => no remap

prev = os.getcwd()
try:
    os.chdir(PP_SRC)
    sys.path.insert(0, PP_SRC)
    spec = importlib.util.spec_from_file_location(
        'placepoints_app', os.path.join(PP_SRC, 'app.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client = mod.app.test_client()

    # 1) Configure (loads CPC frames, no OCR queue => phase=review).
    r = client.post('/api/configure', json={
        'selected_frames': selected_frames,
        'export_dir': export_dir,
        'species_filter': [],
        'review_batch_size': 999999,
        'reference_mode': False,
        'shuffle': False,
    })
    ok(r.status_code == 200, f'/api/configure status ({r.status_code}): {r.get_data(as_text=True)[:200]}')
    cfg_resp = r.get_json() or {}
    ok(cfg_resp.get('cpc_loaded', 0) == len(FRAMES),
       f'/api/configure cpc_loaded={cfg_resp.get("cpc_loaded")} (expected {len(FRAMES)})')
    ok(cfg_resp.get('ocr_remaining', -1) == 0,
       f'/api/configure ocr_remaining={cfg_resp.get("ocr_remaining")} (expected 0, CPC-only)')

    # 2) Export ALL in one headless call.
    r = client.post('/api/export_all', json={})
    ok(r.status_code == 200, f'/api/export_all status ({r.status_code}): {r.get_data(as_text=True)[:200]}')
    resp = r.get_json() or {}
    ok(resp.get('ok') is True, f'/api/export_all ok flag missing: {resp}')
    ok(resp.get('exported_frames', 0) == len(FRAMES),
       f'/api/export_all exported_frames={resp.get("exported_frames")} (expected {len(FRAMES)})')

    expected_years = sorted({yr for (yr, *_rest) in FRAMES})
    ok(sorted(resp.get('years', [])) == expected_years,
       f'/api/export_all years={resp.get("years")} (expected {expected_years})')

    # 3) sam_click_prompts.json exists for EVERY year and holds the frame.
    for (year, basename, *_rest) in FRAMES:
        sam_path = os.path.join(export_dir, str(year), 'ids', 'sam_click_prompts.json')
        ok(os.path.exists(sam_path), f'sam_click_prompts.json missing for {year}: {sam_path}')
        if os.path.exists(sam_path):
            with open(sam_path) as f:
                sam = json.load(f)
            raw_name = basename + '.jpeg'
            ok(raw_name in sam, f'{raw_name} not in {year} sam_click_prompts.json (keys={list(sam.keys())})')

    # 4) raw/ copies present for every frame.
    for (year, basename, *_rest) in FRAMES:
        raw_copy = os.path.join(export_dir, str(year), 'raw', basename + '.jpeg')
        ok(os.path.exists(raw_copy), f'raw copy missing for {year}: {raw_copy}')

except Exception as e:  # noqa: BLE001
    failed += 1
    print('  FAIL: export_all flow:', repr(e))
finally:
    os.chdir(prev)

print(f'test_export_all: {passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
