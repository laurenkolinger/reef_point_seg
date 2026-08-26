#!/usr/bin/env python3
"""Headless routing-accountability verification for TCRMPclip_placePoints.

Run with the unified env python:
    env/bin/python scripts/TCRMPclip_placePoints/tests/test_routing_report.py

Verifies Task 2 (reporting only): configure() accounts for EVERY selected
frame so none is silently lost. CPC-only fixture (NO GPU / OCR needed):
  * one routable CPC frame (raw image + CPC points present)
  * one genuinely-missing CPC frame (no raw image, no CPC points)

Asserts:
  * routing_report.json + routing_report.csv are written to the export dir
  * the report records both frames (routed + cpc_missing)
  * the /api/configure response dropped[] contains the missing frame
  * the reconciliation holds: routed + already_processed + dropped == len(df)
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


# ── Fixture: one routable + one genuinely-missing CPC frame ─────────────
GOOD = (2014, 'TCRMP20140926_clip_BWR_T101', 1, 1, 'BWR', '2014-09-26')
MISSING = (2016, 'TCRMP20160815_clip_GHOST_T102', 1, 2, 'GHOST', '2016-08-15')
FRAMES = [GOOD, MISSING]
TARGET = 'DUMMY'

tmp = tempfile.mkdtemp(prefix='pp_routing_report_')
selected_frames = os.path.join(tmp, 'selected_frames.csv')
export_dir = os.path.join(tmp, 'export')
cpc_dir = os.path.join(tmp, 'cpc_all')
clip_dir = os.path.join(tmp, 'clip')

# 1) selected_frames.csv (both rows)
with open(selected_frames, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['basename', 'year', 'route', 'date', 'site',
                'transect', 'frame', 'species_present'])
    for (year, basename, transect, frame, site, date) in FRAMES:
        w.writerow([basename, year, 'cpc', date, site,
                    transect, frame, TARGET])

try:
    from PIL import Image as PILImage
except Exception as e:  # noqa: BLE001
    print('  FAIL: Pillow unavailable for fixture:', repr(e))
    print('test_routing_report: 0 passed, 1 failed')
    sys.exit(1)

# 2) Only the GOOD frame gets CPC points + a raw image. The MISSING frame
#    gets neither, so it must drop as cpc_missing.
(gyear, gbase, gtransect, gframe, gsite, gdate) = GOOD
ids_dir = os.path.join(cpc_dir, str(gyear), 'ids')
os.makedirs(ids_dir, exist_ok=True)
with open(os.path.join(ids_dir, 'point_coords.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'raw_image', 'label', 'species_code', 'species_name',
        'category', 'x', 'y', 'source'])
    w.writeheader()
    for (lbl, x, y) in [('A', 100.0, 120.0), ('B', 200.0, 220.0)]:
        w.writerow({
            'raw_image': gbase + '.jpeg', 'label': lbl,
            'species_code': TARGET, 'species_name': 'Dummy species',
            'category': 'coral', 'x': x, 'y': y, 'source': 'cpc',
        })
year_clip = os.path.join(clip_dir, f'TCRMP{gyear}_clip')
os.makedirs(year_clip, exist_ok=True)
PILImage.new('RGB', (640, 480), (10, 20, 30)).save(
    os.path.join(year_clip, gbase + '.jpeg'))

# ── Env BEFORE importing app.py ──
os.environ['TCRMP_SELECTED_FRAMES'] = selected_frames
os.environ['TCRMP_EXPORT_DIR'] = export_dir
os.environ['TCRMP_CPC_DIR'] = cpc_dir
os.environ['TCRMP_CLIP_DIR'] = clip_dir
os.environ['TCRMP_TARGET_SPECIES'] = TARGET
os.environ['TCRMP_REMAP_LOG'] = os.path.join(tmp, '_no_remap.json')

prev = os.getcwd()
try:
    os.chdir(PP_SRC)
    sys.path.insert(0, PP_SRC)
    spec = importlib.util.spec_from_file_location(
        'placepoints_app_rr', os.path.join(PP_SRC, 'app.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client = mod.app.test_client()

    r = client.post('/api/configure', json={
        'selected_frames': selected_frames,
        'export_dir': export_dir,
        'species_filter': [],
        'review_batch_size': 999999,
        'reference_mode': False,
        'shuffle': False,
    })
    ok(r.status_code == 200,
       f'/api/configure status ({r.status_code}): {r.get_data(as_text=True)[:200]}')
    resp = r.get_json() or {}

    # 1) Exactly one routed (the GOOD frame), one cpc_missing.
    ok(resp.get('cpc_loaded', -1) == 1,
       f'cpc_loaded={resp.get("cpc_loaded")} (expected 1)')
    ok(resp.get('cpc_missing', -1) == 1,
       f'cpc_missing={resp.get("cpc_missing")} (expected 1)')

    # 2) dropped[] contains the missing frame, not the good one.
    dropped = resp.get('dropped', [])
    ok(len(dropped) == 1, f'dropped len={len(dropped)} (expected 1): {dropped}')
    if dropped:
        d = dropped[0]
        ok(d.get('basename') == MISSING[1],
           f'dropped basename={d.get("basename")} (expected {MISSING[1]})')
        ok(d.get('reason') == 'cpc_missing',
           f'dropped reason={d.get("reason")} (expected cpc_missing)')

    # 3) Reconciliation holds.
    ok(resp.get('reconciled') is True,
       f'reconciled flag={resp.get("reconciled")} (expected True)')

    # 4) routing_report.json + .csv written and record BOTH frames.
    json_path = os.path.join(export_dir, 'routing_report.json')
    csv_path = os.path.join(export_dir, 'routing_report.csv')
    ok(os.path.exists(json_path), f'routing_report.json missing: {json_path}')
    ok(os.path.exists(csv_path), f'routing_report.csv missing: {csv_path}')

    if os.path.exists(json_path):
        with open(json_path) as f:
            report = json.load(f)
        ok(len(report) == len(FRAMES),
           f'routing_report rows={len(report)} (expected {len(FRAMES)})')
        by_base = {e['basename']: e for e in report}
        ok(by_base.get(gbase, {}).get('outcome') == 'routed',
           f'{gbase} outcome={by_base.get(gbase, {}).get("outcome")} (expected routed)')
        ok(by_base.get(MISSING[1], {}).get('outcome') == 'cpc_missing',
           f'{MISSING[1]} outcome={by_base.get(MISSING[1], {}).get("outcome")} '
           '(expected cpc_missing)')
        # Reconciliation invariant cross-check against the report itself.
        routed_n = sum(1 for e in report if e['outcome'] == 'routed')
        ap_n = sum(1 for e in report if e['outcome'] == 'already_processed')
        drop_n = sum(1 for e in report
                     if e['outcome'] not in ('routed', 'already_processed'))
        ok(routed_n + ap_n + drop_n == len(report),
           f'report partition {routed_n}+{ap_n}+{drop_n} != {len(report)}')

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        ok(len(rows) == len(FRAMES),
           f'routing_report.csv rows={len(rows)} (expected {len(FRAMES)})')
        ok(set(rows[0].keys()) >= {
            'frame_id', 'basename', 'year', 'route', 'outcome',
            'resolved_path', 'note'},
           f'routing_report.csv header missing fields: {rows[0].keys() if rows else None}')

except Exception as e:  # noqa: BLE001
    failed += 1
    print('  FAIL: routing report flow:', repr(e))
finally:
    os.chdir(prev)

print(f'test_routing_report: {passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
