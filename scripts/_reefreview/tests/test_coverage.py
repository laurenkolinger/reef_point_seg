"""
Unit tests for the coverage planner (_reefreview.coverage).
Run: env/bin/python scripts/_reefreview/tests/test_coverage.py
"""

import os
import sys
import csv
import collections
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _SCRIPTS)

from _reefreview import coverage
from _reefreview.library import Library

_R = []
def check(c, m):
    if not c: raise AssertionError(m)
def run(fn):
    try: fn(); _R.append((fn.__name__, True, '')); print(f"  PASS {fn.__name__}")
    except Exception as e: _R.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}")); print(f"  FAIL {fn.__name__}: {e}")

REAL_AP = '/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/all_points.csv'
COLS = ['date', 'year', 'site', 'transect', 'frame', 'point_label', 'species_code', 'species_name', 'category']


def _write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({**{c: '' for c in COLS}, **r})


def _synth_csv(path):
    rows = []
    def add(site, t, fr, code, n, yr='2020', date='2020-10-22'):
        for i in range(n):
            rows.append({'date': date, 'year': yr, 'site': site, 'transect': t,
                         'frame': fr, 'point_label': chr(65 + i), 'species_code': code})
    add('SCP', '1', '1', 'PA', 2); add('SCP', '1', '1', 'OFAV', 1)   # img1: PA+OFAV
    add('SCP', '1', '2', 'PA', 3)                                     # img2: PA
    add('BIT', '2', '1', 'OFAV', 2); add('BIT', '2', '1', 'AA', 1)   # img3: OFAV+AA
    _write_csv(path, rows)

K1 = ('20201022', 'SCP', '1', '1')
K2 = ('20201022', 'SCP', '1', '2')
K3 = ('20201022', 'BIT', '2', '1')


def test_identity_keys_and_filename_join():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        census, meta = coverage.build_census(ap)
        check(meta['n_images'] == 3, meta)
        check(census[K1]['PA'] == 2 and census[K1]['OFAV'] == 1, dict(census[K1]))
        # the census key equals the key parsed from the matching clip filename
        check(coverage.key_from_filename('TCRMP20201022_clip_SCP_T101.jpeg') == K1,
              coverage.key_from_filename('TCRMP20201022_clip_SCP_T101.jpeg'))


def test_filters_and_year_coerce():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv')
        _write_csv(ap, [
            {'date': '2014-09-30', 'year': '2014.0', 'site': 'BWR', 'transect': '1', 'frame': '1',
             'point_label': 'A', 'species_code': 'PA'},
            {'date': '2020-10-22', 'year': '', 'site': 'SCP', 'transect': '1', 'frame': '1',
             'point_label': 'A', 'species_code': 'PA'}])
        c1, _ = coverage.build_census(ap, year_min=2018)   # '2014.0' must coerce
        check(len(c1) == 1 and K1 in c1, list(c1))
        c2, _ = coverage.build_census(ap, sites=['bwr'])    # case-insensitive
        check(len(c2) == 1 and ('20140930', 'BWR', '1', '1') in c2, list(c2))


def test_remap():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv')
        _write_csv(ap, [
            {'date': '2014-09-30', 'year': '2014', 'site': 'BWR', 'transect': '1', 'frame': '1',
             'point_label': 'A', 'species_code': 'MFAV'},
            {'date': '2014-09-30', 'year': '2014', 'site': 'BWR', 'transect': '1', 'frame': '1',
             'point_label': 'B', 'species_code': 'OFAV'}])
        census, _ = coverage.build_census(ap, remap={'MFAV': 'OFAV'})
        k = ('20140930', 'BWR', '1', '1')
        check(census[k]['OFAV'] == 2 and 'MFAV' not in census[k], dict(census[k]))


def test_load_remap():
    with tempfile.TemporaryDirectory() as t:
        import json
        p = os.path.join(t, 'remap_log.json')
        json.dump({'remaps': [{'old_code': 'MFAV', 'new_code': 'OFAV'},
                              {'old_code': 'X', 'new_code': 'X'}]}, open(p, 'w'))
        check(coverage.load_remap(p) == {'MFAV': 'OFAV'}, coverage.load_remap(p))
        check(coverage.load_remap('/nope.json') == {}, 'missing -> {}')


def test_report_counts_cooc_shortfall():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        census, _ = coverage.build_census(ap)
        rep = coverage.coverage_report(census, ['PA', 'OFAV', 'AA'],
                                       target_per_code={'PA': 10, 'OFAV': 2, 'AA': 5})
        pc = rep['per_code']
        check(pc['PA']['census_images'] == 2 and pc['PA']['census_instances'] == 5, pc['PA'])
        check(pc['OFAV']['census_instances'] == 3, pc['OFAV'])
        check(rep['cooccurrence']['PA']['OFAV'] == 1, rep['cooccurrence'])
        check(rep['shortfall'] == {'PA': 5, 'OFAV': 0, 'AA': 4}, rep['shortfall'])


def test_empty_codes_guard():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        census, _ = coverage.build_census(ap)
        check(coverage.coverage_report(census, []).get('empty') is True, 'empty codes')
        check(coverage.coverage_report(census, ['', None]).get('empty') is True, 'blank codes')


def test_non_identity_census_raises():
    bad = {'TCRMP20201022_clip_SCP_T101.jpeg': collections.Counter({'PA': 2})}
    try:
        coverage.coverage_report(bad, ['PA'])
        raise AssertionError("expected ValueError for basename-keyed census")
    except ValueError:
        pass


def test_non_identity_census_guard_checks_all_keys():
    # A stray basename key beyond the first few must still be caught (the old
    # guard only sampled 5 keys).
    census = {('20201022', 'SCP', '1', str(i)): collections.Counter({'PA': 1})
              for i in range(20)}
    census['TCRMP20201022_clip_SCP_T199.jpeg'] = collections.Counter({'PA': 1})  # stray at the end
    try:
        coverage.coverage_report(census, ['PA'])
        raise AssertionError("expected ValueError for a stray basename key")
    except ValueError:
        pass


def test_catalog_segmented_vs_gap():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        census, _ = coverage.build_census(ap)
        lib = Library(os.path.join(t, '_lib')); lib.ensure()
        lib.upsert({'uid': 'U1', 'code': 'PA', 'mode': 'EXPERT',
                    'source_image': 'TCRMP20201022_clip_SCP_T101.jpeg'})  # img1 PA mask
        rep = coverage.coverage_report(census, ['PA', 'OFAV'], library=lib)
        check(rep['per_code']['PA']['segmented_images'] == 1, rep['per_code']['PA'])
        check(rep['per_code']['PA']['gap_images'] == 1, rep['per_code']['PA'])     # img2 PA, no mask
        check(rep['per_code']['OFAV']['segmented_images'] == 0, rep['per_code']['OFAV'])
        check(coverage.gap_images(census, 'PA', lib) == [K2], coverage.gap_images(census, 'PA', lib))


def test_review_code_not_segmented():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        census, _ = coverage.build_census(ap)
        lib = Library(os.path.join(t, '_lib')); lib.ensure()
        lib.upsert({'uid': 'U1', 'code': 'REVIEW', 'mode': 'USER',
                    'source_image': 'TCRMP20201022_clip_SCP_T101.jpeg'})
        rep = coverage.coverage_report(census, ['PA'], library=lib)
        check(rep['per_code']['PA']['segmented_images'] == 0, 'REVIEW not segmented')
        check(rep['per_code']['PA']['gap_images'] == 2, rep['per_code']['PA'])


def test_select_minimal_cooc():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        census, _ = coverage.build_census(ap)
        sel = coverage.select_images(census, {'PA': 1, 'OFAV': 1})   # img1 has both
        check(sel['images_reviewed'] == 1 and not sel['capped'], sel)
        check(sel['shortfall'] == {'PA': 0, 'OFAV': 0}, sel['shortfall'])


def test_select_cap_and_unreachable():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        census, _ = coverage.build_census(ap)
        sel = coverage.select_images(census, {'PA': 99}, max_images=1)
        check(sel['images_reviewed'] == 1 and sel['capped'] and sel['shortfall']['PA'] > 0, sel)
        sel2 = coverage.select_images(census, {'AA': 99})  # unreachable -> terminates
        check(sel2['reached']['AA'] == 1 and sel2['shortfall']['AA'] == 98 and not sel2['capped'], sel2)


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as t:
        ap = os.path.join(t, 'ap.csv'); _synth_csv(ap)
        c1, m1 = coverage.build_census_cached(ap)
        check('cached_to' in m1 and os.path.exists(m1['cached_to']), m1)
        c2, _ = coverage.build_census_cached(ap)   # hits cache
        check(c2[K1]['PA'] == 2 and isinstance(c2[K1], collections.Counter), dict(c2[K1]))


def test_real_smoke():
    if not os.path.exists(REAL_AP):
        print("  SKIP test_real_smoke (no real all_points.csv)"); return
    import time
    t0 = time.time(); census, meta = coverage.build_census(REAL_AP); dt = time.time() - t0
    check(meta['n_images'] > 1000 and dt < 30, (meta, dt))
    rep = coverage.coverage_report(census, ['PA', 'OFAV', 'OFRA'])
    check(rep['per_code']['PA']['census_images'] > 0, 'PA present')
    check(rep['cooccurrence']['PA'].get('OFAV', 0) > 0, 'PA/OFAV co-occur')
    print(f"  (real: {meta['n_images']} imgs, {dt:.1f}s, PA imgs={rep['per_code']['PA']['census_images']}, "
          f"PA&OFAV={rep['cooccurrence']['PA'].get('OFAV')})")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    print(f"Running {len(tests)} coverage tests...\n")
    for fn in tests:
        run(fn)
    passed = sum(1 for _, ok, _ in _R if ok)
    failed = [(n, e) for n, ok, e in _R if not ok]
    print(f"\n==== {passed}/{len(_R)} passed ====")
    if failed:
        for n, e in failed:
            print(f"\n--- {n} ---\n{e}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == '__main__':
    main()
