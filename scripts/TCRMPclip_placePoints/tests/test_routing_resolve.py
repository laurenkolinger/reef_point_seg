"""Step 4 routing image resolution (placePoints).

Routing must (1) resolve images by a whole-tree recursive index keyed on
basename (season-agnostic), and (2) trust Step 3's already-resolved
source_image/pts_image paths when they name a real file, falling back to the
index when the column is blank/stale/NaN/absent.
No-pytest harness: run with env/bin/python <this file>.
"""
import os, sys, tempfile, shutil
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))
import app as ppapp


def check(c, m):
    if not c:
        raise AssertionError(m)


def _tree():
    d = tempfile.mkdtemp()

    def touch(*p):
        fp = os.path.join(d, *p)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, 'w').close()
        return fp

    raw = touch('TCRMP2017_clip', 'TCRMP20180118_clip_CRB', 'TCRMP20180118_clip_CRB_T203.jpg')
    return d, raw


def test_index_finds_cross_season_frame():
    d, raw = _tree()
    try:
        idx = ppapp.build_clip_index(d)
        got = idx.get('TCRMP20180118_clip_CRB_T203')
        check(got == raw, f'index miss: {got} != {raw}')
    finally:
        shutil.rmtree(d)


def test_resolve_prefers_csv_source_image_when_present():
    d, raw = _tree()
    try:
        idx = ppapp.build_clip_index(d)
        row = {'source_image': raw, 'pts_image': ''}
        got = ppapp._resolve_raw(row, 'TCRMP20180118_clip_CRB_T203', idx)
        check(got == raw, f'should trust csv path, got {got}')
    finally:
        shutil.rmtree(d)


def test_resolve_falls_back_to_index_when_csv_blank_or_stale():
    d, raw = _tree()
    try:
        idx = ppapp.build_clip_index(d)
        # stale dropbox-era path no longer on disk -> index fallback
        row = {'source_image': '/nonexistent/old/dropbox/path.jpg', 'pts_image': ''}
        got = ppapp._resolve_raw(row, 'TCRMP20180118_clip_CRB_T203', idx)
        check(got == raw, f'stale csv path should fall back to index, got {got}')
        # no column at all (standalone selected_frames.csv) -> index fallback
        got2 = ppapp._resolve_raw({}, 'TCRMP20180118_clip_CRB_T203', idx)
        check(got2 == raw, f'missing column should fall back to index, got {got2}')
    finally:
        shutil.rmtree(d)


def test_resolve_handles_nan_cell():
    """Adversarial: pandas reads a blank CSV cell as float('nan'); the resolver
    must treat that as 'no value' and fall back to the index, not crash."""
    d, raw = _tree()
    try:
        idx = ppapp.build_clip_index(d)
        row = {'source_image': float('nan'), 'pts_image': float('nan')}
        got = ppapp._resolve_raw(row, 'TCRMP20180118_clip_CRB_T203', idx)
        check(got == raw, f'NaN cell should fall back to index, got {got}')
        check(ppapp._resolve_pts(row, 'TCRMP20180118_clip_CRB_T203', idx) is None,
              'no pts in tree -> None even with NaN cell')
    finally:
        shutil.rmtree(d)


def test_cpc_index_built_and_canonical():
    """build_cpc_index maps .cpc by stem; collisions prefer the flat copy."""
    d = tempfile.mkdtemp()
    try:
        def touch(*p):
            fp = os.path.join(d, *p)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, 'w').close()
            return fp
        flat = touch('TCRMP2017_clip', 'TCRMP20180118_clip_CRB', 'TCRMP20180118_clip_CRB_T203.cpc')
        touch('TCRMP2017_clip', 'TCRMP20180118_clip_CRB', 'JPEG', 'TCRMP20180118_clip_CRB_T203.cpc')
        cidx = ppapp.build_cpc_index(d)
        check(cidx.get('TCRMP20180118_clip_CRB_T203') == flat, f'cpc index miss/non-canonical: {cidx}')
        # image index must NOT contain the .cpc
        iidx = ppapp.build_clip_index(d)
        check('TCRMP20180118_clip_CRB_T203' not in iidx, 'cpc leaked into image index')
    finally:
        shutil.rmtree(d)


if __name__ == '__main__':
    fails = []
    for fn in (test_index_finds_cross_season_frame,
               test_resolve_prefers_csv_source_image_when_present,
               test_resolve_falls_back_to_index_when_csv_blank_or_stale,
               test_resolve_handles_nan_cell,
               test_cpc_index_built_and_canonical):
        try:
            fn(); print(f'  PASS {fn.__name__}')
        except Exception as e:
            fails.append((fn.__name__, e)); print(f'  FAIL {fn.__name__}: {e}')
    print(f'\ntest_routing_resolve: {5 - len(fails)} passed, {len(fails)} failed')
    sys.exit(1 if fails else 0)
