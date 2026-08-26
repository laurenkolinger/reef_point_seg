"""Season-agnostic recursive clip-image resolver (Step 3).

TCRMP images are filed by survey SEASON; the folder year often differs from the
frame's date year (a 2018-01 frame lives under TCRMP2017_clip, a 2021 frame under
TCRMP2020_clip/Annual). Basenames are globally unique, so resolution must be a
whole-tree recursive scan keyed on basename, not a TCRMP{date_year}_clip guess.
No-pytest harness: run with env/bin/python <this file>.
"""
import os, sys, tempfile, shutil
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))
import select_images as si


def check(c, m):
    if not c:
        raise AssertionError(m)


def _make_tree():
    """Mimic the real season-folder layout."""
    d = tempfile.mkdtemp()

    def touch(*parts):
        p = os.path.join(d, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').close()
        return p

    touch('TCRMP2017_clip', 'TCRMP20180118_clip_CRB', 'TCRMP20180118_clip_CRB_T203.jpg')
    touch('TCRMP2020_clip', 'Annual', 'TCRMP20210312_clip_LBH', 'TCRMP20210312_clip_LBH_T103.jpg')
    touch('TCRMP2020_clip', 'Annual', 'TCRMP20210312_clip_LBH', 'TCRMP20210312_clip_LBH_T103_pts.jpg')
    touch('TCRMP2014_clip', 'TCRMP20140926_clip_BWR', 'TCRMP20140926_clip_BWR_T615.jpeg')
    return d


def test_finds_frame_filed_under_wrong_season_year():
    d = _make_tree()
    try:
        idx = si.build_clip_index(d)
        p, ext = si.find_source_image('TCRMP20180118_clip_CRB_T203', idx)
        check(p is not None, '2018 frame under TCRMP2017_clip not found')
        check(p.endswith('TCRMP20180118_clip_CRB_T203.jpg'), f'wrong path {p}')
        check(ext == 'jpg', f'wrong ext {ext}')
    finally:
        shutil.rmtree(d)


def test_finds_pts_in_period_subfolder():
    d = _make_tree()
    try:
        idx = si.build_clip_index(d)
        pts = si.find_pts_image('TCRMP20210312_clip_LBH_T103', idx)
        check(pts is not None and pts.endswith('_pts.jpg'), f'pts not found: {pts}')
        raw, _ = si.find_source_image('TCRMP20210312_clip_LBH_T103', idx)
        check(raw is not None and raw.endswith('T103.jpg'), f'raw resolved wrong: {raw}')
    finally:
        shutil.rmtree(d)


def test_jpeg_extension_variant_found():
    d = _make_tree()
    try:
        idx = si.build_clip_index(d)
        p, ext = si.find_source_image('TCRMP20140926_clip_BWR_T615', idx)
        check(p is not None, 'jpeg variant not found')
    finally:
        shutil.rmtree(d)


def test_genuinely_absent_returns_none():
    d = _make_tree()
    try:
        idx = si.build_clip_index(d)
        p, ext = si.find_source_image('TCRMP20251023_clip_FLC_T307', idx)
        check(p is None and ext is None, f'absent frame should be None, got {p}')
    finally:
        shutil.rmtree(d)


def test_pts_lookup_does_not_match_raw_and_vice_versa():
    """Adversarial: a raw lookup must not return the _pts file, and a pts lookup
    must not return the raw file."""
    d = _make_tree()
    try:
        idx = si.build_clip_index(d)
        raw, _ = si.find_source_image('TCRMP20210312_clip_LBH_T103', idx)
        check('_pts' not in os.path.basename(raw), f'raw matched a pts file: {raw}')
        # a frame with only a raw (no pts) returns None for pts
        check(si.find_pts_image('TCRMP20140926_clip_BWR_T615', idx) is None,
              'pts should be None when only a raw exists')
    finally:
        shutil.rmtree(d)


def test_collision_prefers_canonical_flat_copy():
    """When the same stem exists flat AND under a JPEG/ re-export or an _edit
    variant subfolder, the flat original wins (deterministic), so we never
    annotate a re-encoded/cropped copy whose points would misalign."""
    d = tempfile.mkdtemp()
    try:
        def touch(*parts):
            p = os.path.join(d, *parts)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'w').close()
            return p
        flat = touch('TCRMP2012_clip', 'TCRMP20130322_clip_SVN', 'TCRMP20130322_clip_SVN_T310.jpg')
        touch('TCRMP2012_clip', 'TCRMP20130322_clip_SVN', 'JPEG', 'TCRMP20130322_clip_SVN_T310.jpg')
        touch('TCRMP2012_clip', 'TCRMP20130322_clip_SVN_edit', 'TCRMP20130322_clip_SVN_T310.jpg')
        idx = si.build_clip_index(d)
        got, _ = si.find_source_image('TCRMP20130322_clip_SVN_T310', idx)
        check(got == flat, f'expected flat canonical copy, got {got}')
    finally:
        shutil.rmtree(d)


def test_hidden_dirs_and_dotfiles_skipped():
    """.AppleDouble junk and dot-files are not indexed."""
    d = tempfile.mkdtemp()
    try:
        def touch(*parts):
            p = os.path.join(d, *parts)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'w').close()
            return p
        touch('TCRMP2014_clip', 'F', 'TCRMP20140926_clip_BWR_T615.jpg')
        touch('TCRMP2014_clip', 'F', '.AppleDouble', 'TCRMP20140926_clip_BWR_T615.jpg')
        touch('TCRMP2014_clip', 'F', '._TCRMP20140926_clip_BWR_T615.jpg')
        idx = si.build_clip_index(d)
        got, _ = si.find_source_image('TCRMP20140926_clip_BWR_T615', idx)
        check('.AppleDouble' not in got, f'indexed AppleDouble junk: {got}')
        check('._' not in os.path.basename(got), f'indexed dotfile: {got}')
    finally:
        shutil.rmtree(d)


if __name__ == '__main__':
    fails = []
    for fn in (test_finds_frame_filed_under_wrong_season_year,
               test_finds_pts_in_period_subfolder,
               test_jpeg_extension_variant_found,
               test_genuinely_absent_returns_none,
               test_pts_lookup_does_not_match_raw_and_vice_versa,
               test_collision_prefers_canonical_flat_copy,
               test_hidden_dirs_and_dotfiles_skipped):
        try:
            fn(); print(f'  PASS {fn.__name__}')
        except Exception as e:
            fails.append((fn.__name__, e)); print(f'  FAIL {fn.__name__}: {e}')
    print(f'\ntest_image_resolver: {7 - len(fails)} passed, {len(fails)} failed')
    sys.exit(1 if fails else 0)
