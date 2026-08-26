"""Step 4.test routing clip-dir resolution.

Routing must read images from the LOCAL full-set clip dir under the repo's
supporting_data (the same dir Step 3 reads), never the (absent) Dropbox path,
and must skip a stale saved clip_dir that no longer exists on disk.
No-pytest harness: run with env/bin/python <this file>.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import app as appmod


def check(c, m):
    if not c:
        raise AssertionError(m)


EXPECTED_LOCAL = os.path.join(appmod.REPO_DIR, 'supporting_data', 'TCRMP_clip')


def test_local_full_set_exists():
    check(os.path.isdir(EXPECTED_LOCAL), f'local full-set clip dir must exist: {EXPECTED_LOCAL}')


def test_clipdir_falls_back_to_local_when_saved_is_missing():
    chosen = appmod._resolve_routing_clip_dir(
        step4test_cfg={}, s4cfg={'clip_dir': '/home/bizon/UVI Dropbox/NOPE/TCRMP_clip'})
    check(chosen == EXPECTED_LOCAL, f'expected local fallback, got {chosen}')


def test_clipdir_honors_existing_panel_value():
    chosen = appmod._resolve_routing_clip_dir(step4test_cfg={'clip_dir': EXPECTED_LOCAL}, s4cfg={})
    check(chosen == EXPECTED_LOCAL, f'panel value should win when it exists, got {chosen}')


def test_clipdir_never_returns_dropbox():
    chosen = appmod._resolve_routing_clip_dir(step4test_cfg={}, s4cfg={})
    check('Dropbox' not in chosen, f'must not return a Dropbox path: {chosen}')
    check(os.path.isdir(chosen), f'returned dir must exist on disk: {chosen}')


if __name__ == '__main__':
    fails = []
    for fn in (test_local_full_set_exists,
               test_clipdir_falls_back_to_local_when_saved_is_missing,
               test_clipdir_honors_existing_panel_value,
               test_clipdir_never_returns_dropbox):
        try:
            fn(); print(f'  PASS {fn.__name__}')
        except Exception as e:
            fails.append((fn.__name__, e)); print(f'  FAIL {fn.__name__}: {e}')
    print(f'\ntest_step4test_clipdir: {4 - len(fails)} passed, {len(fails)} failed')
    sys.exit(1 if fails else 0)
