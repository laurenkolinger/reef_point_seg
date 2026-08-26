#!/usr/bin/env python3
"""Launch / render verification for the Place Points + Segment Images apps.

Run with the unified env python:
    env/bin/python scripts/TCRMPclip_placePoints/tests/verify_apps.py

Checks (no GPU / SAM3 needed):
  * Place Points Flask app imports, GET / renders 200 with the new name + UI,
    and /static/pp_core.js is served.
  * Both templates compile as Jinja (my CSS/markup edits didn't break syntax).
  * The renamed identifiers are present and the old ones are gone.
"""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
PP_SRC = os.path.normpath(os.path.join(HERE, '..', 'src'))
SCRIPTS = os.path.normpath(os.path.join(HERE, '..', '..'))
SG_TPL = os.path.join(SCRIPTS, 'TCRMPclip_segmentImages', 'src', 'templates', 'index.html')
PP_TPL = os.path.join(PP_SRC, 'templates', 'index.html')

passed = 0
failed = 0


def ok(cond, msg):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print('  FAIL:', msg)


def must(text, needle, label):
    ok(needle in text, f'{label}: expected to find {needle!r}')


def must_not(text, needle, label):
    ok(needle not in text, f'{label}: did NOT expect {needle!r}')


# ── 1) Jinja compile both templates (catches markup/CSS syntax breakage) ──
try:
    from jinja2 import Environment
    env = Environment()
    for tpl in (PP_TPL, SG_TPL):
        env.from_string(open(tpl, encoding='utf-8').read())
    passed += 1
except Exception as e:  # noqa: BLE001
    failed += 1
    print('  FAIL: jinja compile:', e)

# ── 2) Static string assertions on rendered template sources ──
pp = open(PP_TPL, encoding='utf-8').read()
must(pp, 'TCRMP Place Points', 'place: new title')
must_not(pp, 'Route Chosen Images', 'place: old name gone')
must(pp, "filename='pp_core.js'", 'place: pp_core.js included')
must(pp, 'id="kbd-help"', 'place: K cheat-sheet present')
must(pp, 'Press <kbd>K</kbd>', 'place: press-K hint present (sticky toggle)')
must(pp, 'SEL_COLOR', 'place: pink selection color present')
must_not(pp, 'Fit View', 'place: Fit View button removed')
must_not(pp, 'qa-name', 'place: species-name span removed from bar')
must_not(pp, "case 'n': case 'N':", 'place: N frame shortcut removed')
must_not(pp, 'lockPoint', 'place: dead lockPoint removed')
# REVIEW is a normal quick-add label now — the special review MODE is gone.
must_not(pp, 'reviewMode', 'place: review MODE removed')
must_not(pp, 'setReviewMode', 'place: setReviewMode removed')
must_not(pp, 'armReviewMode', 'place: armReviewMode removed')
must_not(pp, 'addReviewPoint', 'place: addReviewPoint removed')
must_not(pp, 'review-active', 'place: review-mode banner class removed')
must(pp, 'REVIEW_IDX', 'place: REVIEW is a quick-add label index')
must(pp, 'relabelSelectedPoint', 'place: select+key relabel present')
must(pp, 'deleteSelectedSet', 'place: multi-select delete present')
must(pp, 'confirmAndAdvance', 'place: Enter confirm+advance+center present')
must(pp, 'doneAndExit', 'place: Done/close-to-menu button present')
must(pp, 'nextLabelMonotonic', 'place: monotonic next-label wired in')

sg = open(SG_TPL, encoding='utf-8').read()
must(sg, 'id="kbd-help"', 'segment: K cheat-sheet mirrored')
# The segment tool (another cluster) may use either the legacy hold-K hint or
# the newer press-K wording; accept either so this suite isn't coupled to its
# state. We only assert SOME K hint exists.
ok(('Hold <kbd>K</kbd>' in sg) or ('Press <kbd>K</kbd>' in sg), 'segment: a K cheat-sheet hint present')
must_not(sg, 'routeChosenImages', 'segment: old dir token gone from UI copy')
must(sg, 'Load from Place Points', 'segment: input label renamed')

# ── 3) Import the Place Points Flask app + exercise routes ──
prev = os.getcwd()
try:
    os.chdir(PP_SRC)
    sys.path.insert(0, PP_SRC)
    spec = importlib.util.spec_from_file_location('placepoints_app', os.path.join(PP_SRC, 'app.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client = mod.app.test_client()

    r = client.get('/')
    ok(r.status_code == 200, f'GET / status ({r.status_code})')
    body = r.get_data(as_text=True)
    must(body, 'TCRMP Place Points', 'GET /: rendered new name')
    must(body, 'kbd-help', 'GET /: rendered K cheat-sheet')
    must_not(body, 'Route Chosen Images', 'GET /: old name gone')

    rs = client.get('/static/pp_core.js')
    ok(rs.status_code == 200, f'GET /static/pp_core.js status ({rs.status_code})')
    must(rs.get_data(as_text=True), 'function colLabel', 'pp_core.js served with helpers')
except Exception as e:  # noqa: BLE001
    failed += 1
    print('  FAIL: Place Points app import/serve:', repr(e))
finally:
    os.chdir(prev)

print(f'verify_apps: {passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
