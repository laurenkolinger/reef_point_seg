#!/usr/bin/env python3
"""Module: test_review_lock_points

Purpose: prove the 2026-09-06 expert-review flag lock holds in the archived
    Place Points tool. Lauren Olinger suspended expert-review flagging in every
    annotator on 2026-09-06: no new review tag may reach a mask. Place Points
    writes the POINT that the annotator later turns into a flagged mask, so the
    tag is refused here too. These cases prove three things: the REVIEW quick-add
    button is still on the page but inert and explained, the R key no longer arms
    the REVIEW label, and the batch export writes no review tag into
    ids/sam_click_prompts.json while every existing point keeps its geometry and
    still displays.

Inputs: the live template at ../src/templates/index.html, the live export module
    at ../src/export.py, and the Place Points Flask app at ../src/app.py. Node is
    required for the browser-side cases (the board already depends on it for
    test_pp_core.js).

Outputs: one PASS or FAIL line per assertion, one CASE line per test, a summary,
    and exit status 1 when anything failed.

Run:
    env/bin/python scripts/TCRMPclip_placePoints/tests/test_review_lock_points.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', 'src'))
TEMPLATE_PATH = os.path.join(SRC, 'templates', 'index.html')
EXPORT_PATH = os.path.join(SRC, 'export.py')
APP_PATH = os.path.join(SRC, 'app.py')

sys.path.insert(0, SRC)
import export as E  # noqa: E402  (path is set above on purpose)

# The tooltip is fixed copy from the change map. It must match byte for byte on
# the rendered page, so it lives here as one string and nowhere else.
LOCKED_TOOLTIP = (
    'VICARIUS has temporarily suspended expert-review flagging: the REVIEW label '
    'cannot be armed, and the R key no longer arms it. Points placed before '
    '2026-09-06 keep their REVIEW code, and the annotator no longer turns a '
    'REVIEW point into a flagged mask.'
)

# The tooltip the same button carries once the switch is back on. Written for
# this change rather than copied from the pre-lock page: the old copy ran to
# three sentences and carried an em dash, both of which the platform prose rules
# refuse.
UNLOCKED_TOOLTIP = (
    'Tap R or click here to arm the REVIEW label, then click the image to '
    'place a REVIEW point; tap again to clear. VICARIUS ships a REVIEW point '
    'to an outside taxonomic expert for identification and never uses it as a '
    'training label.'
)

# A frame name the export helper can turn into a raw image name.
PTS_FILENAME = 'TCRMP20171202_clip_CKR_T403_pts.jpg'
RAW_FILENAME = 'TCRMP20171202_clip_CKR_T403.jpeg'
EXPORT_YEAR = 2017
BULK_POINT_COUNT = 1000
# The node harness stubs TARGET_LABELS with two species, so REVIEW sits at
# index 2 in every harness run.
REVIEW_IDX_WHEN_STUBBED = 2

_fail = 0
_case_fail = 0


def check(cond, msg):
    """Record one assertion and print its verdict.

    Args:
        cond: the assertion result, read as a boolean.
        msg: plain-language description of what was asserted.

    Returns:
        The boolean the assertion was given, so a caller can branch on it.
    """
    global _fail, _case_fail
    print(('  PASS ' if cond else '  FAIL ') + msg)
    if not cond:
        _fail += 1
        _case_fail += 1
    return bool(cond)


def read_text(path):
    """Read a UTF-8 text file whole, with an error that names the file.

    Args:
        path: absolute path to the file.

    Returns:
        The file content as a string.

    Raises:
        RuntimeError: when the file cannot be read, naming path and cause.
    """
    try:
        with open(path, encoding='utf-8') as handle:
            return handle.read()
    except OSError as err:
        raise RuntimeError(f'cannot read {path}: {err}') from err


TEMPLATE = read_text(TEMPLATE_PATH)
EXPORT_SOURCE = read_text(EXPORT_PATH)

_rendered_cache = {}
_app_cache = {}


def app_module():
    """Import the Place Points Flask app once per run.

    The app is imported from its own directory because it resolves sibling
    modules by relative import path.

    Returns:
        The imported app module.

    Raises:
        RuntimeError: when the app cannot be imported, naming the cause.
    """
    if 'module' in _app_cache:
        return _app_cache['module']
    import importlib.util
    previous_cwd = os.getcwd()
    try:
        os.chdir(SRC)
        spec = importlib.util.spec_from_file_location('placepoints_app', APP_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as err:
        raise RuntimeError(f'cannot import the Place Points app: {err!r}') from err
    finally:
        os.chdir(previous_cwd)
    _app_cache['module'] = module
    return module


def rendered_page():
    """Render GET / through the real Place Points Flask app, once per run.

    This is the SHIPPED page: whatever app.py passes the template today. It
    proves what an operator actually sees, which is the locked state while
    app.py passes no switch at all.

    Returns:
        The rendered HTML of GET / as a string.

    Raises:
        RuntimeError: when the app does not answer 200.
    """
    if 'html' in _rendered_cache:
        return _rendered_cache['html']
    module = app_module()
    response = module.app.test_client().get('/')
    if response.status_code != 200:
        raise RuntimeError(f'GET / answered {response.status_code}, expected 200')
    _rendered_cache['html'] = response.get_data(as_text=True)
    return _rendered_cache['html']


def rendered_source(review_tagging_enabled=False):
    """Render the page template with the expert-review switch in one state.

    The template is the single source of the button's state, so a case that
    reads only the shipped page cannot tell a switched control from a dead one
    typed into the markup. This renders through the app's own Jinja environment
    (inside a request context, because the page resolves a static URL) with the
    switch set either way, so both states are testable before app.py passes it.

    Args:
        review_tagging_enabled: True renders the unlocked page, False the
            locked one, and None omits the variable altogether so the
            template's own fallback decides.

    Returns:
        The rendered HTML as a string.

    Raises:
        RuntimeError: when the template cannot be rendered, naming the cause.
    """
    cache_key = f'source:{review_tagging_enabled}'
    if cache_key in _rendered_cache:
        return _rendered_cache[cache_key]
    module = app_module()
    context = {
        'target_species': module.ALL_TARGET_SPECIES,
        'target_labels': module.ALL_TARGET_LABELS,
        'default_frames': module.DEFAULT_SELECTED_FRAMES,
        'default_export': module.DEFAULT_EXPORT_DIR,
        'orchestrated': False,
        'orchestrator_url': '',
        'default_reference_mode': False,
        'default_shuffle': False,
        'default_batch_size': '10',
    }
    if review_tagging_enabled is not None:
        context['review_tagging_enabled'] = review_tagging_enabled
    try:
        with module.app.test_request_context('/'):
            html = module.app.jinja_env.get_template('index.html').render(**context)
    except Exception as err:
        raise RuntimeError(
            f'cannot render index.html with review_tagging_enabled='
            f'{review_tagging_enabled!r}: {err!r}') from err
    _rendered_cache[cache_key] = html
    return html


def js_function(name, source=None):
    """Extract one top-level JavaScript function declaration from the page.

    Args:
        name: the function name, for example 'setQuickLabel'.
        source: the page text to slice. Defaults to the locked rendering, which
            is what an operator is served today.

    Returns:
        The full source text of the declaration, opening brace to closing brace.

    Raises:
        RuntimeError: when the function is absent or its braces do not close.

    Example:
        >>> js_function('isReviewPoint').startswith('function isReviewPoint')
        True
    """
    source = rendered_source() if source is None else source
    marker = f'function {name}('
    start = source.find(marker)
    if start < 0:
        raise RuntimeError(f'{name} is not declared in {TEMPLATE_PATH}')
    open_brace = source.find('{', start)
    if open_brace < 0:
        raise RuntimeError(f'{name} has no body in {TEMPLATE_PATH}')
    depth = 0
    for index in range(open_brace, len(source)):
        char = source[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise RuntimeError(f'{name} body never closes in {TEMPLATE_PATH}')


def js_const(name, source=None):
    """Extract one top-level `const NAME = ...;` line from the page.

    The switch is a Jinja expression in the template source, so the constants
    are read out of a RENDERED page: what node runs has to be what a browser
    runs.

    Args:
        name: the constant name, for example 'REVIEW_IDX'.
        source: the page text to search. Defaults to the locked rendering.

    Returns:
        The declaration line as source text.

    Raises:
        RuntimeError: when the constant is not declared exactly once.
    """
    source = rendered_source() if source is None else source
    found = re.findall(rf'^\s*const {re.escape(name)}\s*=.*?;\s*$', source, re.M)
    if len(found) != 1:
        raise RuntimeError(
            f'expected exactly one `const {name}` in {TEMPLATE_PATH}, found {len(found)}')
    return found[0].strip()


# Stubs for the browser objects the extracted functions touch. Nothing here
# reimplements a rule under test: every decision comes from the real template
# source spliced in below.
NODE_STUBS = """
'use strict';
let numKeyDown = null;
let selectedPoint = -1;
let sortedIndices = [];
let loupeEnabled = false;
let aKeyDown = false;
let auxMode = false;
let currentData = null;
let lastStatus = '';
const canvas = {style: {}};
const PP = null;
const TARGET_LABELS = [
  {code: 'PA', name: 'Porites astreoides', category: 'Coral'},
  {code: 'OFAV', name: 'Orbicella faveolata', category: 'Coral'},
];
function setStatus(text) { lastStatus = String(text); }
function highlightQuickBtn() {}
function renderPointList() {}
function render() {}
function sortedPointIndices() { return []; }
function updateLoupeOverlay() {}
function toggleAuxMode() { auxMode = false; }
function flashLetter() {}
function nextAddLabel() { return 'A'; }
"""


def run_node_harness(body, review_tagging_enabled=False):
    """Run the page's own functions under node with a stub browser.

    Args:
        body: JavaScript assertions appended after the stubs and the page
            source. It must call `report(object)` exactly once.
        review_tagging_enabled: which rendering to take the functions and
            constants from. False, the shipped state, is the default.

    Returns:
        The object the harness reported, decoded from JSON.

    Raises:
        RuntimeError: when node is missing, exits nonzero, or reports nothing.
    """
    page = rendered_source(review_tagging_enabled)
    pieces = [
        NODE_STUBS,
        js_const('REVIEW_TAGGING_ENABLED', page),
        js_const('REVIEW_LOCKED_MSG', page),
        js_const('REVIEW_LABEL', page),
        js_const('QUICK_LABELS', page),
        js_const('REVIEW_IDX', page),
        js_function('reviewLabelLocked', page),
        js_function('setQuickLabel', page),
        js_function('toggleQuickLabel', page),
        js_function('relabelSelectedPoint', page),
        js_function('applyQuickKey', page),
        js_function('addPointWithLabel', page),
        js_function('isReviewPoint', page),
        'function report(o) { console.log("RESULT:" + JSON.stringify(o)); }',
        body,
    ]
    handle = tempfile.NamedTemporaryFile(
        'w', suffix='.js', delete=False, encoding='utf-8')
    try:
        handle.write('\n'.join(pieces))
        handle.close()
        try:
            result = subprocess.run(
                ['node', handle.name], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as err:
            raise RuntimeError(f'node could not run the harness: {err}') from err
    finally:
        os.unlink(handle.name)
    if result.returncode != 0:
        raise RuntimeError(
            f'node exited {result.returncode}: {result.stderr.strip()}')
    for line in result.stdout.splitlines():
        if line.startswith('RESULT:'):
            return json.loads(line[len('RESULT:'):])
    raise RuntimeError(f'the node harness reported nothing: {result.stdout.strip()}')


def review_point(**overrides):
    """Build one point dict that carries the expert-review tag both ways.

    Args:
        **overrides: fields to replace on the returned point.

    Returns:
        A point dict shaped the way the Place Points front end writes one.
    """
    point = {
        'label': 'A',
        'x': 1871,
        'y': 778,
        'species_code': 'REVIEW',
        'species_name': 'Flagged for expert review',
        'category': 'Review',
        'review': True,
        'added': True,
        'point_type': 1,
    }
    point.update(overrides)
    return point


def export_points(points, log_lines):
    """Run one batch export of a single frame into a throwaway directory.

    Args:
        points: the list of point dicts to export.
        log_lines: a list that collects every log line the export emits.

    Returns:
        A (stats, sam_points, csv_text) tuple, where sam_points is the exported
        prompt list for the frame and csv_text is point_coords.csv verbatim.
    """
    detection = {
        'filename': PTS_FILENAME,
        'pts_path': '',
        'raw_path': '',
        'year': EXPORT_YEAR,
        'date_str': '20171202',
        'site': 'CKR',
        't_id': '403',
        'points': points,
    }
    with tempfile.TemporaryDirectory() as out_dir:
        stats = E.export_batch(
            [detection], out_dir, log_fn=log_lines.append, lores_mode=False)
        ids_dir = os.path.join(out_dir, str(EXPORT_YEAR), 'ids')
        with open(os.path.join(ids_dir, 'sam_click_prompts.json'), encoding='utf-8') as f:
            prompts = json.load(f)
        csv_text = read_text(os.path.join(ids_dir, 'point_coords.csv'))
    return stats, prompts[RAW_FILENAME]['points'], csv_text


# ── The annotator's two gates, copied verbatim ───────────────────────
# A prompt point written here has to pass two independent gates in
# scripts/TCRMPclip_combinedAnnotate/src/app.py before it becomes a mask. These
# copies are byte-faithful to that file so this suite fails if an export shape
# stops satisfying them. Their source lines are named in each docstring.

ANNOTATOR_TARGET_CODES = {'PA', 'OFAV'}


def annotator_is_review(point):
    """The annotator's read predicate, verbatim from its app.py:84-88."""
    return bool(point.get('review')) or point.get(
        'species', point.get('species_code', '')) == 'REVIEW'


def annotator_is_non_target(point):
    """The annotator's non-target test, verbatim from its app.py:827-829."""
    return point.get('non_target') is True or point.get(
        'species', point.get('species_code', '')) == '?'


def annotator_selects(point):
    """Gate 1, the queue filter under Target species only: its app.py:835-839.

    A point enters the frame's segmentation queue when its species is a target
    code, or it is a non-target prompt, or it still carries the review tag.
    """
    return (point.get('species', point.get('species_code', '')) in ANNOTATOR_TARGET_CODES
            or annotator_is_non_target(point) or annotator_is_review(point))


def annotator_treats_as_negative(point):
    """Gate 2, the prompt policy, verbatim from its app.py:1164-1170.

    A negative point produces NO mask of its own and subtracts its region from
    neighbouring masks, so an exported point must never read as one.
    """
    if annotator_is_review(point):
        return False
    return point.get('non_target') is True or point.get(
        'species', point.get('species_code', '')) == '?'


# ── Cases ────────────────────────────────────────────────────────────

def review_button_tag(page):
    """Return the REVIEW quick-add button's opening tag from one rendering.

    Args:
        page: the rendered HTML.

    Returns:
        The opening tag text, or None when the button is not on the page.
    """
    found = re.search(r'<button[^>]*id="qa-review"[^>]*>', page)
    return found.group(0) if found else None


def test_review_button_is_present_and_disabled():
    """The REVIEW quick-add button stays on the page, greyed and inert.

    Read in BOTH states, because a case that reads only the shipped page passes
    just as well against a button with `disabled` typed into the markup, which
    is exactly what this page carried until 2026-09-06. The disabled attribute
    and the click handler have to move when the switch moves.
    """
    states = (('locked', rendered_source(False), True),
              ('unlocked', rendered_source(True), False),
              ('the fallback rendering', rendered_source(None), True),
              ('the shipped page', rendered_page(), True))
    for label, page, want_disabled in states:
        check('id="qa-review"' in page,
              f'{label}: the REVIEW quick-add button is still rendered')
        tag = review_button_tag(page)
        if not check(tag is not None, f'{label}: the REVIEW button tag can be read'):
            continue
        is_disabled = re.search(r'\sdisabled[\s>]', tag) is not None
        check(is_disabled is want_disabled,
              f'{label}: the disabled attribute is '
              f'{"present" if want_disabled else "absent"}')
        check(('onclick=' in tag) is not want_disabled,
              f'{label}: the click handler is '
              f'{"absent" if want_disabled else "present"}')
        check('<span class="qa-code">REVIEW</span>' in page,
              f'{label}: the button still reads REVIEW, so the label is not gone')
    locked = rendered_source(False)
    check('.qa-btn:disabled' in locked,
          'a CSS rule paints the disabled quick-add button')
    check('not-allowed' in locked, 'the disabled rule sets the not-allowed cursor')


def test_review_button_tooltip_is_the_locked_message():
    """The button explains the suspension, and only while the lock holds.

    Both tooltips are pinned, in both directions: the locked copy has to be off
    the unlocked page and the arming copy off the locked one. Asserting only
    that the locked sentence is somewhere on the shipped page passed even when
    the sentence was typed into the markup for good.
    """
    locked = rendered_source(False)
    unlocked = rendered_source(True)
    check(f'title="{LOCKED_TOOLTIP}"' in locked,
          'the locked tooltip is the title attribute, not loose text')
    check(LOCKED_TOOLTIP not in unlocked,
          'the locked tooltip is gone once the switch is back on')
    check(f'title="{UNLOCKED_TOOLTIP}"' in unlocked,
          'the unlocked page explains how to arm the label instead')
    check(UNLOCKED_TOOLTIP not in locked,
          'the arming tooltip is withheld while the lock holds')
    check(LOCKED_TOOLTIP in rendered_page(),
          'the page an operator is served today carries the locked tooltip')
    for page, label in ((locked, 'locked'), (unlocked, 'unlocked')):
        check('NEVER used as training labels' not in page,
              f'{label}: the old REVIEW tooltip is gone, not merely appended to')
    for text, label in ((LOCKED_TOOLTIP, 'locked'), (UNLOCKED_TOOLTIP, 'unlocked')):
        check(text.count('. ') == 1 and text.endswith('.'),
              f'the {label} tooltip is two complete sentences ending in a full stop')
        check(not text.startswith('You'),
              f'the {label} tooltip does not open with the word You')
        check('\u2014' not in text and '\u2013' not in text,
              f'the {label} tooltip carries no dash glyph')
        check('UVI' not in text, f'the {label} tooltip does not name the university')


def test_place_points_button_state_follows_the_switch():
    """One value decides the whole REVIEW control, in both directions.

    Place Points used to type the disabled attribute and the locked tooltip
    into its markup and `const REVIEW_TAGGING_ENABLED = false` into its script,
    so flipping the switch in src/export.py left the button permanently dead
    and still labelled unavailable. The template now renders the attribute, the
    tooltip, the script constant and the keyboard help from one value, the way
    the Place plus Segment template does, and unlocking it arms the label
    again for real.
    """
    raw_tag = review_button_tag(TEMPLATE)
    if check(raw_tag is not None,
             'the REVIEW button is still declared in the template source'):
        check('review_tagging_enabled' in raw_tag,
              'the button tag chooses its attributes from the switch instead of '
              'typing them into the markup')
    check('const REVIEW_TAGGING_ENABLED = false;' not in TEMPLATE,
          'the template no longer types the script constant as a literal false')
    check(LOCKED_TOOLTIP not in TEMPLATE or 'review_tagging_enabled' in TEMPLATE,
          'the locked tooltip is chosen by the switch, not fixed in the markup')

    locked = rendered_source(False)
    unlocked = rendered_source(True)
    check('const REVIEW_TAGGING_ENABLED = false;' in locked,
          'the locked page hands the browser a false switch')
    check('const REVIEW_TAGGING_ENABLED = true;' in unlocked,
          'the unlocked page hands the browser a true switch')
    check('no longer arms the REVIEW label' in locked,
          'the locked page explains the suspension in the keyboard help')
    check('no longer arms the REVIEW label' not in unlocked,
          'the unlocked page drops the suspension from the keyboard help')
    check('arm REVIEW' in unlocked,
          'the unlocked page tells the operator R arms REVIEW again')

    armed = run_node_harness("""
const out = {};
out.lockedHelper = reviewLabelLocked(REVIEW_IDX);
numKeyDown = null; selectedPoint = -1; currentData = {points: []};
applyQuickKey(REVIEW_IDX);
out.armedAfterKey = numKeyDown;
addPointWithLabel(10, 20, REVIEW_IDX);
out.pointCount = currentData.points.length;
out.placedSpecies = currentData.points.length ? currentData.points[0].species_code : '';
out.placedReview = currentData.points.length ? currentData.points[0].review : null;
currentData = {points: [{label: 'B', x: 1, y: 2, species_code: 'PA',
                         species_name: 'Porites astreoides', category: 'Coral'}]};
selectedPoint = 0; numKeyDown = null;
out.relabelReturn = relabelSelectedPoint(REVIEW_IDX);
out.relabelledSpecies = currentData.points[0].species_code;
report(out);
""", review_tagging_enabled=True)
    check(armed['lockedHelper'] is False,
          'unlocked: the lock helper stops refusing the REVIEW index')
    check(armed['armedAfterKey'] == REVIEW_IDX_WHEN_STUBBED,
          f'unlocked: the R key arms the REVIEW label again, got '
          f'{armed["armedAfterKey"]!r}')
    check(armed['pointCount'] == 1 and armed['placedSpecies'] == 'REVIEW',
          'unlocked: a click places a REVIEW point again')
    check(armed['placedReview'] is True,
          'unlocked: the placed point carries the review flag again')
    check(armed['relabelReturn'] is True and armed['relabelledSpecies'] == 'REVIEW',
          'unlocked: a selected point can be relabelled REVIEW again')


def test_r_key_no_longer_arms_review():
    """Pressing R leaves the REVIEW label unarmed and relabels nothing."""
    outcome = run_node_harness("""
const out = {};
// Nothing armed, nothing selected: the R key routes through applyQuickKey.
numKeyDown = null; selectedPoint = -1; currentData = {points: []};
applyQuickKey(REVIEW_IDX);
out.armedAfterKey = numKeyDown;
out.statusAfterKey = lastStatus;
// A point is selected and nothing is armed: R would have relabelled it.
currentData = {points: [{label: 'A', x: 1, y: 2, species_code: 'PA',
                         species_name: 'Porites astreoides', category: 'Coral'}]};
selectedPoint = 0; numKeyDown = null;
out.relabelReturn = relabelSelectedPoint(REVIEW_IDX);
out.speciesAfterRelabel = currentData.points[0].species_code;
out.reviewKeyAfterRelabel = Object.prototype.hasOwnProperty.call(currentData.points[0], 'review');
// The button click path is the same refusal.
numKeyDown = null; toggleQuickLabel(REVIEW_IDX);
out.armedAfterButton = numKeyDown;
// A species label still arms, and Escape still disarms.
numKeyDown = null; selectedPoint = -1;
applyQuickKey(0); out.armedAfterSpecies = numKeyDown;
setQuickLabel(null); out.armedAfterDisarm = numKeyDown;
// Even with the label somehow armed, a click may not create a REVIEW point.
currentData = {points: []};
addPointWithLabel(10, 20, REVIEW_IDX);
out.pointsAfterReviewClick = currentData.points.length;
addPointWithLabel(10, 20, 0);
out.pointsAfterSpeciesClick = currentData.points.length;
out.speciesOfPlacedPoint = currentData.points.length ? currentData.points[0].species_code : '';
report(out);
""")
    check(outcome['armedAfterKey'] is None, 'the R key leaves the REVIEW label unarmed')
    check(outcome['statusAfterKey'].startswith('Flagging for expert review'),
          'the R key explains the refusal in the status line')
    check(outcome['relabelReturn'] is False,
          'relabelling a selected point to REVIEW is refused')
    check(outcome['speciesAfterRelabel'] == 'PA',
          'the selected point keeps the species it already had')
    check(outcome['reviewKeyAfterRelabel'] is False,
          'the selected point gains no review key')
    check(outcome['armedAfterButton'] is None,
          'clicking the REVIEW button arms nothing either')
    check(outcome['armedAfterSpecies'] == 0,
          'a species quick-add key still arms normally')
    check(outcome['armedAfterDisarm'] is None, 'disarming still works')
    check(outcome['pointsAfterReviewClick'] == 0,
          'a click cannot place a REVIEW point even if the label were armed')
    check(outcome['pointsAfterSpeciesClick'] == 1
          and outcome['speciesOfPlacedPoint'] == 'PA',
          'a click still places an ordinary species point')


def test_review_idx_is_still_defined():
    """REVIEW_IDX and REVIEW_LABEL survive, so verify_apps.py line 73 holds."""
    check('REVIEW_IDX' in TEMPLATE, 'REVIEW_IDX is still in the template source')
    check('REVIEW_LABEL' in TEMPLATE, 'REVIEW_LABEL is still in the template source')
    outcome = run_node_harness("""
report({
  reviewIdx: REVIEW_IDX,
  labelCount: TARGET_LABELS.length,
  quickCode: QUICK_LABELS[REVIEW_IDX] ? QUICK_LABELS[REVIEW_IDX].code : null,
  labelName: REVIEW_LABEL.name,
});
""")
    check(outcome['reviewIdx'] == outcome['labelCount'],
          'REVIEW_IDX still indexes the slot after the species labels')
    check(outcome['quickCode'] == 'REVIEW',
          'QUICK_LABELS still names REVIEW at that index')
    check(outcome['labelName'] == 'Flagged for expert review',
          'REVIEW_LABEL still carries its full name')


def test_export_writes_no_review_tag():
    """A tagged point exports without the tag, and the drop is counted."""
    log_lines = []
    stats, points, _csv = export_points([review_point()], log_lines)
    check(len(points) == 1, 'the frame still exports exactly one prompt point')
    exported = points[0]
    check('review' not in exported, 'the exported prompt carries no review key')
    check(exported['species'] != 'REVIEW',
          'the exported prompt does not name REVIEW as its species')
    check(exported['species'] == '',
          'the exported prompt reads as unlabelled instead')
    check(stats.get('review_tags_dropped') == 1,
          'the export counts the one suppressed review tag')
    dropped_lines = [line for line in log_lines if 'review' in line.lower()]
    check(len(dropped_lines) == 1,
          'the export says once how many review tags it suppressed')
    if dropped_lines:
        check('1' in dropped_lines[0] and 'suspend' in dropped_lines[0].lower(),
              'that line gives the count and the reason')
    quiet_lines = []
    quiet_stats, _pts, _csv2 = export_points(
        [review_point(species_code='PA', species_name='Porites astreoides',
                      category='Coral', review=False)], quiet_lines)
    check(quiet_stats.get('review_tags_dropped') == 0,
          'an untagged batch reports no suppressed tags')
    check(not [line for line in quiet_lines if 'suspend' in line.lower()],
          'an untagged batch prints no suspension line')


def test_export_keeps_the_point_geometry():
    """Only the tag is removed: position, letter and record survive."""
    log_lines = []
    _stats, points, csv_text = export_points([review_point()], log_lines)
    exported = points[0]
    check(exported['label'] == 'A', 'the letter identifier survives the export')
    check(exported['x'] == 1871 and exported['y'] == 778,
          'the point keeps its exact position')
    check(exported['point_type'] == 1, 'the point stays a positive prompt')
    check(exported.get('added') is True, 'the manually-added marker survives')
    check(exported['name'] == 'Flagged for expert review',
          'the point keeps the name the operator saw')
    check('REVIEW' in csv_text,
          'point_coords.csv still records what the operator placed')
    mixed_log = []
    _s, mixed, _c = export_points(
        [review_point(),
         {'label': 'B', 'x': 10, 'y': 20, 'species_code': 'PA',
          'species_name': 'Porites astreoides', 'category': 'Coral'}],
        mixed_log)
    check(len(mixed) == 2, 'a mixed frame exports both points')
    check(mixed[1]['species'] == 'PA', 'the ordinary point is untouched')
    check('review' not in mixed[1], 'the ordinary point gains nothing')


def test_existing_review_point_still_displays():
    """A point placed before the lock still reads and draws as REVIEW."""
    outcome = run_node_harness("""
report({
  byBoolean: isReviewPoint({label: 'A', x: 1, y: 2, review: true}),
  byCode: isReviewPoint({label: 'B', x: 1, y: 2, species_code: 'REVIEW'}),
  ordinary: isReviewPoint({label: 'C', x: 1, y: 2, species_code: 'PA'}),
  nothing: isReviewPoint(null),
});
""")
    check(outcome['byBoolean'] is True, 'an existing review boolean still reads as REVIEW')
    check(outcome['byCode'] is True, 'an existing REVIEW code still reads as REVIEW')
    check(outcome['ordinary'] is False, 'an ordinary point is not mistaken for one')
    check(not outcome['nothing'], 'a missing point answers falsy instead of crashing')
    check("ctx.fillText('REVIEW'" in TEMPLATE,
          'the canvas still draws the REVIEW crosshair label')
    check(".pspecies.review" in TEMPLATE,
          'the Labels pane still styles a REVIEW row')
    check("or p.get('review')" in EXPORT_SOURCE,
          'the overlay image still keeps review points visible')


def test_exported_point_is_not_a_negative_refinement_click():
    """The de-tagged point must never read as a negative SAM3 click.

    Marking it non_target would put it back in the queue but turn it into a
    negative click: no mask of its own, and it erodes the masks around it (the
    annotator's app.py:1153-1156 and 1164-1170). That is worse than being left
    out, so this case pins the export against that shape.
    """
    _stats, points, _csv = export_points([review_point()], [])
    exported = points[0]
    check('non_target' not in exported,
          'the exported prompt carries no non_target key')
    check(exported['species'] != '?',
          'the exported prompt is not spelled as a non-target question mark')
    check(annotator_is_non_target(exported) is False,
          'the annotator does not read it as a non-target prompt')
    check(annotator_treats_as_negative(exported) is False,
          'the annotator does not turn it into a negative refinement click')


def test_exported_point_through_the_annotator_selection_filter():
    """Pin what gate 1 does with the de-tagged point, in both launch modes.

    In the shipped mode the queue filter is bypassed: TCRMP_MANUAL_ANNOTATE
    defaults to 1 and the orchestrator hard-sets it, and that branch carries
    img_data['points'] whole as reference points (the annotator's app.py:868-889),
    so the point reaches the operator with its position intact.

    In the hand-launched SAM3 batch path the filter does apply, and a point that
    has lost its tag matches none of the three arms while a species filter is
    on. That gap is the annotator's to close, in the filter at its app.py:835-851;
    it cannot be closed from here without either inventing a species or marking
    the point non_target, which the case above rules out.
    """
    _stats, points, _csv = export_points([review_point()], [])
    exported = points[0]
    check(annotator_selects(exported) is False,
          'gate 1 does not select the de-tagged point while a species filter is '
          'on (known gap, closed in the annotator app.py:835-851, not here)')
    unfiltered = list(points)
    check(len(unfiltered) == 1 and unfiltered[0] is exported,
          'with no filter chosen the annotator takes every point, including this one')
    still_tagged = dict(exported, species='REVIEW')
    check(annotator_selects(still_tagged) is True,
          'the same point would be selected if the tag had been left on, which '
          'is what the lock deliberately gives up')
    ordinary = dict(exported, species='PA')
    check(annotator_selects(ordinary) is True,
          'an ordinary species point is unaffected by the lock')


def test_lock_constant_is_the_single_switch():
    """One constant per surface turns the lock off, and nothing else does."""
    check(E.REVIEW_TAGGING_ENABLED is False, 'the export lock is on by default')
    assignments = re.findall(r'^REVIEW_TAGGING_ENABLED\s*=', EXPORT_SOURCE, re.M)
    check(len(assignments) == 1,
          'export.py assigns the switch exactly once at module level')
    declarations = re.findall(r'const REVIEW_TAGGING_ENABLED\s*=', TEMPLATE)
    check(len(declarations) == 1, 'the template declares the switch exactly once')
    for name in ('setQuickLabel', 'relabelSelectedPoint', 'addPointWithLabel'):
        check('reviewLabelLocked(' in js_function(name),
              f'{name} asks the one lock helper instead of testing its own rule')
    original = E.REVIEW_TAGGING_ENABLED
    try:
        E.REVIEW_TAGGING_ENABLED = True
        entry = E.build_sam_entry(RAW_FILENAME, [review_point()])
        unlocked = entry['points'][0]
        check(unlocked.get('review') is True and unlocked['species'] == 'REVIEW',
              'flipping the one constant restores the old export behaviour')
    finally:
        E.REVIEW_TAGGING_ENABLED = original
    check(E.REVIEW_TAGGING_ENABLED is False, 'the switch is restored after the flip')


def test_every_review_spelling_is_refused():
    """Case, whitespace, either key and truthy values are all refused."""
    spellings = [
        ({'species_code': 'REVIEW'}, 'the plain REVIEW code'),
        ({'species_code': 'review'}, 'a lower case review code'),
        ({'species_code': ' Review '}, 'a padded mixed case review code'),
        ({'species': 'REVIEW'}, 'the prompt-shaped species key'),
        ({'review': True}, 'the review boolean alone'),
        ({'review': 1}, 'a truthy number in the review key'),
        ({'review': 'no'}, 'any truthy string in the review key'),
    ]
    for extra, label in spellings:
        point = {'label': 'A', 'x': 1, 'y': 2, 'species_code': '', 'species_name': '',
                 'category': ''}
        point.update(extra)
        check(E.review_tag_requested(point) is True, f'{label} counts as a review tag')
        exported = E.build_sam_entry(RAW_FILENAME, [point])['points'][0]
        check('review' not in exported and exported['species'] != 'REVIEW',
              f'{label} exports with no tag')
    keeps_species = E.build_sam_entry(
        RAW_FILENAME,
        [{'label': 'A', 'x': 1, 'y': 2, 'species_code': 'PA', 'species_name': '',
          'category': 'Coral', 'review': True}])['points'][0]
    check(keeps_species['species'] == 'PA',
          'a real species code is kept when only the boolean asked for the tag')
    check('review' not in keeps_species, 'and the boolean is still dropped')


def test_review_predicate_survives_malformed_points():
    """Hostile and empty records are answered, never raised on."""
    for junk in (None, 'REVIEW', 42, [], set()):
        try:
            answer = E.review_tag_requested(junk)
        except Exception as err:  # noqa: BLE001  (the point of the case)
            check(False, f'{junk!r} raised {err!r} instead of answering')
            continue
        check(answer is False, f'{junk!r} is not a review tag')
    check(E.review_tag_requested({}) is False, 'an empty point is not a review tag')
    check(E.review_tag_requested({'review': None}) is False,
          'a null review key is not a review tag')
    check(E.review_tag_requested({'species_code': None}) is False,
          'a null species code is not a review tag')
    check(E.count_review_tag_requests([]) == 0, 'an empty frame counts zero')
    check(E.count_review_tag_requests(None) == 0, 'a missing point list counts zero')


def test_export_counts_every_dropped_tag_in_a_large_batch():
    """A big frame is counted exactly, not sampled."""
    points = []
    for index in range(BULK_POINT_COUNT):
        letter = f'P{index}'
        if index % 2 == 0:
            points.append(review_point(label=letter, x=index, y=index))
        else:
            points.append({'label': letter, 'x': index, 'y': index,
                           'species_code': 'PA', 'species_name': '', 'category': 'Coral'})
    log_lines = []
    stats, exported, _csv = export_points(points, log_lines)
    check(len(exported) == BULK_POINT_COUNT, 'every point still exports')
    check(stats['review_tags_dropped'] == BULK_POINT_COUNT // 2,
          'the count matches the number of tagged points exactly')
    tagged = [p for p in exported if 'review' in p or p.get('species') == 'REVIEW']
    check(not tagged, 'not one prompt in the batch carries a tag')
    check(len([line for line in log_lines if 'suspend' in line.lower()]) == 1,
          'the suspension is reported once for the whole export, not per point')


CASES = [
    test_review_button_is_present_and_disabled,
    test_review_button_tooltip_is_the_locked_message,
    test_place_points_button_state_follows_the_switch,
    test_r_key_no_longer_arms_review,
    test_review_idx_is_still_defined,
    test_export_writes_no_review_tag,
    test_export_keeps_the_point_geometry,
    test_existing_review_point_still_displays,
    test_exported_point_is_not_a_negative_refinement_click,
    test_exported_point_through_the_annotator_selection_filter,
    test_lock_constant_is_the_single_switch,
    test_every_review_spelling_is_refused,
    test_review_predicate_survives_malformed_points,
    test_export_counts_every_dropped_tag_in_a_large_batch,
]


def main():
    """Run every case, print a verdict per case, and return an exit status."""
    global _case_fail
    failed_cases = []
    for case in CASES:
        _case_fail = 0
        print(f'\n== {case.__name__} ==')
        try:
            case()
        except Exception as err:  # noqa: BLE001  (one broken case must not hide the rest)
            print(f'  FAIL {case.__name__} raised {err!r}')
            globals()['_fail'] += 1
            _case_fail += 1
        verdict = 'PASS' if _case_fail == 0 else 'FAIL'
        if _case_fail:
            failed_cases.append(case.__name__)
        print(f'{verdict} {case.__name__}')
    print(f'\nreview lock (Place Points): {len(CASES) - len(failed_cases)}'
          f'/{len(CASES)} cases passed, {_fail} assertions failed')
    if failed_cases:
        print('failed cases: ' + ', '.join(failed_cases))
    return 1 if _fail else 0


if __name__ == '__main__':
    sys.exit(main())
