# CA/tests/test_annotator_copy.py
"""The annotator says what it is, names every code it can resolve, and tells the
truth about its keys and about rejecting a mask.

Task T11 of the VEGA student-ready sweep, 2026-09-16 (LO). A student spends most
of their annotating time in this one window, so four defects in its copy were
fixed and are held here:

  1. Bare codes. The label popup and both mask lists printed OFRA, PA, OA and
     nothing else. Each code now shows the name the project's label list gives
     it, the code beside it as a chip, and a scientific name set in italic. A
     code that resolves to nothing stays bare and says so in its hover text.
  2. The page never said what it was. The tab title and the review top bar were
     the fixed string "Annotate Images" in both the project case and the
     standalone case, so two windows open at once could not be told apart. Both
     now carry the project folder, or the standalone session name.
  3. A dead key. Expert-review flagging has been suspended since 2026-09-06 and
     the R key no longer arms it, yet two controls still printed R as though it
     did. The key badge is gone while the lock holds and returns with the
     feature. Every other key the on-screen sheet claims is checked against what
     the page actually binds.
  4. Rejecting a mask reads as final and is not. The row vanishes because
     "hide rejected" is ticked by default, and the copy said the mask was
     deleted. A reject is a status: the checkbox tooltip, a standing line in the
     Masks panel, the key sheet and both status lines now say how to bring one
     back, and acceptMask is exercised to prove it really does.

Plus the platform prose rules over the whole rendered page: no em dash, no en
dash, no emoji, never "UVI", never the word "land".

No pytest, GPU-free. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_annotator_copy.py
"""
# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import csv
import json
import os
import re
import subprocess
import sys
import tempfile

from jinja2 import Environment, FileSystemLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_TPL_DIR = os.path.join(_SRC, "templates")
_TPL = os.path.join(_TPL_DIR, "index.html")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MASTER_CODES = os.path.join(_REPO, "supporting_data", "master_codes.csv")
sys.path.insert(0, _SRC)

# A launcher-shaped export path: the annotator is given no project name, only
# this, so the project folder is read back out of it.
PROJECT_EXPORT = ("/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess"
                  "/run_26-09-15_LO_train_coral_intel_subset/step4test_combinedAnnotate")
PROJECT_FOLDER = "26-09-15_LO_train_coral_intel_subset"

# Three labels that exercise the whole name rule: a binomial, an English common
# name that happens to be two words, and a code the label list cannot resolve.
TARGET_LABELS = [
    {"code": "PA", "name": "Porites astreoides", "category": "Coral"},
    {"code": "BALL", "name": "Ball sponge", "category": "Sponge"},
    {"code": "ZZZ", "name": "", "category": ""},
]

# Every key the on-screen sheet claims that the keydown switch must bind. Tab,
# Enter, Escape, Delete, the arrows and Space are named by word in the sheet and
# are checked separately by _bound_keys.
SHEET_LETTER_KEYS = set("ABCDFGHKMNOPRSTUXZ")

_fail = 0


def check(condition, message):
    """Record one assertion, printing PASS or FAIL and counting the failures."""
    global _fail
    print(("  PASS " if condition else "  FAIL ") + message)
    if not condition:
        _fail += 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _template_source():
    """Return the raw Jinja template text.

    Returns:
        str: the whole of src/templates/index.html.
    """
    with open(_TPL) as handle:
        return handle.read()


def _context(**overrides):
    """Build a complete render context for the annotator page.

    Rendering the template directly, rather than through the app's test client,
    is what lets one test choose the export path (and so the project folder) and
    the session mode. One test below still goes through the real route.

    Args:
        **overrides: context values to replace.

    Returns:
        dict: every variable index.html reads.
    """
    ctx = dict(
        default_input="/mnt/rip/routed_input",
        default_export=PROJECT_EXPORT,
        default_categories=["Target species only"],
        all_categories=["Target species only", "Coral", "Sponge"],
        target_species=[lbl["code"] for lbl in TARGET_LABELS],
        target_labels=TARGET_LABELS,
        default_review_dir="/mnt/tear/REVIEW_reefpointseg",
        default_contacts="lauren.olinger@uvi.edu",
        target_species_only=True,
        reference_default=True,
        default_confidence=0.4,
        default_min_area=200,
        default_merge_dist=30,
        default_overlap="larger_wins",
        default_mask_size="generous",
        default_thin_ratio=0.1,
        default_simplify=0.004,
        default_exemplar=0.5,
        default_dev_tracker="cuda:1",
        default_dev_exemplar="cuda:0",
        orchestrator_url="http://127.0.0.1:5050",
        orchestrated=False,
        session_mode="configure",
        review_locked_msg="Flagging for expert review is temporarily unavailable."
                          " VICARIUS is accepting no new review tags.",
        review_tagging_enabled=False,
        default_session_name="26-09-16_LO",
    )
    ctx.update(overrides)
    return ctx


def _render(**overrides):
    """Render index.html with a controlled context.

    Args:
        **overrides: context values to replace, see _context.

    Returns:
        str: the rendered HTML.
    """
    env = Environment(loader=FileSystemLoader(_TPL_DIR))
    return env.get_template("index.html").render(**_context(**overrides))


def _tag_carrying(html, needle):
    """Return the single opening tag that contains `needle`.

    Args:
        html: the document to search.
        needle: a substring appearing inside exactly one opening tag.

    Returns:
        str: the opening tag text, angle brackets included.

    Raises:
        AssertionError: when the needle is absent.
    """
    pos = html.find(needle)
    if pos < 0:
        raise AssertionError("%r is not in the page" % needle)
    start = html.rfind("<", 0, pos)
    end = html.find(">", pos)
    if start < 0 or end < 0:
        raise AssertionError("%r is not inside a tag" % needle)
    return html[start:end + 1]


def _element(html, needle):
    """Return the whole element, opening tag through closing tag, holding `needle`.

    Args:
        html: the document to search.
        needle: a substring appearing inside exactly one element.

    Returns:
        str: the element's markup.

    Raises:
        AssertionError: when the needle is absent.
    """
    pos = html.find(needle)
    if pos < 0:
        raise AssertionError("%r is not in the page" % needle)
    start = html.rfind("<", 0, pos)
    end = html.find("</", pos)
    if start < 0 or end < 0:
        raise AssertionError("%r has no element around it" % needle)
    return html[start:html.find(">", end) + 1]


def _block(html, start_marker, end_marker):
    """Return the slice of `html` between two markers.

    Args:
        html: the document to search.
        start_marker: substring that opens the block.
        end_marker: substring that closes it.

    Returns:
        str: everything from the start marker to the end marker.

    Raises:
        AssertionError: when either marker is absent.
    """
    start = html.find(start_marker)
    if start < 0:
        raise AssertionError("%r is not in the page" % start_marker)
    end = html.find(end_marker, start)
    if end < 0:
        raise AssertionError("%r does not follow %r" % (end_marker, start_marker))
    return html[start:end]


def _extract_function(name, src=None):
    """Slice one top-level JavaScript function verbatim out of the template.

    Every function in this template closes with a brace in column one, so the
    first '\\n}\\n' after the declaration is its end.

    Args:
        name: the function name, without 'function'.
        src: template text to search; read from disk when omitted.

    Returns:
        str: the function source, declaration through closing brace.
    """
    src = src if src is not None else _template_source()
    marker = "\nfunction %s(" % name
    start = src.index(marker) + 1
    end = src.index("\n}\n", start) + len("\n}\n")
    return src[start:end]


def _find_node():
    """Return a usable node binary path, or None when node is unavailable."""
    for candidate in ("node", "/usr/bin/node"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except Exception:
            continue
    return None


def _run_node(harness, label):
    """Run one JavaScript harness under node and report its exit status.

    Args:
        harness: the complete script to run.
        label: the assertion message for the PASS or FAIL line.
    """
    node = _find_node()
    if node is None:
        print("    (node unavailable, skipping) " + label)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(harness)
        path = handle.name
    try:
        result = subprocess.run([node, path], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            print("    " + line)
        if result.stderr.strip():
            print("    [stderr] " + result.stderr.strip())
        check(result.returncode == 0, label)
    finally:
        os.unlink(path)


def _name_helpers_js(src=None):
    """Return the four label-name helpers as one runnable JavaScript block.

    Returns:
        str: isScientificName, labelNameFor, labelNameHtml and labelRowTitle,
            preceded by the word list and the category map they read.
    """
    src = src if src is not None else _template_source()
    words = _block(src, "const BINOMIAL_COMMON_WORDS =", "\n\n")
    words = re.sub(r"{{.*?}}", json.dumps(_binomial_common_words(src)), words,
                   flags=re.DOTALL)
    category_map = _block(src, "const CATEGORY_CODE_NAMES = {", "};") + "};"
    return "\n".join([
        words,
        category_map,
        # An empty stand-in, not the real seeded object (2026-09-27 pixel
        # review added this third name source): labelNameFor only reads it,
        # never assumes it holds anything, so an always-empty registry here
        # is a valid "fresh page load, nothing learned yet" state and keeps
        # this harness independent of TARGET_LABELS's declaration order,
        # which callers of this helper set on their own.
        "const _knownLabelNames = {};",
        _extract_function("escHtml", src),
        _extract_function("isScientificName", src),
        _extract_function("labelNameFor", src),
        _extract_function("labelNameHtml", src),
        _extract_function("labelRowTitle", src),
    ])


def _binomial_common_words(src=None):
    """Return the shared word list out of the Jinja constant at the template head.

    Returns:
        list[str]: the English common nouns that keep a two-word name roman.
    """
    src = src if src is not None else _template_source()
    block = _block(src, "{%- set BINOMIAL_COMMON_WORDS = [", "] -%}")
    return re.findall(r"'([a-z.]+)'", block)


def _master_code_rows():
    """Return the whole label vocabulary the platform ships.

    Returns:
        list[dict]: rows of code, category, name; empty when the file is absent.
    """
    if not os.path.isfile(_MASTER_CODES):
        return []
    with open(_MASTER_CODES) as handle:
        return list(csv.DictReader(handle))


def _keydown_source(src=None):
    """Return the body of the page's keydown listener.

    Returns:
        str: everything from the listener's declaration to the keyup listener
            that follows it.
    """
    src = src if src is not None else _template_source()
    return _block(src, "document.addEventListener('keydown'",
                  "document.addEventListener('keyup'")


def _bound_keys(src=None):
    """Return every single-letter key the keydown listener actually binds.

    Reads both shapes the listener uses: the `case 'x': case 'X':` arms of its
    switch, and the `e.key === 'x'` comparisons in the branches that run before
    the switch (the cheat sheet toggle, the export dialog, the label popup).

    Returns:
        set[str]: uppercase letters.
    """
    body = _keydown_source(src)
    keys = set(re.findall(r"case '([A-Za-z])':", body))
    keys |= set(re.findall(r"e\.key === '([A-Za-z])'", body))
    keys |= set(re.findall(r"e\.key\.toLowerCase\(\) === '([a-z])'", body))
    keys |= set(re.findall(r"catKeys\[e\.key\.toLowerCase\(\)\]", body) and
                re.findall(r"\{([a-z]):'", body))
    return {k.upper() for k in keys}


def _visible_text(html):
    """Return the page stripped of tags, with every title attribute kept.

    Both halves matter: the prose rules apply to on-screen text and to hover
    text alike.

    Args:
        html: the rendered page.

    Returns:
        str: the visible text and every tooltip, newline separated.
    """
    titles = re.findall(r'title="([^"]*)"', html)
    body = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.DOTALL)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    body = re.sub(r"<[^>]+>", " ", body)
    return "\n".join([body] + titles)


# ── 1. The window says which project or session it is on ─────────────────────

def test_project_window_names_its_project():
    """The tab title and the top bar both carry the project folder."""
    page = _render(session_mode="configure")
    title = re.search(r"<title>(.*?)</title>", page, flags=re.DOTALL).group(1)
    check(PROJECT_FOLDER in title, "the tab title names the project, got %r" % title)
    check(title.startswith(PROJECT_FOLDER),
          "the project comes first, so a truncated tab still shows it")
    check("Annotate Images" in title, "the tab title still names the tool")
    chip = _element(page, 'id="topbar-session"')
    check(PROJECT_FOLDER in chip, "the review top bar names the project")
    check("hidden" not in _tag_carrying(page, 'id="topbar-session"'),
          "the top bar chip is shown when there is a project to name")
    check('title="%s"' % PROJECT_FOLDER in _tag_carrying(page, 'id="topbar-session"'),
          "the chip carries the whole value in hover text, for when it ellipsizes")
    glyph = _tag_carrying(page, 'id="topbar-session-help"')
    check("VEGA" in glyph, "the help glyph names VEGA as the actor")
    check('title="VEGA' in glyph, "the help glyph opens with the actor, not with You")
    check("text-overflow: ellipsis" in page.split("<script>")[0],
          "the chip truncates rather than pushing the top bar wider")


def test_a_project_window_and_a_standalone_window_differ():
    """The two windows a student can legitimately have open are not twins."""
    project = _render(session_mode="configure")
    standalone = _render(session_mode="standalone")
    p_title = re.search(r"<title>(.*?)</title>", project, flags=re.DOTALL).group(1)
    s_title = re.search(r"<title>(.*?)</title>", standalone, flags=re.DOTALL).group(1)
    check(p_title != s_title, "the two tab titles differ, got %r and %r"
          % (p_title, s_title))
    check(PROJECT_FOLDER not in s_title,
          "a standalone window never borrows the project folder of an export path")
    tag = _tag_carrying(standalone, 'id="topbar-session"')
    check("hidden" in tag,
          "the standalone chip stays hidden until the session name is known")


def test_standalone_session_name_is_kept_and_put_back():
    """The name typed at Start survives the reload Start performs."""
    src = _template_source()
    check("const SESSION_NAME_KEY = 'sa_session_name';" in src,
          "the stored key is named once")
    check("localStorage.setItem(SESSION_NAME_KEY, sessionName);" in src,
          "saStart stores the session name it just started")
    check("fd.append('session', sessionName);" in src,
          "the stored name is the same value the server was sent")
    harness = """
%(key)s
%(fn)s
let fails = 0;
let store = {};
let titleSet = '';
const chip = {textContent: '', title: '', classes: ['topbar-session', 'hidden'],
  classList: {remove(c) { chip.classes = chip.classes.filter(x => x !== c); }}};
global.localStorage = {
  getItem: k => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = v; },
  removeItem: k => { delete store[k]; },
};
global.document = {
  getElementById: id => (id === 'topbar-session' ? chip : null),
  set title(v) { titleSet = v; },
  get title() { return titleSet; },
};
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
// A project window ignores the store entirely.
let SESSION_MODE = 'configure';
store[SESSION_NAME_KEY] = 'leftover';
_applySessionLabel({configured: true});
want(chip.textContent === '', 'a project window never reads the standalone store');
want(store[SESSION_NAME_KEY] === 'leftover', 'a project window never clears it either');

// A standalone window that has not started a session drops a stale name.
SESSION_MODE = 'standalone';
_applySessionLabel({configured: false});
want(store[SESSION_NAME_KEY] === undefined,
     'an unstarted standalone window drops a stale session name');
want(chip.classes.includes('hidden'), 'and shows no chip at all');

// A started session is named, in the chip and in the tab title.
store[SESSION_NAME_KEY] = '26-09-16_LO';
_applySessionLabel({configured: true});
want(chip.textContent === '26-09-16_LO', 'a started session names the chip');
want(chip.title === '26-09-16_LO', 'and carries the whole value in hover text');
want(!chip.classes.includes('hidden'), 'and the chip is shown');
want(titleSet === '26-09-16_LO | Annotate Images', 'and the tab title leads with it');

// Storage that refuses (a private window) must not break the page.
global.localStorage = {
  getItem() { throw new Error('denied'); },
  setItem() { throw new Error('denied'); },
  removeItem() { throw new Error('denied'); },
};
let threw = false;
try { _applySessionLabel({configured: true}); } catch (e) { threw = true; }
want(threw === false, 'blocked storage is caught, not thrown at the operator');
process.exit(fails ? 1 : 0);
""" % {"key": "const SESSION_NAME_KEY = 'sa_session_name';",
       "fn": _extract_function("_applySessionLabel", src)}
    _run_node(harness, "the standalone session label is stored, restored and safe")


def test_the_real_route_still_renders():
    """The app's own GET / renders the page, so nothing added breaks the route."""
    os.environ.setdefault("TCRMP_TARGET_SPECIES", "DICT")
    os.environ.setdefault("TCRMP_MANUAL_ANNOTATE", "1")
    import app as A
    A.app.config["TESTING"] = True
    response = A.app.test_client().get("/")
    check(response.status_code == 200,
          "GET / returns 200, got %s" % response.status_code)
    page = response.get_data(as_text=True)
    check('id="topbar-session"' in page, "the session chip is rendered by the real route")
    check("<title>" in page and "Annotate Images" in page,
          "the real route still titles the page")


# ── 2. Full names, never bare codes ──────────────────────────────────────────

def test_label_popup_shows_names_beside_codes():
    """Every popup row carries the name its code resolves to, or says it cannot."""
    page = _render()
    popup = _block(page, '<div id="cat-popup"', '<div id="mode-indicator">')
    check("Porites astreoides" in popup, "the popup shows the binomial for PA")
    check('<i class="sci-name">Porites astreoides</i>' in popup,
          "a scientific name is set in italic")
    check("Ball sponge" in popup, "the popup shows the English name for BALL")
    check('<i class="sci-name">Ball sponge</i>' not in popup,
          "an English common name is not set in italic")
    check('<span class="cat-code">PA</span>' in popup,
          "the code rides beside the name as a chip")
    zzz_row = _element(popup, "assignNewMaskSpecies('ZZZ')")
    check("<span class=\"cat-name\">" not in zzz_row,
          "a code that resolves to nothing shows no invented name")
    check("records no name for the code ZZZ" in zzz_row,
          "and says in hover text why it stands bare")
    check("Target labels" in popup, "the popup heading is sentence case")
    check("TARGET SPECIES" not in popup, "the old uppercase heading is gone")
    for clipped in (">Macro<", ">Gorg<"):
        check(clipped not in popup,
              "the category chips spell their names out, found %r" % clipped)


def test_quick_add_bar_shows_names_the_same_way():
    """The arm-a-label bar follows the same italic rule as the popup."""
    page = _render()
    bar = _block(page, '<div class="quickadd-bar"', '<div class="review-body">')
    check('<i class="sci-name">Porites astreoides</i>' in bar,
          "the quick-add bar italicises a scientific name")
    check('<i class="sci-name">Ball sponge</i>' not in bar,
          "and leaves an English common name roman")
    check("font-style: italic" not in
          _block(page, ".quickadd-bar .qa-btn .qa-name {", "}"),
          "the bar no longer italicises every name it is handed")
    tag = _tag_carrying(bar, 'data-code="PA"')
    check("Porites astreoides, code PA" in tag,
          "the hover text names the label before its code")


def test_the_name_rule_is_one_rule():
    """The template macro and its JavaScript twin agree on the whole vocabulary."""
    rows = _master_code_rows()
    if not rows:
        print("    (master_codes.csv absent, skipping) one name rule")
        return
    names = [row["name"] for row in rows]
    env = Environment(loader=FileSystemLoader(_TPL_DIR))
    macro_tpl = ("{% from 'index.html' import label_name %}"
                 "{% for n in names %}{{ label_name(n) }}\n{% endfor %}")
    rendered = env.from_string(macro_tpl).render(names=names).strip().split("\n")
    macro_italic = [n for n, out in zip(names, rendered) if out.startswith("<i")]
    harness = """
%(helpers)s
const names = %(names)s;
const italic = names.filter(n => isScientificName(n));
console.log(JSON.stringify(italic));
""" % {"helpers": _name_helpers_js(), "names": json.dumps(names)}
    node = _find_node()
    if node is None:
        print("    (node unavailable, skipping) one name rule")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(harness)
        path = handle.name
    try:
        result = subprocess.run([node, path], capture_output=True, text=True)
        check(result.returncode == 0, "the JavaScript name rule runs: %s" % result.stderr)
        js_italic = json.loads(result.stdout.strip() or "[]")
    finally:
        os.unlink(path)
    check(sorted(js_italic) == sorted(macro_italic),
          "the macro and the script italicise the same %d of %d names (macro only: %s,"
          " script only: %s)"
          % (len(macro_italic), len(names),
             sorted(set(macro_italic) - set(js_italic)),
             sorted(set(js_italic) - set(macro_italic))))
    # The rule must be right, not merely consistent: a binomial is italic and an
    # English two-word name is not.
    for name in ("Porites astreoides", "Orbicella faveolata", "Agaricia spp.",
                 "Briareum asbestinum"):
        check(name in js_italic, "%r is treated as a scientific name" % name)
    for name in ("Ball sponge", "Sea fan", "Bleached coral", "Transect tape",
                 "Filamentous cyanobacteria", "Dead coral w/ turf algae"):
        check(name not in js_italic, "%r is not treated as a scientific name" % name)


def test_mask_list_rows_resolve_their_codes():
    """Both lists put the resolved name on the row and explain a bare code."""
    src = _template_source()
    check(src.count("${labelNameHtml(m.species, m.name)}") == 2,
          "the mask list and the scan-candidate list both render the name")
    check(src.count("labelRowTitle(m.species, m.name)") == 2,
          "both lists carry the label in hover text")
    harness = """
%(helpers)s
const TARGET_LABELS = [
  {code: 'PA', name: 'Porites astreoides', category: 'Coral'},
  {code: 'BALL', name: 'Ball sponge', category: 'Sponge'}
];
let fails = 0;
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
want(labelNameFor('PA') === 'Porites astreoides', 'a target code resolves to its name');
want(labelNameHtml('PA').includes('sci-name'), 'and its name is italic');
want(labelNameFor('BALL') === 'Ball sponge', 'an English name resolves too');
want(!labelNameHtml('BALL').includes('sci-name'), 'and is not italic');
want(labelNameFor('SPO') === 'Sponge', 'a category code resolves from the popup words');
want(labelNameFor('MACA') === 'Macroalgae', 'so does the macroalgae category code');
want(labelNameFor('REVIEW') === 'Flagged for expert review', 'so does the review tag');
want(labelNameFor('ZZZ') === '', 'an unknown code resolves to nothing');
want(labelNameHtml('ZZZ') === '', 'and adds no second line to its row');
want(labelRowTitle('ZZZ').includes('records no name for the code ZZZ'),
     'and says so in hover text');
want(labelRowTitle('').includes('no label yet'), 'an unlabelled mask says it has no label');
want(labelNameFor('PA', 'PA') === 'Porites astreoides',
     'the label list beats a mask that saved its own code as a name');
want(labelNameFor('QQQ', 'QQQ') === '', 'a code repeated as a name is not a name');
want(labelNameFor('QQQ', 'Some hand typed name') === 'Some hand typed name',
     'a real saved name is used when nothing else resolves');
want(labelNameHtml('PA', '<img onerror=x>').includes('Porites'), 'escaping is exercised');
want(!labelNameHtml('QQQ', '<b>x</b>').includes('<b>'), 'a saved name is escaped, not injected');
process.exit(fails ? 1 : 0);
""" % {"helpers": _name_helpers_js(src)}
    _run_node(harness, "code resolution covers every case a row can hold")


# ── 3. The keys the page claims ──────────────────────────────────────────────

def test_no_control_asks_for_a_key_that_does_nothing():
    """While expert-review flagging is suspended, nothing prints R as a live key.

    Two different treatments for the two REVIEW controls (2026-09-27 review of
    the VEGA annotator tidy-up plan): the quickadd bar's button, a control
    nobody could click, is left out of the markup entirely while locked (a
    real deletion, reversed the moment the feature returns); the label popup's
    chip, sitting among species codes an operator is actively choosing from,
    keeps the original disabled-not-removed treatment.
    """
    locked = _render(review_tagging_enabled=False)
    check('id="qa-btn-review"' not in locked,
          "the quick-add REVIEW button is not on the page while locked")
    chip = _element(locked, 'id="cat-btn-review"')
    check("<kbd>" not in chip, "the suspended popup REVIEW chip shows no key badge")
    check("Tap R" not in locked, "no control invites a tap on R")
    check("R now arms REVIEW" not in locked, "no tooltip claims R arms REVIEW")

    unlocked = _render(review_tagging_enabled=True)
    check("qa-key" in _element(unlocked, 'id="qa-btn-review"'),
          "the quick-add button comes back live, key badge included, so this "
          "is a flag on the feature, not a deletion of it")
    check("<kbd>r</kbd>" in _element(unlocked, 'id="cat-btn-review"'),
          "the popup chip's key badge comes back too")


def test_the_key_sheet_matches_what_the_page_binds():
    """Every letter the on-screen sheet claims is a key the listener handles."""
    page = _render()
    sheet = _block(page, '<div id="kbd-help"', '<!-- End-of-batch confirm')
    claimed = {k.upper() for k in re.findall(r"<kbd>([A-Za-z])</kbd>", sheet)}
    bound = _bound_keys()
    check(claimed <= bound,
          "every claimed letter is bound (claimed and unbound: %s)"
          % sorted(claimed - bound))
    check(SHEET_LETTER_KEYS <= bound,
          "the keys this suite knows about are all bound (missing: %s)"
          % sorted(SHEET_LETTER_KEYS - bound))
    # The R row is dropped from the sheet entirely while locked (2026-09-27
    # pixel review, defect 18: an inert row describing a key that currently
    # does nothing was itself a defect, not a fix), and reappears, arming
    # wording included, once expert-review flagging is back on.
    r_row = [line for line in sheet.splitlines() if "<kbd>R</kbd>" in line]
    check(len(r_row) == 0, "the sheet has no row for R while locked, got %r" % r_row)
    unlocked_sheet = _block(_render(review_tagging_enabled=True),
                             '<div id="kbd-help"', '<!-- End-of-batch confirm')
    r_row_unlocked = [line for line in unlocked_sheet.splitlines() if "<kbd>R</kbd>" in line]
    check(len(r_row_unlocked) == 1, "the sheet has exactly one row for R once unlocked")
    check("arm REVIEW" in r_row_unlocked[0],
          "the R row says the key arms REVIEW once unlocked, got %r"
          % r_row_unlocked[0][-90:])
    # P is the frame timer, not a marker toggle; G is the reference points.
    p_row = [line for line in sheet.splitlines() if "<kbd>P</kbd>" in line]
    check(any("timer" in line for line in p_row), "the P row names the frame timer")
    g_row = [line for line in sheet.splitlines() if "<kbd>G</kbd>" in line]
    check(any("reference points" in line for line in g_row),
          "the G row names the reference points")
    # The popup's own keys are documented where the popup is opened.
    c_row = [line for line in sheet.splitlines() if "<kbd>C</kbd>" in line]
    check(any("s / m / g / t" in line for line in c_row),
          "the C row names the category keys the popup takes")


# ── 4. Rejecting a mask is reversible and says so ────────────────────────────

def test_rejecting_a_mask_never_reads_as_final():
    """The checkbox, the panel, the key sheet and both status lines agree."""
    page = _render()
    tooltip = _tag_carrying(page, 'id="chk-hide-rejected"')
    label = _tag_carrying(page, 'class="hide-rej"')
    check("is not deleted" in label,
          "the hide-rejected tooltip says a rejected mask is not deleted")
    check("press Enter to accept it again" in label,
          "and says exactly how to bring one back")
    check("stay rejected and excluded from export" not in page,
          "the old tooltip, which read as final, is gone")
    check("Rejected is not final" not in page,
          "the standing help-hint line is gone (2026-09-27 pixel review, "
          "defect 11): it repeated a fact the hide-rejected tooltip, "
          "checked above, already states")
    sheet = _block(page, '<div id="kbd-help"', '<!-- End-of-batch confirm')
    check("on a rejected row" in sheet, "the key sheet has a row for bringing one back")
    check("never deleted" in sheet, "and the reject row says a reject is not a delete")
    src = _template_source()
    check("Deleted ${n} mask" not in src,
          "the multi-select status line no longer calls a reject a delete")
    check("Rejected ${n} mask" in src, "it calls it a reject")
    reject = _extract_function("rejectMask", src)
    check("Untick hide rejected" in reject,
          "a single reject says how to undo itself at the moment it happens")
    check("checked" in tooltip, "hide rejected is still ticked by default")


def test_enter_really_does_bring_a_rejected_mask_back():
    """acceptMask flips a rejected mask to accepted, which is what the copy claims."""
    src = _template_source()
    harness = """
let currentData = {masks: [
  {id: 1, label: 'A', species: 'PA', status: 'rejected'},
  {id: 2, label: 'B', species: '', status: 'rejected'}
]};
let selectedMask = 0;
let stickyAddMode = false;
let stickyBoxMode = false;
let autoAdvance = false;
let pendingAddMaskIdx = -1;
const REVIEW_TAGGING_ENABLED = false;
let persisted = [];
let popupShown = 0;
let statusLines = [];
function persistMaskStatus(id, s, prev) { persisted.push([id, s, prev]); }
function editBlockedByLoad() { return false; }
function flashMode() {}
function resumeStickyMode() {}
function setInteractionMode() {}
function renderMaskList() {}
function render() {}
function advanceToNextOrFrame() {}
function showCatPopup() { popupShown++; }
function setStatus(s) { statusLines.push(s); }
%(accept)s
let fails = 0;
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
acceptMask();
want(currentData.masks[0].status === 'accepted',
     'Enter on a rejected, labelled mask accepts it again');
want(persisted.length === 1 && persisted[0][2] === 'rejected',
     'the server is told the previous status was rejected');
selectedMask = 1;
acceptMask();
want(currentData.masks[1].status === 'rejected',
     'a rejected mask with no label is not silently accepted');
want(popupShown === 1, 'it opens the label popup instead');
process.exit(fails ? 1 : 0);
""" % {"accept": _extract_function("acceptMask", src)}
    _run_node(harness, "Enter on a rejected row really does accept it again")


# ── Prose over the whole page ────────────────────────────────────────────────

def test_the_page_obeys_the_prose_rules():
    """No em dash, no en dash, no emoji, no UVI, no banned word, anywhere."""
    src = _template_source()
    for name, glyph in (("em dash", "—"), ("en dash", "–")):
        check(glyph not in src, "no %s anywhere in the template" % name)
    for mode in ("configure", "standalone"):
        page = _render(session_mode=mode)
        text = _visible_text(page)
        check("UVI" not in text.replace("uvi.edu", ""),
              "no UVI in %s mode" % mode)
        check(not re.search(r"\bland(s|ed|ing)?\b", text, re.I),
              "no banned word in %s mode" % mode)
        stray = sorted({c for c in text if 0x1F000 <= ord(c) <= 0x1FAFF
                        or 0x2600 <= ord(c) <= 0x27BF})
        check(not stray, "no emoji in %s mode, found %r" % (mode, stray))


def test_the_added_tooltips_read_like_tooltips():
    """The tooltips this change writes open with an actor and finish a sentence."""
    page = _render()
    added = [
        _tag_carrying(page, 'id="topbar-session-help"'),
        _tag_carrying(page, 'class="hide-rej"'),
        _tag_carrying(page, 'data-code="PA"'),
    ]
    for tag in added:
        tooltip = re.search(r'title="([^"]*)"', tag).group(1)
        label = tooltip[:44]
        check(not tooltip.startswith("You"), "does not open with You: %s" % label)
        check(tooltip.rstrip().endswith("."), "ends in a full stop: %s" % label)
        check(tooltip.count(".") <= 2, "at most two sentences: %s" % label)
        for screen_word in (" above", " below", " this tab", " on screen"):
            check(screen_word not in tooltip.lower(),
                  "no screen reference: %s" % label)


def test_the_inline_script_still_parses():
    """The page script passes node --check after the Jinja is stripped."""
    node = _find_node()
    if node is None:
        print("    (node unavailable, skipping) node --check")
        return
    src = _template_source()
    blocks = re.findall(r"<script>(.*?)</script>", src, flags=re.DOTALL)
    js = "\n".join(blocks)
    js = re.sub(r"{%.*?%}", "", js, flags=re.DOTALL)
    js = re.sub(r"{{.*?}}", "0", js, flags=re.DOTALL)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(js)
        path = handle.name
    try:
        result = subprocess.run([node, "--check", path], capture_output=True, text=True)
        check(result.returncode == 0, "node --check passes: %s" % result.stderr.strip())
    finally:
        os.unlink(path)


def _run(case):
    """Run one case and keep going when a missing element aborts it.

    Args:
        case: the zero-argument test function to run.
    """
    global _fail
    print(case.__name__)
    try:
        case()
    except Exception as exc:
        print("  FAIL %s raised %s: %s" % (case.__name__, type(exc).__name__, exc))
        _fail += 1


if __name__ == "__main__":
    for _case in (
        test_project_window_names_its_project,
        test_a_project_window_and_a_standalone_window_differ,
        test_standalone_session_name_is_kept_and_put_back,
        test_the_real_route_still_renders,
        test_label_popup_shows_names_beside_codes,
        test_quick_add_bar_shows_names_the_same_way,
        test_the_name_rule_is_one_rule,
        test_mask_list_rows_resolve_their_codes,
        test_no_control_asks_for_a_key_that_does_nothing,
        test_the_key_sheet_matches_what_the_page_binds,
        test_rejecting_a_mask_never_reads_as_final,
        test_enter_really_does_bring_a_rejected_mask_back,
        test_the_page_obeys_the_prose_rules,
        test_the_added_tooltips_read_like_tooltips,
        test_the_inline_script_still_parses,
    ):
        _run(_case)
    print("PASS" if _fail == 0 else "%d FAILED" % _fail)
    sys.exit(1 if _fail else 0)
