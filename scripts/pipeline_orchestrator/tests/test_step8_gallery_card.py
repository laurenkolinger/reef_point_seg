"""
Module: test_step8_gallery_card
Purpose: house checks for the VEGA inference gallery card (the Step 7: Model
         Inference panel's results block): the Jinja fragment renders with no
         variables, every control carries a tooltip on a keyboard-focusable
         glyph that obeys the platform sentence rules, the root class appears
         exactly once, there is no inline script, the stylesheet is scoped
         under the root class, the JavaScript never builds inline handlers,
         and none of the three files carries banned prose.
Inputs:  scripts/pipeline_orchestrator/templates/_step8_gallery.html
         scripts/pipeline_orchestrator/static/orchestrator_gallery.css
         scripts/pipeline_orchestrator/static/orchestrator_gallery.js
Outputs: PASS/FAIL lines and a non-zero exit on any failure.

House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step8_gallery_card.py
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

import os
import re
import sys
import traceback
from html.parser import HTMLParser

import jinja2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_TEMPLATES = os.path.join(_ORCH, "templates")
_FRAGMENT = "_step8_gallery.html"
_CSS = os.path.join(_ORCH, "static", "orchestrator_gallery.css")
_JS = os.path.join(_ORCH, "static", "orchestrator_gallery.js")

ROOT_CLASS = "vega-gallery"
ROOT_ID = "s8-gallery-card"
GLYPH_CLASS = "vg-tip"
CONTROL_TAGS = ("button", "select", "input", "textarea", "a")
EM_DASH = "\u2014"
EN_DASH = "\u2013"
BANNED_WORD = re.compile(r"\bland(s|ed|ing)?\b", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?](\s|$)")
MAX_SENTENCES = 2
MIN_WORDS = 4
TERMS_OF_ART = ("confidence", "overlap threshold", "blind spot")
# The guard that keeps a long option in the run picker from widening the host
# panel (pixel review of 2026-09-05, workflow page design section 8a).
CONTAINMENT_GUARD = "contain: inline-size"

_fail = 0
_RESULTS = []


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, print PASS or FAIL, and remember the outcome."""
    global _fail
    try:
        fn()
        _RESULTS.append((fn.__name__, True))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001 - a test harness reports every failure kind
        _fail += 1
        _RESULTS.append((fn.__name__, False))
        print(f"  FAIL {fn.__name__}: {e}")
        traceback.print_exc()


# -- sources -----------------------------------------------------

_CACHE = {}


def _read(path):
    """Read a file once and cache it."""
    if path not in _CACHE:
        with open(path, encoding="utf-8") as fh:
            _CACHE[path] = fh.read()
    return _CACHE[path]


def _render_fragment():
    """Render the fragment with no variables; StrictUndefined proves it needs none."""
    if "html" not in _CACHE:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(_TEMPLATES),
            undefined=jinja2.StrictUndefined,
            autoescape=True,
        )
        _CACHE["html"] = env.get_template(_FRAGMENT).render()
    return _CACHE["html"]


class _Collector(HTMLParser):
    """Collect every start tag with its attributes, in document order."""

    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _tags():
    """Parsed start tags of the rendered fragment."""
    if "tags" not in _CACHE:
        parser = _Collector()
        parser.feed(_render_fragment())
        _CACHE["tags"] = parser.tags
    return _CACHE["tags"]


def _classes(attrs):
    """Class tokens of one element."""
    return (attrs.get("class") or "").split()


def _is_glyph(tag, attrs):
    """Is this element a tooltip glyph (a button carrying the glyph class)?"""
    return tag == "button" and GLYPH_CLASS in _classes(attrs)


def _sentence_count(text):
    """Number of sentence terminators followed by a space or the end."""
    return len(SENTENCE_END.findall(text))


def _check_tooltip_text(text, where):
    """Apply the platform tooltip sentence rules to one title string."""
    check(text and text.strip() == text, f"{where}: tooltip is empty or has stray whitespace")
    check(text.endswith("."), f"{where}: tooltip must end with a period: {text!r}")
    n = _sentence_count(text)
    check(1 <= n <= MAX_SENTENCES, f"{where}: tooltip has {n} sentences, want 1 or 2: {text!r}")
    check(len(text.split()) >= MIN_WORDS, f"{where}: tooltip is a fragment: {text!r}")
    check(not text.startswith("You"), f"{where}: tooltip opens with You: {text!r}")
    check(not re.search(r"\b(above|below|this tab|this window)\b", text, re.IGNORECASE),
          f"{where}: tooltip carries a screen reference: {text!r}")
    check(text[0].isupper(), f"{where}: tooltip must start with a capital: {text!r}")


# -- tests -------------------------------------------------------


def test_fragment_renders_with_no_variables():
    html = _render_fragment()
    check(len(html) > 200, "fragment rendered to almost nothing")
    check("{{" not in html and "{%" not in html, "unrendered Jinja left in the output")


def test_root_class_and_id_exactly_once():
    html = _render_fragment()
    roots = [a for t, a in _tags() if ROOT_CLASS in _classes(a)]
    check(len(roots) == 1, f"root class {ROOT_CLASS} appears on {len(roots)} elements, want 1")
    ids = [a for t, a in _tags() if a.get("id") == ROOT_ID]
    check(len(ids) == 1, f"root id {ROOT_ID} appears {len(ids)} times, want 1")
    check(html.count(ROOT_CLASS) == 1, f"the string {ROOT_CLASS} appears {html.count(ROOT_CLASS)} times in the markup, want 1")


def test_no_inline_script_or_handlers():
    html = _render_fragment()
    check("<script" not in html.lower(), "fragment carries an inline script tag")
    check("javascript:" not in html.lower(), "fragment carries a javascript: URL")
    for tag, attrs in _tags():
        for name in attrs:
            check(not name.lower().startswith("on"), f"<{tag}> carries an inline handler attribute {name}")
        check("style" not in attrs, f"<{tag}> carries an inline style; styling lives in the scoped stylesheet")


def test_every_control_has_a_tooltip_on_a_focusable_glyph():
    tags = _tags()
    glyphs = [a for t, a in tags if _is_glyph(t, a)]
    check(glyphs, "no tooltip glyphs in the fragment")
    for attrs in glyphs:
        check(attrs.get("type") == "button", "glyph must be type=button so it never submits")
        check("tabindex" not in attrs or attrs["tabindex"] != "-1", "glyph must stay in the tab order")
        _check_tooltip_text(attrs.get("title", ""), "glyph")
    # Every field (select, input, textarea) has a glyph pointing at it by id.
    field_ids = [a.get("id") for t, a in tags if t in ("select", "input", "textarea")]
    check(field_ids, "no fields found in the fragment")
    pointed = {a.get("data-tip-for") for a in glyphs}
    for fid in field_ids:
        check(fid, "a field has no id")
        check(fid in pointed, f"field #{fid} has no ? glyph carrying its tooltip (data-tip-for)")
        labels = [a for t, a in tags if t == "label" and a.get("for") == fid]
        check(len(labels) == 1, f"field #{fid} needs exactly one label[for]")
    # Every other control (a non-glyph button or link) carries its own tooltip.
    for tag, attrs in tags:
        if tag in CONTROL_TAGS and not _is_glyph(tag, attrs) and tag not in ("select", "input", "textarea"):
            _check_tooltip_text(attrs.get("title", ""), f"<{tag} id={attrs.get('id')}>")
            if tag == "button":
                check(attrs.get("type") == "button", f"button #{attrs.get('id')} must be type=button")


def test_heading_glyph_explains_terms_of_art_once():
    tags = _tags()
    info = [a for t, a in tags if _is_glyph(t, a) and a.get("data-tip-kind") == "section"]
    check(len(info) == 1, f"want exactly one section (i) glyph, found {len(info)}")
    text = info[0].get("title", "").lower()
    for term in TERMS_OF_ART:
        check(term in text, f"section glyph does not explain {term!r}")
    headings = [a for t, a in tags if t in ("h2", "h3", "h4")]
    check(len(headings) == 1 and headings[0].get("id"), "the card has one heading with an id")
    root = [a for t, a in tags if ROOT_CLASS in _classes(a)][0]
    check(root.get("aria-labelledby") == headings[0]["id"], "root is labelled by its heading")


def test_ids_used_by_the_script_exist_in_the_fragment():
    js = _read(_JS)
    ids_block = re.search(r"IDS\s*=\s*Object\.freeze\(\{(.*?)\}\)", js, re.DOTALL)
    check(ids_block, "orchestrator_gallery.js has no frozen IDS block")
    wanted = re.findall(r"['\"](s8-gallery-[\w-]+)['\"]", ids_block.group(1))
    check(len(wanted) >= 4, f"expected at least four ids in the IDS block, found {wanted}")
    present = {a.get("id") for t, a in _tags()}
    for wid in wanted:
        check(wid in present, f"id {wid} named in the script is missing from the fragment")


def test_stylesheet_is_scoped_under_the_root_class():
    css = _read(_CSS)
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    # Drop @media wrappers so their inner rules are checked like any other.
    body = re.sub(r"@media[^{]*\{", "", body)
    selectors = re.findall(r"([^{}]+)\{", body)
    check(selectors, "no CSS rules found")
    for sel in selectors:
        for part in sel.split(","):
            part = part.strip()
            if not part or part.startswith("@"):
                continue
            check(part.startswith("." + ROOT_CLASS), f"CSS selector not scoped under .{ROOT_CLASS}: {part!r}")


def test_root_block_never_widens_the_host_panel():
    """A select contributes its longest option to the ancestor's minimum width.

    The host's .main-panel is flex:1 with the default min-width:auto, so that
    minimum sizes the panel and pushes the whole page sideways at 946 px. The
    root block contains its inline size, the same guard the Step 5 runs card
    uses, so the picker truncates inside itself instead.
    """
    body = re.sub(r"/\*.*?\*/", "", _read(_CSS), flags=re.DOTALL)
    root = re.search(r"\.vega-gallery\s*\{([^{}]*)\}", body)
    check(root, "no .vega-gallery root block in the stylesheet")
    check(CONTAINMENT_GUARD in root.group(1),
          f"the root block must declare {CONTAINMENT_GUARD!r} so a long run option "
          f"never widens the host panel; found: {root.group(1).strip()!r}")
    picker = re.search(r"\.vega-gallery \.vg-select\s*\{([^{}]*)\}", body)
    check(picker, "no .vg-select block in the stylesheet")
    check("min-width: 0" in picker.group(1),
          "the run picker keeps min-width: 0 so it shrinks inside its own row")


def test_script_never_builds_inline_handlers_or_raw_urls():
    js = _read(_JS)
    check(not re.search(r"on(click|error|load|mouseover)\s*=", js), "script builds an inline handler attribute")
    check("encodeURIComponent" in js, "script does not encode URL parts")
    check("escapeHtml" in js, "script has no HTML escaper")
    check("window.VegaGallery" in js, "script does not expose window.VegaGallery")
    check("module.exports" in js, "script is not requireable from Node")
    check("BLIND_SPOT_NO_DETECTION_FRACTION = 0.2" in js, "blind-spot threshold constant missing or changed")


def test_no_banned_prose_in_any_of_the_three_files():
    for label, text in (("fragment", _read(os.path.join(_TEMPLATES, _FRAGMENT))),
                        ("stylesheet", _read(_CSS)), ("script", _read(_JS))):
        check(EM_DASH not in text, f"{label}: em dash present")
        check(EN_DASH not in text, f"{label}: en dash present")
        check("UVI" not in text, f"{label}: the string UVI is present")
        check(not BANNED_WORD.search(text), f"{label}: the banned word is present")
        check(all(ord(c) < 128 for c in text), f"{label}: non-ASCII character (emoji or typographic mark) present")
        check("TODO" not in text and "placeholder" not in text.lower(), f"{label}: placeholder text present")


def test_verdict_and_toggle_tooltips_in_the_script_obey_the_rules():
    js = _read(_JS)
    titles = re.findall(r"^\s*(?:yes|no|skip|TOGGLE_IMAGE_TOOLTIP|OVERLAP_TOOLTIP|CODE_ONLY_TOOLTIP_PREFIX)\w*:?\s*=?\s*['\"](.+?)['\"],?\s*$", js, re.MULTILINE)
    check(len(titles) >= 4, f"expected the verdict, toggle and overlap tooltips in the script, found {len(titles)}")
    for text in titles:
        if text.endswith("."):
            _check_tooltip_text(text, "script tooltip")
    check("VEGA appends" in js and "VEGA records" in js, "verdict tooltips must say what VEGA appends or records")


if __name__ == "__main__":
    print(f"test_step8_gallery_card: fragment {os.path.join(_TEMPLATES, _FRAGMENT)}")
    for _fn in (
        test_fragment_renders_with_no_variables,
        test_root_class_and_id_exactly_once,
        test_no_inline_script_or_handlers,
        test_every_control_has_a_tooltip_on_a_focusable_glyph,
        test_heading_glyph_explains_terms_of_art_once,
        test_ids_used_by_the_script_exist_in_the_fragment,
        test_stylesheet_is_scoped_under_the_root_class,
        test_root_block_never_widens_the_host_panel,
        test_script_never_builds_inline_handlers_or_raw_urls,
        test_no_banned_prose_in_any_of_the_three_files,
        test_verdict_and_toggle_tooltips_in_the_script_obey_the_rules,
    ):
        run(_fn)
    passed = sum(1 for _, okd in _RESULTS if okd)
    print(f"\n{passed}/{len(_RESULTS)} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)
