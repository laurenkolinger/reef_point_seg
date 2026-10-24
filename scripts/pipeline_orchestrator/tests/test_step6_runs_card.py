"""
T10 of the 2026-09-04 VEGA fold-in: the Step 5 cross-project runs card.

House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_runs_card.py

The card is three new files the host page includes without editing them
here: the var-free Jinja fragment templates/_step6_runs.html, the
stylesheet static/orchestrator_runs.css (every rule scoped under the one
root class .vega-runs) and the script static/orchestrator_runs.js (an IIFE
exposing window.VegaRuns, with pure render functions the Node suite
tests/test_vega_runs.js exercises).

Covers:
  FRAGMENT  - renders through jinja2 with StrictUndefined and no variables;
              the root class and the card ids appear exactly once; no inline
              script, inline handler or inline style; every control carries a
              tooltip on a keyboard-focusable element; the section glyph is
              present once and labelled.
  PROSE     - no em or en dash, no non-ASCII (so no emoji), never the word
              "land" or its forms, never "UVI", in all three files; every
              tooltip is one or two complete sentences, never opens with
              "You", ends with a full stop and names no screen position.
  CSS       - every selector starts with .vega-runs; no rule leaks onto the
              host page.
  JS        - IIFE, strict mode, the wave-2 contract URL and the three event
              names, the plan's initStep6Runs entry point, a JSDoc block on
              every function, no dead export; the Trained on column and its
              filter, named by the route's field or the open project's own
              record and never by a folder; and the comparability apparatus
              off the card, its glyph included, while the payload keeps every
              field the route sends (Lauren, 2026-09-16).
  NODE      - the Node suite passes end to end under /usr/bin/node.
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
import subprocess
import sys
import traceback
from html.parser import HTMLParser

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_FRAGMENT = os.path.join(_ORCH, "templates", "_step6_runs.html")
_CSS = os.path.join(_ORCH, "static", "orchestrator_runs.css")
_JS = os.path.join(_ORCH, "static", "orchestrator_runs.js")
_NODE_SUITE = os.path.join(_HERE, "test_vega_runs.js")
_NODE = "/usr/bin/node"

ROOT_CLASS = "vega-runs"
CARD_ID = "s6-runs-card"
BODY_ID = "s6-runs-body"
RUNS_URL = "/api/step/6/all_runs"
EVENT_NAMES = ("vega:evaluate-run", "vega:promote-run", "vega:finetune-from")
EM_DASH = "\u2014"
EN_DASH = "\u2013"
BANNED_WORD = re.compile(r"\bland(s|ed|ing)?\b", re.IGNORECASE)
SCREEN_REFERENCE = re.compile(r"\b(above|below|this tab|this window)\b", re.IGNORECASE)
SENTENCE = re.compile(r"[^.!?]+[.!?]")
MAX_JSDOC_DISTANCE_LINES = 8

_fail = 0
_RESULTS = []


def check(cond, msg):
    """Raise AssertionError with ``msg`` when ``cond`` is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one named test, record PASS or FAIL, never stop the suite."""
    global _fail
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001 - the harness reports every failure kind
        _fail += 1
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


_CACHE = {}


def _read(path):
    """Return the text of ``path``, cached, failing loudly when it is missing."""
    if path not in _CACHE:
        check(os.path.isfile(path), f"missing file: {path}")
        with open(path, encoding="utf-8") as fh:
            _CACHE[path] = fh.read()
    return _CACHE[path]


def _rendered_fragment():
    """The fragment rendered by jinja2 with StrictUndefined and no variables."""
    if "rendered" not in _CACHE:
        import jinja2
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
        _CACHE["rendered"] = env.from_string(_read(_FRAGMENT)).render()
    return _CACHE["rendered"]


class _Tags(HTMLParser):
    """Collect every start tag with its attributes, in document order."""

    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _tags():
    parser = _Tags()
    parser.feed(_rendered_fragment())
    return parser.tags


def _titles_in(text):
    """Every title="..." value in ``text`` (HTML or JS source), unescaped for quotes."""
    found = re.findall(r'title="([^"]*)"', text)
    return [t.replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&") for t in found]


def _assert_tooltip_rules(title, where):
    """The platform tooltip sentence rules for one title string."""
    check(title.strip(), f"{where}: empty tooltip")
    check(not title.startswith("You"), f"{where}: tooltip opens with You: {title!r}")
    check(title.rstrip().endswith("."), f"{where}: tooltip does not end with a full stop: {title!r}")
    n = len(SENTENCE.findall(title))
    check(1 <= n <= 2, f"{where}: tooltip has {n} sentences, want one or two: {title!r}")
    check(not SCREEN_REFERENCE.search(title), f"{where}: tooltip references the screen: {title!r}")


def _assert_prose_clean(text, where):
    """No dashes of either kind, no non-ASCII, no banned word, no UVI."""
    check(EM_DASH not in text, f"{where}: em dash present")
    check(EN_DASH not in text, f"{where}: en dash present")
    non_ascii = sorted({ch for ch in text if ord(ch) > 127})
    check(not non_ascii, f"{where}: non-ASCII characters present (emoji or typographic marks): {non_ascii!r}")
    hit = BANNED_WORD.search(text)
    check(hit is None, f"{where}: banned word at offset {hit.start() if hit else -1}: {hit.group(0) if hit else ''!r}")
    check("UVI" not in text, f"{where}: the string UVI is present")


# ── FRAGMENT ─────────────────────────────────────────────────────────

def test_fragment_renders_with_no_variables():
    src = _read(_FRAGMENT)
    check("{{" not in src and "{%" not in src, "the fragment must be var-free: no {{ }} or {% %}")
    html = _rendered_fragment()
    check(len(html.strip()) > 0, "rendered fragment is empty")
    check("{#" not in html, "a Jinja comment leaked into the render")


def test_root_class_present_exactly_once():
    html = _rendered_fragment()
    roots = [t for t in _tags() if ROOT_CLASS in (t[1].get("class") or "").split()]
    check(len(roots) == 1, f"root class {ROOT_CLASS} appears {len(roots)} times, want exactly once")
    check(roots[0][1].get("id") == CARD_ID, f"root element id is {roots[0][1].get('id')!r}, want {CARD_ID}")
    check(html.count(f'id="{CARD_ID}"') == 1, "card id must appear once")
    check(html.count(f'id="{BODY_ID}"') == 1, "body id must appear once")
    bodies = [t for t in _tags() if t[1].get("data-role") == "body"]
    check(len(bodies) == 1 and bodies[0][1].get("id") == BODY_ID, "exactly one data-role=body element, carrying the body id")


def test_fragment_has_no_inline_script_handlers_or_styles():
    html = _rendered_fragment()
    check("<script" not in html.lower(), "the fragment must carry no inline script")
    check("javascript:" not in html.lower(), "no javascript: URL")
    for tag, attrs in _tags():
        for name in attrs:
            check(not name.startswith("on"), f"<{tag}> carries an inline handler {name}")
            check(name != "style", f"<{tag}> carries an inline style; styling lives in orchestrator_runs.css")


def test_every_control_has_a_tooltip_on_a_focusable_glyph():
    tags = _tags()
    buttons = [(t, a) for t, a in tags if t == "button"]
    check(len(buttons) >= 2, f"expected the glyph and the Refresh button, found {len(buttons)} buttons")
    for tag, attrs in buttons:
        check(attrs.get("type") == "button", f"<button> without type=button: {attrs}")
        check((attrs.get("title") or "").strip(), f"button without a tooltip: {attrs}")
        check(attrs.get("tabindex") != "-1", f"button removed from the tab order: {attrs}")
    glyphs = [(t, a) for t, a in buttons if "vega-runs-tip" in (a.get("class") or "").split()]
    check(len(glyphs) == 1, f"expected one section glyph, found {len(glyphs)}")
    check((glyphs[0][1].get("aria-label") or "").strip(), "the section glyph needs an aria-label")
    check(re.search(r'class="vega-runs-tip"[^>]*>\s*<span>i</span>', _rendered_fragment()), "the section glyph shows the letter i")
    for tag, attrs in tags:
        if "title" in attrs:
            focusable = tag == "button" or (tag == "a" and "href" in attrs) or attrs.get("tabindex") == "0"
            check(focusable, f"<{tag}> carries a title but is not keyboard-focusable: {attrs}")
    inputs = [t for t, _ in tags if t in ("input", "select", "textarea")]
    check(not inputs, f"the fragment declares form fields ({inputs}) that would each need a ? glyph")


def test_glyph_defines_the_terms_of_art_once():
    """The terms the card still uses: the champion, the two score families and
    the model lab, plus what the table is (every trained run VEGA can find,
    grouped by what kind of model it is, with the data it learned from).
    "comparable" left this list on 2026-09-16 with the comparability
    apparatus, and on 2026-09-17 the clause left the glyph too: a tooltip
    that teaches comparability on a card that no longer shows it teaches the
    wrong thing."""
    glyph_titles = [a.get("title") for t, a in _tags() if "vega-runs-tip" in (a.get("class") or "").split()]
    text = " ".join(glyph_titles)
    for term in ("champion", "mAP50", "model lab", "grouped"):
        check(term in text, f"the section glyph must define {term!r}")
    check("outline mAP50" in text and "box mAP50" in text, "the glyph names both score families")
    check("learned from" in text or "trained on" in text, "the glyph says the table names the data each run learned from")
    check("comparab" not in text.lower(), f"the glyph must teach nothing about comparability: {text!r}")
    check("held-out" not in text, "nor about the held-out images a comparison would need")


def test_no_comparability_copy_anywhere_in_the_card():
    """Lauren, 2026-09-16: comparing model runs is not what this card is for.
    No user-facing string in the three files says anything about it: not the
    fragment's tooltip, not a TEXT or TIPS sentence, not a class the
    stylesheet paints. The comments that record why it left may still use the
    word, and normaliseRun may still read the two payload fields."""
    html = _rendered_fragment().lower()
    check("comparab" not in html, "the fragment carries no comparability copy")
    src = _read(_JS)
    for name in ("TEXT", "TIPS"):
        block = re.search(r"var " + name + r" = \{(.*?)\n  \};", src, re.DOTALL)
        check(block is not None, f"orchestrator_runs.js must keep its {name} strings in one object")
        strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', block.group(1))
        for text in strings:
            check("comparab" not in text.lower(), f"{name} carries comparability copy: {text!r}")
    css = re.sub(r"/\*.*?\*/", "", _read(_CSS), flags=re.DOTALL)
    check("comparab" not in css.lower() and "is-attention" not in css, "the stylesheet paints no comparability state")
    check("is-attention" not in src, "and the script names no attention chip, in code or in a docstring")


def test_root_is_a_labelled_region_and_aria_live_sits_on_the_summary():
    tags = _tags()
    root = next(a for t, a in tags if a.get("id") == CARD_ID)
    check(root.get("role") == "region", "the card root must be role=region")
    check(root.get("aria-labelledby") == "s6-runs-title", "the region is labelled by the card heading")
    body = next(a for t, a in tags if a.get("id") == BODY_ID)
    check("aria-live" not in body, "aria-live belongs on the summary line the script renders, not on the whole body")
    check("aria-labelledby" not in body, "the table, not the body, carries aria-labelledby")
    src = _read(_JS)
    check('class="vega-runs-summary" aria-live="polite"' in src, "the summary line is rendered as the live region")
    check('TITLE_ID = "s6-runs-title"' in src
          and '<table class="vega-runs-table" aria-labelledby="\' + TITLE_ID + \'" role="table">' in src,
          "the table is labelled by the heading id (held in TITLE_ID) and declares role=table, "
          "which the stacked narrow layout would otherwise strip")
    check(src.count("aria-live") == 1, "aria-live appears once in the script")


# ── PROSE ────────────────────────────────────────────────────────────

def test_no_banned_prose_in_the_three_files():
    for path in (_FRAGMENT, _CSS, _JS):
        _assert_prose_clean(_read(path), os.path.basename(path))


def test_fragment_tooltips_follow_the_sentence_rules():
    titles = _titles_in(_rendered_fragment())
    check(len(titles) >= 2, f"expected at least two tooltips in the fragment, found {len(titles)}")
    for title in titles:
        _assert_tooltip_rules(title, "fragment")


def test_js_tooltip_strings_follow_the_sentence_rules():
    src = _read(_JS)
    block = re.search(r"var TIPS = \{(.*?)\n  \};", src, re.DOTALL)
    check(block is not None, "orchestrator_runs.js must keep its tooltip strings in one TIPS object")
    strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', block.group(1))
    tips = [s.replace('\\"', '"') for s in strings if len(s) > 20]
    check(len(tips) >= 8, f"expected at least eight tooltip strings in TIPS, found {len(tips)}")
    for tip in tips:
        _assert_tooltip_rules(tip, "TIPS")
    check(all("best.pt" not in t for t in tips), "a file name with a dot inside a tooltip breaks the sentence count; say weights file")


def test_no_filler_copy_in_the_fragment():
    html = _rendered_fragment()
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split(" ") if w]
    check(len(words) <= 12, f"the fragment's visible copy is {len(words)} words; labels only, explanations live in tooltips")
    check("All trained models" in html, "the card heading reads All trained models")


# ── CSS ──────────────────────────────────────────────────────────────

def _selectors(css):
    """Every selector list in ``css`` outside comments, @media bodies included."""
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    body = re.sub(r"@media[^{]*\{", "", body)
    body = re.sub(r"@container[^{]*\{", "", body)
    body = re.sub(r"@keyframes[^{]*\{[^}]*\}[^}]*\}", "", body)
    out = []
    for block in re.findall(r"([^{}]+)\{", body):
        for sel in block.split(","):
            sel = sel.strip()
            if sel:
                out.append(sel)
    return out


def _container_block(css):
    """The body of the stylesheet's @container at-rule, brace-matched.

    Parameters:
        css: the stylesheet text.
    Returns:
        the declarations inside the at-rule, or '' when it is absent. Used to
        prove a width rule follows the card's own width rather than the
        viewport's.
    """
    start = css.find("@container")
    if start < 0:
        return ""
    open_brace = css.find("{", start)
    if open_brace < 0:
        return ""
    depth = 0
    for i in range(open_brace, len(css)):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace + 1:i]
    return ""


def test_css_is_scoped_under_the_root_class():
    css = _read(_CSS)
    selectors = _selectors(css)
    check(len(selectors) >= 10, f"expected a real stylesheet, found {len(selectors)} selectors")
    for sel in selectors:
        check(sel.startswith("." + ROOT_CLASS), f"selector leaks outside the root class: {sel!r}")
    check("!important" not in css, "no !important overrides of the host stylesheet")
    for token in ("--surface-light", "--text-dim", "--border", "--pink", "--warning", "--error", "--font-mono"):
        check(f"var({token}," in css, f"stylesheet should read the host token {token}")
    check("focus-visible" in css, "keyboard focus ring rule missing")
    check(".vega-runs-score[tabindex]:focus-visible" in css, "the focusable score value needs the focus ring too")
    check("table-layout: fixed" in css, "the table uses fixed layout so the name column absorbs the squeeze")
    check("overflow-x: auto" in css or "overflow-x:auto" in css, "the table must scroll inside its own wrapper")


def test_css_narrow_card_rules_are_scoped_and_named():
    """The 2026-09-15 pixel review: at 946 px the table was cut at the card's
    right edge, losing the Actions column. The fix stacks each record below the
    width the three columns need, keyed off the card's own width."""
    css = _read(_CSS)
    at = re.search(r"@container\s+vega-runs\s*\(max-width:\s*(\d+)px\)\s*\{", css)
    check(at is not None, "the stylesheet must declare a named @container rule for the narrow card")
    wrap = re.search(r"\.vega-runs-wrap\s*\{([^}]*)\}", css)
    check(wrap is not None, "the stylesheet must declare .vega-runs-wrap")
    decls = wrap.group(1)
    check("container-type: inline-size" in decls,
          "the wrapper must be an inline-size query container so the layout follows the card, not the viewport")
    check("container-name: vega-runs" in decls,
          "the query container must be named so no unrelated ancestor answers the query")
    check("contain: inline-size" in decls,
          "inline-size containment stays: it is what keeps the host page from scrolling sideways")
    table = re.search(r"\.vega-runs-table\s*\{([^}]*)\}", css)
    floor = re.search(r"min-width:\s*(\d+)px", table.group(1))
    check(floor is not None, "the wide table must still declare the width its three columns need")
    check(int(at.group(1)) == int(floor.group(1)) - 1,
          f"the stack must begin one pixel below the table floor: query {at.group(1)}, floor {floor.group(1)}")
    check("@media" not in css,
          "every width rule in this card keys off the card's own width: the VEGA window is resized "
          "inside the VICARIUS desktop, so a viewport query measures a different number")
    container = _container_block(css)
    check(re.search(r"\.vega-runs-trained\s*\{[^}]*max-width", container) is not None,
          "the training data name's cap belongs in the container block with the other width rules")
    check("td:nth-child(5)" in container,
          "the stacked record must place all five cells, the actions last")


def test_css_reads_every_host_token_through_a_fallback():
    """The 2026-09-16 pixel review: on the running orchestrator :root carried
    none of the design's names, so every declaration reading one was dropped
    and the card rendered with no panel, no hairlines, no state colours and no
    magenta edge. Each token is now read once, at the top of the root rule,
    with the design's own value as its fallback: the host still wins wherever
    it defines the name, and the card paints itself where it does not."""
    css = _read(_CSS)
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    reads = sorted(set(re.findall(r"var\((--[a-z-]+?)(?=[,)])", body)))
    host = [t for t in reads if not t.startswith("--vr-")]
    check(host, "the stylesheet must still read the host tokens")
    for token in host:
        pattern = r"var\(" + re.escape(token) + r",\s*[^)]+\)"
        check(re.search(pattern, body) is not None,
              f"the host token {token} is read with no fallback, so the card loses it "
              f"on any page whose :root does not carry that name")
        check(len(re.findall(r"var\(" + re.escape(token) + r"[,)]", body)) == 1,
              f"the host token {token} is read in more than one place; every read goes "
              f"through the one alias layer at the top of the root rule")
    aliases = sorted(set(re.findall(r"var\((--vr-[a-z-]+?)(?=[,)])", body)))
    check(len(aliases) >= 10, f"expected the card to paint through its alias layer, found {aliases}")
    for alias in aliases:
        check(re.search(re.escape(alias) + r":\s*var\(", body) is not None,
              f"{alias} is used but never declared from a host token")


def test_css_has_no_hardcoded_colours():
    body = re.sub(r"/\*.*?\*/", "", _read(_CSS), flags=re.DOTALL)
    hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", body)
    check(not hexes, f"hardcoded colours instead of tokens or rgba washes: {hexes}")


# ── JS ───────────────────────────────────────────────────────────────

def test_js_is_an_iife_exposing_vega_runs():
    src = _read(_JS)
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL).lstrip()
    check(code.startswith("(function () {"), "orchestrator_runs.js must open with an IIFE after its header comment")
    check(code.rstrip().endswith("})();") or code.rstrip().endswith("}());"), "the IIFE must close and invoke itself")
    check('"use strict";' in src, "strict mode")
    check("window.VegaRuns = VegaRuns" in src, "window.VegaRuns is exposed for the host page")
    check("module.exports = VegaRuns" in src, "module.exports is set so Node can test the pure functions")
    check(f'RUNS_URL = "{RUNS_URL}"' in src, f"the contract URL {RUNS_URL} is a named constant")
    for name in EVENT_NAMES:
        check(f'"{name}"' in src, f"event name {name} present")
    check("function initStep6Runs" in src and "window.initStep6Runs = " in src, "the plan's initStep6Runs entry point is exposed")
    check(f'CARD_ID = "{CARD_ID}"' in src and "getElementById(CARD_ID)" in src,
          "initStep6Runs finds the card by its contract id, held in the named constant CARD_ID")
    check('data-role="body"' in src, "the script targets the fragment's data-role=body element")
    check("var TIPS" in src and "var TEXT" in src, "copy strings live in named constant objects")
    check("innerHTML" in src, "the body is filled with the rendered string")
    check("fetchJson" in src, "the fetch is injectable")


def test_js_carries_the_imported_run_contract():
    """A run copied into a lab from a training run made elsewhere must say so.
    Each imported run's provenance.json carries imported_from and imported_at,
    and the row's origin_kind says the run was copied in; the card reads all
    three off the all_runs row."""
    src = _read(_JS)
    check('ORIGIN_KIND_IMPORTED = "imported"' in src, "the imported origin kind is a named constant")
    for field in ("origin_kind:", "imported_from:", "imported_at:"):
        check(field in src, f"normaliseRun must carry {field.rstrip(':')} through to the renderers")
    check("function importedTip" in src, "the imported tooltip is built by its own documented function")
    check("is-imported" in src, "the imported marker has its own chip class")
    check("is-imported" in _read(_CSS), "the imported chip is styled in the card stylesheet")


def test_js_states_that_the_imported_fact_is_per_run():
    """lab.json carries origin_kind once for the whole lab, so a run trained
    into an imported lab afterwards would inherit a claim its own record does
    not make. The card's chip says VEGA did not train this run, so the source
    of that claim is written down: the run's own provenance.json, or the run's
    name in the lab's imported_runs list."""
    src = _read(_JS)
    flowed = re.sub(r"\s+", " ", src)
    check("provenance.json" in src, "the card names the per-run file the imported fact comes from")
    check("imported_runs" in src, "the card names the lab's imported_runs list as the other per-run source")
    check("never from the lab's flag alone" in flowed,
          "the card's comment must forbid deriving a per-run claim from the lab-level flag")


def test_js_gives_a_disabled_action_one_owner_for_its_reason():
    """A disabled button dispatches no pointer events, so the reason rides on a
    focusable wrapper. Measured on the live page on 2026-09-15: the hit-tested
    element at the centre of a disabled Promote is the button itself, so a title
    left on the button would be the one a tooltip lookup resolves and the
    wrapper's would never be read. The disabled button carries none."""
    src = _read(_JS)
    check('vega-runs-action" tabindex="0"' in src, "a disabled action is wrapped in a focusable element carrying its reason")
    check(".vega-runs-action:focus-visible" in _read(_CSS) or ".vega-runs-action" in _read(_CSS),
          "the wrapper is styled and shows a focus ring")
    check('"no ledger row"' not in src,
          "the opaque phrase no ledger row must not survive as a copy string; "
          "the comment explaining the ledger may still use the words")
    check("score it again to make it the best so far" in src, "the meta line names the action that lifts the block")
    check('"evaluate to enable Promote"' not in src,
          "the instruction must not read as work never done: the stamp beside it says the run was evaluated")
    body = re.search(r"function actionButton\(.*?\n  \}", src, re.DOTALL)
    check(body is not None, "actionButton must still build the row buttons")
    check('var title = disabled ? ""' in body.group(0),
          "a disabled button must emit no title of its own; the wrapper is the one element that owns the reason "
          "(the Node suite proves it on the rendered markup)")


def test_js_carries_the_training_data_column_and_filter():
    """Lauren, 2026-09-16: "maybe just make a table that includes the training
    data subset (so we can filter by those) ... in some applications simple is
    better." The card names the data subset each run trained on in its own
    column and filters by it. The name is the route's own trained_on field,
    or the open project's own training_source_name from the state route for
    the runs that project trained; it is never the folder that holds the run,
    and a record naming none reads "not recorded" rather than a guess."""
    src = _read(_JS)
    css = _read(_CSS)
    check("COLUMN_COUNT = 5" in src, "the record spans five columns, the fifth being Trained on")
    check('headTrained: "Trained on"' in src and 'labelTrained: "Trained on"' in src,
          "the column heading and the filter label are the same words")
    check('notRecorded: "not recorded"' in src,
          "a run whose record names no training data says so, rather than showing a blank")
    check('TRAINED_ID = "s6-runs-trained"' in src, "the filter control has its contract id")
    check('STORE_TRAINED = STORE_PREFIX + "trainedOn"' in src,
          "and the choice is remembered under the one storage prefix the registry spec asks for")
    for name in ("function trainedOn", "function trainedValue", "function trainedLabel",
                 "function trainedTip", "function renderTrainedCell", "function compareTrained",
                 "function projectSourceOf"):
        check(name in src, f"{name} must build the column")
    for gone in ("function trainedDisplayName", "PROJECT_STAMP_PATTERN", "dataset_for_run"):
        check(gone not in src, f"the folder-name fallback must be gone from the card: {gone}")
    for field in ("trained_on:", "trained_on_kind:"):
        check(field in src, f"normaliseRun must carry {field.rstrip(':')} through to the renderers")
    for field in ("training_source_name", "training_source_label"):
        check(field in src, f"the card reads {field} from the state route for the open project's own runs")
    for kind in ('TRAINED_KIND_BBOX = "bbox"', 'TRAINED_KIND_COMPOSITE = "composite"', 'TRAINED_KIND_EXPORT = "export"'):
        check(kind in src, f"the three kinds of training data are named constants: {kind}")
    for tip in ("trainedOnBoxDataset:", "trainedOnTrainingSet:", "trainedOnExport:",
                "trainedOnRecorded:", "trainedOnUnknown:", "trainedFilter:"):
        check(tip in src, f"TIPS must carry {tip.rstrip(':')}")
    check(".vega-runs-col-trained" in css and ".vega-runs-trained" in css,
          "the column and its cell are styled in the card stylesheet")
    check(re.search(r"\.vega-runs-trained\s*\{[^}]*font-size:\s*13px", css) is not None,
          "and the value reads at the 13 px the legibility floor sets for a table cell")
    width = re.search(r"\.vega-runs-col-trained\s*\{\s*width:\s*(\d+)px", css)
    check(width is not None and int(width.group(1)) >= 200,
          "the column is wide enough for a data set name of ordinary length to show whole")


def test_the_comparability_apparatus_is_off_the_card():
    """Lauren ruled on 2026-09-16 that comparing model runs is not what this
    card is for. Nothing about comparability is drawn any more: no state chip,
    no summary count, no dimmed score, no tooltip. The route is untouched, so
    the card still reads both fields the payload sends."""
    src = _read(_JS)
    css = _read(_CSS)
    for gone in ('"not comparable"', "familyMismatch", "splitUnknown", "splitMismatch",
                 "comparabilityOther", "comparabilityReason", "comparabilityShort",
                 "attentionKind", "STATE_ATTENTION", "notRanked"):
        check(gone not in src, f"the comparability apparatus must be gone from the card: {gone}")
    check("is-attention" not in css, "and so must the ink that painted it")
    check("is-dim" not in css, "a score is never dimmed for a comparison the card no longer makes")
    check("comparability_reason: str(r.comparability_reason)" in src,
          "while normaliseRun still reads the route's own reason, so the payload is untouched")
    check("comparable_to_champion" in src, "and its verdict")


def test_every_js_function_has_a_docstring():
    lines = _read(_JS).splitlines()
    missing = []
    for i, line in enumerate(lines):
        if not re.match(r"\s*function \w+\(", line):
            continue
        window = lines[max(0, i - MAX_JSDOC_DISTANCE_LINES):i]
        if not any("*/" in w for w in window):
            missing.append(line.strip())
    check(not missing, f"functions without a JSDoc block: {missing}")


def test_js_exports_match_the_fragment_contract():
    src = _read(_JS)
    exported = re.search(r"var VegaRuns = \{(.*?)\n  \};", src, re.DOTALL)
    check(exported is not None, "the export object VegaRuns = {...} must be present")
    names = set(re.findall(r"(\w+):", exported.group(1))) | set(re.findall(r"^\s*(\w+),?$", exported.group(1), re.MULTILINE))
    for name in ("render", "load", "init", "handleClick", "dispatchAction", "formatAst", "esc"):
        check(name in names, f"VegaRuns must export {name}")
    check('ACTION_REFRESH = "refresh"' in src and 'data-action="refresh"' in _rendered_fragment(),
          "the fragment's Refresh button and the script agree on data-action=refresh")


# ── NODE ─────────────────────────────────────────────────────────────

def test_node_suite_passes():
    check(os.path.isfile(_NODE_SUITE), f"missing Node suite {_NODE_SUITE}")
    check(os.path.isfile(_NODE), f"node not found at {_NODE}")
    proc = subprocess.run([_NODE, _NODE_SUITE], capture_output=True, text=True, timeout=120, check=False)
    tail = (proc.stdout + proc.stderr)[-2000:]
    check(proc.returncode == 0, f"Node suite failed (exit {proc.returncode}):\n{tail}")
    check(", 0 failed" in proc.stdout, f"Node suite tally does not read 0 failed:\n{tail}")


if __name__ == "__main__":
    print("FRAGMENT:")
    run(test_fragment_renders_with_no_variables)
    run(test_root_class_present_exactly_once)
    run(test_fragment_has_no_inline_script_handlers_or_styles)
    run(test_every_control_has_a_tooltip_on_a_focusable_glyph)
    run(test_glyph_defines_the_terms_of_art_once)
    run(test_root_is_a_labelled_region_and_aria_live_sits_on_the_summary)
    print("PROSE:")
    run(test_no_banned_prose_in_the_three_files)
    run(test_fragment_tooltips_follow_the_sentence_rules)
    run(test_js_tooltip_strings_follow_the_sentence_rules)
    run(test_no_filler_copy_in_the_fragment)
    print("CSS:")
    run(test_css_is_scoped_under_the_root_class)
    run(test_css_narrow_card_rules_are_scoped_and_named)
    run(test_css_reads_every_host_token_through_a_fallback)
    run(test_css_has_no_hardcoded_colours)
    print("JS:")
    run(test_js_is_an_iife_exposing_vega_runs)
    run(test_no_comparability_copy_anywhere_in_the_card)
    run(test_js_carries_the_imported_run_contract)
    run(test_js_states_that_the_imported_fact_is_per_run)
    run(test_js_gives_a_disabled_action_one_owner_for_its_reason)
    run(test_js_carries_the_training_data_column_and_filter)
    run(test_the_comparability_apparatus_is_off_the_card)
    run(test_every_js_function_has_a_docstring)
    run(test_js_exports_match_the_fragment_contract)
    print("NODE:")
    run(test_node_suite_passes)

    print(f"\n{len(_RESULTS) - _fail}/{len(_RESULTS)} passed.")
    if _fail:
        print("\nFAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"--- {name} ---\n{detail}")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
