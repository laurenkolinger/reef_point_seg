# CA/tests/test_annotator_design.py
"""The annotator window obeys the VICARIUS workflow-page design language.

Task T-annotator of the VEGA student-ready sweep, 2026-09-16 (LO). The four copy
defects are held by test_annotator_copy.py; this suite holds the design pass that
followed, against version 2 of
docs/superpowers/specs/2026-09-02-workflow-page-design.md:

  1. The legibility floor. No rendered text below 12 px anywhere on any screen,
     no explanatory text below 13 px, no hint prose on a hardcoded hex grey, and
     no text faded below full opacity.
  2. Rows. The mask list and the exemplar-candidate list lead with the full name
     the code resolves to, carry the code beside it as a chip, put the facts on a
     meta line, stripe every other row and wash a row on hover.
  3. Chrome. Section headings are Syne sentence case with an "i" glyph; uppercase
     is left to the window title and the buttons.
  4. No paragraph on screen, and every field label carries a "?" glyph.
  5. The window says who built it, on every screen it can show.

It reads the rendered page, not the source alone, so a rule written in CSS but
beaten by a later rule is still caught. Rendering only: no SAM3, no GPU, no port.

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_annotator_design.py
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

from jinja2 import Environment, FileSystemLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_TPL_DIR = os.path.join(_SRC, "templates")

# The type scale's floors, from spec section 3 ("Legibility floor").
MIN_TEXT_PX = 12
MIN_EXPLANATORY_PX = 13
# Buttons are the one element the scale puts below the floor (11 px, mini 10 px).
BUTTON_MIN_PX = 10

FAILURES = []


def check(cond, name):
    """Record one assertion and print its verdict."""
    print("  %s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILURES.append(name)


def _template_source():
    """The template exactly as it sits on disk."""
    with open(os.path.join(_TPL_DIR, "index.html")) as handle:
        return handle.read()


def _render(session_mode="configure", orchestrated=False):
    """Render the page with a stub context, the way app.py's index route does."""
    labels = [
        {"code": "PA", "name": "Porites astreoides", "category": "Coral"},
        {"code": "BALL", "name": "Ball sponge", "category": "Sponge"},
        {"code": "ZZZ", "name": "", "category": ""},
    ]
    env = Environment(loader=FileSystemLoader(_TPL_DIR))
    return env.get_template("index.html").render(
        default_input="/tmp/in", default_export="/tmp/inprocess/run_demo/4test",
        default_categories=["Target species only"],
        all_categories=["Target species only", "Coral"],
        target_species=[lbl["code"] for lbl in labels], target_labels=labels,
        default_review_dir="/tmp/review", default_contacts="a@b.c",
        target_species_only=True, reference_default=True,
        default_confidence=0.4, default_min_area=100, default_merge_dist=20,
        default_overlap="larger_wins", default_mask_size="generous",
        default_thin_ratio=0.12, default_simplify=1.5, default_exemplar=0.5,
        default_dev_tracker="cuda", default_dev_exemplar="cuda",
        orchestrator_url="http://127.0.0.1:5050", orchestrated=orchestrated,
        session_mode=session_mode, review_locked_msg="locked",
        review_tagging_enabled=False, default_session_name="26-09-16_LO_demo",
    )


def _style_block(page):
    """The page's own stylesheet, without the markup around it."""
    start = page.index("<style>") + len("<style>")
    return page[start:page.index("</style>", start)]


def _declarations(css, prop):
    """Every (selector, value) pair in the stylesheet for one property."""
    out = []
    for rule in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = rule.group(1).strip(), rule.group(2)
        for hit in re.finditer(r"(?<![a-z-])%s:\s*([^;]+)" % prop, body):
            out.append((selector, hit.group(1).strip()))
    return out


# ── 1. The legibility floor ──────────────────────────────────────────────────

def test_no_rule_sets_text_below_the_floor():
    """Every font-size in the stylesheet clears 12 px, buttons excepted."""
    css = _style_block(_render())
    button_selectors = ("btn", "kbd", "qa-key", "mode-btn", "orch-retry")
    offenders = []
    for selector, value in _declarations(css, "font-size"):
        match = re.match(r"^(\d+(?:\.\d+)?)px$", value)
        if not match:
            continue
        size = float(match.group(1))
        floor = (BUTTON_MIN_PX if any(tok in selector for tok in button_selectors)
                 else MIN_TEXT_PX)
        if size < floor:
            offenders.append("%s -> %spx" % (selector.replace("\n", " "), value))
    check(not offenders, "no stylesheet rule sets text below the floor: %s"
          % (offenders or "none"))


def test_no_inline_style_sets_text_below_the_floor():
    """The inline styles in the markup clear the floor too."""
    page = _render()
    markup = page[:page.index("<script>")]
    offenders = [m for m in re.findall(r"font-size:\s*(\d+)px", markup)
                 if int(m) < MIN_TEXT_PX]
    check(not offenders,
          "no inline style sets text below %dpx: %s" % (MIN_TEXT_PX, offenders))


def test_hint_prose_is_never_a_hardcoded_grey():
    """Explanatory prose uses --text-dim, which holds the contrast floor."""
    css = _style_block(_render())
    greys = []
    for selector, value in _declarations(css, "color"):
        hit = re.match(r"^#([0-9a-fA-F]{6})$", value)
        if not hit:
            continue
        r, g, b = (int(hit.group(1)[i:i + 2], 16) for i in (0, 2, 4))
        # A text grey: the three channels within 24 of each other, and in the
        # range ink is actually set in. A near-black sitting on a coloured fill
        # (the amber REVIEW chip's #1a1205) is a foreground, not hint prose.
        if (max(r, g, b) - min(r, g, b) <= 24
                and 0x40 <= max(r, g, b) < 0xF0):
            greys.append("%s -> %s" % (selector.replace("\n", " "), value))
    check(not greys, "no hardcoded grey ink is left: %s" % (greys or "none"))


def test_nothing_readable_is_faded():
    """No rule fades text below full opacity, which would undo the floor."""
    css = _style_block(_render())
    # A keyframe stop is a selector shaped "50%"; a disabled button is meant
     # to look unavailable. Neither is readable prose being faded.
    faded = [(s, v) for s, v in _declarations(css, "opacity")
             if re.match(r"^0?\.\d+$", v) and "::" not in s
             and "body" not in s and "indicator" not in s
             and not s.rstrip().endswith("%") and ":disabled" not in s]
    check(not faded, "no control fades its own text: %s" % (faded or "none"))


# ── 2. The rows ──────────────────────────────────────────────────────────────

def test_a_row_leads_with_the_name_and_carries_the_code_as_a_chip():
    """Both lists put the resolved name before the code in the row markup."""
    src = _template_source()
    for which, anchor in (("mask list", "onclick=\"selectMaskFromList"),
                          ("candidate list", "onclick=\"selectScanCandidate")):
        row = src[src.index(anchor):]
        row = row[:row.index("</li>")]
        name_at = row.index("${labelNameHtml(m.species, m.name)}")
        code_at = row.index('class="m-species"')
        meta_at = row.index('class="m-meta"')
        check(name_at < code_at,
              "%s: the name comes before the code" % which)
        check(code_at < meta_at,
              "%s: the facts come after the code, on the meta line" % which)


def test_the_code_chip_looks_like_a_chip():
    """The code carries the ring and the mono 12 px the design gives a chip."""
    css = _style_block(_render())
    rule = css[css.index(".mask-list li .m-species {"):]
    rule = rule[:rule.index("}")]
    check("font-size: 12px" in rule, "the code chip is 12 px")
    check("Space Mono" in rule, "the code chip is set in Space Mono")
    check("border: 1px solid var(--border-dim)" in rule, "the code chip has a ring")
    check("var(--text-dim)" in rule, "the code chip is secondary ink")


def test_a_long_list_stripes_and_hovers():
    """A list this long carries an alternate row wash and a hover wash."""
    css = _style_block(_render())
    check(".mask-list li:nth-child(even)" in css, "the mask list stripes")
    check(".mask-list li:hover" in css, "and washes a row on hover")
    row = css[css.index(".mask-list li {"):]
    row = row[:row.index("}")]
    check("min-height: 46px" in row, "a row keeps the 46 px definition-row height")
    check("border-bottom: 1px solid var(--border-dim)" in row,
          "rows are separated by hairlines, not by gaps")


def test_an_unlabelled_row_says_so_in_words():
    """A mask with no label reads as a state, not as a punctuation mark."""
    src = _template_source()
    check(src.count('class="m-unlabeled">no label yet<') == 2,
          "both lists word the unlabelled state")
    check("m.species || '?'" not in src,
          "the bare question mark that stood in for a code is gone")


# ── 3. Chrome ────────────────────────────────────────────────────────────────

def test_section_headings_are_syne_sentence_case_with_a_glyph():
    """Each section names itself in Syne and explains itself on an "i" glyph."""
    page = _render()
    css = _style_block(page)
    for selector, floor in ((".panel-header h3 {", 15),
                            (".panel-sub summary h3 {", 13),
                            ("#kbd-help .kh-group h3 {", 13)):
        rule = css[css.index(selector):]
        rule = rule[:rule.index("}")]
        size = float(re.search(r"font-size:\s*(\d+)px", rule).group(1))
        check("Syne" in rule, "%s is set in Syne" % selector.strip(" {"))
        check(size >= floor, "%s is at least %dpx, got %s"
              % (selector.strip(" {"), floor, size))
        check("text-transform: uppercase" not in rule,
              "%s is sentence case" % selector.strip(" {"))
    for glyph_id in ("masks-help", "exemplar-help", "quickadd-help"):
        check('id="%s"' % glyph_id in page, "the %s glyph is present" % glyph_id)


def test_uppercase_is_left_to_the_title_and_the_buttons():
    """No label, kicker or status line shouts.

    kh-head h2 and overlay-card h2 dropped from the allow-list 2026-09-27
    (pixel review): both headings' own text ("Keyboard", "End of batch") was
    already written in sentence case, so the CSS transform was the only thing
    shouting it; removing the transform is the fix, not a new exemption.
    """
    css = _style_block(_render())
    allowed = ("h1", "btn", "kbd", "mode-indicator", "reviewer-field input",
               "setup-card h1", "setup-card h2", "progress-wrap h2")
    shouting = [s.replace("\n", " ") for s, v in _declarations(css, "text-transform")
                if v == "uppercase" and not any(tok in s for tok in allowed)]
    check(not shouting, "nothing but the title and the buttons is uppercase: %s"
          % (shouting or "none"))


# ── 4. No paragraph, and a glyph on every field ──────────────────────────────

def test_no_explanation_is_left_on_screen():
    """The setup form explains itself in tooltips, not in a paragraph."""
    page = _render()
    markup = page[:page.index("<script>")]
    check("Random per-session" not in markup,
          "the shuffle paragraph moved off the screen")
    check("still tracked on disk" not in markup,
          "and none of its sentences were left behind")
    card = markup[markup.index('id="startup"'):]
    labels = re.findall(r"<label[^>]*>([^<]{3,40})</label>", card)
    check(len(labels) >= 5, "the setup card still has its labels, found %d"
          % len(labels))


def test_a_dead_control_never_promises_an_override():
    """The SAM3 panel shows what is in force and invites no edit that goes nowhere.

    /api/configure (src/app.py) reads input_dir, export_dir, categories,
    target_species_only, review_batch_size, shuffle, review_dir and contacts. The
    ten SAM3 fields are read by nothing, so the panel is read-only and says so.
    """
    page = _render()
    block = page[page.index('id="adv-sam3-details"'):]
    block = block[:block.index("</details>")]
    check("override here" not in block,
          "the panel no longer promises an override it cannot send")
    inputs = re.findall(r"<input[^>]*id=\"inp-[a-z-]+\"[^>]*>", block)
    selects = re.findall(r"<select[^>]*id=\"inp-[a-z-]+\"[^>]*>", block)
    check(len(inputs) == 8 and all("readonly" in tag for tag in inputs),
          "its eight number and text fields are read-only, found %d" % len(inputs))
    check(len(selects) == 2 and all("disabled" in tag for tag in selects),
          "and its two selects are disabled, found %d" % len(selects))
    check('id="adv-sam3-help"' in block, "a glyph says where the values come from")
    with open(os.path.join(_SRC, "app.py")) as handle:
        app = handle.read()
    configure = app[app.index("def configure("):]
    configure = configure[:configure.index("\ndef ")]
    # What the route takes OFF the request is the whole contract; the same names
    # appear later as cfg values it logs, which is exactly the point.
    read_from_request = set(re.findall(r"data\.get\('([a-z_]+)'", configure))
    for field in ("confidence_threshold", "min_mask_area", "merge_distance",
                  "thin_mask_ratio", "overlap_strategy", "mask_size",
                  "polygon_simplify_epsilon", "exemplar_threshold",
                  "sam3_device_tracker", "sam3_device_exemplar"):
        check(field not in read_from_request,
              "/api/configure reads no %s from the request, so read-only is honest"
              % field)


# ── 5. Who built it ──────────────────────────────────────────────────────────

def test_every_screen_says_who_built_it():
    """Lauren Olinger is named, with a mailto, on each card and on the panel."""
    for mode, expected in (("configure", 2), ("standalone", 2)):
        page = _render(session_mode=mode)
        markup = page[:page.index("<script>")]
        check(markup.count('class="built-by"') == expected,
              "%s mode carries %d attribution blocks, found %d"
              % (mode, expected, markup.count('class="built-by"')))
        check(markup.count("mailto:lauren.olinger@uvi.edu") >= expected,
              "%s mode links the address on each" % mode)
        check("UVI" not in re.sub(r"lauren\.olinger@uvi\.edu", "", markup),
              "%s mode never prints UVI as an identity" % mode)


# ── The relic sweep ──────────────────────────────────────────────────────────

def test_no_style_is_left_for_a_chip_nothing_renders():
    """The expert/pending review chips were CSS with no emitter; they are gone."""
    src = _template_source()
    for dead in ("m-rev", "rev-chip", "rev-legend"):
        check(dead not in src, "the dead %s rule is gone" % dead)


def test_every_handler_the_markup_names_exists():
    """No control calls a function this page does not define."""
    src = _template_source()
    markup = src[src.index("</head>"):src.index("<script>")]
    calls = sorted(set(re.findall(
        r'on(?:click|change|input|keydown)="([A-Za-z_$][A-Za-z0-9_$]*)\(', markup)))
    defined = set(re.findall(
        r"(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)", src))
    defined |= set(re.findall(
        r"(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(", src))
    missing = [call for call in calls if call not in defined]
    check(calls, "the markup wires handlers at all, found %d" % len(calls))
    check(not missing, "every handler resolves: %s" % (missing or "none"))


# ── The page still parses in both modes ──────────────────────────────────────

def test_the_page_renders_in_both_modes():
    """A project window and a standalone window both render without error."""
    for mode in ("configure", "standalone", "resume"):
        page = _render(session_mode=mode)
        check(len(page) > 100000, "%s mode renders a full page (%d bytes)"
              % (mode, len(page)))
        check("{{" not in page and "{%" not in page,
              "%s mode leaves no unrendered Jinja" % mode)


# ── Hostile and empty label data ─────────────────────────────────────────────

def test_a_hostile_label_never_reaches_the_page_as_markup_or_code():
    """A label code and name are operator data, so both are escaped both ways.

    The standalone card lets a student type a label code by hand, and the master
    codes CSV is an editable file, so neither is trusted. Flask autoescapes the
    markup; the page script reads both through tojson, which also fixes the older
    bug where a code holding an apostrophe arrived as "&#39;" and stopped
    matching its own masks.
    """
    from flask import Flask, render_template
    app = Flask(__name__, template_folder=_TPL_DIR)
    base = dict(
        default_input="", default_export="", default_categories=[],
        all_categories=[], default_review_dir="", default_contacts="",
        target_species_only=True, reference_default=False,
        default_confidence=0, default_min_area=0, default_merge_dist=0,
        default_overlap="", default_mask_size="", default_thin_ratio=0,
        default_simplify=0, default_exemplar=0, default_dev_tracker="",
        default_dev_exemplar="", orchestrator_url="", orchestrated=True,
        session_mode="configure", review_locked_msg="",
        review_tagging_enabled=False, default_session_name="")
    hostile = [
        {"code": "A'-alert(1)-'B", "name": "<b>Evil</b> name", "category": ""},
        {"code": "</script><img src=x onerror=alert(1)>", "name": "", "category": ""},
    ]
    with app.app_context():
        page = render_template("index.html", target_species=[h["code"] for h in hostile],
                               target_labels=hostile, **base)
    check("<b>Evil</b>" not in page, "a hostile name never reaches the markup raw")
    check("<img src=x onerror" not in page,
          "a hostile code never reaches the markup raw")
    line = page[page.index("const TARGET_LABELS"):]
    line = line[:line.index("];") + 2]
    check("&#39;" not in line,
          "a code with an apostrophe is not left HTML-escaped inside the script")
    check("\\u0027" in line, "it is JavaScript-escaped instead, got %r" % line[:120])
    check("</script>" not in line, "no code can close the script element")


def test_the_page_survives_an_empty_and_an_enormous_label_list():
    """No labels at all, and sixty of them, both render."""
    for count in (0, 60):
        labels = [{"code": "C%02d" % i, "name": "Genus species%02d" % i,
                   "category": "Coral"} for i in range(count)]
        env = Environment(loader=FileSystemLoader(_TPL_DIR))
        page = env.get_template("index.html").render(
            default_input="", default_export="", default_categories=[],
            all_categories=[], target_species=[lbl["code"] for lbl in labels],
            target_labels=labels, default_review_dir="", default_contacts="",
            target_species_only=True, reference_default=False,
            default_confidence=0, default_min_area=0, default_merge_dist=0,
            default_overlap="", default_mask_size="", default_thin_ratio=0,
            default_simplify=0, default_exemplar=0, default_dev_tracker="",
            default_dev_exemplar="", orchestrator_url="", orchestrated=True,
            session_mode="configure", review_locked_msg="",
            review_tagging_enabled=False, default_session_name="")
        check(len(page) > 100000, "%d labels render a full page" % count)
        check("{{" not in page, "%d labels leave no unrendered Jinja" % count)
        check(page.count('class="qa-btn"') == count,
              "%d labels make %d quick-add buttons, found %d"
              % (count, count, page.count('class="qa-btn"')))


def main():
    """Run every check and exit non-zero on the first failure recorded."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        print(test.__name__)
        test()
    if FAILURES:
        print("\nFAILED: %d" % len(FAILURES))
        for name in FAILURES:
            print("  - %s" % name)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
