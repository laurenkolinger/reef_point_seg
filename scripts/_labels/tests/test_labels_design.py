#!/usr/bin/env python
"""
Module:  test_labels_design.py
Purpose: Pin the Label Manager page (scripts/_labels) to version 2 of the
         VICARIUS workflow page design, docs/superpowers/specs/2026-09-02-
         workflow-page-design.md: the 12 px legibility floor, no hardcoded
         grey, no paragraph on screen, no user-agent button chrome, striping
         and hover on every long list, sortable column heads, filters that
         remember themselves under the vega.labels. prefix, names beside
         codes, tooltip sentence rules, and the banned words.
Inputs:  the template, stylesheet and script under scripts/_labels, read from
         disk, plus one render of the page through the Flask test client with
         an empty paths provider (no project file is read or written).
Outputs: one ok or FAIL line per check on stdout, a PASS or "N FAILED" tally,
         exit code 0 on success and 1 on any failure.

House harness, no pytest. Run with:
  env/bin/python scripts/_labels/tests/test_labels_design.py
"""

from __future__ import annotations

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

import re
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
LABELS_DIR = THIS.parent.parent                 # .../scripts/_labels
SCRIPTS_DIR = LABELS_DIR.parent                 # .../scripts
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

TEMPLATE = LABELS_DIR / "templates" / "labels" / "index.html"
STYLESHEET = LABELS_DIR / "static" / "labels.css"
SCRIPT = LABELS_DIR / "static" / "labels.js"

# Version 2 legibility floor: nothing user-visible under 12 px.
FLOOR_PX = 12.0
# Classes the stylesheets cover, so a button carrying one of them never
# renders in the browser's own black-on-white chrome.
COVERED_BUTTON_CLASSES = {"btn", "tip", "lb-sort"}
# Browser storage keys the page may use, all under one prefix.
PREF_PREFIX = "vega.labels."
# Words and characters that never appear in VICARIUS output.
BANNED_WORD_RE = re.compile(r"\bland(s|ed|ing)?\b", re.IGNORECASE)
BANNED_STRINGS = ("UVI", "—", "–")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SECTION_RE = re.compile(r"<section\b.*?</section>", re.DOTALL)
# A paragraph tag, not <pre> or <path>: "<p" followed by a space or ">".
P_TAG_RE = re.compile(r"<p[\s>]")
TITLE_RE = re.compile(r'title="([^"]*)"')
JS_TITLE_RE = re.compile(r"\.title = '([^']*)'")
BUTTON_TAG_RE = re.compile(r"<button\b([^>]*)>", re.DOTALL)
CLASS_ATTR_RE = re.compile(r'class="([^"]*)"')
JS_BUTTON_RE = re.compile(r"el\('button', '([^']*)'")
JS_ID_RE = re.compile(r"\$\('([a-z0-9-]+)'\)")
FONT_SIZE_RE = re.compile(r"font-size:\s*([0-9.]+)px")
FONT_SHORTHAND_RE = re.compile(r"font:\s*(?:[0-9]{3}\s+)?([0-9.]+)px")
HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")

_fail = 0


def check(cond: bool, msg: str) -> None:
    """Print one ok or FAIL line and count the failure."""
    global _fail
    if cond:
        print(f"  ok   {msg}")
    else:
        _fail += 1
        print(f"  FAIL {msg}")


def read(path: Path) -> str:
    """Read one of the page's files as text, failing loudly when it is gone."""
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def strip_root_block(css: str) -> str:
    """Return the stylesheet without its :root token block, so a hex literal
    outside the tokens can be found."""
    return re.sub(r":root\s*\{.*?\n\}", "", css, count=1, flags=re.DOTALL)


def sentences(text: str) -> int:
    """Count sentences in a tooltip: terminal punctuation followed by a space
    or the end of the string."""
    return len(re.findall(r"[.!?](?:\s|$)", text.strip()))


# ── stylesheet ───────────────────────────────────────────────────────────────

def test_stylesheet() -> None:
    print("\n[stylesheet: legibility floor and tokens]")
    css = read(STYLESHEET)
    sizes = [float(v) for v in FONT_SIZE_RE.findall(css)]
    sizes += [float(v) for v in FONT_SHORTHAND_RE.findall(css)]
    check(len(sizes) > 0, f"labels.css declares font sizes ({len(sizes)} found)")
    small = sorted({s for s in sizes if s < FLOOR_PX})
    check(not small, f"no font-size below {FLOOR_PX:g} px in labels.css (found {small or 'none'})")

    check("font-size: 14px" in css and "html, body" in css,
          "the page body is 14 px")
    outside = HEX_RE.findall(strip_root_block(css))
    check(not outside, f"no hex colour outside the :root tokens (found {outside or 'none'})")
    check("#666" not in css and "#999" not in css and "#888" not in css,
          "no hardcoded hex grey anywhere")
    check("var(--ink-dim)" in css, "hint prose uses the dim ink token")

    check("tbody tr:nth-child(even)" in css, "table rows stripe")
    check("tbody tr:hover" in css, "table rows wash on hover")
    check(".lb-history-list li:nth-child(even)" in css and ".lb-history-list li:hover" in css,
          "the remap history list stripes and hovers")

    for selector in (".lb .btn", ".lb .btn-mini", ".lb .tip span", ".lb .foot"):
        rule = re.search(re.escape(selector) + r"[^{]*\{[^}]*\}", css)
        check(bool(rule) and "12px" in rule.group(0),
              f"{selector} is raised to the 12 px floor on this page")

    check("color-scheme: dark" in css, "form controls and scrollbars declare the dark colour scheme")
    check(".lb-sort" in css and 'th[aria-sort="ascending"]' in css and 'th[aria-sort="descending"]' in css,
          "sort heads carry a styled button and an arrow per sort direction")
    check(":focus-visible" in css and "var(--pink)" in css, "a visible focus ring is declared")
    check("min-height: 28px" in css, "hit targets on this page's own controls are at least 28 px")
    check(".lb-name" in css and ".sci" in css and "font-style: italic" in css,
          "a species binomial reads in italics")
    check(re.search(r"\.lb-code\s*\{[^}]*border", css) is not None,
          "the code is a chip on a ring")
    for token in ("--font-body:", "--font-display:", "--font-mono:"):
        check(token in css, f"labels.css declares {token.rstrip(':')}")


# ── template ─────────────────────────────────────────────────────────────────

def test_template() -> None:
    print("\n[template: no paragraphs, styled buttons, sort heads]")
    html = read(TEMPLATE)
    bodies = SECTION_RE.findall(html)
    check(len(bodies) == 3, f"three sections on the page (got {len(bodies)})")
    for body in bodies:
        head = re.search(r'id="([a-z-]+)"', body)
        name = head.group(1) if head else "?"
        check(not P_TAG_RE.search(body), f"section {name} has no <p> element")
    check(not P_TAG_RE.search(html), "no <p> element anywhere on the page")

    buttons = BUTTON_TAG_RE.findall(html)
    check(len(buttons) >= 10, f"the template has buttons to check ({len(buttons)})")
    bare = []
    for attrs in buttons:
        m = CLASS_ATTR_RE.search(attrs)
        classes = set(m.group(1).split()) if m else set()
        if not classes & COVERED_BUTTON_CLASSES:
            bare.append(attrs.strip()[:60])
    check(not bare, f"every button carries a covered class (bare: {bare or 'none'})")

    check(html.count('class="tip"') >= 4, "the title and each section carry an i or ? glyph")
    check('data-sort="name"' in html and 'data-sort="code"' in html and 'data-sort="category"' in html,
          "the vocabulary heads name their sort columns")
    check(html.count('class="lb-sort"') == 3, "three sort buttons, one per named column")
    first_th = re.search(r"<thead>.*?<th\b[^>]*class=\"([^\"]*)\"", html, re.DOTALL)
    check(bool(first_th) and "col-name" in first_th.group(1),
          "the name is the first (primary) column of the vocabulary table")
    check('href="{{ url_for(\'static\', filename=\'vicarius_workflow.css\') }}' in html,
          "the page links the shared workflow block")
    check("vicarius_tooltips.js" in html, "the page loads the shared tooltip bubble")
    check("Lauren Olinger" in html, "the footer names Lauren Olinger")

    print("\n[template: tooltip sentence rules]")
    titles = TITLE_RE.findall(html)
    titles = [t for t in titles if t and not t.startswith("{{")]
    check(len(titles) >= 12, f"tooltips found to check ({len(titles)})")
    opens_you = [t for t in titles if t.startswith("You")]
    check(not opens_you, f"no tooltip opens with You ({len(opens_you)} do)")
    unterminated = [t[:40] for t in titles if not t.rstrip().endswith(".")]
    check(not unterminated, f"every tooltip ends with a full stop ({unterminated or 'all do'})")
    long_ones = [t[:40] for t in titles if sentences(t) > 2]
    check(not long_ones, f"no tooltip runs past two sentences ({long_ones or 'none do'})")
    screen_refs = [t[:40] for t in titles if re.search(r"\b(above|below|this tab|this window)\b", t)]
    check(not screen_refs, f"no tooltip refers to the screen ({screen_refs or 'none do'})")

    print("\n[template: banned words]")
    check(not BANNED_WORD_RE.search(html), "the banned word is absent from the template")
    for s in BANNED_STRINGS:
        check(s not in html, f"template has no {s!r}")
    check(not EMOJI_RE.search(html), "template has no emoji")


# ── script ───────────────────────────────────────────────────────────────────

def test_script() -> None:
    print("\n[script: ids, storage, sorting, names]")
    js = read(SCRIPT)
    html = read(TEMPLATE)
    ids = sorted(set(JS_ID_RE.findall(js)))
    missing = [i for i in ids if f'id="{i}"' not in html]
    check(len(ids) >= 30 and not missing,
          f"every id the script reads exists in the template ({len(ids)} ids, missing {missing or 'none'})")

    js_buttons = JS_BUTTON_RE.findall(js)
    check(js_buttons and all("btn" in c.split() for c in js_buttons),
          f"every button the script creates carries .btn ({len(js_buttons)} created)")
    check("el('p'" not in js, "the script creates no <p> element")

    check(f"'{PREF_PREFIX}'" in js, f"browser storage keys share the {PREF_PREFIX} prefix")
    check(js.count("localStorage") == 3,
          f"localStorage is touched only inside the two pref helpers ({js.count('localStorage')} uses)")
    for helper in ("function readPref(", "function writePref("):
        block = re.search(re.escape(helper) + r".*?\n  \}", js, re.DOTALL)
        check(bool(block) and "try {" in block.group(0) and "catch" in block.group(0),
              f"{helper.split()[1]} wraps storage in try and catch")
    for key in ("'search'", "'category'", "'sort'"):
        check(f"readPref({key})" in js or f"writePref({key}" in js,
              f"the {key} choice is remembered")

    check("function sortedRows(" in js, "rows are sorted by a named function")
    check("'aria-sort'" in js, "the active sort head sets aria-sort")
    check("localeCompare" in js, "sorting compares text locale-aware")
    check("NON_TAXON_CATEGORIES" in js and "function isScientific(name, category)" in js,
          "italics depend on the category as well as the name shape")
    check("el('span', 'lb-code'" in js, "the code renders as a chip span")
    check("stamp24(" in js and "(AST)" in js, "the footer stamp is 24 hour and names AST")

    js_titles = JS_TITLE_RE.findall(js)
    check(js_titles and not [t for t in js_titles if t.startswith("You")],
          f"no script-made tooltip opens with You ({len(js_titles)} checked)")
    check(not BANNED_WORD_RE.search(js), "the banned word is absent from the script")
    for s in BANNED_STRINGS:
        check(s not in js, f"script has no {s!r}")
    check(not EMOJI_RE.search(js), "script has no emoji")


# ── render ───────────────────────────────────────────────────────────────────

def test_render() -> None:
    print("\n[render: the page answers 200 with no paragraph]")
    from flask import Flask

    from _labels import make_blueprint

    app = Flask("labels_design_test")
    app.register_blueprint(make_blueprint(lambda: {}, log_fn=lambda m: None),
                           url_prefix="/labels")
    client = app.test_client()
    r = client.get("/labels/")
    html = r.get_data(as_text=True)
    check(r.status_code == 200, f"GET /labels/ is 200 (got {r.status_code})")
    check(not P_TAG_RE.search(html), "the rendered page has no <p> element")
    check('id="vocab-table"' in html and 'id="vocab-tbody"' in html,
          "the rendered page keeps the vocabulary table ids")
    check("/labels/static/labels.css" in html, "the rendered page links its own stylesheet")
    r2 = client.get("/labels/api/vocabulary")
    check(r2.status_code == 200 and r2.get_json().get("rows") == [],
          "an unconfigured vocabulary answers an empty list, not an error")


def main() -> None:
    """Run every group and exit non-zero on any failure."""
    print("label manager design, spec version 2")
    test_stylesheet()
    test_template()
    test_script()
    test_render()
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
