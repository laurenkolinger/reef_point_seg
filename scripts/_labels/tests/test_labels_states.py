#!/usr/bin/env python
"""
Module:  test_labels_states.py
Purpose: Pin the Label Manager's empty states, the wash of the row being
         edited, the action row spacing, the named remap tooltips and the
         cache-bust pin, the defects and notes the 2026-09-17 review of the
         page raised (.superpowers/sdd/2026-09-16-vega-student-ready).
         Static checks over the template, stylesheet and script on disk, plus
         one render through the Flask test client with an empty paths
         provider (no project file is read or written).
Inputs:  scripts/_labels/templates/labels/index.html, static/labels.css,
         static/labels.js, and pipeline_orchestrator/app.py for ASSET_VERSION.
         LABELS_DIR in the environment overrides the location of the _labels
         package when the suite runs from outside scripts/_labels/tests.
Outputs: one ok or FAIL line per check, a PASS or "N FAILED" tally, exit
         code 0 on success and 1 on any failure.

House harness, no pytest. Run with:
  env/bin/python scripts/_labels/tests/test_labels_states.py
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

import os
import re
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
LABELS_DIR = Path(os.environ.get("LABELS_DIR") or THIS.parent.parent)
SCRIPTS_DIR = LABELS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

TEMPLATE = LABELS_DIR / "templates" / "labels" / "index.html"
STYLESHEET = LABELS_DIR / "static" / "labels.css"
SCRIPT = LABELS_DIR / "static" / "labels.js"
ORCHESTRATOR_APP = SCRIPTS_DIR / "pipeline_orchestrator" / "app.py"

ASSET_VERSION_RE = re.compile(r'^ASSET_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)
PIN_RE = re.compile(r"\?v=([A-Za-z0-9.-]+)")
# The stripe rule and the two rules that must be able to beat it.
STRIPE_RULE = ".lb-table tbody tr:nth-child(even) td"
HOVER_RULE = ".lb-table tbody tr:hover td"
EDITING_RULE = ".lb-table tbody tr.lb-row-editing td"
# The copy of the two empty lines.
MINT_LINE = "No labels yet. Use Add label to mint the first species code."
NOMATCH_LINE = "No label matches these filters. Press Clear filters to show all"

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


def squeeze(text: str) -> str:
    """Collapse runs of whitespace so template copy can be matched across lines."""
    return re.sub(r"\s+", " ", text)


def test_empty_states() -> None:
    print("\n[empty states: mint line for no labels, filter line for no match]")
    html = squeeze(read(TEMPLATE))
    js = read(SCRIPT)
    check('id="vocab-empty"' in html and MINT_LINE in html,
          "the mint line stays for a vocabulary with no labels")
    check('id="vocab-nomatch"' in html and NOMATCH_LINE in html,
          "a second line says the filters hid the rows and how to get them back")
    check('id="vocab-nomatch-total"' in html, "the filter line carries the vocabulary total")
    nomatch = re.search(r'id="vocab-nomatch"[^>]*>(.*?)</div>', html)
    check(bool(nomatch) and "Clear filters" in nomatch.group(1),
          "the filter line names the Clear filters button by its label")
    check("show($('vocab-empty'), total === 0)" in js,
          "the mint line shows only when the vocabulary total is zero")
    check("show($('vocab-nomatch'), total > 0 && shown === 0)" in js,
          "the filter line shows only when labels exist and none match")
    check("if (totalNode) totalNode.textContent = String(total)" in js,
          "the filter line's total is the vocabulary total, written only when the node exists")
    check("show($('vocab-empty'), rows.length === 0)" not in js,
          "the mint line no longer keys on the filtered rows")


def test_editing_wash() -> None:
    print("\n[editing wash: one look on odd and even rows]")
    css = read(STYLESHEET)
    js = read(SCRIPT)
    stripe_at = css.find(STRIPE_RULE)
    hover_at = css.find(HOVER_RULE)
    editing_at = css.find(EDITING_RULE)
    check(stripe_at != -1, "the stripe rule is present")
    check(editing_at != -1, "the editing wash is written at the stripe rule's specificity")
    check(editing_at > stripe_at and editing_at > hover_at,
          "the editing wash comes after the stripe and hover rules, so it wins on an even row")
    check(not re.search(r"^\.lb-row-editing td\b", css, re.MULTILINE),
          "the short .lb-row-editing td selector that lost to the stripe is gone")
    check(".lb-row-confirm" not in css and "lb-row-confirm" not in js,
          "no separate confirm row: nothing takes an nth-child slot under the edited row")
    check("insertBefore(noteTr" not in js, "the script inserts no row after the edited one")
    check("tdName.appendChild(errLine)" in js,
          "the save error line lives inside the edited row's name cell")
    check(".lb-row-editing .lb-inline-error" in css, "the in-row error line has its own spacing rule")


def test_action_row_spacing() -> None:
    print("\n[spacing: the Add label action row is not the table's first row]")
    css = read(STYLESHEET)
    wrap = re.search(r"\.lb-table-wrap\s*\{([^}]*)\}", css)
    check(bool(wrap), "the vocabulary table wrap has a rule")
    body = wrap.group(1) if wrap else ""
    check("margin-top: 16px" in body, "the table sits 16 px under the action row")
    check("border-top: 1px solid var(--hair-dim)" in body,
          "a hairline separates the action row from the table")
    addrow = re.search(r"\.lb-addrow\s*\{([^}]*)\}", css)
    check(bool(addrow) and "margin: 16px 0 0" in addrow.group(1),
          "the add form keeps the same 16 px step under the action row")


def test_remap_tooltips() -> None:
    print("\n[remap tooltips: full names, never a bare code]")
    js = read(SCRIPT)
    check("function describeCode(code, name)" in js,
          "a helper renders a code with its full name for sentences")
    check("describeCode(p.old, '')" in js and "describeCode(p.new, p.newName)" in js,
          "the merge tooltip names both codes in full")
    check("recorded as ' + p.old + ' becomes ' + p.new" not in js,
          "the old bare-code merge sentence is gone")
    check("p.old + ' keeps its code" not in js, "the old bare-code rename sentence is gone")
    check("its name is now ' + nowName" in js,
          "the rename tooltip says which name the code now carries")


def test_cache_pin() -> None:
    print("\n[cache pin: the template's ?v= matches the orchestrator's ASSET_VERSION]")
    html = read(TEMPLATE)
    app_py = read(ORCHESTRATOR_APP)
    m = ASSET_VERSION_RE.search(app_py)
    check(bool(m), "pipeline_orchestrator/app.py declares ASSET_VERSION")
    version = m.group(1) if m else ""
    pins = PIN_RE.findall(html)
    check(len(pins) == 7, f"the template pins seven assets (four of its own and the three guided-tour files, 2026-09-17) (found {len(pins)})")
    stale = sorted({p for p in pins if p != version})
    check(not stale, f"every pin is {version} (stale: {stale or 'none'})")


def test_render() -> None:
    print("\n[render: both empty lines are on the page, hidden until the script decides]")
    from flask import Flask

    from _labels import make_blueprint

    app = Flask("labels_states_test")
    app.register_blueprint(make_blueprint(lambda: {}, log_fn=lambda m: None),
                           url_prefix="/labels")
    r = app.test_client().get("/labels/")
    html = r.get_data(as_text=True)
    check(r.status_code == 200, f"GET /labels/ is 200 (got {r.status_code})")
    for node_id in ("vocab-empty", "vocab-nomatch", "vocab-nomatch-total"):
        check(re.search(rf'id="{node_id}"', html) is not None, f"the render carries #{node_id}")
    check(re.search(r'id="vocab-nomatch"\s+hidden', html) is not None,
          "the filter line starts hidden")


def main() -> None:
    """Run every group and exit non-zero on any failure."""
    print("label manager states, 2026-09-17 review close-out")
    test_empty_states()
    test_editing_wash()
    test_action_row_spacing()
    test_remap_tooltips()
    test_cache_pin()
    test_render()
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
