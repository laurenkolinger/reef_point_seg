#!/usr/bin/env python
"""No-pytest harness for the matrix frontend (template + static JS/CSS).

Run with:
  env/bin/python scripts/_matrix/tests/test_matrix_ui.py

Checks (spec 2026-08-26 matrix cell review / stats composite, section 7 F):
  - node --check on every JS file under static/
  - template markers for the tabs, the cell drawer and the stats view
  - every <button>, <input>, <select> in the template carries a title
  - every JS-built control (control(...) helper, createElement) carries a title
  - banned-string sweep (em dash, emoji, the banned word) on every owned file
  - no inline project-name formatting (names come from project.json via the API)
  - the drawer and stats modules code against the spec 4.1 routes
  - GET /matrix/ renders through the blueprint with the markers present

Exits nonzero on any failed assertion. Uses a throwaway registry dir that is
removed at exit (never the real inprocess/_mask_registry).
"""

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, ".."))
_SCRIPTS = os.path.abspath(os.path.join(_PKG, ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Registry redirect BEFORE any _matrix import (builder imports MaskRegistry).
_TMP_REG = tempfile.mkdtemp(prefix="mx_ui_reg_")
os.environ["TCRMP_MASK_REGISTRY_DIR"] = _TMP_REG
os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")


def _cleanup():
    shutil.rmtree(_TMP_REG, ignore_errors=True)


atexit.register(_cleanup)

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


STATIC = os.path.join(_PKG, "static")
TEMPLATE = os.path.join(_PKG, "templates", "matrix", "index.html")
BLUEPRINT_PY = os.path.join(_PKG, "blueprint.py")
JS_FILES = {
    "matrix.js": os.path.join(STATIC, "matrix.js"),
    "matrix_cell.js": os.path.join(STATIC, "matrix_cell.js"),
    "matrix_stats.js": os.path.join(STATIC, "matrix_stats.js"),
}
CSS_FILE = os.path.join(STATIC, "matrix.css")
OWNED = dict(JS_FILES)
OWNED["index.html"] = TEMPLATE
OWNED["matrix.css"] = CSS_FILE
OWNED["test_matrix_ui.py"] = os.path.abspath(__file__)


# ── 1. node --check ──────────────────────────────────────────────────
def test_node_check():
    node = shutil.which("node")
    check(node is not None, "node is on PATH")
    if not node:
        return
    for name, path in JS_FILES.items():
        check(os.path.isfile(path), f"{name} exists")
        r = subprocess.run([node, "--check", path], capture_output=True, text=True)
        check(r.returncode == 0, f"node --check {name}: {r.stderr.strip()[:200] or 'ok'}")


# ── 2. template markers ──────────────────────────────────────────────
TEMPLATE_IDS = [
    # tabs + views
    "mx-tab-matrix", "mx-tab-stats", "mx-tab-training",
    "mx-view-matrix", "mx-view-stats", "mx-view-training",
    # drawer
    "mx-cell", "mx-cell-title", "mx-cell-label", "mx-cell-outcome", "mx-cell-close",
    "mx-by", "mx-cell-status", "mx-cell-error", "mx-cell-opacity", "mx-cell-others",
    "mx-cell-canvas", "mx-cell-caption", "mx-cell-tip", "mx-cell-tabs", "mx-cell-masks",
    "mx-cell-prefer", "mx-cell-clear", "mx-cell-pref", "mx-cell-projects",
    # stats
    "mx-stats-head", "mx-stats-status", "mx-stats-refresh", "mx-stats-error",
    "mx-stats-kpis", "mx-stats-species", "mx-stats-species-all", "mx-stats-species-none",
    "mx-stats-legend", "mx-chart-species", "mx-chart-year", "mx-chart-site",
    "mx-chart-images-year", "mx-chart-images-site", "mx-stats-reviewers",
    "mx-composites", "mx-composite-name", "mx-stats-by", "mx-composite-labels",
    "mx-composite-build", "mx-composite-result", "mx-hygiene", "mx-purge", "mx-purge-result",
    # existing grid + popover (kept)
    "mx-grid", "mx-tbody", "mx-pop", "mx-rescan", "mx-project",
    # Task I: composite-to-project linkage (Limit to projects, drawer line)
    "mx-composite-projects-label", "mx-composite-projects-all", "mx-composite-projects-none",
    "mx-composite-projects", "mx-cell-composites",
    # Task 4 (2026-09-02 stage-rail redesign): rail + panes + a few new ids
    "mx-train-rail", "mx-stage-1", "mx-stage-2", "mx-stage-3", "mx-pane-1", "mx-pane-2", "mx-pane-3",
    "mx-train-by", "mx-train-projects", "mx-stats-totals", "mx-stats-excluded", "mx-cell-where", "mx-cell-species",
]


def _rendered_page():
    """GET /matrix/ through the real blueprint (a throwaway inprocess root,
    stubbed open_folder_fn), so id checks run against what render_template
    actually produces from the new base/shared/v context vars, not just the
    raw Jinja source (which still carries the unrendered {{ v }} markers)."""
    from flask import Flask
    from _matrix import make_blueprint

    root = tempfile.mkdtemp(prefix="mx_ui_tplroot_")
    try:
        app = Flask(__name__)
        app.register_blueprint(
            make_blueprint(lambda: {"inprocess_root": root}, log_fn=lambda m: None,
                           open_folder_fn=lambda p: (False, "host opening stubbed in tests")),
            url_prefix="/matrix",
        )
        r = app.test_client().get("/matrix/")
        return r.data.decode("utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 2. template markers: ids, shared assets, cache-bust, rail, tooltips ──
def test_template_markers():
    tpl = _read(TEMPLATE)          # raw Jinja source (still carries {{ v }})
    bp_src = _read(BLUEPRINT_PY)   # blueprint source (MATRIX_ASSET_VERSION)
    html = _rendered_page()        # rendered page (base/shared/v substituted)
    for i in TEMPLATE_IDS:
        check(html.count('id="%s"' % i) == 1, "template id %s appears exactly once" % i)
    check('vicarius_workflow.css?v={{ v }}' in tpl and 'vicarius_tooltips.js?v={{ v }}' in tpl,
          "template links the shared assets with the version")
    check(tpl.count('?v={{ v }}') == 9, "nine versioned asset links (2 shared, 1 css, 3 js, 3 tour) and no literal version string")
    check('20260902-theme-unify' not in tpl, "old hand-written cache-bust gone")
    check(re.search(r'MATRIX_ASSET_VERSION\s*=\s*"20260917-plain"', bp_src) is not None,
          "blueprint holds the version constant")
    check(tpl.index('vicarius_tooltips.js') < tpl.index('matrix_cell.js') < tpl.index('matrix_stats.js') < tpl.index('matrix.js'),
          "script order: tooltips, cell, stats, matrix")
    check('role="tablist"' in tpl and 'aria-controls="mx-view-stats"' in tpl, "view tabs are a tablist")
    check(tpl.count('id="mx-stage-') == 3 and 'aria-orientation="vertical"' in tpl, "three rail stages in a vertical tablist")
    check('<p class="mx-stats-note">' not in tpl, "no explanatory paragraph on screen")
    # I5 (2026-09-03 whole-batch review): every "?"/"i" glyph explains its
    # control or section; a titleless glyph leaves the reader with no way
    # to learn what it does short of guessing.
    for m in re.finditer(r'<button\b[^>]*class="tip"[^>]*>', tpl):
        check('title="' in m.group(0), "tip glyph carries a title: %s" % m.group(0)[:70].replace("\n", " "))
    for t in re.findall(r'title="([^"]*)"', tpl):
        check(not t.startswith("You"), "tooltip does not open with You: %s" % t[:40])
        check(EM_DASH not in t, "no em dash in tooltip: %s" % t[:40])


# ── 2b. copy-rule sweep (design spec section 6 rules + section 7 rule 2/6):
# every title="..." in the template and every title: / .title = literal in
# the three scripts stays within the six textual bans. test_template_markers
# above already enforces two of the six (no leading "You", no em dash) on
# the template's own title= attributes; this sweep re-checks those two
# there for symmetry and adds the remaining four ("this tab", "above", the
# banned geography word LAND_RE already guards elsewhere in this file,
# "UVI"), then extends all six to the JS-built tooltips.
TITLE_JS_ANCHOR_RE = re.compile(r"title\s*:\s*|\.title\s*=\s*")
THIS_TAB_RE = re.compile(r"\bthis tab\b", re.IGNORECASE)
ABOVE_RE = re.compile(r"\babove\b", re.IGNORECASE)


def _js_title_literals(src):
    """Every quoted string literal inside a title: ... or .title = ...
    expression, one entry per literal segment so a concatenated title
    (subject + ' has ' + fmtN(v) + ...) is checked piece by piece rather
    than only at its first quote. Returns (is_lead, text) pairs; is_lead is
    True only for the segment that opens the expression (a variable-led
    title, e.g. 'title: title' or 'subject + ...', is skipped for the
    "never opens with You" rule since its real first word cannot be read
    statically; every segment still gets the other five rules)."""
    out = []
    for m in TITLE_JS_ANCHOR_RE.finditer(src):
        window = src[m.end():m.end() + 500]
        stop = window.find(";")
        expr = window[:stop] if stop != -1 else window[:200]
        lits = re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", expr)
        lits = [a or b for a, b in lits]
        leads_with_literal = expr.lstrip()[:1] in ("'", '"')
        for i, lit in enumerate(lits):
            out.append((leads_with_literal and i == 0, lit))
    return out


def test_copy_rules():
    tpl = _read(TEMPLATE)
    pairs = [(True, t) for t in re.findall(r'title="([^"]*)"', tpl)]
    for name, path in JS_FILES.items():
        pairs.extend(_js_title_literals(_read(path)))
    # Re-pinned 2026-09-03 (Task 12 tooltip audit): the audit dropped 5
    # redundant tooltips and trimmed 2 more (keep/trim/drop verdicts in
    # task-12-tooltip-audit.md), taking the scanned literal count to 98;
    # the floor is a plausibility check that the sweep is scanning real
    # content, not a pin on the audited total, so it moves down with room
    # to spare rather than tracking the exact count.
    check(len(pairs) > 80, f"copy-rule sweep scanned a plausible number of tooltip literals ({len(pairs)})")
    for is_lead, text in pairs:
        short = text[:50]
        if is_lead:
            check(not text.startswith("You"), "copy rule: tooltip does not open with You: %s" % short)
        check(EM_DASH not in text, "copy rule: no em dash: %s" % short)
        check(THIS_TAB_RE.search(text) is None, 'copy rule: no "this tab": %s' % short)
        check(ABOVE_RE.search(text) is None, 'copy rule: no "above": %s' % short)
        check(LAND_RE.search(text) is None, "copy rule: banned geography word absent: %s" % short)
        check("UVI" not in text, 'copy rule: no "UVI": %s' % short)


# TAG_RE stays module-level: test_js_tooltips (JS section, not touched here)
# still scans string-built <button|input|select> tags for a title= attribute.
TAG_RE = re.compile(r"<(button|input|select)\b([^>]*)>", re.IGNORECASE | re.DOTALL)


def test_js_tooltips():
    for name in ("matrix_cell.js", "matrix_stats.js"):
        src = _read(JS_FILES[name])
        # Controls are built only through control(tag, attrs) which refuses a
        # missing title at runtime; no bare createElement of a control tag.
        for tag in ("button", "input", "select"):
            check(f"createElement('{tag}')" not in src and f'createElement("{tag}")' not in src,
                  f"{name}: no bare createElement('{tag}') (use control())")
        check("function control(" in src and "control without tooltip" in src,
              f"{name}: control() helper enforces a title")
        calls = [m.start() for m in re.finditer(r"\bcontrol\('(button|input|select)'", src)]
        check(len(calls) >= 2, f"{name}: builds controls through control() ({len(calls)})")
        for pos in calls:
            window = src[pos:pos + 600]
            check("title:" in window, f"{name}: control() call at offset {pos} passes title:")
        # HTML string templates: any control tag literal must carry title=
        for tag, attrs in TAG_RE.findall(src):
            check("title=" in attrs, f"{name}: string-built <{tag}> carries title=")
    # matrix.js keeps createElement; each control gets a title within a few lines
    src = _read(JS_FILES["matrix.js"])
    lines = src.splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"createElement\('(button|input|select)'\)", line)
        if not m:
            continue
        window = "\n".join(lines[i:i + 8])
        check(".title =" in window or "setAttribute('title'" in window,
              f"matrix.js line {i + 1}: created <{m.group(1)}> gets a title")


# ── 4. banned strings ────────────────────────────────────────────────
# Built from code points so this file itself stays clean under the sweep.
EM_DASH = chr(0x2014)
EMOJI_RE = re.compile("[" + chr(0x1F000) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + "]")
LAND_RE = re.compile(r"\b" + "l" + "and" + r"\b", re.IGNORECASE)


def test_banned_strings():
    for name, path in OWNED.items():
        src = _read(path)
        check(EM_DASH not in src, f"{name}: no em dash")
        check(EMOJI_RE.search(src) is None, f"{name}: no emoji")
        hits = LAND_RE.findall(src)
        check(not hits, f"{name}: banned word absent ({len(hits)} hits)")


# ── 5. no inline project-name formatting ─────────────────────────────
def test_no_inline_project_name():
    for name, path in JS_FILES.items():
        src = _read(path)
        check("project_name" not in src, f"{name}: never reads the registry's project_name")
        check("run_dir.replace" not in src and "run_dir.split" not in src,
              f"{name}: never derives a display name from the run folder")
    cell = _read(JS_FILES["matrix_cell.js"])
    check("p.name || p.project_id" in cell, "drawer displays the API's name (project.json) with id fallback")


# ── 6. contract coverage ─────────────────────────────────────────────
def test_contract_routes():
    cell = _read(JS_FILES["matrix_cell.js"])
    for route in ("/api/cell?image=", "/api/open-folder", "/api/prefer'", "/api/prefer/clear"):
        check(route in cell, f"matrix_cell.js calls {route}")
    for key in ("polygon_px", "is_label", "image_url", "outcome_for_label", "opened_by",
                "preference", "chosen_by", "source_type", "reviewer", "area"):
        check(key in cell, f"matrix_cell.js reads {key}")
    check("'mx.by'" in cell, "drawer remembers initials in localStorage (mx.by)")
    check("Escape" in cell, "Esc closes the drawer")
    check("404" in cell, "drawer handles a 404 from a not-yet-live route")
    check("getContext('2d')" in cell and "drawImage" in cell, "drawer draws on a canvas")
    check("function inPoly" in cell and "hitTest" in cell, "drawer hit-tests masks by point-in-polygon")
    check("unrecorded" in cell, "blank reviewer renders as unrecorded")
    stats = _read(JS_FILES["matrix_stats.js"])
    for route in ("/api/stats?status=", "/api/composites", "/api/composite'", "/api/registry/purge-test-rows"):
        check(route in stats, f"matrix_stats.js calls {route}")
    for key in ("by_year", "by_site", "images_by_year", "images_by_site", "reviewers",
                "unknown_project_rows", "test_residue_rows", "n_negatives", "skipped_review_only"):
        check(key in stats, f"matrix_stats.js reads {key}")
    check("<svg" in stats, "stats charts are inline SVG")
    check("window.confirm(" in stats, "purge sits behind a confirm")
    check("'mx.by'" in stats, "stats shares the initials key (mx.by)")
    check("function fmtAst(" in stats and "' AST'" in stats, "stats stamps the server time as AST (fmtAst, YYYY-MM-DD HH:MM AST)")
    for bad in ("toISOString", "toLocaleTimeString", "getUTC", "toUTCString", "'UTC'"):
        check(bad not in stats and bad not in cell, f"no browser-side {bad} timestamps")
    # palette: the old 8-slot literal pin retired 2026-09-03 (Task 10) -
    # Task 9's palette correction (matrix_stats.js SLOTS) changed slots 4-8
    # to named extension hues that design spec 9.2 explicitly does not
    # spec-pin (only the first 3, Porites/Siderastrea/Orbicella, are
    # contrast-validated); pinning the full 8-entry literal here duplicated
    # test_task9_stats_tab's own "species hues per the spec lead the
    # palette" check (first-3 prefix) while also freezing the unpinned
    # tail. Tail-folding behavior itself is unchanged and stays pinned:
    check("OTHER_COLOR" in stats and "'Other'" in stats, "stats folds the tail into Other")
    # "charts label the y axis" (rotate(-90)) retired 2026-09-03 (Task 10):
    # design spec 9.2 ("no axis titles; the caption names measure and
    # dimension") replaced the rotated axis-title text with the h2 figure
    # captions Task 9 built (fig-cap), so no axis-label string exists to pin.
    check("Table view" in stats, "every chart has a table view")
    main = _read(JS_FILES["matrix.js"])
    check("window.MatrixCell.open(" in main, "matrix.js opens the drawer on activation")
    check("e.key !== 'Enter' && e.key !== ' '" in main, "Enter and Space activate a cell")
    check("window.MatrixStats.activate()" in main, "matrix.js activates the stats view")
    check("'mx.tab'" in main, "tab choice remembered in localStorage (mx.tab)")
    check("elTbody.addEventListener('mouseover'" in main and "function popoverHtml" in main,
          "existing hover popover kept")


# ── 6b. review fixes (2026-08-26 wf2 findings M1/M2/L1-L8) ──────────
def _order(src, *needles):
    """True when every needle occurs and in the given order."""
    pos = -1
    for n in needles:
        i = src.find(n, pos + 1)
        if i < 0:
            return False
        pos = i
    return True


def test_review_fixes():
    cell = _read(JS_FILES["matrix_cell.js"])
    stats = _read(JS_FILES["matrix_stats.js"])
    main = _read(JS_FILES["matrix.js"])
    html = _read(TEMPLATE)
    css = _read(CSS_FILE)
    # M1 UI: the composite result's Open folder keeps posting {path}; a failure
    # shows the server path (or out_dir) and points at Copy path.
    check("postJson('/api/open-folder', { path: j.out_dir || '' })" in stats,
          "M1: composite Open folder posts {path: out_dir}")
    check("var p = err.path || j.out_dir || '';" in stats and "use Copy path" in stats,
          "M1: open-folder failure shows err.path or out_dir and points at Copy path")
    # L4 UI: a 409 carries out_dir; parse() folds it into err.path and the
    # build failure renders it with a Copy path button.
    check("err.path = (j && (j.path || j.out_dir)) || '';" in stats, "L4: parse reads err.out_dir as the path")
    # Re-pinned 2026-09-03 (Task 10): the literal message renamed from
    # "composite" to "training set" per the section 6 vocabulary rule
    # (Task 6); resultError() itself and the path/Copy-path behavior it
    # renders are unchanged, so only the string changed here.
    check("function resultError(" in stats and "resultError('Could not build the training set: '" in stats,
          "L4: build failure goes through resultError (path + Copy path)")
    check("err.status === 409" in stats, "L4: 409 gets a name-collision hint")
    # M2: stacked charts take the union of parsed values and by_* keys, unknown last.
    check("categoriesFor(d.years, d.by_year)" in stats and "categoriesFor(d.sites, d.by_site)" in stats,
          "M2: stacked charts build categories from years/sites plus by_* keys")
    check("function unknownLast(" in stats and ".sort(unknownLast)" in stats, "M2: unknown bucket sorts last")
    for key in ("unknown_year_masks", "unknown_site_masks", "unlabeled_rows"):
        check(key in stats, f"M2: hygiene line reads {key}")
    check("Number(ex.unlabeled_rows) || 0" in stats, "M2: unlabeled_rows absent counts as 0")
    # L6 UI: no species -> build disabled with a reason.
    check("el.compBuild.disabled = !species.length;" in stats, "L6: build button disabled when species is empty")
    check(stats.count("Nothing to build") >= 2, "L6: the empty-species reason shows in the result line and on click")
    check("!((S.data && S.data.species) || []).length" in stats, "L6: after a build the button re-enables only with species")
    # L1: ledger-only sources get a project card.
    check("ledger_only: true" in cell and "s.name || s.project_id" in cell, "L1: ledger-only source becomes a card named from the source")
    check("no segmentation data for this image" in cell, "L1: ledger-only card carries the note")
    check(_order(cell, "var all = ps.concat(ledgerOnly);", "all.forEach(function (p) {", "'Open project folder'", "'Copy path'"),
          "L1: ledger-only cards keep Open project folder and Copy path")
    # L2: prefer button reset before any early return; prefer() reports no project.
    check("function resetPrefer(" in cell and "el.prefer.textContent = 'Use these masks';" in cell, "L2: resetPrefer helper")
    check(_order(cell, "function renderMasks() {", "resetPrefer();", "if (!p) return;"),
          "L2: renderMasks resets the button before its early return")
    check(_order(cell, "function open(image, label", "resetPrefer();", "var seq = ++S.reqSeq;"),
          "L2: open() resets the button before fetching")
    check("No project holds this image, so there are no masks to choose." in cell, "L2: prefer() reports a missing project")
    # L3 UI: stale preference.
    check("!!pref.stale || !found" in cell, "L3: stale when flagged or when no listed project matches")
    # Re-pinned 2026-09-03 (I3, whole-batch review): the vocabulary rule
    # (design spec section 6, "training set, not composite") reached this
    # line too, so "composites" became "training sets".
    check("training sets fall back to the strongest project" in cell and "no longer exists" in cell, "L3: stale note text")
    check("composites take" not in cell, "I3: no-choice line reads training sets, not composites")
    check("composites fall back" not in cell, "I3: stale-note line reads training sets, not composites")
    check("el.clear.classList.toggle('is-prominent', stale)" in cell and ".btn.is-prominent" in css,
          "L3: Clear choice becomes prominent (JS + CSS)")
    # L7: focus management.
    check("el.close.focus()" in cell, "L7: open() focuses the close button")
    check("returnTo" in cell and "back.focus()" in cell and "document.contains(back)" in cell, "L7: close() restores focus to the opener")
    check("window.MatrixCell.open(td.dataset.image, td.dataset.label, td)" in main, "L7: matrix.js passes the activating cell")
    # L8: no native title on the canvas; caption under the tools row.
    canvas = re.search(r"<canvas\b[^>]*>", html)
    check(canvas is not None and "title=" not in canvas.group(0), "L8: canvas carries no title attribute")
    check(canvas is not None and 'aria-describedby="mx-cell-caption"' in canvas.group(0), "L8: canvas is described by the caption")
    # Re-pinned 2026-09-03 (Task 10): controller ruling 1 (Task 4) put the
    # canvas first (canvas-wrap -> caption -> tools row), inverting the old
    # tools -> caption -> canvas order this check pinned; "class=\"mx-cell-
    # tools\"" no longer exists at all (the opacity/other-labels controls
    # now live in a plain "dl.choices", the shared tools-row pattern), so
    # the new anchors are the ids that actually carry this order today.
    check(_order(html, 'id="mx-cell-canvas-wrap"', 'id="mx-cell-caption"', 'id="mx-cell-opacity"'),
          "L8: caption sits between the canvas and the tools row (canvas first, per ruling 1)")
    check("Hover a mask for its uid, status, source, reviewer and area." in html and ".mx-cell-caption" in css,
          "L8: caption text kept and styled")


# ── 6c. 2026-09-01 quality sweep (Task 5: drawer band, popover, tabs, jargon) ─
def test_quality_sweep_task5():
    main = _read(JS_FILES["matrix.js"])
    cell = _read(JS_FILES["matrix_cell.js"])
    stats = _read(JS_FILES["matrix_stats.js"])
    html = _read(TEMPLATE)
    css = _read(CSS_FILE)

    # Drawer band: the 460px/899px/900px/484px pins retired 2026-09-03
    # (Task 5, page css rewrite). The drawer is now a real grid column
    # (--drawer-w: 440px) beside the grid at >= 1000px, with body.mx-cell-
    # open collapsing .page's own padding-right; full width below 1000px.
    # See test_task5_page_css for the current breakpoint checks.

    # Popover vs drawer z-order: the popover never shows while the drawer is
    # open (chosen fix over clamping positionPop to the drawer's edge).
    m = re.search(r"function showPop\(td\) \{(.*?)\n  \}", main, re.DOTALL)
    check(m is not None, "showPop() function found")
    if m:
        body = m.group(1)
        check("MatrixCell" in body and "isOpen()" in body and "hidePop()" in body,
              "showPop() bails to hidePop() while the drawer is open")
        check(body.index("isOpen()") < body.index("cellAt("), "the drawer-open guard runs before any popover content is built")

    # Dead "Found by model" legend row + outcome-filter option removed; the
    # ladder itself (builder.py, out of this task's ownership) is untouched.
    check('data-outcome="found_model"' not in html, "found_model legend row removed")
    check('value="found_model"' not in html, "found_model outcome-filter option removed")
    check('data-outcome="found_expert"' in html, "the remaining legend rows are still present")
    # M7 (2026-09-03 whole-batch review): every legend item explains its
    # outcome in its own tooltip (design spec 9.1); restored from the
    # approved mockup's six li.lg controls, verbatim.
    check(html.count('class="lg"') == 6, "six legend items present")
    legend_section = html.split('id="mx-legend"', 1)[1].split("</ul>", 1)[0]
    check(legend_section.count("title=") == 6, "all six li.lg carry a title")

    # Jargon: empty state and SOURCE_WORDS renamed to current step names.
    check("Annotate a project (step 4, Place + Segment)" in html and "to populate the matrix." in html,
          "empty state uses current step-4 naming")
    check("Step 5 or the 4.test" not in html, "stale empty-state phrasing removed")
    check("'4.test combined annotator'" not in main and "'Step 5 review'" not in main,
          "SOURCE_WORDS no longer carries the stale display strings")
    check("step 4, Place + Segment" in main, "SOURCE_WORDS renamed to the current step-4 name")

    # Third tab: Training Sets (#training deep link), composites + build form
    # and registry hygiene moved out of STATS; STATS keeps KPIs/charts/reviewers.
    check('id="mx-tab-training"' in html and 'id="mx-view-training"' in html, "Training Sets tab exists")
    check("h === 'training'" in main, "matrix.js applies the #training deep link")
    check("saved === 'training'" in main, "matrix.js restores the training tab from localStorage")
    stats_section = html.split('id="mx-view-stats"', 1)[1].split('id="mx-view-training"', 1)[0]
    training_section = html.split('id="mx-view-training"', 1)[1]
    check("mx-composite-build" not in stats_section, "composite build form no longer lives in STATS")
    check("mx-hygiene" not in stats_section and "mx-purge" not in stats_section,
          "registry hygiene no longer lives in STATS")
    check("mx-stats-kpis" in stats_section and "mx-chart-year" in stats_section and "mx-stats-reviewers" in stats_section,
          "STATS keeps KPIs, charts and the reviewers table")
    check("mx-composite-build" in training_section and "mx-composites" in training_section,
          "Training Sets carries the composites list and build form")
    check(_order(training_section, "<details", 'id="mx-hygiene"', 'id="mx-purge"', "</details>"),
          "registry hygiene sits in a <details> block at the bottom of Training Sets")
    check("if (which === 'stats' || which === 'training')" in main,
          "either Stats or Training Sets triggers the shared MatrixStats.activate() load")

    # partially_registered (Task 4 contract): "(partial)" suffix + tooltip
    # on the project isolate dropdown.
    check("partially_registered" in main, "matrix.js reads the partially_registered flag")
    check("' (partial)'" in main, "partial projects get a (partial) suffix")
    check("Some of this project\\'s reviewed cells are not yet backed by the canonical registry" in main,
          "partial-project tooltip text matches the pinned contract (cell-granular wording)")

    # Provenance filter radio group on the composite build form (Task 6
    # /matrix/api/composite contract): source_types null/hand/hand+model.
    check(html.count('name="mx-composite-source"') == 3, "three provenance radios in the build form")
    for val in ("all", "hand", "hand_model"):
        check(f'value="{val}"' in training_section, f"provenance radio {val} present")
    check("SOURCE_TYPE_MAP" in stats, "matrix_stats.js maps the radio value to a source_types list")
    check("['manual_box', 'manual_click', 'edit']" in stats, "hand-drawn-only source_types list matches the pin")
    check("['manual_box', 'manual_click', 'edit', 'model']" in stats, "hand-drawn+model source_types list matches the pin")
    check("source_types: selectedSourceType()" in stats, "build posts source_types")
    # Rebuild-existing-name checkbox.
    check('id="mx-composite-rebuild"' in html, "Rebuild existing name checkbox present")
    check("rebuild: !!(el.compRebuild" in stats, "build posts rebuild: true/false")

    # Drawer linkage: "Open this project in the pipeline" behind a styled confirm.
    check("Open in pipeline" in cell, "drawer offers Open in pipeline")
    check("window.confirm('Switches the pipeline window to this project and closes any open annotator.')" in cell,
          "Open in pipeline confirms with the pinned wording")
    check("project_dir=" in cell and "window.open(url, 'reefpointseg_orchestrator')" in cell,
          "Open in pipeline opens the orchestrator's ?project_dir= auto-open route")
    check(_order(cell, "function openInPipeline(p) {", "window.confirm(", "window.open("),
          "the confirm runs before the window opens")

    # Every new control still carries a tooltip (spot checks beyond the
    # blanket sweep in test_template_tooltips/test_js_tooltips).
    # Both re-pinned 2026-09-03 (Task 10) to the rail mockup's verbatim
    # plain-language text (Task 4 report): the old sentences described the
    # same controls in the pre-redesign register-detail voice.
    check('title="Only masks a person drew or edited by hand go in."' in html, "hand-drawn radio has a plain-language tooltip")
    check('title="If the name is taken and this is unticked, VICARIUS keeps the existing set and refuses the build."' in html,
          "rebuild checkbox has a plain-language tooltip")


# ── 6c2. 2026-09-03 stage-rail redesign: page css (Task 5) ───────────
def test_task5_page_css():
    css = _read(CSS_FILE)
    check(':root' not in css, "page css declares no tokens (shared stylesheet owns them)")
    check("font-family: 'Inter'" not in css and "font-family: 'Syne'" not in css,
          "page css uses font tokens only")
    # Fix round 2 (2026-09-03, reviewer): the brief's .mx-grid/.mx-cell/
    # .figure named the mockup's classes, but the template and scripts
    # never emit those three as a literal class (mx-grid and mx-cell are
    # ids only; .figure is never emitted at all, real markup carries .fig).
    # Checking the classes actually emitted instead of a header-comment
    # mention: .grid (the coverage table), .drawer (the cell drawer),
    # .fig (each stats figure).
    for sel in (".board", ".grid", ".mx-colhead", ".lbl-stack", ".drawer", ".drawer-where",
                ".sci", ".fig", ".hbar", ".ylabs", ".excl"):
        check(sel in css, "page rule %s present" % sel)
    # Re-pinned 2026-09-03 (M4, whole-batch review): 999px so this query
    # never overlaps the drawer layout's "@media (min-width: 1000px)".
    check("@media (max-width: 999px)" in css, "page breakpoint at 999px, no overlap with the 1000px min-width query")
    check("899px" not in css and "1420px" not in css and "820px" not in css, "old breakpoints gone")
    check("var(--c-manual)" in css or "#4bbf73" in css, "found-by-hand outcome color bound")
    for tok in ("--c-not-found", "--c-found-manual", "--c-found-ai", "--c-found-expert", "--c-pending-expert"):
        check(tok in css, "outcome token %s defined on the page" % tok)
    # Fix round 1 (2026-09-03): drawer viewer chrome (canvas wrap, opacity
    # row, hover tip) styled from the mockup's .canvas/opacity/tip rules.
    # Re-pinned 2026-09-03 (M1, whole-batch review): ".mx-cell-tools" was
    # pruned as dead CSS (the opacity row is a real dl.choices dt/dd pair,
    # never that scaffolding class); "input[type=\"range\"]" is the page
    # rule that actually styles the drawer's mask-opacity control today.
    for sel in (".mx-cell-viewer", ".mx-cell-canvas-wrap", 'input[type="range"]'):
        check(sel in css, "page rule %s present" % sel)


# ── 6d. 2026-09-02 typography alignment with the orchestrator ────────
# The matrix used to load Space Mono only and set it on the page root, so
# every paragraph, control and label rendered monospace. The orchestrator
# (scripts/pipeline_orchestrator/static/style.css) splits three roles:
# Inter for body prose, Syne for display headings, Space Mono for buttons,
# micro-labels and data chrome. These checks pin that split here.
def test_typography():
    html = _read(TEMPLATE)
    css = _read(CSS_FILE)

    # Template loads all three families with display=swap, behind preconnects.
    check('rel="preconnect" href="https://fonts.googleapis.com"' in html, "preconnect to fonts.googleapis.com")
    check('rel="preconnect" href="https://fonts.gstatic.com"' in html, "preconnect to fonts.gstatic.com")
    for fam in ("family=Inter:wght@400;500;600;700",
                "family=Space+Mono:wght@400;700",
                "family=Syne:wght@600;700;800"):
        check(fam in html, f"font link requests {fam}")
    check("display=swap" in html, "font link uses display=swap")

    # Font-role tokens and the html/body base rule moved to the shared
    # stylesheet 2026-09-03 (Task 5, page css rewrite): matrix.css declares
    # no tokens of its own (they live in vicarius_workflow.css) and never
    # redeclares html, body. Both pins retired here; the body-stack string
    # stays as a page-css comment, so this one check is left in place.
    check("'Inter', ui-sans-serif, -apple-system, sans-serif" in css,
          "body stack matches the orchestrator's Inter stack")

    # Display type: section heads still owned by this page. .mx-title and
    # .mx-tab moved to the shared chrome (.title/.tab) 2026-09-03; .mx-cell-
    # section and .mx-chart no longer exist, replaced by .dsec-head/.fig-cap.
    # Fix round 1 (2026-09-03, coordinator review): the replacements were
    # dropped without a re-pin when the old selectors were retired here, so
    # the "section heads and chart captions use the display font" contract
    # went unguarded; re-pinned below (.dsec-head h3 in the loop, .fig-cap /
    # .fig-cap h2 as a pair right after it).
    for sel in (".mx-stats-section h2", ".mx-hygiene-details > summary", ".dsec-head h3"):
        block = css.split(sel + " {", 1)[1].split("}", 1)[0] if (sel + " {") in css else ""
        check("var(--font-display)" in block, f"{sel} uses the display font token")
    # .fig-cap h2 declares no font-family of its own (it reads "font:
    # inherit" and takes the display font from its parent .fig-cap), so it
    # cannot be pinned through the single-block loop above (its own block
    # never contains the literal token); checked as a pair instead.
    fig_cap_block = css.split(".fig-cap {", 1)[1].split("}", 1)[0] if ".fig-cap {" in css else ""
    check("var(--font-display)" in fig_cap_block, ".fig-cap uses the display font token")
    fig_cap_h2_block = css.split(".fig-cap h2 {", 1)[1].split("}", 1)[0] if ".fig-cap h2 {" in css else ""
    check("font: inherit" in fig_cap_h2_block, ".fig-cap h2 inherits the display font from .fig-cap")

    # The 0.22em/0.18em letter-spacing bans retired 2026-09-03: both
    # trackings were retired from the shared chrome long before this page
    # css rewrite, so the pins had nothing left to guard.

    # Mono stays on buttons, inputs, micro-labels and data chrome. .mx-btn,
    # .mx-rescan and .ctrl input[type="search"] moved to the shared .btn /
    # select,input[type=search] rules 2026-09-03; .mx-cell-title is now an
    # id only (its class is .drawer-title). Retired here. .mx-cell-tip
    # retired from this loop in Fix round 1 (2026-09-03): the coordinator's
    # instruction gave it the shared .bubble's look (Inter), not mono, so
    # it moved out of the anchors that must stay mono.
    # .mx-table, .mx-kpi-value and .mx-cell-chip dropped from this anchor
    # loop 2026-09-03 (Task 10 dead-CSS prune): each rule was removed from
    # matrix.css because grep confirmed no script or template emits the
    # class any more (renderKpis/tableOf already moved to .tot/.totals and
    # .t/.tbl; every chip in the drawer and legend is .lchip/.chip, never
    # .mx-cell-chip) - see the CSS pruning note in matrix.css's own header.
    for anchor in (".mx-pop {", ".mx-cell-tab {"):
        idx = css.find(anchor)
        check(idx >= 0, f"{anchor} rule found")
        if idx >= 0:
            block = css[idx:].split("}", 1)[0]
            check("var(--font-mono)" in block, f"{anchor} keeps the mono font token")
    # Re-pinned 2026-09-03 (Task 10): matrix.css declares no font tokens of
    # its own (checked above), so the SVG text rule reaches Space Mono
    # through the shared var(--font-mono) token rather than the literal
    # family name; the old literal-string pin predates that tokenization.
    check(".mx-svg text" in css and "var(--font-mono)" in css.split(".mx-svg text", 1)[1].split("}", 1)[0],
          "SVG chart text stays mono (--font-mono, i.e. Space Mono; it is data)")
    # ".mx-colhead .col-code" dropped 2026-09-03 (M1, whole-batch review):
    # the column head builder finished moving onto the mockup's .lbl-stack/
    # .lbl-name/.id.code shape, and the code chip's mono font now comes
    # from the shared .id class (vicarius_workflow.css), not a page rule.
    for sel in (".mx-rowhead .row-name", ".row-pct"):
        block = css.split(sel + " {", 1)[1].split("}", 1)[0] if (sel + " {") in css else ""
        check("var(--font-mono)" in block, f"{sel} (grid stem, data) keeps mono")

    # Prose reads as prose: no leftover "font-family: inherit", which used to
    # mean monospace and would now silently mean Inter on data chrome.
    check("font-family: inherit" not in css, "no font-family: inherit left in matrix.css")

    # Numerics line up.
    check(css.count("tabular-nums") >= 2, "numeric columns and KPI values use tabular-nums")

    # The JS modules never hardcode a font: canvas and SVG text take theirs
    # from matrix.css so this swap is a one-file change.
    for name, path in JS_FILES.items():
        src = _read(path)
        check("font-family" not in src, f"{name} hardcodes no font-family")
        check("ctx.font" not in src, f"{name} sets no canvas font")


# ── 6e. Task I: composite-to-project linkage (matrix side) ───────────
# Backend contract this codes against (pinned, not verified server-side
# here): POST /matrix/api/composite takes an optional "project_ids" list;
# POST /matrix/api/composite/create-project {name, by} -> {run_dir,
# project_id, display_name} | 404 | 400; each /api/composites entry may
# carry source_projects [{project_id, n_masks}], linked_projects [run dir
# basenames], project_filter; /api/cell may carry an optional "composites"
# list.
def test_task_i_composite_linkage():
    html = _read(TEMPLATE)
    stats = _read(JS_FILES["matrix_stats.js"])
    cell = _read(JS_FILES["matrix_cell.js"])
    css = _read(CSS_FILE)

    # 1. Build form: "Limit to projects" multi-select, all/none actions,
    # tooltip carrying the pinned wording (checked by default is a runtime
    # behavior, not a static marker: renderCompProjects ticks every project
    # unless S.compProjSel says otherwise, checked via the JS source below).
    training_section = html.split('id="mx-view-training"', 1)[1]
    # Re-pinned 2026-09-03 (Task 10): the rail mockup renamed the label
    # "Limit to projects" to plain "Projects" (Task 4 report, expected
    # failure #4 - matches the mockup's "Projects" checklist, same job).
    check('id="mx-composite-projects-label"' in training_section and
          '>Projects<' in training_section, "Projects control present (label matches the mockup)")
    check('id="mx-composite-projects-all"' in training_section and
          'id="mx-composite-projects-none"' in training_section,
          "Projects checklist gets all/none actions like the labels box")
    # Re-pinned 2026-09-03 (Task 10): the tooltip moved off an inline
    # title= on the box div onto a "?" glyph button next to the dt label
    # (Task 4 report, expected failure #5 - every filter/checklist tooltip
    # in the redesign uses the uniform "?" glyph pattern now).
    m = re.search(r'id="mx-composite-projects-label">.*?<button[^>]*title="([^"]+)"', training_section, re.DOTALL)
    check(m is not None, "Projects checklist carries a '?' glyph tooltip")
    if m:
        check("Tick the projects whose masks may go into the training set" in m.group(1),
              "tooltip matches the mockup's verbatim wording")
    check("renderCompProjects" in stats and "S.compProjSel" in stats, "matrix_stats.js renders the project checklist from state")
    check("cb.checked = S.compProjSel[p.project_id] !== false;" in stats, "every project defaults to checked")
    check("function selectedProjectIds()" in stats and "chosen.length === all.length) ? null : chosen" in stats,
          "all-checked sends project_ids: null, matching the labels/pattern")
    check("project_ids: selectedProjectIds()" in stats, "the build POST carries project_ids")
    check("getJson('/api/data')" in stats, "the project list for the multi-select and name lookups comes from /api/data")

    # 2. Create-project button + fetch call.
    check("'/api/composite/create-project'" in stats, "matrix_stats.js posts to the pinned create-project route")
    check("postJson('/api/composite/create-project', { name: c.name, by: by })" in stats,
          "create-project POST body carries name and by, per the pinned contract")
    check("'Create training project'" in stats, "the primary action button is labelled")
    # Re-pinned 2026-09-03 (Task 10): rail mockup's verbatim tooltip (Task 6
    # report), replacing the old pipeline-project/symlink/no-duplication
    # sentence.
    check("'VICARIUS creates the project from this set and lists it in stage 3.'" in stats,
          "button tooltip matches the rail mockup's verbatim wording")
    # Re-pinned 2026-09-03 (Task 10): stage-rail success copy (Task 6
    # report), replacing the old "Open it from the reef_point_seg module
    # window..." sentence.
    check("Created ' + (j.display_name || j.project_id || '') +" in stats and
          "'. It is listed in stage 3.'" in stats,
          "success message matches the pinned wording")
    # Task 6 gave create-project its own dedicated initials field
    # (#mx-train-by / getTrainBy / saveTrainBy, mirrored from but no longer
    # sharing #mx-by), so the comment and the two order checks below are
    # re-pinned to the identifiers createTrainingProject actually reads.
    # (The old getBy()/el.by()/saveBy() pins still returned PASS, but only
    # by coincidentally matching an unrelated later function in the same
    # file, buildComposite's own pre-flight, not createTrainingProject's -
    # _order() finds each needle anywhere after the previous one, and
    # buildComposite's "var by = getBy(); ... el.by.focus(); ... saveBy();"
    # sits later in the file. Re-pinned so the check demonstrates what it
    # claims to.)
    check(_order(stats, "function createTrainingProject(c, btn, note) {", "var by = getTrainBy();",
                 "el.trainBy.focus();", "saveTrainBy();"),
          "create-project reuses its own dedicated initials field (getTrainBy/saveTrainBy)")
    # Fix round 1: the shared #mx-stats-by/#mx-by field is looser (maxlength
    # 8, digits allowed, used by Build training set and the drawer's Use
    # these masks) than the live create-project route, which enforces
    # exactly 1-4 letters server-side (blueprint.py _INITIALS_RE) and 400s
    # otherwise. createTrainingProject pre-flights the same pattern
    # client-side against its own #mx-train-by field so a value that works
    # everywhere else on the tab does not fail only here, discovered
    # server-side; it never tightens the shared fields themselves.
    check("var TRAIN_INITIALS_RE = /^[A-Za-z]{1,4}$/;" in stats,
          "create-project pre-flights against the route's exact 1-4-letter pattern")
    check(_order(stats, "function createTrainingProject(c, btn, note) {", "var by = getTrainBy();",
                 "if (!by)", "if (!TRAIN_INITIALS_RE.test(by)) {", "saveTrainBy();", "postJson('/api/composite/create-project'"),
          "the pre-flight regex check runs after the empty check and before saveTrainBy()/the POST")
    check("Training projects need 1 to 4 letter initials (they go into the folder name)" in stats,
          "pre-flight failure shows the pinned inline message")
    check(_order(stats, "if (!TRAIN_INITIALS_RE.test(by)) {", "el.trainBy.focus();",
                 "'Training projects need 1 to 4 letter initials (they go into the folder name)'", "return;"),
          "a regex mismatch focuses the initials field and returns without posting")
    check("function getBy() { return (el.by.value || '').trim().toUpperCase().slice(0, 8); }" in stats,
          "the shared field itself keeps its own looser rule (maxlength 8, no letters-only trim)")
    # Re-pinned 2026-09-03 (Task 10): Task 6's rail replaced the old
    # always-both-buttons layout (a disabled primary + a small "create
    # another") with a real mutually exclusive "or" - the primary button is
    # not rendered at all once a project is linked, "Create another"
    # (capital C) stands alone in its place, so there is no disabled button
    # to pin any more.
    check("if (linked.length) {" in stats and "'Create another'" in stats and "'Create training project'" in stats,
          "Create training project is replaced by a smaller Create another once a project is linked")
    # Re-pinned 2026-09-03 (Task 10): the old single concatenated string
    # ("Training project already created: " + linked.join(', ')) is gone;
    # the same names now render as a dedicated .id span inside the row's
    # state pill (renderComposites).
    check("mk('span', { 'class': 'id' }, linked.join(', '))" in stats,
          "the linked project name(s) show in the row-state's .id span")

    # 3. Source / linked info on each composite row. sourceLine/linkedLine
    # and the old infoTd table cell (pre-Task-6 table markup) were inlined
    # into compositeMeta(c), the row-meta builder Task 6's li.row structure
    # actually calls; sourceLine survives in spirit as compositeMeta (same
    # job, source_projects -> a details.src disclosure), so it is re-pinned
    # rather than deleted. linkedLine does not survive as a separate named
    # function (linked_projects handling is inlined directly into
    # renderComposites' own row-state block, checked above), so its two
    # single-string-format pins ("built from: ", "training projects: " +
    # lp.join(', ')) are retired outright: the rail's li.row splits the
    # same information across compositeMeta's details.src summary ("N
    # source projects") and the row-state pill, and neither carries a
    # single prefixed-string literal to pin any more.
    check("function compositeMeta(c)" in stats and "c.source_projects" in stats,
          "compositeMeta (the row-meta builder that replaced sourceLine) reads source_projects")
    check("'sources recorded at next build'" in stats, "fallback text matches the pin when source_projects is absent")
    check("var linked = c.linked_projects || [];" in stats, "renderComposites reads linked_projects")
    check("row.appendChild(compositeMeta(c));" in stats,
          "each composite row renders its row-meta (compositeMeta), replacing the old infoTd table cell")

    # 4. Cache-bust: covered by test_template_markers (2026-09-03 stage-rail
    # rewrite retired the theme-unify string this used to repeat here).

    # 5. Cell drawer: "Feeds training sets" line, from an optional per-cell
    # field only (never a client-side guess against the composites list).
    check('id="mx-cell-composites"' in html, "drawer carries the training-sets line container")
    # Re-pinned 2026-09-03 (Task 10): class="mx-cell-actions" was renamed to
    # the shared class="actions" (Task 4 report, expected failure #6 - the
    # actions row now matches every other actions row on the page).
    check(_order(html, 'id="mx-cell-masks"', 'id="mx-cell-composites"', 'class="actions"'),
          "the training-sets line sits directly under the masks list")
    check("function renderTrainSets()" in cell, "matrix_cell.js has a dedicated render function")
    check("S.data && S.data.composites" in cell, "reads an optional composites field on the /api/cell payload")
    # "line text matches the pinned format" (the old 'In training sets: '
    # literal) retired outright 2026-09-03 (Task 10): Task 8 renamed the
    # line to "Feeds training sets: ", and that wording is already pinned
    # by test_task8_drawer's "feeds line wording" check, so this was a
    # duplicate assertion, not a distinct one.
    check("Per-cell membership cannot be derived from the /api/composites listing" in cell,
          "code comment records why this never guesses from the composites listing")
    check("renderTrainSets();" in cell and _order(cell, "renderTabs();", "renderProjects();", "renderTrainSets();"),
          "renderAll() calls renderTrainSets alongside the other per-cell renders")
    check(".mx-cell-composites" in css, "drawer line gets its own CSS rule")


# ── 6f. Task 6 (2026-09-03 stage rail): Training Sets rail ───────────
def test_task6_training_rail():
    stats_js = _read(JS_FILES["matrix_stats.js"])
    matrix_js = _read(JS_FILES["matrix.js"])
    tpl = _read(TEMPLATE)
    check("function showStage(" in stats_js and "window.MatrixStats" in stats_js and "showStage:" in stats_js, "stage switching exposed")
    check("'mx.stage'" in stats_js, "last stage remembered")
    check("function renderRail(" in stats_js and "sets built" in stats_js and "needs a project" in stats_js and "to train" in stats_js, "rail counts rendered")
    check("function renderTrainingProjects(" in stats_js and "linked_project_dirs" in stats_js and "project_dir=" in stats_js, "stage 3 lists training projects with Open in pipeline")
    check("Hand-drawn and model-predicted" in tpl, "third source option renamed")
    check("Build training set" in tpl and "'Create training project'" in stats_js and "'Create another'" in stats_js, "stage verbs")
    check("h === 'training-2'" in matrix_js and "h === 'training-3'" in matrix_js, "deep links to stages 2 and 3")
    check("mx-train-by" in stats_js, "stage 2 initials field mirrored")
    check("It starts at the Train step" not in stats_js, "old success copy gone")
    check("el.compResult.className = 'result'" in stats_js, "build result line resets to plain 'result' before a fresh attempt")


def test_categories_node():
    """Run the chart-category helpers from matrix_stats.js under node with the
    M2 fixture shape: unknown buckets join the parsed years/sites, last."""
    node = shutil.which("node")
    if not node:
        check(False, "node available for the categories check")
        return
    src = _read(JS_FILES["matrix_stats.js"])
    funcs = re.findall(r"^  function (?:numericThenText|unknownLast|categoriesFor)\([^)]*\) \{\n.*?\n  \}\n",
                       src, re.MULTILINE | re.DOTALL)
    check(len(funcs) == 3, f"extracted the three category helpers ({len(funcs)})")
    if len(funcs) != 3:
        return
    script = "".join(funcs) + """
var out = [
  categoriesFor([2014, 2015], { PA: { '2014': 2, 'unknown': 1, '2015': 1 } }),
  categoriesFor(['BLK', 'CRS'], { PA: { unknown: 1 }, OFAV: { ABC: 2 } }),
  categoriesFor([2016], {}),
  categoriesFor(null, null),
  ['unknown', '2015', 'x', '2014', '10'].sort(unknownLast)
];
process.stdout.write(JSON.stringify(out));
"""
    fd, path = tempfile.mkstemp(prefix="mx_cats_", suffix=".js")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(script)
        r = subprocess.run([node, path], capture_output=True, text=True)
    finally:
        os.remove(path)
    check(not os.path.exists(path), "categories node script removed")
    check(r.returncode == 0, f"node ran the category helpers: {r.stderr.strip()[:200] or 'ok'}")
    if r.returncode != 0:
        return
    out = json.loads(r.stdout)
    check(out[0] == ["2014", "2015", "unknown"], f"year categories include unknown last: {out[0]}")
    check(out[1] == ["ABC", "BLK", "CRS", "unknown"], f"site categories merge by_site keys, unknown last: {out[1]}")
    check(out[2] == ["2016"], f"no by_* keys keeps the parsed list: {out[2]}")
    check(out[3] == [], f"null inputs give no category: {out[3]}")
    check(out[4] == ["10", "2014", "2015", "x", "unknown"], f"unknownLast orders numbers, text, unknown: {out[4]}")


# ── 7. render through the blueprint ──────────────────────────────────
def test_render_via_blueprint():
    try:
        from flask import Flask
    except Exception as e:  # pragma: no cover
        check(False, f"flask importable: {e}")
        return
    from _matrix import make_blueprint

    root = tempfile.mkdtemp(prefix="mx_ui_root_")
    try:
        app = Flask(__name__)
        # open_folder_fn stubbed (2026-09-02 hotfix): the blueprint's
        # default_open_folder spawns the real xdg-open, and this suite keeps
        # subprocess.run live for the node checks, so it cannot block Popen.
        app.register_blueprint(
            make_blueprint(lambda: {"inprocess_root": root}, log_fn=lambda m: None,
                           open_folder_fn=lambda p: (False, "host opening stubbed in tests")),
            url_prefix="/matrix",
        )
        c = app.test_client()
        r = c.get("/matrix/")
        check(r.status_code == 200, f"GET /matrix/ -> {r.status_code}")
        body = r.data.decode("utf-8")
        for i in ("mx-tab-matrix", "mx-tab-stats", "mx-tab-training", "mx-cell", "mx-cell-canvas", "mx-by",
                  "mx-view-stats", "mx-view-training", "mx-chart-year", "mx-composite-build", "mx-purge",
                  "mx-composite-projects", "mx-cell-composites"):
            check(f'id="{i}"' in body, f"rendered page has id={i}")
        check('window.MATRIX_BASE = "/matrix"' in body, "MATRIX_BASE renders as the mount prefix")
        rendered_canvas = re.search(r"<canvas\b[^>]*>", body)
        check(rendered_canvas is not None and "title=" not in rendered_canvas.group(0),
              "rendered canvas carries no native title (L8)")
        for src in ("matrix_cell.js", "matrix_stats.js", "matrix.css"):
            rs = c.get(f"/matrix/static/{src}")
            check(rs.status_code == 200, f"GET /matrix/static/{src} -> {rs.status_code}")
        rd = c.get("/matrix/api/data")
        check(rd.status_code == 200, f"GET /matrix/api/data (empty root) -> {rd.status_code}")
        # Routes owned by M may or may not be live yet; the frontend copes with
        # either, so only report which ones exist here.
        rules = {str(x) for x in app.url_map.iter_rules()}
        for want in ("/matrix/api/cell", "/matrix/api/stats", "/matrix/api/preferences",
                     "/matrix/api/composites", "/matrix/api/open-folder"):
            print(f"[INFO] route {want}: {'present' if want in rules else 'absent (backend pending)'}")
        # When the backend routes are live, confirm the error shapes the drawer
        # and stats view rely on (JSON bodies with an error key).
        if "/matrix/api/cell" in rules:
            rc = c.get("/matrix/api/cell?image=no_such_image&label=PA")
            check(rc.status_code == 404, f"unknown cell -> 404 ({rc.status_code})")
            check(rc.is_json and "error" in (rc.get_json() or {}), "unknown cell 404 body is JSON with error")
        if "/matrix/api/stats" in rules:
            rs2 = c.get("/matrix/api/stats?status=accepted")
            check(rs2.status_code == 200 and rs2.is_json, f"stats on an empty root -> {rs2.status_code} JSON")
            j = rs2.get_json() or {}
            for k in ("totals", "species", "years", "sites", "by_year", "by_site",
                      "images_by_year", "images_by_site", "reviewers", "excluded"):
                check(k in j, f"stats JSON carries {k}")
        if "/matrix/api/open-folder" in rules:
            ro = c.post("/matrix/api/open-folder", json={"project_id": "nope"})
            check(ro.status_code in (400, 403, 404, 500) and ro.is_json and "error" in (ro.get_json() or {}),
                  f"open-folder on an unknown project -> error JSON ({ro.status_code})")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        check(not os.path.exists(root), "temp inprocess root removed")


# ── 8. Task 7: Matrix tab (legend, filters, grid heads, name helpers) ──
def test_task7_matrix_tab():
    main = _read(JS_FILES["matrix.js"])
    check("window.MatrixNames" in main and "function whereFromStem(" in main,
          "name helpers exposed")
    check("TCRMP(\\d{8})_clip_([A-Za-z0-9]+)_T(\\d+)" in main, "basename parser")
    check("lbl-stack" in main and "id code" in main,
          "column heads carry full name and code chip")
    check("Reviewers have looked for " in main, "column head tooltip from the mockup")
    check("The full image name is " in main, "row tooltip from the mockup")
    check("(ringed" not in main and "(dimmed" not in main,
          "isolate note carries counts only")
    check("Show or hide the " not in main, "label column checkboxes carry no tooltip")
    check("of reviewed cells found" in main, "headline wording kept")
    # I1 (2026-09-03 whole-batch review): a #cell= deep link selects the
    # Matrix view before opening the drawer, so a Stats or Training tab
    # remembered in localStorage cannot hide it (the drawer sits inside
    # #mx-view-matrix).
    m = re.search(r"function applyHash\(\) \{(.*?)\n  \}", main, re.DOTALL)
    check(m is not None, "applyHash() function found")
    if m:
        check(_order(m.group(1), "setView('matrix', false)", "MatrixCell.open("),
              "the #cell= branch calls setView('matrix', false) before MatrixCell.open(")


# ── 9. Task 8: cell drawer (subtitle, species, masks, projects, feeds) ──
def test_task8_drawer():
    cell = _read(JS_FILES["matrix_cell.js"])
    tpl = _read(TEMPLATE)
    check("mx-cell-where" in cell and "window.MatrixNames.where(" in cell,
          "drawer subtitle resolved from the stem")
    check("mx-cell-species" in cell and "window.MatrixNames.label(" in cell, "species name line")
    check("label.htmlFor" in cell or "'for'" in cell, "mask tick boxes bound to visible labels")
    check("Feeds training sets: " in cell, "feeds line wording")
    check("Use the ticked masks" in cell and "Clear the saved choice" in cell,
          "action tooltips match verbs")
    check("el.close.focus()" in cell and "returnTo" in cell and "document.contains(back)" in cell,
          "focus contract kept")
    check("function inPoly" in cell and "hitTest" in cell and "getContext('2d')" in cell,
          "canvas viewer kept")
    check("Hover a mask for its uid, status, source, reviewer and area." in tpl,
          "caption text kept (visually hidden)")
    check("mk('label', { 'class': 'mx-cell-mask'" not in cell and
          "mk('div', { 'class': 'mx-cell-mask'" in cell,
          "mask row is a div, not a nested label (single labelling relationship)")
    check("mx-train-by" in cell, "saveBy mirrors initials into the training-project stage field too")
    check("mx-btn" not in cell, "no leftover mx-btn/mx-btn-mini classes (drawer buttons use btn/btn-mini)")


# ── 9b. Task 7b: attention first, clear filters, copy as CSV (registry
# spec v2, sections 3 and 12) ─────────────────────────────────────────
def test_task7b_attention():
    tpl = _read(TEMPLATE)
    matrix_js = _read(JS_FILES["matrix.js"])
    check('value="needs_attention"' in tpl and 'id="mx-attention"' in tpl and 'id="mx-clear-filters"' in tpl and 'id="mx-copy-csv"' in tpl,
          "attention count, needs-attention filter, clear filters and copy as CSV present")
    check("function attentionCount(" in matrix_js and "need attention" in matrix_js, "attention count rendered")
    check("function clearFilters(" in matrix_js and "function copyCsv(" in matrix_js and "Copied " in matrix_js, "clear filters and copy as CSV wired")
    check("'needs_attention'" in matrix_js and "pending_expert" in matrix_js, "needs attention maps to pending_expert")
    m = re.search(r"function copyCsv\(\) \{(.*?)\n  \}\n", matrix_js, re.DOTALL)
    check(m is not None, "copyCsv() function found")
    if m:
        body = m.group(1)
        check("cellAt(" in body, "copyCsv resolves the true outcome via cellAt() (fix round 1)")
        check("dataset.outcome" not in body, "copyCsv no longer reads dataset.outcome (fix round 1)")


# ── 10. Task 9 (2026-09-02 stage-rail): Stats tab charts ─────────────
def test_task9_stats_tab():
    stats_js = _read(JS_FILES["matrix_stats.js"])
    css = _read(CSS_FILE)
    check("SLOTS = ['#3987e5', '#d95926', '#199e70'" in stats_js, "species hues per the spec lead the palette")
    check("function hbar(" in stats_js and "ylabs" in stats_js and "class=\"yl\"" in stats_js, "horizontal charts carry an HTML label column")
    check("preserveAspectRatio" in stats_js or "width=\"100%\"" in stats_js, "svg scales to its column")
    check("Table view" in stats_js and "twrap" in stats_js, "table views scroll in their own wrapper")
    check("function renderExcluded(" in stats_js and "masks with no label" in stats_js, "Left out figure rendered")
    check("mx-chart-tip" not in stats_js, "custom hover tip replaced by title bubbles")
    check(" has " in stats_js and " of the " in stats_js, "bar tooltips are sentences")
    check("reviewers recorded" in stats_js and "unrecorded" in stats_js, "totals and reviewer wording")
    check("function fmtAst(" in stats_js and "' AST'" in stats_js and "toISOString" not in stats_js, "AST stamps kept")
    # Fix round 1 (2026-09-03, reviewer): chart text had no font-size (it
    # inherited the 13px body size); grid/axis strokes were hex constants.
    check(".mx-svg text" in css, "page css sizes every chart text element")
    # Re-pinned 2026-09-17 (T-matrix-nudge, spec v2 legibility floor): chart
    # text moved from 10px to 12px on 2026-09-16, and the last 10px rule on
    # the page (the Tick all / Untick all buttons) was raised to 12px, so the
    # old "font-size: 10px" substring pin would now match nothing. The pin
    # reads the .mx-svg text block itself and guards the floor page-wide.
    svg_text_block = css.split(".mx-svg text {", 1)[1].split("}", 1)[0] if ".mx-svg text {" in css else ""
    check("font-size: 12px" in svg_text_block, "chart text pinned to 12px (spec v2 floor)")
    check("font-size: 10px" not in css and "font-size: 11px" not in css and "11px/" not in css and "10px/" not in css,
          "no page rule sizes text below the 12px floor (spec v2 section 3)")
    check('class="grid"' in stats_js, "grid lines carry class=\"grid\", not a hex stroke")
    check("GRID" not in stats_js, "the old GRID hex constant is gone")


def main():
    test_node_check()
    test_template_markers()
    test_copy_rules()
    test_js_tooltips()
    test_banned_strings()
    test_no_inline_project_name()
    test_contract_routes()
    test_review_fixes()
    test_quality_sweep_task5()
    test_task5_page_css()
    test_typography()
    test_task_i_composite_linkage()
    test_task6_training_rail()
    test_categories_node()
    test_render_via_blueprint()
    test_task7_matrix_tab()
    test_task7b_attention()
    test_task8_drawer()
    test_task9_stats_tab()
    _cleanup()
    check(not os.path.exists(_TMP_REG), "temp registry dir removed")
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED")
        for f in _FAILURES:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS")


# 2026-09-17: design spec version 2 legibility rules the matrix now carries (LO).
_css_v2 = open(CSS_FILE, encoding="utf-8").read()
check("--row-wash: color-mix(in srgb, var(--text) 7%, transparent)" in _css_v2, "matrix.css names the row wash token")
check("--row-hover: color-mix(in srgb, var(--text) 13%, transparent)" in _css_v2, "matrix.css names the row hover token")
check(".page .result, .page .row-state { font-size: 13px; }" in _css_v2, "result and state lines sit at the 13 px explanatory floor")
check('.t tbody tr:nth-child(even) td' in _css_v2, "the grid stripes its even rows")

if __name__ == "__main__":
    main()
