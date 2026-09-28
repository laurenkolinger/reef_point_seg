"""
T11 of the 2026-09-04 VEGA fold-in: the Step 6 (internal step 7) evaluation
report card. House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step7_report_card.py

Covers:
  FRAGMENT  - templates/_step7_report.html renders through jinja2 with no
              variables; the root class and the card id appear exactly once;
              every control is a keyboard-focusable glyph carrying a title;
              no inline script, style or event handler; the hooks the loader
              needs (the body container, the title id) exist.
  PROSE     - no em or en dash, no emoji, never the banned word or the
              banned initialism in the fragment, the stylesheet, the script or either
              test; every tooltip (fragment and script) is one or two
              complete sentences, never opens with "You", never points at
              the screen.
  SCOPE     - every selector in orchestrator_report.css starts with
              .vega-report (inside media blocks too); the script exposes the
              contract, carries a JSDoc block on every function and has no
              console or alert calls.
  RENDER    - the plan's two named behaviours, driven through node against a
              fixture: a preview image link renders as an img, and a figure
              whose url is null is not rendered.
"""

import json
import os
import re
import subprocess
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_TEMPLATES = os.path.join(_ORCH, "templates")
_STATIC = os.path.join(_ORCH, "static")
_FRAGMENT = os.path.join(_TEMPLATES, "_step7_report.html")
_CSS = os.path.join(_STATIC, "orchestrator_report.css")
_JS = os.path.join(_STATIC, "orchestrator_report.js")
_NODE_SUITE = os.path.join(_HERE, "test_vega_report.js")
_NODE = "/usr/bin/node"
_NODE_TIMEOUT_S = 30

ROOT_CLASS = "vega-report"
CARD_ID = "s7-report-card"
EM_DASH = "\u2014"
EN_DASH = "\u2013"
BANNED_WORD_RE = re.compile(r"\bl" + r"and(?:s|ed|ing)?\b", re.IGNORECASE)
BANNED_INITIALISM = "U" + "VI"
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
SCREEN_REFERENCES = ("above", "below", "this tab", "this window", "this page")
CONTROL_TAG_RE = re.compile(r"<(button|a|input|select|textarea|summary)\b[^>]*>", re.IGNORECASE)
GLYPH_RE = re.compile(r'<button type="button" class="tip" title="([^"]*)" aria-label="([^"]*)"><span>([?i])</span></button>')
JS_FUNCTION_RE = re.compile(r"^\s*(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE)
JS_ARROW_RE = re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>", re.MULTILINE)
RUN_DIR = "/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess/_models/lab_a/runs/run_a"

_fail = 0
_RESULTS = []


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, record its outcome and print a PASS or FAIL line."""
    global _fail
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001
        _fail += 1
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _read(path):
    """Return the text of path, failing the test with a clear message when it is missing."""
    check(os.path.isfile(path), f"missing deliverable: {path}")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _render_fragment():
    """Render the fragment through jinja2 with StrictUndefined and no variables."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(_TEMPLATES),
        undefined=jinja2.StrictUndefined,
        autoescape=True,
    )
    return env.get_template(os.path.basename(_FRAGMENT)).render()


def _node(script):
    """Run a short node script against the report module and return its stdout."""
    check(os.path.isfile(_NODE), f"node not found at {_NODE}")
    proc = subprocess.run(
        [_NODE, "-e", script],
        cwd=_STATIC,
        capture_output=True,
        text=True,
        timeout=_NODE_TIMEOUT_S,
    )
    check(proc.returncode == 0, f"node exited {proc.returncode}: {proc.stderr.strip()[:800]}")
    return proc.stdout


def _render_with_node(fixture, options=None):
    """Render a fixture through VegaReport.render in node and return the HTML."""
    script = (
        f"const R = require({json.dumps(_JS)});"
        f"process.stdout.write(R.render({json.dumps(fixture)}, {json.dumps(options or {})}));"
    )
    return _node(script)


def _image_url(rel):
    """The image route shape from the wave-2 API contract."""
    from urllib.parse import quote

    return f"/api/step/7/image?run_dir={quote(RUN_DIR, safe='')}&rel={quote(rel, safe='')}"


def _fixture():
    """A detector evaluation in the exact shape of GET /api/step/7/report."""
    return {
        "ok": True,
        "run_name": "run_a",
        "run_dir": RUN_DIR,
        "out_dir": RUN_DIR + "/eval",
        "evaluated_at": "2026-09-03T20:49:29-04:00",
        "report_md": (
            "# Evaluation report for run `run_a`\n\n"
            "## Prediction samples (1 images)\n\n"
            "- **Acropora-palmata_881.jpg** drew 1 instance; the model produced 2 detections.\n"
            "  ![Acropora-palmata_881.jpg](previews/Acropora-palmata_881_sbs.jpg)\n"
        ),
        "metrics": {
            "overall": {
                "box_map50": 0.2088, "box_map50_95": 0.1769, "box_precision": 0.3346, "box_recall": 0.2512,
                "mask_map50": None, "mask_map50_95": None, "mask_precision": None, "mask_recall": None,
            },
            "per_class": {
                "Acropora-palmata": {"box_precision": 0.7727, "box_recall": 0.5862, "box_map50": 0.5459, "box_map50_95": 0.4382},
            },
            "per_class_instances": {"Acropora-palmata": 35, "Agaricia-undata": 0},
            "absent_classes": ["Agaricia-undata"],
            "metric_family": "box",
            "class_names": {"0": "Acropora-palmata", "1": "Agaricia-undata"},
        },
        "previews": [
            {"filename": "Acropora-palmata_881.jpg", "url": _image_url("previews/Acropora-palmata_881_sbs.jpg"), "n_gt": 1, "n_pred": 2},
        ],
        "figures": {
            "box_pr": _image_url("val_run/BoxPR_curve.png"),
            "box_f1": None,
            "mask_pr": None,
            "mask_f1": None,
            "confusion_norm": _image_url("val_run/confusion_matrix_normalized.png"),
            "confusion_abs": None,
            "results": None,
        },
        "pdf_url": None,
    }


def _prose_problems(text, label):
    """Return the house prose violations found in text, each prefixed with label."""
    problems = []
    if EM_DASH in text:
        problems.append(f"{label}: em dash")
    if EN_DASH in text:
        problems.append(f"{label}: en dash")
    if EMOJI_RE.search(text):
        problems.append(f"{label}: emoji")
    m = BANNED_WORD_RE.search(text)
    if m:
        problems.append(f"{label}: banned word at {text[max(0, m.start() - 30):m.end() + 30]!r}")
    if BANNED_INITIALISM in text:
        problems.append(f"{label}: banned initialism")
    return problems


def _tooltip_problems(text):
    """Return the tooltip sentence-rule violations for one title string."""
    problems = []
    t = text.strip()
    if not t:
        return ["empty tooltip"]
    if t.startswith("You"):
        problems.append("opens with You")
    if not t.endswith("."):
        problems.append("does not end as a complete sentence")
    sentences = [s for s in SENTENCE_SPLIT_RE.split(t) if s.strip()]
    if not 1 <= len(sentences) <= 2:
        problems.append(f"{len(sentences)} sentences, want 1 or 2")
    lowered = t.lower()
    for ref in SCREEN_REFERENCES:
        if re.search(r"\b" + re.escape(ref) + r"\b", lowered):
            problems.append(f"screen reference {ref!r}")
    return [f"{p}: {t[:70]!r}" for p in problems]


def _css_selectors(css):
    """Return every selector list in css, media blocks included, comments stripped."""
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    selectors = []
    stack = []
    buf = ""
    for ch in body:
        if ch == "{":
            head = buf.strip()
            buf = ""
            if head.startswith("@"):
                stack.append("at")
            else:
                selectors.append(head)
                stack.append("rule")
        elif ch == "}":
            buf = ""
            if stack:
                stack.pop()
        else:
            buf += ch
    check(not stack, "unbalanced braces in the stylesheet")
    return selectors


# ── FRAGMENT ────────────────────────────────────────────────────────


def test_fragment_renders_with_no_variables():
    html = _render_fragment()
    check(html.strip(), "fragment renders to something")
    check("{{" not in html and "{%" not in html, "no jinja syntax survives rendering")


def test_root_class_and_card_id_exactly_once():
    html = _render_fragment()
    check(html.count(f'class="{ROOT_CLASS}"') == 1, "root class exactly once")
    check(html.count(f'id="{CARD_ID}"') == 1, "card id exactly once")
    check(re.search(r'<section class="vega-report" id="s7-report-card"', html) is not None,
          "the root is a section carrying both the class and the id")


def test_loader_hooks_exist():
    html = _render_fragment()
    check(html.count('class="vega-report-body"') == 1, "one body container for VegaReport.load to fill")
    check('id="s7-report-title"' in html and 'aria-labelledby="s7-report-title"' in html,
          "the section is labelled by its title")
    check('<h3 class="vega-report-title" id="s7-report-title">Evaluation report</h3>' in html,
          "title text is exactly Evaluation report")
    check(html.count('class="vega-report-status"') == 1, "one status line for the loading, empty and error sentences")
    check('<div class="vega-report-status" role="status" aria-live="polite">' in html, "the status line is the live region")
    check(re.search(r'<section[^>]*aria-live', html) is None, "the whole section is not a live region")


def test_every_control_is_a_focusable_glyph_with_a_title():
    html = _render_fragment()
    controls = CONTROL_TAG_RE.findall(html)
    glyphs = GLYPH_RE.findall(html)
    check(len(controls) >= 1, "the fragment carries at least the i glyph")
    check(len(glyphs) == len(controls),
          f"every control is a glyph button with title and aria-label ({len(glyphs)} glyphs, {len(controls)} controls)")
    check(html.count('title="') == len(glyphs), "no title sits anywhere but on a glyph")
    check(any(kind == "i" for _, _, kind in glyphs), "an i glyph explains the card")
    check("tabindex" not in html, "glyphs are buttons, so no tabindex is needed")
    for title, aria, _ in glyphs:
        check(aria.startswith("About "), f"aria-label names the subject: {aria!r}")
        check(len(title) > 40, f"glyph title explains, not restates: {title!r}")


def test_no_inline_script_style_or_handlers():
    html = _render_fragment()
    check("<script" not in html, "no inline script")
    check("<style" not in html, "no inline style block")
    check(re.search(r"\son[a-z]+=", html, re.IGNORECASE) is None, "no inline event handler")
    check('style="' not in html, "no inline style attribute; the stylesheet owns the look")


def test_fragment_terms_of_art_on_the_i_glyph():
    html = _render_fragment()
    title = [t for t, _, kind in GLYPH_RE.findall(html) if kind == "i"][0]
    for term in ("precision", "recall", "mAP50", "mAP50-95", "IoU", "split"):
        check(term in title, f"the i glyph explains {term}")


# ── PROSE ────────────────────────────────────────────────────────────


def test_no_banned_prose_in_deliverables():
    problems = []
    for path in (_FRAGMENT, _CSS, _JS, _NODE_SUITE, os.path.abspath(__file__)):
        problems += _prose_problems(_read(path), os.path.basename(path))
    check(not problems, "; ".join(problems))


def test_fragment_tooltips_follow_the_sentence_rules():
    html = _render_fragment()
    problems = []
    for title in re.findall(r'title="([^"]*)"', html):
        problems += _tooltip_problems(title)
    check(not problems, "; ".join(problems))


def test_js_tooltips_follow_the_sentence_rules():
    raw = _node(f"const R = require({json.dumps(_JS)}); process.stdout.write(JSON.stringify(R.TOOLTIPS));")
    tips = json.loads(raw)
    flat = []

    def walk(node):
        """Flatten nested tooltip objects into a list of strings."""
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        else:
            flat.append(str(node))

    walk(tips)
    check(len(flat) >= 9, f"the script carries the pdf, figure, full-text and six column tooltips ({len(flat)})")
    problems = []
    for t in flat:
        problems += _tooltip_problems(t)
        problems += _prose_problems(t, "tooltip")
    check(not problems, "; ".join(problems))
    for tip in flat:
        check(tip[0].isalpha(), f"tooltip opens with a word, not a symbol: {tip[:60]!r}")


# ── SCOPE ────────────────────────────────────────────────────────────


def test_css_is_scoped_under_the_root_class():
    css = _read(_CSS)
    selectors = _css_selectors(css)
    check(selectors, "the stylesheet has rules")
    bad = []
    for group in selectors:
        for sel in group.split(","):
            s = sel.strip()
            if not s.startswith("." + ROOT_CLASS):
                bad.append(s)
    check(not bad, f"unscoped selectors: {bad}")
    check("@import" not in css and "url(" not in css, "no external fetches from the stylesheet")
    for token in ("--surface-light", "--text-dim", "--border", "--font-mono", "--pink"):
        check(f"var({token})" in css, f"stylesheet reads the page token {token}")
    check("prefers-reduced-motion" in css, "transitions are wrapped for reduced motion")
    check(":focus-visible" in css, "focus is visible")


def test_js_exposes_the_contract():
    raw = _node(
        f"const R = require({json.dumps(_JS)});"
        "process.stdout.write(JSON.stringify(['render','markdownToHtml','load','loadStep7ReportCard']"
        ".map(n => typeof R[n])));"
    )
    check(json.loads(raw) == ["function"] * 4, f"contract exports are functions: {raw}")
    js = _read(_JS)
    check("root.VegaReport = api" in js, "window.VegaReport is set")
    check("root.loadStep7ReportCard = api.loadStep7ReportCard" in js, "window.loadStep7ReportCard is set for the panel wiring")
    check('"use strict"' in js, "strict mode")


def test_every_js_function_has_a_docstring():
    js = _read(_JS)
    lines = js.split("\n")
    missing = []
    for regex in (JS_FUNCTION_RE, JS_ARROW_RE):
        for m in regex.finditer(js):
            line_no = js[:m.start()].count("\n")
            prev = line_no - 1
            while prev >= 0 and not lines[prev].strip():
                prev -= 1
            if prev < 0 or not lines[prev].rstrip().endswith("*/"):
                missing.append(f"{m.group(1)} (line {line_no + 1})")
    check(not missing, f"functions without a JSDoc block: {missing}")


def test_js_has_no_debug_or_dead_calls():
    js = _read(_JS)
    check("console." not in js, "no console calls")
    check("alert(" not in js, "no alert calls")
    check("TODO" not in js and "FIXME" not in js, "no placeholders")
    check("document.write" not in js and "eval(" not in js, "no document.write or eval")


def test_node_suite_exists_beside_this_one():
    check(os.path.isfile(_NODE_SUITE), f"node suite present at {_NODE_SUITE}")
    src = _read(_NODE_SUITE)
    check(src.startswith("#!/usr/bin/env node"), "node suite carries the node shebang")
    for name in ("test_markdown_escapes_html_before_rendering", "test_markdown_renders_tables_lists_headings",
                 "test_image_rule_rewrites_previews_and_drops_others",
                 "test_per_class_table_shows_full_name_and_chip_and_absent_rows", "test_family_aware_top_line",
                 "test_figures_with_null_urls_are_omitted", "test_sort_by_column_is_stable",
                 "test_empty_and_error_states", "test_pdf_link_only_when_present"):
        check(name in src, f"node suite carries {name}")


# ── RENDER (through node) ───────────────────────────────────────────


def test_report_card_renders_an_image_link_as_an_img():
    html = _render_with_node(_fixture())
    check("![" not in html, "no literal markdown image syntax on the page")
    preview = _image_url("previews/Acropora-palmata_881_sbs.jpg").replace("&", "&amp;")
    check(f'<img class="vega-report-mdimg" src="{preview}"' in html,
          "the report text's preview link renders as an img pointing at the image route")
    check(f'<img src="{preview}" alt="Acropora-palmata_881.jpg" loading="lazy">' in html,
          "the previews entry renders as an img")
    check("the human labeler drew 1 instance; the model produced 2 detections" in html, "preview caption")
    check("(s)" not in html, "no caption falls back to the (s) plural construct")
    check("evaluated 2026-09-03 20:49 AST" in html, "AST timestamp in the heading")
    check("No PDF was built for this evaluation." in html, "absent PDF sentence")
    check("N/A" not in html, "never N/A")
    check('<td><em class="vega-report-binomial">Acropora palmata</em></td>' in html,
          "full species name, no chip for a class name that is itself the binomial")
    check("not present in this split" in html, "absent species row")


def test_report_card_hides_missing_figures():
    html = _render_with_node(_fixture())
    check(html.count('<figure class="vega-report-fig">') == 2, "only the two figures with a url render")
    check("Precision-recall curve, boxes" in html and "Confusion matrix, normalized" in html, "present figures captioned")
    for caption in ("F1 against confidence, boxes", "Precision-recall curve, outlines", "F1 against confidence, outlines",
                    "Confusion matrix, counts", "Training curves by epoch"):
        check(caption not in html, f"null figure not rendered: {caption}")
    check("placeholder" not in html.lower(), "no placeholder for a missing figure")


if __name__ == "__main__":
    print("FRAGMENT:")
    run(test_fragment_renders_with_no_variables)
    run(test_root_class_and_card_id_exactly_once)
    run(test_loader_hooks_exist)
    run(test_every_control_is_a_focusable_glyph_with_a_title)
    run(test_no_inline_script_style_or_handlers)
    run(test_fragment_terms_of_art_on_the_i_glyph)
    print("PROSE:")
    run(test_no_banned_prose_in_deliverables)
    run(test_fragment_tooltips_follow_the_sentence_rules)
    run(test_js_tooltips_follow_the_sentence_rules)
    print("SCOPE:")
    run(test_css_is_scoped_under_the_root_class)
    run(test_js_exposes_the_contract)
    run(test_every_js_function_has_a_docstring)
    run(test_js_has_no_debug_or_dead_calls)
    run(test_node_suite_exists_beside_this_one)
    print("RENDER:")
    run(test_report_card_renders_an_image_link_as_an_img)
    run(test_report_card_hides_missing_figures)

    print(f"\n{len(_RESULTS) - _fail}/{len(_RESULTS)} passed.")
    if _fail:
        print("\nFAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"--- {name} ---\n{detail}")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
