"""
Task T10 of the 2026-09-16 VEGA student-ready build: the pipeline window's step
panels, converted to the VICARIUS workflow page design.

Design authority: docs/superpowers/specs/2026-09-02-workflow-page-design.md
(tokens in section 2, the type scale in section 3, the copy rules in section 6,
the tooltip rules in section 7, the accessibility floor in section 8). Its
section 10 names "the pipeline orchestrator's step panels" as the surface this
suite guards.

House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_workflow_design_conformance.py

The page is rendered through Jinja with stand-in variables (the same names
app.py's index() passes) so this suite never imports app.py, which another task
may be editing. It renders the page twice, once as a segmentation project and
once as a detection project, because the panels now carry copy that is written
per family.

Covers:
  SHARED BLOCK  - the shared stylesheet and the shared tooltip script are
                  linked, both pinned with the asset version, and the fonts
                  come from exactly the link the spec names.
  PLACEHOLDERS  - no internal step-directory token renders on screen, and no
                  internal step key or output-directory name reaches the reader.
  TYPE SCALE    - version 2 of the design: field labels read 14 px, stage
                  headings 17 px, and no declaration anywhere on the page sets
                  type below the 12 px legibility floor. No field label, chip,
                  column head or status line is uppercased; uppercase is left
                  to the window title, the tabs and the buttons.
  CONTRAST      - hint prose reads --text-dim and never a hardcoded grey.
  DEFINITION    - every choice is a definition row: a fixed label column with
    ROWS          the label and its "?" glyph, the control beside it.
  STAGE NUMBER  - a panel heading is numbered off the rail this project shows,
                  in the shape static/orchestrator.js renumbers.
  FAMILY COPY   - a detection project is never offered mask guidance or a mask
                  default.
  ROW STRIPING  - the lists longer than three rows stripe and wash on hover.
  LABELS        - every form control the reader can operate carries a visible
                  label bound to it, by for= or by a wrapping label.
  TOOLTIPS      - every field has one explanation, not two that can drift: the
                  glyph beside a label and the control it explains say the same
                  thing, and no tooltip opens with "You".
  PROSE         - no emoji, em dash, en dash, the banned word, or "UVI" in the
                  template or in the page stylesheet.
  ID CONTRACT   - every element id static/orchestrator.js reads still exists in
                  the template.
  RELICS        - the removed Expert Review surface leaves nothing behind.
"""

import os
import re
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
_TPL = os.path.join(_ORCH, "templates", "index.html")
_CSS = os.path.join(_ORCH, "static", "style.css")
_SHARED_CSS = os.path.join(_ORCH, "static", "vicarius_workflow.css")
_SHARED_JS = os.path.join(_ORCH, "static", "vicarius_tooltips.js")
_JS = os.path.join(_ORCH, "static", "orchestrator.js")

ASSET_STAMP = "conformance-stamp"

EM = "—"
EN = "–"
DASH_ENTITIES = ("&mdash;", "&ndash;", "&#8212;", "&#8211;", "&#x2014;", "&#x2013;")

# The font link the spec's section 2 names, verbatim.
FONT_LINK = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
             "&family=Space+Mono:wght@400;700&family=Syne:wght@600;700;800&display=swap")

# Emoji and pictographs. U+2605 BLACK STAR is excepted: it is the time-cost
# marker beside the Train panel's settings and is written as an entity.
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-☄☆-⛿✀-➿]")

# Controls whose accessible name comes from something other than a <label>,
# with the reason. Everything else on the page must have a bound label.
LABEL_EXEMPT = {
    "s3-species": "hidden store read by collectConfig(), never shown",
    "s6-device-custom": "revealed only when Device reads Custom; the Device label names it",
    "s8-device-custom": "revealed only when Device reads Custom; the Device label names it",
}

_fail = 0
_RESULTS = []


def check(cond, msg):
    """Raise the message as an assertion when the condition is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one check function, record its verdict, and keep going."""
    global _fail
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _fail += 1
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ── sources ─────────────────────────────────────────────────────────
_CACHE = {}


def _read(path):
    """The file at path, cached for the run."""
    if path not in _CACHE:
        with open(path, encoding="utf-8") as fh:
            _CACHE[path] = fh.read()
    return _CACHE[path]


def _render(project_state):
    """templates/index.html rendered through Jinja with stand-ins for every
    variable app.py's index() passes.

    Parameters:
        project_state: the project payload the page branches its copy on; {}
            renders the page as a segmentation project.
    Returns:
        The rendered HTML as a string.
    Example:
        _render({"training_source": {"task_default": "detect"}})
    """
    key = ("render", repr(project_state))
    if key not in _CACHE:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(
            loader=FileSystemLoader(os.path.join(_ORCH, "templates")),
            autoescape=True,
        )
        env.globals["url_for"] = lambda endpoint, **kw: f"/stub/{endpoint}/{kw.get('filename', '')}"
        tpl = env.get_template("index.html")
        _CACHE[key] = tpl.render(
            asset_version=ASSET_STAMP,
            repo_root="/stub/repo",
            projects_dir="/stub/projects",
            supporting_data_dir="/stub/supporting_data",
            all_points_csv="/stub/all_points.csv",
            master_codes_csv="/stub/master_codes.csv",
            project_loaded=False,
            project_state=project_state,
        )
    return _CACHE[key]


def _seg():
    """The page as a segmentation project sees it."""
    return _render({})


def _detect():
    """The page as a detection project sees it."""
    return _render({"training_source": {"task_default": "detect"}})


def _box():
    """The page as a project minted from a staged box dataset sees it.

    This is the family whose rail drops the four annotation stages, so it is
    the only one whose stage numbers differ from the internal step keys.
    """
    return _render({
        "is_training_project": True,
        "project_type": "bbox_training",
        "training_source": {"kind": "bbox", "task_default": "detect"},
    })


def _both():
    """Both renderings, named, for a check that must hold on each."""
    return (("segmentation", _seg()), ("detection", _detect()))


def _tags(html, name):
    """Every opening tag of one element name in the HTML, as strings."""
    return re.findall(r"<" + name + r"\b[^>]*>", html, re.I)


def _attr(tag, name):
    """One attribute's value from an opening tag, or None when it is absent."""
    m = re.search(r'\b' + name + r'="([^"]*)"', tag)
    return m.group(1) if m else None


# ── SHARED BLOCK ────────────────────────────────────────────────────
def test_shared_stylesheet_and_tooltip_script_are_linked():
    """The two files the spec calls the shared block in executable form are
    linked by the page, not merely served beside it."""
    tpl = _read(_TPL)
    check('href="/static/vicarius_workflow.css?v={{ asset_version }}"' in tpl,
          "index.html must link the shared stylesheet with the asset version")
    check('src="/static/vicarius_tooltips.js?v={{ asset_version }}"' in tpl,
          "index.html must load the shared tooltip script with the asset version")
    check(os.path.isfile(_SHARED_CSS), "static/vicarius_workflow.css missing on disk")
    check(os.path.isfile(_SHARED_JS), "static/vicarius_tooltips.js missing on disk")
    html = _seg()
    check(f"/static/vicarius_workflow.css?v={ASSET_STAMP}" in html,
          "the shared stylesheet link does not render")
    check(f"/static/vicarius_tooltips.js?v={ASSET_STAMP}" in html,
          "the shared tooltip script tag does not render")
    # The shared block loads first so the page's own rules win on conflict,
    # and the bubble script loads before the page scripts that build markup.
    check(tpl.index("vicarius_workflow.css") < tpl.index("/static/style.css"),
          "the shared stylesheet must load before style.css")
    check(tpl.index("vicarius_tooltips.js") < tpl.index("/static/orchestrator.js"),
          "the tooltip script must load before orchestrator.js")


def test_fonts_are_the_link_the_spec_names():
    """Section 2 of the spec gives one font link; the page uses it verbatim."""
    tpl = _read(_TPL)
    check(FONT_LINK in tpl, "the font link must be exactly the one the spec names")
    check("Space+Grotesk" not in tpl, "Space Grotesk is not in the spec's font link")


def test_style_css_does_not_redeclare_the_shared_palette():
    """The shared block owns the palette. A second copy in style.css is two
    sources of truth for one colour."""
    css = _read(_CSS)
    for token in ("--magenta:", "--bg:", "--surface:", "--text:", "--success:",
                  "--border:", "--radius:", "--radius-lg:"):
        check(token not in css,
              f"style.css redeclares {token}, which the shared block already provides")
    check("--magenta-light:" in css,
          "style.css must keep --magenta-light, the one accent the shared block lacks")


# ── PLACEHOLDERS ────────────────────────────────────────────────────
def test_no_step_directory_placeholder_renders():
    """{step6_dir}, {step7_dir} and {step8_dir} were never substituted and
    printed with their braces, each naming a step number one higher than the
    heading above it (decision D1)."""
    for name, html in _both():
        check("{step" not in html, f"{name}: a literal step-directory token renders")
    tpl = _read(_TPL)
    check("{step" not in tpl, "a literal step-directory token survives in the template")


def test_internal_step_keys_never_reach_the_reader():
    """The internal output-directory names are the code's vocabulary, not the
    reader's; the panels name the step by the number on its own tile."""
    for name, html in _both():
        text = re.sub(r"<!--.*?-->", "", html, flags=re.S)
        for token in ("step6_dir", "step7_dir", "step8_dir",
                      "step6_trainModel", "step7_eval", "step8_inference"):
            check(token not in text, f"{name}: internal token {token} reaches the page")


# ── TYPE SCALE ──────────────────────────────────────────────────────
def test_no_field_label_is_uppercase():
    """Uppercase is reserved for the window title, the tabs and the buttons
    (spec section 3). Every other label reads in sentence case."""
    css = _read(_CSS)
    allowed = (".topbar h1", ".btn", ".setup-container .logo h1")
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = m.group(1).strip(), m.group(2)
        if "text-transform: uppercase" not in body:
            continue
        check(any(a in selector for a in allowed),
              f"style.css uppercases {selector!r}, which is not the title, a tab or a button")
    for name, html in _both():
        check("text-transform:uppercase" not in html.replace(" ", ""),
              f"{name}: the template still uppercases something inline")


def test_labels_read_the_body_font_at_the_spec_size():
    """Field label is Inter 500 14 px (spec section 3, version 2), not the
    13 px of version 1 and not the Space Mono the spec assigns to buttons."""
    css = _read(_CSS)
    i = css.index(".field-label")
    rule = css[css.rindex("\n.", 0, i):css.index("}", i)]
    check("var(--font-body)" in rule, "the field label must read the body font token")
    check("font-size: 14px" in rule, "the field label must be 14 px")
    check("font-weight: 500" in rule, "the field label must be weight 500")


def test_the_stage_heading_reads_the_display_font_at_17px():
    """Section or stage heading is Syne 700 17 px (spec section 3, version 2)."""
    css = _read(_CSS)
    i = css.index(".card h2 {")
    rule = css[i:css.index("}", i)]
    check("var(--font-display)" in rule, "the stage heading must read the display font")
    check("font-size: 17px" in rule, "the stage heading must be 17 px")


def _font_sizes(text, where):
    """Every type size a stylesheet or a style attribute sets, in pixels.

    Only the size is read: the first length in a `font:` shorthand and the
    length in a `font-size:`. A line height or a family is not a type size.

    Parameters:
        text: CSS source, or HTML carrying style attributes.
        where: the name used in a failure message.
    Returns:
        A list of (pixels, snippet) pairs.
    """
    out = []
    for m in re.finditer(r"font(?:-size)?\s*:\s*([^;{}\"]+)", text):
        value = m.group(1)
        px = re.search(r"(\d+(?:\.\d+)?)px", value)
        if px:
            out.append((float(px.group(1)), f"{where}: {value.strip()[:60]}"))
    return out


def test_no_type_is_below_the_legibility_floor():
    """Version 2 of the design sets a floor: no user-visible text below 12 px.
    Twenty-nine hint spans on this page were 10 px, which is the single
    biggest reason Lauren read the panels as hard to read."""
    sizes = _font_sizes(_read(_CSS), "style.css")
    tpl = _read(_TPL)
    sizes += _font_sizes(tpl[tpl.index("<style>"):tpl.index("</style>")], "the page style block")
    for attr in re.findall(r'style="([^"]*)"', tpl):
        sizes += _font_sizes(attr, "an inline style attribute")
    small = [s for s in sizes if s[0] < 12]
    check(len(sizes) > 40, f"parsed too few type sizes ({len(sizes)})")
    check(not small, f"{len(small)} declarations set type under 12 px: {small[:6]}")


def test_hint_prose_reads_the_dim_ink_token():
    """A hint never carries a hardcoded grey. #666 measures 3.03:1 on
    --surface-light and fails the 4.5:1 floor; --text-dim holds 8.0:1."""
    css = _read(_CSS)
    i = css.index(".hint {")
    rule = css[i:css.index("}", i)]
    check("var(--text-dim)" in rule, "the hint line must read --text-dim")
    # A colour literal, never a character entity: "&#9998;" carries "#999"
    # inside it and is the pencil glyph on the Edit Masks tile.
    grey_re = re.compile(r"(?<![&\w])#(?:666|777|888|999)(?![0-9a-fA-F])", re.I)
    for name, text in (("style.css", css), ("index.html", _read(_TPL))):
        m = grey_re.search(text)
        check(m is None,
              f"{name} hardcodes the grey {m.group(0) if m else ''} instead of --text-dim")


# ── DEFINITION ROWS ─────────────────────────────────────────────────
def test_the_choice_row_is_a_definition_row():
    """Spec section 5.2: a fixed label column holding the label and its "?"
    glyph, the control beside it, hairlines between rows, rows at least 46 px
    tall plus the hairline."""
    css = _read(_CSS)
    i = css.index(".form-group,\n.checkbox-group {")
    rule = css[i:css.index("}", i)]
    check("display: grid" in rule, "a choice row must be a grid")
    check("var(--label-w)" in rule, "the label column must read the --label-w token")
    check("border-top: 1px solid var(--border-dim)" in rule,
          "rows are separated by --border-dim hairlines")
    check("min-height: 47px" in rule, "a row is 46 px tall plus its hairline")
    check(".form-group > .field-head" in css,
          "the label line must be placed in the label column")


def test_every_choice_row_leads_with_its_label_line():
    """A row whose label line is missing leaves the label column empty and its
    control floating, which is how Fill opacity came to sit above an unrelated
    row."""
    # The one row that is a disclosure rather than a choice: the multi-GPU
    # explainer, which sits in the Device row's control column on purpose.
    allowed_without_a_label = 1
    for name, html in _both():
        bodies = re.findall(r'<div class="form-group"[^>]*>(.{0,600}?)<', html, re.S)
        headless = [b for b in re.findall(
            r'<div class="form-group"[^>]*>\s*(<[^>]+>)', html) if "field-head" not in b]
        check(len(headless) <= allowed_without_a_label,
              f"{name}: {len(headless)} choice rows have no label line: {headless[:4]}")
        check(bodies, f"{name}: no choice rows parsed")


def test_every_field_label_carries_one_glyph():
    """Every label line on the page carries the "?" or "i" glyph that holds
    its explanation, because the page keeps no explanation on screen."""
    for name, html in _both():
        heads = re.findall(r'<div class="field-head"[^>]*>(.*?)</div>', html, re.S)
        check(len(heads) > 60, f"{name}: parsed too few label lines ({len(heads)})")
        bare = [re.sub(r"<[^>]+>", " ", h).strip()[:40]
                for h in heads if 'class="tip"' not in h]
        check(not bare, f"{name}: {len(bare)} label lines carry no glyph: {bare[:5]}")
        for head in heads:
            m = re.search(r'<button type="button" class="tip"[^>]*title="([^"]*)"', head)
            check(m is not None and m.group(1).strip(),
                  f"{name}: a label line's glyph carries no explanation")


# ── STAGE NUMBER ────────────────────────────────────────────────────
def test_panel_headings_are_numbered_off_the_rail():
    """A bounding-box project shows three stages, so its Model Inference panel
    is stage 3 and not "Step 7". The heading ships the shape
    static/orchestrator.js renumbers: the word, the number alone inside
    span.stage-num, then the colon and the stage name."""
    detect = _box()
    seg = _seg()
    for key, name, short, full in ((6, "Train Model", 1, 5),
                                   (7, "Evaluate Model", 2, 6),
                                   (8, "Model Inference", 3, 7)):
        want_short = f'<h2>Step <span class="stage-num">{short}</span>: {name}</h2>'
        want_full = f'<h2>Step <span class="stage-num">{full}</span>: {name}</h2>'
        check(want_short in detect,
              f"a detection project's panel {key} heading must read {want_short}")
        check(want_full in seg,
              f"a segmentation project's panel {key} heading must read {want_full}")
    # The rail node paints the same number before any script runs.
    for want in ('data-num="1">1', 'data-num="2">2', 'data-num="3">3'):
        check(want in detect, f"a detection project's rail must paint {want}")
    check("Step 7: Model Inference" not in detect,
          "the old fixed step number survives in the detection heading")


def test_the_stage_number_element_holds_only_the_number():
    """publishStageNumber() in static/orchestrator.js writes the number into
    span.stage-num, so anything else parked there is thrown away on load."""
    for name, html in _both() + (("box dataset", _box()),):
        for body in re.findall(r'<span class="stage-num">(.*?)</span>', html):
            check(body.strip().isdigit(),
                  f"{name}: span.stage-num holds {body!r}, which is not a number")


# ── FAMILY COPY ─────────────────────────────────────────────────────
def test_a_detector_is_never_offered_mask_guidance():
    """The fill-opacity control shipped one sentence to both families, so a
    box-dataset project was told what a segmentation run reads well at and
    was given a mask default it can never use."""
    detect, seg = _detect(), _seg()
    def tip(html, element_id):
        m = re.search(r'id="' + element_id + r'"[^>]*title="([^"]*)"', html)
        check(m is not None, f"no tooltip found on #{element_id}")
        return m.group(1)
    fill = tip(detect, "s8-mask-alpha")
    check("box" in fill.lower(), "the detector's fill tooltip must name the box it fills")
    check("segmentation" not in fill.lower(),
          f"a detector is offered segmentation guidance: {fill}")
    check("0.45" not in fill, f"a detector is offered the mask default: {fill}")
    check("0.15" in fill, f"the detector's own default must be stated: {fill}")
    seg_fill = tip(seg, "s8-mask-alpha")
    check("0.45" in seg_fill, f"a segmentation project keeps its own default: {seg_fill}")
    check("box" in seg_fill.lower(), "the fill tooltip still says it fills boxes too")
    check("detector" in tip(detect, "s8-draw-boxes").lower(),
          "the draw-boxes tooltip must say what it does on a detector")


# ── ROW STRIPING ────────────────────────────────────────────────────
def test_row_lists_stripe_and_wash_on_hover():
    """Version 2 of the design: a list longer than three rows carries visible
    striping and a hover wash, so the eye can track a row across its columns.
    The rounds table read as 18 rows of small mono text with neither."""
    css = _read(_CSS)
    for selector in ("#s7-rounds-table tbody tr:nth-child(even)",
                     "#s7-rounds-table tbody tr:hover",
                     ".remap-table tbody tr:nth-child(even)",
                     ".remap-table tbody tr:hover"):
        check(selector in css, f"style.css must stripe or wash {selector}")


def test_the_time_cost_chips_agree_with_each_other():
    """The chip read 'free' beside one label and 'FREE' beside another, because
    one sat inside a rule that uppercased it and the other did not."""
    for name, html in _both():
        chips = re.findall(r'<span class="cost[^"]*"[^>]*>([^<]*)</span>', html)
        check(chips, f"{name}: no time-cost chips found")
        for chip in chips:
            words = re.sub(r"&#\d+;", "", chip).strip()
            check(words == words.lower(),
                  f"{name}: time-cost chip {words!r} is not sentence case")
        # Each chip is focusable, so the scale is reachable without a mouse.
        for tag in re.findall(r'<span class="cost[^>]*>', html):
            check('tabindex="0"' in tag, f"{name}: a time-cost chip is not focusable: {tag}")


def test_the_time_cost_scale_is_defined_on_the_train_heading():
    """The scale used to exist only as a hover-only title on an unfocusable
    span. It is now stated once, on the Train panel's heading glyph."""
    html = _seg()
    i = html.index('id="panel-6"')
    panel = html[i:html.index('id="panel-', i + 20)]
    head = panel[:panel.index("</div>", panel.index('class="panel-head"'))]
    m = re.search(r'title="([^"]*)"', head)
    check(m is not None, "the Train panel heading needs an information glyph")
    tip = m.group(1).lower()
    check("star" in tip, "the heading glyph must define what a star beside a setting means")


# ── LABELS ──────────────────────────────────────────────────────────
def _controls_and_labels(html):
    """Every operable form control in the page body, and the ids <label for=>
    points at.

    Returns:
        (controls, bound_ids, wrapped_ids): the list of (tag, id) pairs, the
        set of ids a label binds to, and the set of ids wrapped by a label.
    """
    body = html[html.index("<body"):]
    controls = []
    for name in ("input", "select", "textarea"):
        for tag in _tags(body, name):
            if _attr(tag, "type") in ("hidden",):
                continue
            controls.append((tag, _attr(tag, "id")))
    bound = {_attr(t, "for") for t in _tags(body, "label") if _attr(t, "for")}
    wrapped = set()
    for m in re.finditer(r"<label\b[^>]*>(.*?)</label>", body, re.S):
        for tag in re.findall(r"<(?:input|select|textarea)\b[^>]*>", m.group(1)):
            cid = _attr(tag, "id")
            if cid:
                wrapped.add(cid)
            name = _attr(tag, "name")
            if name:
                wrapped.add("name:" + name)
    return controls, bound, wrapped


def test_every_form_control_has_a_bound_label():
    """Spec section 8: every input has a visible label, bound with for= or
    wrapping the control. 79 controls on this page had neither, so clicking
    the label did nothing and a screen reader read the tooltip as the name."""
    for name, html in _both():
        controls, bound, wrapped = _controls_and_labels(html)
        check(len(controls) > 50, f"{name}: parsed too few controls ({len(controls)})")
        unbound = []
        for tag, cid in controls:
            if cid in LABEL_EXEMPT:
                continue
            radio_name = _attr(tag, "name")
            if cid and (cid in bound or cid in wrapped):
                continue
            if radio_name and ("name:" + radio_name) in wrapped:
                continue
            unbound.append(cid or tag[:70])
        check(not unbound, f"{name}: {len(unbound)} controls have no bound label: {unbound[:8]}")


def test_every_label_points_at_a_control_that_exists():
    """A for= naming an id the page does not carry is a label bound to
    nothing, which reads as bound and behaves as unbound."""
    for name, html in _both():
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        for tag in _tags(html, "label"):
            target = _attr(tag, "for")
            if target:
                check(target in ids, f"{name}: <label for={target!r}> names no element")


# ── TOOLTIPS ────────────────────────────────────────────────────────
def test_one_explanation_per_field():
    """A field's glyph and the control it explains must carry the same
    sentence. Two different tooltips on one field (the Cosine schedule
    checkbox carried two) give the reader two answers to one question."""
    for name, html in _both():
        for block in re.findall(r'<div class="form-group"[^>]*>(.*?)</div>\s*</div>', html, re.S):
            tips = {t for t in re.findall(r'title="([^"]+)"', block)}
            # Buttons inside a group (Browse, Refresh) legitimately differ, so
            # only the label glyph and the control it names are compared.
            glyph = re.search(
                r'<button type="button" class="tip"[^>]*title="([^"]+)"[^>]*>'
                r'<span aria-hidden="true">\?</span>', block)
            ctrl = re.search(r'<(?:input|select|textarea)\b[^>]*title="([^"]+)"', block)
            if glyph and ctrl:
                check(glyph.group(1) == ctrl.group(1),
                      f"{name}: a field's glyph and its control disagree:\n"
                      f"  glyph: {glyph.group(1)[:90]}\n  field: {ctrl.group(1)[:90]}")
            check(len(tips) < 40, f"{name}: implausible tooltip count in one group")


def _own_tooltips():
    """Every tooltip string this template owns.

    The rendered page also carries the three card partials, whose tooltips
    belong to their own suites, so the prose rules are read from the template
    source: its literal title= attributes, plus the sentences it sets once into
    a template variable and hands to both a glyph and the control it explains.

    Returns:
        A list of tooltip strings.
    """
    tpl = _read(_TPL)
    tips = [s for s in re.findall(r'title="([^"]*)"', tpl) if "{{" not in s and s.strip()]
    tips += re.findall(r'\{%-? set t = "([^"]+)" -?%\}', tpl)
    check(len(tips) > 80, f"parsed too few tooltips from the template ({len(tips)})")
    return tips


def test_no_tooltip_opens_with_you():
    """Tooltip rule 2: a tooltip never opens with 'You'."""
    for tip in _own_tooltips():
        check(not tip.strip().startswith("You "),
              f"a tooltip opens with You: {tip[:80]!r}")


def test_every_tooltip_is_complete_sentences():
    """Tooltip rule 1 and rule 6: complete sentences, no fragments."""
    for tip in _own_tooltips():
        text = tip.strip()
        check(text.endswith(".") or text.endswith("?"),
              f"a tooltip is not a complete sentence: {text[:80]!r}")
        check(text[0].isupper() or text[0].isdigit(),
              f"a tooltip does not open with a capital: {text[:80]!r}")


def test_no_tooltip_names_an_internal_route_or_library_keyword():
    """A reader gains nothing from a Flask route or an ultralytics keyword;
    spec section 6 keeps internal vocabulary off the screen. The Step 3
    selector flag is the one exception a neighbouring suite pins."""
    # Two selector flags a neighbouring suite pins by name
    # (test_step3_site_controls.py, test_ui_sweep_20260901.py). Both sentences
    # explain the setting in plain words first and name the flag last.
    allowed = ("--target", "--max-site-frac", "--no-site-balance")
    for tip in _own_tooltips():
        check("/api/" not in tip, f"a tooltip names a route: {tip[:80]!r}")
        check("ultralytics `" not in tip,
              f"a tooltip names an ultralytics keyword: {tip[:80]!r}")
        for flag in re.findall(r"(--[a-z][a-z-]+)", tip):
            check(flag in allowed, f"a tooltip names the flag {flag}: {tip[:80]!r}")


# ── PROSE ───────────────────────────────────────────────────────────
def test_no_banned_word_or_character():
    """No emoji, no em dash, no en dash, never the word land, never UVI."""
    sources = {"templates/index.html": _read(_TPL), "static/style.css": _read(_CSS)}
    for name, html in _both():
        sources[f"rendered page ({name})"] = html
    for where, text in sources.items():
        check(EM not in text, f"{where}: em dash")
        check(EN not in text, f"{where}: en dash")
        for ent in DASH_ENTITIES:
            check(ent.lower() not in text.lower(), f"{where}: dash entity {ent}")
        bad = EMOJI_RE.search(text)
        check(bad is None, f"{where}: emoji at " + (f"{bad.start()}: {bad.group(0)!r}" if bad else ""))
        m = re.search(r"\bland(s|ed|ing)?\b", text, re.I)
        check(m is None, f"{where}: the banned word at " + (f"{m.start()}: {m.group(0)!r}" if m else ""))
        check("UVI" not in text, f"{where}: the string UVI")


def test_the_window_names_its_author():
    """Every VICARIUS window Lauren builds names her in small text at the
    bottom of the page."""
    for name, html in _both():
        check("Lauren Olinger" in html, f"{name}: the page footer must name its author")
        check('class="page-foot"' in html, f"{name}: the page footer is missing")


def test_no_paragraph_of_explanation_is_rendered_on_screen():
    """Spec section 6: no paragraph on screen, every explanation a tooltip.
    The step panels opened with four-paragraph descriptions and carried about
    31 grey help paragraphs under the fields."""
    for name, html in _both():
        check('class="step-description"' not in html,
              f"{name}: a step-description block survives")
        body = html[html.index('<div id="app-screen"'):]
        # Prose outside a collapsed <details> is what the rule forbids.
        visible = re.sub(r"<details\b.*?</details>", "", body, flags=re.S)
        visible = re.sub(r"<!--.*?-->", "", visible, flags=re.S)
        for m in re.finditer(r"<p\b[^>]*>(.*?)</p>", visible, re.S):
            words = len(re.sub(r"<[^>]+>", "", m.group(1)).split())
            check(words <= 30,
                  f"{name}: a {words}-word paragraph renders on screen: {m.group(1)[:80]!r}")


def test_no_paragraph_element_in_a_panel_body():
    """Spec section 6, read strictly: a step panel carries no paragraph
    element at all. A state line the scripts write is a one-line div, not a
    <p>, so nothing on the page can grow into a wall of text again. The card
    partials the page includes answer to their own suites."""
    tpl = _read(_TPL)
    body = tpl[tpl.index('<div id="app-screen"'):]
    visible = re.sub(r"<details\b.*?</details>", "", body, flags=re.S)
    visible = re.sub(r"<!--.*?-->", "", visible, flags=re.S)
    paragraphs = re.findall(r"<p\b[^>]*>", visible)
    check(not paragraphs,
          f"{len(paragraphs)} paragraph elements survive in the panels: {paragraphs[:4]}")


def test_no_hardcoded_off_token_colour_in_the_template():
    """The template was the only orchestrator file hardcoding colour: 24 hex
    literals in five off-token values, three of them near-misses of --success,
    --warning and --error."""
    tpl = _read(_TPL)
    hexes = set(re.findall(r"(?<![&#\d])#[0-9a-fA-F]{6}\b", tpl))
    check(not hexes, f"index.html hardcodes colour: {sorted(hexes)}")


# ── ID CONTRACT ─────────────────────────────────────────────────────
def test_every_id_the_scripts_read_still_exists():
    """The id contract of this build: the markup may be restructured, but no
    id static/*.js reads may be renamed or removed."""
    html = _seg()
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    # The three cards and the annotator boot overlay mint their own ids at
    # runtime, so an id the scripts create is not one the template owes.
    minted = {"boot-bar", "boot-hint", "boot-msg", "boot-retry", "boot-title",
              "orch-down-banner", "s4test-route-warnings", "vicarius-bubble"}
    pattern = re.compile(
        r"""getElementById\(\s*['"]([^'"]+)['"]"""
        r"""|querySelector(?:All)?\(\s*['"]#([A-Za-z0-9_-]+)"""
        r"""|\$\(\s*['"]([A-Za-z0-9_-]+)['"]\s*\)""")
    missing = []
    for fname in sorted(os.listdir(os.path.join(_ORCH, "static"))):
        if not fname.endswith(".js"):
            continue
        if fname.startswith("vega_tour"):
            # The tour engine builds ids from prefixes ('panel-' + n); its
            # anchors are pinned against every template by test_vega_tour_content.py.
            continue
        src = _read(os.path.join(_ORCH, "static", fname))
        for m in pattern.finditer(src):
            wanted = m.group(1) or m.group(2) or m.group(3)
            if wanted in minted or wanted in ids:
                continue
            # The card partials own their own ids; only ids nothing renders fail.
            missing.append(f"{fname}:{wanted}")
    check(not missing, f"ids the scripts read but the page no longer has: {sorted(set(missing))}")


# ── RELICS ──────────────────────────────────────────────────────────
def test_the_expert_review_surface_left_nothing_behind():
    """The Expert Review I/O tile was permanently inert while its panel, its
    stylesheet and its script still shipped on every page load. The relic
    sweep rule takes the markup, the assets and the tile together."""
    tpl = _read(_TPL)
    for relic in ('data-step-disabled="expertids"', 'id="panel-expertids"',
                  "expertids.css", "expertids.js", "EXPERTIDS_BASE",
                  "expertids/panel.html", 'id="nav-footer-expertids"'):
        check(relic not in tpl, f"Expert Review relic survives in the template: {relic}")
    css = _read(_CSS)
    check("step-disabled" not in css and "step-disabled" not in tpl,
          "the disabled-tile rule has no tile left to style")


def test_dead_page_css_is_gone():
    """The CSS-only tooltip popover and the inline information circle were
    replaced by the shared block's glyph and bubble."""
    tpl = _read(_TPL)
    for relic in ("has-tip", "info-circle", "picker-hint", "step-description"):
        check(relic not in tpl, f"dead class {relic} survives in the template")
    css = _read(_CSS)
    for relic in (".has-tip", ".info-circle", ".step-description"):
        check(relic not in css, f"dead rule {relic} survives in style.css")


# ── main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("VEGA workflow-design conformance (T10)\n")
    print("SHARED BLOCK:")
    run(test_shared_stylesheet_and_tooltip_script_are_linked)
    run(test_fonts_are_the_link_the_spec_names)
    run(test_style_css_does_not_redeclare_the_shared_palette)
    print("PLACEHOLDERS:")
    run(test_no_step_directory_placeholder_renders)
    run(test_internal_step_keys_never_reach_the_reader)
    print("TYPE SCALE:")
    run(test_no_field_label_is_uppercase)
    run(test_labels_read_the_body_font_at_the_spec_size)
    run(test_the_stage_heading_reads_the_display_font_at_17px)
    run(test_no_type_is_below_the_legibility_floor)
    print("CONTRAST:")
    run(test_hint_prose_reads_the_dim_ink_token)
    print("DEFINITION ROWS:")
    run(test_the_choice_row_is_a_definition_row)
    run(test_every_choice_row_leads_with_its_label_line)
    run(test_every_field_label_carries_one_glyph)
    print("STAGE NUMBER:")
    run(test_panel_headings_are_numbered_off_the_rail)
    run(test_the_stage_number_element_holds_only_the_number)
    print("FAMILY COPY:")
    run(test_a_detector_is_never_offered_mask_guidance)
    print("ROW STRIPING:")
    run(test_row_lists_stripe_and_wash_on_hover)
    run(test_the_time_cost_chips_agree_with_each_other)
    run(test_the_time_cost_scale_is_defined_on_the_train_heading)
    print("LABELS:")
    run(test_every_form_control_has_a_bound_label)
    run(test_every_label_points_at_a_control_that_exists)
    print("TOOLTIPS:")
    run(test_one_explanation_per_field)
    run(test_no_tooltip_opens_with_you)
    run(test_every_tooltip_is_complete_sentences)
    run(test_no_tooltip_names_an_internal_route_or_library_keyword)
    print("PROSE:")
    run(test_no_banned_word_or_character)
    run(test_the_window_names_its_author)
    run(test_no_paragraph_of_explanation_is_rendered_on_screen)
    run(test_no_paragraph_element_in_a_panel_body)
    run(test_no_hardcoded_off_token_colour_in_the_template)
    print("ID CONTRACT:")
    run(test_every_id_the_scripts_read_still_exists)
    print("RELICS:")
    run(test_the_expert_review_surface_left_nothing_behind)
    run(test_dead_page_css_is_gone)

    total = len(_RESULTS)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{passed}/{total} passed.")
    if _fail:
        print("\nFAILURES:")
        for fn, ok, msg in _RESULTS:
            if not ok:
                print(f"\n--- {fn} ---\n{msg}")
    sys.exit(1 if _fail else 0)
