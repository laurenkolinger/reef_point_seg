"""Prior-project masks draw as a read-only echo and never enter this project's
own state.

Task: VEGA annotator tidy-up and no re-screening, 2026-09-27 (LO). Lauren:
"If an image has been previously screened for any target classes, do include
it [in Step 3 selection] ... and include any masks as previews in the
annotator for the present project. When you mouse over that mask you can see
which project it came from. I don't want people retreading each other's
work." The contract (docs/superpowers/plans/2026-09-27-vega-annotator-tidy-
and-no-rescreening.md) hands this page currentData.prior_masks (a list of
{label, name, polygon_px, project_id, project_name, reviewer, at, outcome},
already in this image's pixel space) and currentData.prior_screened ({label:
{outcome, project_id, project_name, reviewer, at}}). app.py (a separate task,
W3) fills those fields in; this suite injects them into currentData directly,
exactly as the plan for this task says to.

Covers:
  1. render() draws prior masks (drawPriorMasks) between the reference points
     and this project's own masks, so an operator's live work always paints
     over another project's echo, never under it.
  2. drawPriorMasks fills every prior mask at a flat 10% alpha in one fixed
     colour (2026-09-27 pixel review, defect 2: a per-label hue at low
     contrast read as nearly invisible against real coral), strokes a dark
     underlay then a bright dash on top for the ordinary case, and a single
     solid line for whichever prior mask is under the pointer.
  3. A prior mask is never selectable: findMaskAt (which drives every click,
     refine, merge and box path) never sees currentData.prior_masks, even
     when a prior polygon sits exactly where the operator clicks.
  4. A prior mask never enters undo, the mask list, saveCurrentMasks (which
     builds the PUT that persists this project's own statuses) or export:
     "prior_masks" appears in this template only in the functions this task
     added to read it.
  5. Hovering a prior mask, with no own mask under the pointer, shows a
     floating tooltip naming the label, its full species name, the source
     project, the reviewer and the date; hovering elsewhere, or hovering an
     own mask that overlaps it, shows nothing.
  6. The "Already screened here" note fills in from currentData.prior_screened
     with the full name, the code, the source project and the date for every
     label, its tooltip names the outcome and who, and its slot holds one
     fixed height whether or not there is anything to say.

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_prior_mask_render.py
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

CHROME = "/usr/bin/google-chrome"
PROBE_TIMEOUT_S = 30
CHROME_ARGS = ["--headless", "--disable-gpu", "--hide-scrollbars", "--no-sandbox"]
_MARK_START = "@@PRIOR_MASK_PROBE_START@@"
_MARK_END = "@@PRIOR_MASK_PROBE_END@@"

# A prior mask from another project, in the shape app.py's contract promises:
# a small square, 40..80 in both axes, already scaled to this image's pixels.
_PRIOR_MASK = {
    "label": "PA", "name": "Porites astreoides",
    "polygon_px": [[40, 40, 80, 40, 80, 80, 40, 80]],
    "project_id": "run_26-09-18_AT_PA_750img", "project_name": "AT_PA_750img_26-09-18",
    "reviewer": "AT", "at": "2026-09-25T09:12:00-04:00", "outcome": "found_manual",
}
_PRIOR_SCREENED = {
    "PA": {"outcome": "found_manual", "project_id": "run_26-09-18_AT_PA_750img",
           "project_name": "AT_PA_750img_26-09-18", "reviewer": "AT",
           "at": "2026-09-25T09:12:00-04:00"},
}

FAILURES = []


def check(cond, name):
    """Record one assertion and print its verdict."""
    print("  %s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILURES.append(name)


def _template_source():
    with open(_TPL, encoding="utf-8") as handle:
        return handle.read()


def _extract_function(name, src=None):
    """Slice one top-level JavaScript function verbatim out of the template.

    Same convention as test_annotator_copy.py's helper of the same name: every
    function in this template closes with a brace in column one.
    """
    src = src if src is not None else _template_source()
    marker = "\nfunction %s(" % name
    if marker not in src:
        marker = "\nasync function %s(" % name
    start = src.index(marker) + 1
    end = src.index("\n}\n", start) + len("\n}\n")
    return src[start:end]


def _render(**overrides):
    labels = [
        {"code": "PA", "name": "Porites astreoides", "category": "Coral"},
        {"code": "BALL", "name": "Ball sponge", "category": "Sponge"},
    ]
    ctx = dict(
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
        orchestrator_url="", orchestrated=False,
        session_mode="configure", review_locked_msg="locked",
        review_tagging_enabled=False, default_session_name="26-09-27_LO_demo",
        session_label="", session_label_tooltip="",
    )
    ctx.update(overrides)
    env = Environment(loader=FileSystemLoader(_TPL_DIR))
    return env.get_template("index.html").render(**ctx)


def _run_probe(probe_js, width=1280, height=900, **render_kwargs):
    """Render the page with `probe_js` appended and return its JSON result.

    Mirrors test_annotator_layout_stability.py's helper of the same name.
    """
    page = _render(**render_kwargs)
    marker = "</body>"
    if marker not in page:
        raise RuntimeError("rendered page has no </body> to inject the probe before")
    page = page.replace(marker, probe_js + "\n" + marker)
    fd, path = tempfile.mkstemp(suffix=".html", prefix="prior_mask_probe_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(page)
        result = subprocess.run(
            [CHROME, *CHROME_ARGS, "--window-size=%d,%d" % (width, height),
             "--dump-dom", "file://" + path],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
        )
    finally:
        os.unlink(path)
    dom = result.stdout
    if _MARK_START not in dom or _MARK_END not in dom:
        raise RuntimeError(
            "the probe never reported back (chrome rc=%s); stderr tail: %r"
            % (result.returncode, result.stderr[-2000:]))
    start = dom.rindex(_MARK_START) + len(_MARK_START)
    end = dom.index(_MARK_END, start)
    return json.loads(dom[start:end])


# ── 1. render() draws prior masks between reference points and own masks ────

def test_render_draws_prior_masks_under_own_masks():
    """drawPriorMasks sits after drawReferencePoints, before the own-mask loop."""
    src = _template_source()
    render_fn = _extract_function("render", src)
    ref_at = render_fn.index("drawReferencePoints(ctx);")
    prior_at = render_fn.index("drawPriorMasks(ctx);")
    masks_at = render_fn.index("const masks = currentData.masks || [];")
    check(ref_at < prior_at < masks_at,
          "render() draws reference points, then prior masks, then its own masks")


# ── 2. Fill and stroke style ─────────────────────────────────────────────────

def _draw_style_probe_js(prior_masks, hovered_index=None):
    """Render drawPriorMasks against a fake ctx that records every fill and
    stroke call: its alpha, style, line width and dash pattern at the moment
    the call fires. hovered_index, if given, sets _hoveredPriorMask to that
    entry of prior_masks before drawing, exercising the hover-highlight path.
    """
    prior = json.dumps(prior_masks)
    hover_setup = (
        "_hoveredPriorMask = currentData.prior_masks[%d];" % hovered_index
        if hovered_index is not None else "_hoveredPriorMask = null;"
    )
    return """
<script>
(function() {
  showScreen('review');
  currentData = {masks: [], prior_masks: %s};
  scale = 1;
  %s
  let dash = [], fillStyle = null, strokeStyle = null, lineWidth = null, alpha = 1;
  const calls = {fills: [], strokes: []};
  const fakeCtx = {
    save() {}, restore() {}, beginPath() {}, closePath() {}, moveTo() {}, lineTo() {},
    fill() { calls.fills.push({alpha, style: fillStyle}); },
    stroke() { calls.strokes.push({alpha, style: strokeStyle, width: lineWidth, dash: dash.slice()}); },
    setLineDash(d) { dash = d; },
    set globalAlpha(v) { alpha = v; }, get globalAlpha() { return alpha; },
    set fillStyle(v) { fillStyle = v; }, get fillStyle() { return fillStyle; },
    set strokeStyle(v) { strokeStyle = v; }, get strokeStyle() { return strokeStyle; },
    set lineWidth(v) { lineWidth = v; }, get lineWidth() { return lineWidth; },
  };
  drawPriorMasks(fakeCtx);
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(calls) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (prior, hover_setup, _MARK_START, _MARK_END)


def test_draw_prior_masks_fills_faintly_and_strokes_a_dark_underlay_then_a_bright_dash():
    """Not hovered: one 10%%-alpha fill, then a dark underlay stroke (3.5px,
    dashed) and a bright stroke on top (2px, same dash), both in the one
    fixed prior-mask colour, never a per-label hue (2026-09-27 pixel review,
    defect 2)."""
    result = _run_probe(_draw_style_probe_js([_PRIOR_MASK]))
    check(len(result["fills"]) == 1, "exactly one fill call for one prior mask")
    fill = result["fills"][0]
    check(abs(fill["alpha"] - 0.10) < 1e-9,
          "the fill alpha is a flat 10%%, got %r" % fill["alpha"])
    check(fill["style"] == "#F5EDF0",
          "the fill uses the one fixed prior-mask colour, got %r" % fill["style"])
    check(len(result["strokes"]) == 2,
          "two strokes for a non-hovered prior mask (dark underlay, bright dash), "
          "got %d" % len(result["strokes"]))
    underlay, bright = result["strokes"]
    check(underlay["style"] == "rgba(0, 0, 0, 0.7)",
          "the underlay is a dark, semi-opaque outline, got %r" % underlay["style"])
    check(abs(underlay["width"] - 3.5) < 1e-9,
          "the underlay is 3.5 canvas units wide at scale 1, got %r" % underlay["width"])
    check(bright["style"] == "#F5EDF0",
          "the top stroke uses the one fixed prior-mask colour, got %r" % bright["style"])
    check(abs(bright["width"] - 2.0) < 1e-9,
          "the top stroke is 2 canvas units wide at scale 1, got %r" % bright["width"])
    for stroke, label in ((underlay, "underlay"), (bright, "bright dash")):
        check(stroke["dash"] == [8.0, 5.0],
              "the %s stroke uses the agreed dash pattern, got %r" % (label, stroke["dash"]))
        check(abs(stroke["alpha"] - 1.0) < 1e-9,
              "the %s stroke is drawn at full opacity, got %r" % (label, stroke["alpha"]))


def test_draw_prior_masks_highlights_the_hovered_one_with_a_solid_line():
    """The hovered prior mask gets one solid 2px line, not the dashed pair."""
    result = _run_probe(_draw_style_probe_js([_PRIOR_MASK], hovered_index=0))
    check(len(result["strokes"]) == 1,
          "one stroke call for the hovered prior mask, got %d" % len(result["strokes"]))
    stroke = result["strokes"][0]
    check(stroke["dash"] == [], "the hovered outline is solid, not dashed, got %r" % stroke["dash"])
    check(stroke["style"] == "#F5EDF0",
          "the hovered outline uses the same fixed colour, got %r" % stroke["style"])
    check(abs(stroke["width"] - 2.0) < 1e-9,
          "the hovered outline is 2 canvas units wide at scale 1, got %r" % stroke["width"])
    check(abs(stroke["alpha"] - 1.0) < 1e-9, "the hovered outline is full opacity")


def test_prior_mask_color_is_the_same_fixed_colour_for_every_label():
    """Every prior mask reads the same fixed colour, regardless of its label
    (2026-09-27 pixel review, defect 2: the old per-label hue was replaced
    because it was reading as nearly invisible against real coral photos)."""
    ball_mask = dict(_PRIOR_MASK, label="BALL", name="Ball sponge")
    result = _run_probe(_draw_style_probe_js([_PRIOR_MASK, ball_mask]))
    fill_styles = {f["style"] for f in result["fills"]}
    stroke_styles = {s["style"] for s in result["strokes"]}
    check(fill_styles == {"#F5EDF0"},
          "every prior mask fills with the one fixed colour, got %r" % fill_styles)
    check(stroke_styles == {"rgba(0, 0, 0, 0.7)", "#F5EDF0"},
          "no label-specific hue appears in any stroke, got %r" % stroke_styles)


# ── 3. Never selectable ──────────────────────────────────────────────────────

def _hit_test_probe_js():
    prior = json.dumps([_PRIOR_MASK])
    return """
<script>
(function() {
  showScreen('review');
  offsetX = 0; offsetY = 0; scale = 1;
  currentData = {masks: [], prior_masks: %s};
  selectedMask = -1;
  const out = {};
  // A point squarely inside the prior mask's polygon (40..80 both axes).
  out.findMaskAt = findMaskAt(60, 60);
  out.findPriorMaskAt = findPriorMaskAt(60, 60) ? findPriorMaskAt(60, 60).label : null;

  const c = document.getElementById('canvas');
  const r = c.getBoundingClientRect();
  const fire = (type, x, y) => c.dispatchEvent(new MouseEvent(type, {
    bubbles: true, clientX: r.left + x, clientY: r.top + y, button: 0,
    buttons: type === 'mousedown' ? 1 : 0,
  }));
  fire('mousedown', 60, 60);
  fire('mouseup', 60, 60);
  out.selectedAfterClick = selectedMask;

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (prior, _MARK_START, _MARK_END)


def test_a_prior_mask_is_never_hit_tested_or_selectable():
    """findMaskAt, and a real click at the same point, both see nothing there."""
    result = _run_probe(_hit_test_probe_js())
    check(result["findMaskAt"] == -1,
          "findMaskAt never sees a prior mask, got index %r" % result["findMaskAt"])
    check(result["findPriorMaskAt"] == "PA",
          "findPriorMaskAt does see it, so the point itself is a valid hit")
    check(result["selectedAfterClick"] == -1,
          "a real click on a prior mask selects nothing, got %r"
          % result["selectedAfterClick"])


# ── 4. Never in undo, the mask list, save, or export ─────────────────────────

def test_prior_masks_appears_only_where_this_task_added_it():
    """The read-only echo touches no editing, persistence, or export path."""
    src = _template_source()
    reading_functions = ("drawPriorMasks", "findPriorMaskAt", "updatePriorMaskHover",
                         "_updatePriorScreenedNote", "render")
    never_functions = ("undoRefine", "saveCurrentMasks", "renderMaskList",
                       "findMaskAt", "doExport", "mergeSameId", "reloadSegmentations")
    for name in never_functions:
        body = _extract_function(name, src)
        check("prior" not in body.lower(), "%s never mentions a prior mask" % name)
    for name in reading_functions:
        # Sanity: the functions this task actually wrote/touched do mention it,
        # so the negative check above has teeth.
        body = _extract_function(name, src)
        check("prior" in body.lower(),
              "%s is one of the functions that legitimately reads prior masks" % name)


# ── 5. Hover tooltip ──────────────────────────────────────────────────────────

def _hover_probe_js():
    prior = json.dumps([_PRIOR_MASK])
    return """
<script>
(function() {
  showScreen('review');
  offsetX = 0; offsetY = 0; scale = 1;
  currentData = {masks: [], prior_masks: %s};
  const tip = document.getElementById('prior-mask-tip');
  const out = {};

  updatePriorMaskHover(300, 300, 60, 60);
  out.overPrior = {
    visible: tip.classList.contains('visible'), text: tip.textContent,
    nameIsItalic: !!tip.querySelector('i.sci-name'),
  };

  updatePriorMaskHover(300, 300, 500, 500);
  out.awayFromPrior = tip.classList.contains('visible');

  currentData = {masks: [{id: 1, status: 'accepted', species: 'PA',
                          polygon_px: [[40, 40, 80, 40, 80, 80, 40, 80]]}],
                 prior_masks: %s};
  updatePriorMaskHover(300, 300, 60, 60);
  out.ownMaskWins = tip.classList.contains('visible');

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (prior, prior, _MARK_START, _MARK_END)


def test_hovering_a_prior_mask_shows_a_floating_tooltip_naming_it():
    """The tooltip leads with the full species name, then the code, the
    project, the reviewer and the date; it never shows over empty space or an
    own mask. Exact copy (2026-09-27 review): "Porites astreoides (PA):
    screened in AT_PA_750img_26-09-18 by AT on 2026-09-25"."""
    result = _run_probe(_hover_probe_js())
    over = result["overPrior"]
    check(over["visible"], "the tooltip shows while hovering the prior mask")
    check(over["text"] == (
        "Porites astreoides (PA): screened in AT_PA_750img_26-09-18 "
        "by AT on 2026-09-25"
    ), "the tooltip text matches the agreed copy exactly, got %r" % over["text"])
    check(over["text"].startswith("Porites astreoides (PA)"),
          "the full species name leads, with the code beside it in parens")
    check(over["nameIsItalic"],
          "the species name in the floating tip is in italics (2026-09-27 pixel review)")
    check(not result["awayFromPrior"],
          "the tooltip hides once the pointer leaves the prior mask")
    check(not result["ownMaskWins"],
          "an own mask at the same spot wins: the tooltip never shows over it")


def test_hover_tooltip_omits_by_x_when_reviewer_is_blank():
    """A blank reviewer drops the whole "by X" clause instead of "by ?".

    2026-09-27 review: an older or imported prior mask can carry no reviewer
    at all; printing "by ?" reads as a person named question mark, not as
    missing data. Exact copy: "Porites astreoides (PA): screened in
    18aprilpm_PA on 2026-04-18".
    """
    blank_reviewer_mask = dict(_PRIOR_MASK)
    blank_reviewer_mask.update(
        reviewer="", project_name="18aprilpm_PA", at="2026-04-18T09:12:00-04:00")
    prior = json.dumps([blank_reviewer_mask])
    probe = """
<script>
(function() {
  showScreen('review');
  offsetX = 0; offsetY = 0; scale = 1;
  currentData = {masks: [], prior_masks: %s};
  const text = priorMaskTipText(currentData.prior_masks[0]);
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify({text}) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (prior, _MARK_START, _MARK_END)
    result = _run_probe(probe)
    check(result["text"] == "Porites astreoides (PA): screened in 18aprilpm_PA on 2026-04-18",
          "the blank-reviewer tooltip drops \"by\" entirely, got %r" % result["text"])
    check(" by " not in result["text"], "no dangling by clause with nothing after it")


def test_prior_mask_tip_sits_below_the_settings_popover_in_z_order():
    """#prior-mask-tip never covers the popover it can appear alongside.

    2026-09-27 review: both used z-index 260, so a lingering hover tip could
    sit on top of the popover and cover Merge same ID / Reload IDs.
    """
    src = _template_source()
    tip_rule = src[src.index("#prior-mask-tip {"):]
    tip_rule = tip_rule[:tip_rule.index("}") + 1]
    popover_rule = src[src.index(".settings-popover {"):]
    popover_rule = popover_rule[:popover_rule.index("}") + 1]
    tip_z = int(re.search(r"z-index:\s*(\d+)", tip_rule).group(1))
    popover_z = int(re.search(r"z-index:\s*(\d+)", popover_rule).group(1))
    check(tip_z < popover_z,
          "the tip's z-index (%d) is below the popover's (%d)" % (tip_z, popover_z))


def test_the_tip_is_hidden_on_mouseleave_frame_switch_and_popover_open():
    """A stale tip never survives past the moment its content is no longer true.

    2026-09-27 review found three gaps: the canvas mouseleave handler hid the
    magnifier but not the tip; selectImage (a frame switch) left it showing a
    polygon and a screen position that belong to the previous frame; and
    opening the settings popover left it floating over housekeeping buttons
    the popover can cover.
    """
    src = _template_source()
    mouseleave = src[src.index("canvas.addEventListener('mouseleave'"):]
    mouseleave = mouseleave[:mouseleave.index("});") + 3]
    check("hidePriorMaskTip();" in mouseleave,
          "the canvas mouseleave handler hides the tip")
    select_image = _extract_function("selectImage", src)
    check("hidePriorMaskTip();" in select_image,
          "selectImage hides the tip on every frame switch")
    open_popover = _extract_function("openSettingsPopover", src)
    check("hidePriorMaskTip();" in open_popover,
          "openSettingsPopover hides the tip before it can be covered")

    # Behavioral: a real mouseleave dispatched at the canvas actually clears
    # the visible class, not just a source-text match above.
    probe = """
<script>
(function() {
  showScreen('review');
  const tip = document.getElementById('prior-mask-tip');
  tip.classList.add('visible');
  canvas.dispatchEvent(new MouseEvent('mouseleave', {bubbles: true}));
  const out = {hiddenAfterLeave: !tip.classList.contains('visible')};
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)
    result = _run_probe(probe)
    check(result["hiddenAfterLeave"],
          "a real mouseleave event actually clears the tip's visible class")


# ── 6. The "Already screened here" note ──────────────────────────────────────
# Moved 2026-09-27 (LO review) from a fixed-height top-bar row into the Masks
# panel header, under the "x/y accepted, n pending" count: a variable-length
# note can wrap freely there without ever moving the canvas, since the panel
# is a fixed-width column beside it, not a row above it.

def _note_probe_js():
    screened = json.dumps(_PRIOR_SCREENED)
    return """
<script>
(function() {
  showScreen('review');
  const el = document.getElementById('prior-screened-note');
  const wrapEl = document.getElementById('canvas-wrap');
  const out = {};

  currentData = {masks: [], prior_screened: {}};
  _updatePriorScreenedNote();
  out.hiddenWhenEmpty = {
    collapsed: getComputedStyle(el).display === 'none', text: el.textContent,
    top: Math.round(wrapEl.getBoundingClientRect().top),
  };

  currentData = {masks: [], prior_screened: %s};
  _updatePriorScreenedNote();
  // Read the chip through the DOM (querySelector), not el.innerHTML: a raw
  // '<' or '>' inside a JSON string value survives JSON.stringify but is
  // then HTML-entity-escaped by dump-dom's own serialization of the <pre>
  // text node carrying it, which corrupts the JSON before Python parses it.
  const chip = el.querySelector('.code-chip');
  out.shownWhenPresent = {
    text: el.textContent, title: el.title,
    hasCodeChip: !!chip, chipText: chip ? chip.textContent : null,
    top: Math.round(wrapEl.getBoundingClientRect().top),
  };

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (screened, _MARK_START, _MARK_END)


def test_the_already_screened_note_names_the_label_project_and_date():
    """The note reads the full name, the code as a chip, the source project
    and the date (2026-09-27 pixel review, defect 11); its tooltip names the
    outcome and who, not just the visible text repeated."""
    result = _run_probe(_note_probe_js())
    hidden = result["hiddenWhenEmpty"]
    check(hidden["collapsed"], "the note collapses to nothing when nothing was screened")
    check(hidden["text"] == "", "and carries no text either")
    shown = result["shownWhenPresent"]
    check(shown["text"].startswith("Already screened here:"),
          "the note opens with the agreed lead-in, got %r" % shown["text"])
    check("Porites astreoides PA" in shown["text"],
          "the note gives the full name before the code")
    check(shown["hasCodeChip"] and shown["chipText"] == "PA",
          "the code rides as the same .code-chip style a mask-list row uses")
    check("in AT_PA_750img_26-09-18" in shown["text"],
          "the note names the source project, got %r" % shown["text"])
    check("2026-09-25" in shown["text"],
          "the note names the date, got %r" % shown["text"])
    check(shown["title"] ==
          "AT_PA_750img_26-09-18 placed Porites astreoides here by hand, "
          "no AI assistance on 2026-09-25",
          "the tooltip names the outcome and who, not just the visible text, "
          "got %r" % shown["title"])
    check(hidden["top"] == shown["top"],
          "the canvas never moves whether the note is empty or full "
          "(empty top=%dpx, full top=%dpx)" % (hidden["top"], shown["top"]))


def main():
    """Run every check; skip cleanly with no failure if Chrome is absent."""
    if not (os.path.exists(CHROME) and os.access(CHROME, os.X_OK)):
        print("SKIP: %s not found; these probes need headless Chrome" % CHROME)
        return 0
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
