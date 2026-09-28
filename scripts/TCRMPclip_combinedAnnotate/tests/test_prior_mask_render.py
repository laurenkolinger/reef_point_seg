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
  2. drawPriorMasks fills at 35% of the current mask opacity and strokes with
     a dashed line, distinct per label code.
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
     with the full name, the code and the source project for every label, and
     its slot holds one fixed height whether or not there is anything to say.

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_prior_mask_render.py
"""
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

def _draw_style_probe_js():
    prior = json.dumps([_PRIOR_MASK])
    return """
<script>
(function() {
  showScreen('review');
  currentData = {masks: [], prior_masks: %s};
  maskOpacity = 0.5;
  scale = 1;
  // Records only what this test asserts: the fill alpha at the moment
  // fillStyle is set (canvas has no "this call is a fill" event, so the
  // alpha is captured at the point the following draw call will read it),
  // the dash pattern, and the two style strings.
  const calls = {fillAlpha: null, lastAlpha: null, dashes: [],
                fillStyle: null, strokeStyle: null};
  const fakeCtx = {
    save() {}, restore() {}, beginPath() {}, closePath() {},
    moveTo() {}, lineTo() {}, fill() {}, stroke() {},
    setLineDash(d) { calls.dashes.push(d); },
    set globalAlpha(v) { calls.lastAlpha = v; },
    get globalAlpha() { return calls.lastAlpha; },
    set fillStyle(v) { calls.fillStyle = v; calls.fillAlpha = calls.lastAlpha; },
    get fillStyle() { return calls.fillStyle; },
    set strokeStyle(v) { calls.strokeStyle = v; },
    get strokeStyle() { return calls.strokeStyle; },
    set lineWidth(v) {}, get lineWidth() { return 1; },
  };
  drawPriorMasks(fakeCtx);
  const out = {
    fillAlpha: calls.fillAlpha,
    dashed: calls.dashes.length > 0 && calls.dashes[0].some(v => v > 0),
    fillStyle: calls.fillStyle, strokeStyle: calls.strokeStyle,
  };
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (prior, _MARK_START, _MARK_END)


def _hsla_hue(hsla_string):
    """Pull the hue number out of an "hsla(H, S%, L%, A)" string."""
    return hsla_string.split("(", 1)[1].split(",", 1)[0].strip()


def test_draw_prior_masks_fills_at_35_percent_and_strokes_dashed():
    """The fill alpha is 35%% of the live opacity slider; the stroke is dashed.

    Fill and stroke deliberately carry different alphas (1.0 and 0.85, the
    same fill-then-stroke pattern maskColor/maskStrokeColor use for this
    project's own masks), so the two style strings differ; what must match is
    the hue, since both come from the one label-keyed colour.
    """
    result = _run_probe(_draw_style_probe_js())
    check(abs(result["fillAlpha"] - 0.5 * 0.35) < 1e-9,
          "the fill alpha is 35%% of maskOpacity, got %r" % result["fillAlpha"])
    check(result["dashed"], "the stroke uses a dash pattern")
    check(_hsla_hue(result["fillStyle"]) == _hsla_hue(result["strokeStyle"]),
          "the fill and the stroke share one hue for a given label (fill=%r, "
          "stroke=%r)" % (result["fillStyle"], result["strokeStyle"]))
    check(result["fillStyle"].startswith("hsla("),
          "the colour is computed, not a literal swatch: %r" % result["fillStyle"])


def test_prior_mask_color_is_stable_per_label_and_differs_across_labels():
    """Two masks with the same label match; two different labels do not."""
    src = _template_source()
    fn = _extract_function("priorMaskColor", src)
    harness = """
%s
const pa1 = priorMaskColor('PA', 1.0);
const pa2 = priorMaskColor('PA', 1.0);
const ball = priorMaskColor('BALL', 1.0);
console.log(JSON.stringify({pa1, pa2, ball}));
""" % fn
    out = _run_node(harness)
    check(out["pa1"] == out["pa2"], "the same label code always gets the same colour")
    check(out["pa1"] != out["ball"], "a different label code gets a different colour")


def _find_node():
    for candidate in ("node", "/usr/bin/node"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except Exception:
            continue
    return None


def _run_node(harness):
    """Run a bare JavaScript harness under node and parse its one JSON line."""
    node = _find_node()
    if node is None:
        print("    (node unavailable, skipping)")
        return {"pa1": None, "pa2": None, "ball": None}
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(harness)
        path = handle.name
    try:
        result = subprocess.run([node, path], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("node failed: %s" % result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])
    finally:
        os.unlink(path)


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
  out.overPrior = {visible: tip.classList.contains('visible'), text: tip.textContent};

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


def test_the_already_screened_note_names_the_label_and_project():
    """The note reads the full name, the code as a chip, and the source project."""
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
    check("(AT_PA_750img_26-09-18)" in shown["text"],
          "the note names the source project in parentheses")
    check(shown["title"] == shown["text"],
          "the full text is on the title too, for when it wraps or ellipses")
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
