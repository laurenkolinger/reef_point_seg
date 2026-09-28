"""The review screen holds its layout still through a session, and a mouse
gesture that starts on the canvas and ends off it never leaves the view stuck
mid-gesture.

Task: VEGA combined annotator layout-drift and stuck-pan fix, 2026-09-26 (LO),
widened after an independent review of the first pass. Two kinds of defect,
found by reading src/templates/index.html directly:

  1. Progressive jump. `.review-topbar` wraps onto a second row once its
     content is wider than the window, and several elements in that row grow
     wider as a session runs, none reserving its own width: the status
     message `#status-text` (a few words early, a long confirmation sentence
     late, and a fixed-but-too-narrow slot still cut a normal message down
     to its first ~24 characters, hiding the useful part); the per-frame
     timer `#frame-timer` ("--" before the first frame, a full "this Xm Ys
     (paused) . avg Xm Ys (n=NNN, target 15s)" line late, with no cap so a
     rarer, longer reading kept growing it); the processing badge
     `#bg-processing-badge` (toggled with `display`, which drops it from
     layout instead of reserving its slot); and the current frame's name in
     `#image-filename` (an unbounded `max-width` instead of a fixed one). The
     row toggled between one line and two, or grew past its own cap, as the
     session went on, and the image moved down and back up under it every
     time. The fix: every one of these gets a fixed width (and #status-text
     its own full-width row), sized to its longest realistic text, ellipsis
     for the rarer text that still does not fit.
  2. Stuck gesture. The canvas mouseup handler is bound to the canvas element
     alone. A press that starts on the image and is released anywhere else
     (the mask panel, a toolbar, outside the browser window) never reaches
     it, so `isPanning`, `pendingClick`, a freehand draw, or a box drag is
     left open. The mousemove handler's pan-promotion check does not require
     a live mousedown, so the next hover with no button held pans the image,
     and the click after that is swallowed ending the stale gesture instead
     of doing what the user clicked for.

Every probe runs against a static jinja2 render of the template, loaded once
in headless Chrome per case. Rendering and DOM probing only: no SAM3, no GPU,
no live annotator, and `orchestrator_url` is deliberately left empty (unlike
test_annotator_design.py's stub context) so the render never asks a live
orchestrator on :5050 for its tour stylesheet; the one network dependency
this suite keeps, in common with every other headless-Chrome page in this
codebase, is the Google Fonts request the template itself always makes.

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_annotator_layout_stability.py
"""
import json
import os
import subprocess
import sys
import tempfile

from jinja2 import Environment, FileSystemLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_TPL_DIR = os.path.join(_SRC, "templates")

CHROME = "/usr/bin/google-chrome"
PROBE_TIMEOUT_S = 30
CHROME_ARGS = ["--headless", "--disable-gpu", "--hide-scrollbars", "--no-sandbox"]
# Markers the probe result is wrapped in. Neither string is a substring of the
# other (unlike the more obvious "PROBE"/"ENDPROBE" pair, where "ENDPROBE"
# itself contains "PROBE"), so the LAST occurrence of the start marker in the
# dumped DOM is unambiguously the live result appended to <body>, never a
# match inside the probe's own <script> source text, which also has to spell
# both literals in order to write them.
_MARK_START = "@@ANNOTATOR_PROBE_START@@"
_MARK_END = "@@ANNOTATOR_PROBE_END@@"

FAILURES = []


def _render(session_mode="standalone", session_label="", session_label_tooltip=""):
    """Render the page with a stub context, `orchestrator_url` empty on purpose.

    Mirrors test_annotator_design.py's `_render()`, minus the orchestrator
    fields: that helper never loads a browser, so its hardcoded
    `http://127.0.0.1:5050` never matters there, but this suite does load one,
    and that port is a real, independently-changing service on this box.

    `session_mode="configure"` with a `session_label` renders a project
    window (the session chip shown, no standalone Import Images control),
    the shape a real TCRMP batch review runs in.
    """
    labels = [
        {"code": "PA", "name": "Porites astreoides", "category": "Coral"},
        {"code": "BALL", "name": "Ball sponge", "category": "Sponge"},
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
        orchestrator_url="", orchestrated=False,
        session_mode=session_mode, review_locked_msg="locked",
        review_tagging_enabled=False, default_session_name="26-09-26_LO_demo",
        session_label=session_label, session_label_tooltip=session_label_tooltip,
    )


def check(cond, name):
    """Record one assertion and print its verdict, matching test_annotator_design.py."""
    print("  %s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILURES.append(name)


def _run_probe(probe_js, width, height=900, **render_kwargs):
    """Render the annotator page with `probe_js` appended, load it in headless
    Chrome at `width` x `height`, and return the JSON payload the probe wrote
    into a <pre> element appended to <body>.

    The probe script is inserted right before </body>, so it runs after every
    top-level `let`/`function` in the annotator's own inline script has
    already executed (classic, non-module <script> tags share one global
    lexical scope in document order), and before DOMContentLoaded fires. That
    lets a probe call the real `setStatus`, `showScreen`, and read the real
    `isPanning`/`offsetX`/`pendingClick` state as bare identifiers, with no
    need to wait on the page's failed same-origin `/api/status` fetch (there
    is no server behind a file:// page; the boot code catches that itself).
    `render_kwargs` passes straight through to `_render` (e.g. `session_mode`).
    """
    page = _render(**render_kwargs)
    marker = "</body>"
    if marker not in page:
        raise RuntimeError("rendered page has no </body> to inject the probe before")
    page = page.replace(marker, probe_js + "\n" + marker)
    fd, path = tempfile.mkstemp(suffix=".html", prefix="annotator_probe_")
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


# ── a. Layout stability across widths and a growing session ─────────────────

# The realistic shortest and longest text each element carries in a session:
# `#img-counter` from frame 1 of a batch to near its end, and `#frame-timer`
# from the first frame (no history yet) to deep into a paused session
# (mirrors the "Frame <fmt> [(paused)]" text `_timerRender` builds in
# src/templates/index.html; the session average, sample count and 15s target
# moved into the element's title, not its visible text, so they are not part
# of this string. Named rather than line-numbered, since it drifts as the
# file changes).
_EARLY_COUNTER, _EARLY_TIMER = "1 / 120", "Frame 4.2s"
_LATE_COUNTER, _LATE_TIMER = "117 / 120", "Frame 1m 12s (paused)"
# A rarer, longer session than _LATE_TIMER models (two-digit minutes): since
# 2026-09-27 #frame-timer is width: auto (it travels with Pause in one
# .timer-pause-group that wraps as a unit, see test_the_timer_and_pause_
# travel_together_without_moving_the_image below), so this is no longer
# expected to render at the same fixed width as _LATE_TIMER; it still must
# never move the image.
_EXTREME_TIMER = "Frame 123m 45s (paused)"
_STATUS_MESSAGES = [
    "",
    "Adding...",
    "PA -> Porites astreoides added: refine now, Enter to accept",
    "x" * 200,
]
# A real TCRMP clip name (TEN_COMMANDMENTS.md naming convention), long enough
# to exercise #image-filename's own shrink-to-fit behaviour in project mode.
_LONG_FILENAME = "TCRMP20251122_clip_CBS_T307.jpeg"
_WIDTHS = (946, 1280, 1920)


def _layout_probe_js():
    stages = json.dumps([
        {"name": "early", "counter": _EARLY_COUNTER, "timer": _EARLY_TIMER},
        {"name": "late", "counter": _LATE_COUNTER, "timer": _LATE_TIMER},
    ])
    messages = json.dumps(_STATUS_MESSAGES)
    return """
<script>
(function() {
  showScreen('review');
  const topbar = document.querySelector('.review-topbar');
  const wrapEl = document.getElementById('canvas-wrap');
  const counterEl = document.getElementById('img-counter');
  const timerEl = document.getElementById('frame-timer');
  const stages = %s;
  const messages = %s;
  const out = [];
  for (const stage of stages) {
    counterEl.textContent = stage.counter;
    timerEl.textContent = stage.timer;
    for (const msg of messages) {
      setStatus(msg);
      out.push({
        stage: stage.name,
        msgLen: msg.length,
        top: Math.round(wrapEl.getBoundingClientRect().top),
        barHeight: Math.round(topbar.getBoundingClientRect().height),
        statusTitle: document.getElementById('status-text').title,
      });
    }
  }
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (stages, messages, _MARK_START, _MARK_END)


def test_the_topbar_does_not_move_the_image_as_a_session_grows():
    """The canvas's top position holds still across every status and stage.

    One invariant per width: every combination of session stage (a fixed
    counter and timer text, early or late in a batch) and status message
    (empty through 200 characters) yields the same image-area top. A
    rendered-page probe catches this because it is the actual flex-wrap
    computation that moves the image, not a CSS rule read off the page.
    """
    js = _layout_probe_js()
    for width in _WIDTHS:
        rows = _run_probe(js, width)
        tops = sorted({r["top"] for r in rows})
        check(len(tops) == 1,
              "%dpx: the image top is the same across every stage and status "
              "message, saw %s" % (width, tops))
        by_stage = {}
        for row in rows:
            by_stage.setdefault(row["stage"], []).append(row)
        long_msg_title = next(r["statusTitle"] for r in by_stage["late"]
                               if r["msgLen"] == 200)
        check(long_msg_title == "x" * 200,
              "%dpx: a status message cut off on screen is still readable "
              "from its title attribute" % width)


def _badge_probe_js():
    return """
<script>
(function() {
  showScreen('review');
  const wrapEl = document.getElementById('canvas-wrap');
  const badge = document.getElementById('bg-processing-badge');
  const cases = [
    {name: 'idle', visible: false, text: 'Processing'},
    {name: 'short', visible: true, text: 'Processing 3/120...'},
    {name: 'full', visible: true, text: 'Processing 120/120...'},
    {name: 'long', visible: true, text: 'Processing 999/999...'},
  ];
  const out = [];
  for (const c of cases) {
    badge.style.visibility = c.visible ? 'visible' : 'hidden';
    badge.textContent = c.text;
    out.push({
      name: c.name,
      top: Math.round(wrapEl.getBoundingClientRect().top),
      badgeWidth: Math.round(badge.getBoundingClientRect().width),
      textFits: badge.scrollWidth <= badge.clientWidth,
      absolute: getComputedStyle(badge).position === 'absolute',
      inCanvas: !!badge.closest('#canvas-wrap'),
    });
  }
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


def test_the_processing_badge_never_moves_the_image_shown_or_hidden():
    """#bg-processing-badge floats over the image, so it never moves it.

    The 5s poll in startReviewWhileProcessing toggles this badge on and off
    and grows its digits through a run; it is positioned absolutely inside
    #canvas-wrap, so none of that touches the top bar or the image top.
    """
    js = _badge_probe_js()
    for width in _WIDTHS:
        rows = _run_probe(js, width)
        by_name = {r["name"]: r for r in rows}
        tops = sorted({r["top"] for r in rows})
        check(len(tops) == 1,
              "%dpx: the image top is the same idle, short, and long, saw %s"
              % (width, tops))
        check(all(r["absolute"] and r["inCanvas"] for r in rows),
              "%dpx: the badge floats over the image area, outside the top "
              "bar, so it takes no row space" % width)
        for name in ("short", "full", "long"):
            check(by_name[name]["textFits"],
                  "%dpx: the badge shows %s whole, no digit cut off"
                  % (width, name))


def test_a_long_filename_keeps_its_full_name_on_hover():
    """#image-filename ellipses a name past 44ch, so its title carries it.

    The frame number sits at the end of the name, the part an ellipsis cuts;
    the code that fills the element in must set the title alongside the text.
    """
    with open(os.path.join(_TPL_DIR, "index.html"), encoding="utf-8") as fh:
        source = fh.read()
    check("filenameEl.title = filename;" in source,
          "the filename element's title is set to the full name with its text")


def _extreme_timer_probe_js():
    stages = json.dumps([
        {"name": "late", "text": _LATE_TIMER},
        {"name": "extreme", "text": _EXTREME_TIMER},
    ])
    return """
<script>
(function() {
  showScreen('review');
  const wrapEl = document.getElementById('canvas-wrap');
  const timerEl = document.getElementById('frame-timer');
  const pauseEl = document.getElementById('frame-timer-pause');
  const stages = %s;
  const out = [];
  for (const stage of stages) {
    timerEl.textContent = stage.text;
    const timerBox = timerEl.getBoundingClientRect();
    const pauseBox = pauseEl.getBoundingClientRect();
    out.push({
      name: stage.name,
      top: Math.round(wrapEl.getBoundingClientRect().top),
      timerWidth: Math.round(timerBox.width),
      timerTop: Math.round(timerBox.top), timerBottom: Math.round(timerBox.bottom),
      pauseTop: Math.round(pauseBox.top), pauseBottom: Math.round(pauseBox.bottom),
    });
  }
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (stages, _MARK_START, _MARK_END)


def test_the_timer_and_pause_travel_together_without_moving_the_image():
    """#frame-timer is width: auto since 2026-09-27 (pixel review, defect 4):
    it and Pause share one .timer-pause-group with margin-left: auto, so a
    wrap always carries both across together instead of stranding Pause on
    its own line. A longer reading is free to widen the timer's own box (this
    is the deliberate trade against the old fixed-width cap, which kept the
    box still but let Pause end up far from it); what must still hold is that
    the timer and Pause stay on the same line as each other, and that neither
    reading moves the canvas.
    """
    js = _extreme_timer_probe_js()
    for width in _WIDTHS:
        rows = _run_probe(js, width)
        by_name = {r["name"]: r for r in rows}
        tops = sorted({r["top"] for r in rows})
        check(len(tops) == 1,
              "%dpx: the image top is the same at the late and the extreme "
              "timer text, saw %s" % (width, tops))
        for name in ("late", "extreme"):
            row = by_name[name]
            # "Same line" means their vertical ranges overlap, not that their
            # tops match exactly: align-items: center in .timer-pause-group
            # centers the shorter timer text against the taller Pause button,
            # so their tops legitimately differ by a few pixels even sharing
            # one line.
            same_line = row["timerTop"] < row["pauseBottom"] and row["pauseTop"] < row["timerBottom"]
            check(same_line,
                  "%dpx, %s reading: the timer (top=%d, bottom=%d) and Pause "
                  "(top=%d, bottom=%d) sit on the same line"
                  % (width, name, row["timerTop"], row["timerBottom"],
                     row["pauseTop"], row["pauseBottom"]))
        check(by_name["extreme"]["timerWidth"] >= by_name["late"]["timerWidth"],
              "%dpx: the timer's own box is free to widen for a longer "
              "reading (late=%dpx, extreme=%dpx), since the group as a whole "
              "wraps, not the timer alone"
              % (width, by_name["late"]["timerWidth"],
                 by_name["extreme"]["timerWidth"]))


def _filename_probe_js():
    names = json.dumps(["", _LONG_FILENAME])
    return """
<script>
(function() {
  showScreen('review');
  const wrapEl = document.getElementById('canvas-wrap');
  const filenameEl = document.getElementById('image-filename');
  const names = %s;
  const out = [];
  for (const name of names) {
    filenameEl.textContent = name;
    out.push({len: name.length,
               top: Math.round(wrapEl.getBoundingClientRect().top)});
  }
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (names, _MARK_START, _MARK_END)


def test_a_realistic_project_filename_does_not_move_the_image():
    """A project window (session_mode="configure") holds still too.

    #image-filename already shrinks (flex-shrink: 1, min-width: 16ch) rather
    than pushing the row wider; this proves it in project mode, where the
    standalone Import Images button is absent and a real TCRMP clip name
    (`_LONG_FILENAME`) is the thing that grows through a session, going from
    no frame selected yet to one loaded.
    """
    js = _filename_probe_js()
    for width in _WIDTHS:
        rows = _run_probe(js, width, session_mode="configure",
                           session_label="TCRMP_MCR_T3",
                           session_label_tooltip="TCRMP, site MCR, transect 3")
        tops = sorted({r["top"] for r in rows})
        check(len(tops) == 1,
              "%dpx, project mode: the image top is the same with no "
              "filename and with %r, saw %s"
              % (width, _LONG_FILENAME, tops))


def _prior_screened_note_probe_js():
    """Toggle #prior-screened-note through _updatePriorScreenedNote() itself.

    Task: VEGA annotator tidy-up and no re-screening, 2026-09-27 (LO), moved
    2026-09-27 (LO review) from a fixed-height top-bar row into the Masks
    panel header, under the "x/y accepted, n pending" count. The note names
    labels another project already screened on this frame
    (currentData.prior_screened); most frames have none, some have one, a
    hostile frame could have every target label screened elsewhere. The panel
    is a fixed-width column beside the canvas, not above it, so none of the
    three cases can move the image; this probe proves that directly rather
    than asserting a reserved height the element no longer needs.
    """
    cases = json.dumps([
        {"name": "none", "screened": {}},
        {"name": "one", "screened": {
            "PA": {"outcome": "found_manual", "project_name": "AT_PA_750img_26-09-18",
                   "project_id": "run_1", "reviewer": "AT", "at": "2026-09-25T09:00:00-04:00"},
        }},
        {"name": "several", "screened": {
            "PA": {"outcome": "found_manual", "project_name": "AT_PA_750img_26-09-18",
                   "project_id": "run_1", "reviewer": "AT", "at": "2026-09-25T09:00:00-04:00"},
            "BALL": {"outcome": "found_ai", "project_name": "LO_BALL_sweep_26-09-10",
                     "project_id": "run_2", "reviewer": "LO", "at": "2026-09-10T09:00:00-04:00"},
            "OFAV": {"outcome": "not_found", "project_name": "AT_OFAV_500img_26-08-30",
                     "project_id": "run_3", "reviewer": "AT", "at": "2026-08-30T09:00:00-04:00"},
        }},
    ])
    return """
<script>
(function() {
  showScreen('review');
  const wrapEl = document.getElementById('canvas-wrap');
  const cases = %s;
  const out = [];
  for (const c of cases) {
    currentData = {masks: [], prior_screened: c.screened};
    _updatePriorScreenedNote();
    const el = document.getElementById('prior-screened-note');
    out.push({
      name: c.name,
      top: Math.round(wrapEl.getBoundingClientRect().top),
      hasText: el.textContent.length > 0,
      collapsed: getComputedStyle(el).display === 'none',
    });
  }
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (cases, _MARK_START, _MARK_END)


def test_the_prior_screened_note_never_moves_the_image():
    """No screening, one label, and three labels never move the canvas.

    Complements test_prior_mask_render.py's content checks: this suite only
    proves the geometry. The note lives in the Masks panel, a fixed-width
    column beside the canvas rather than a row above it, so this also proves
    the architectural claim (moving it there structurally cannot move the
    image) rather than relying on a reserved height the way #status-text
    still does.
    """
    js = _prior_screened_note_probe_js()
    for width in _WIDTHS:
        rows = _run_probe(js, width)
        tops = sorted({r["top"] for r in rows})
        check(len(tops) == 1,
              "%dpx: the image top is the same with no screening, one label, "
              "and several, saw %s" % (width, tops))
        by_name = {r["name"]: r for r in rows}
        check(by_name["none"]["collapsed"] and not by_name["none"]["hasText"],
              "the note collapses to nothing when nothing was screened")
        check(by_name["one"]["hasText"] and by_name["several"]["hasText"],
              "the note shows once at least one label was screened elsewhere")


def test_the_status_line_shows_at_least_80_characters_at_1280():
    """A normal-length status is not clipped down to a useless fragment.

    "Blocked by #14 (Orbicella annularis)" is 37 characters; the fixed-slot
    design that preceded this one cut #status-text to 24, hiding the useful
    part. The full-width row this fix gives it must clear a normal message
    with room to spare: 80 plain characters, well under the topbar's actual
    width, must render with no clipping at all (scrollWidth <= clientWidth).
    """
    probe = """
<script>
(function() {
  showScreen('review');
  setStatus('x'.repeat(80));
  const el = document.getElementById('status-text');
  const out = {scrollWidth: el.scrollWidth, clientWidth: el.clientWidth};
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)
    result = _run_probe(probe, width=1280)
    check(result["scrollWidth"] <= result["clientWidth"] + 1,
          "1280px: 80 characters of status fit with no clipping "
          "(scrollWidth=%d, clientWidth=%d)"
          % (result["scrollWidth"], result["clientWidth"]))


# ── b/c. A gesture never survives past the release that ends it ─────────────

_GESTURE_PROBE_JS = """
<script>
(function() {
  showScreen('review');
  const c = document.getElementById('canvas');
  const r = c.getBoundingClientRect();
  const fire = (target, type, x, y, buttons) => target.dispatchEvent(
    new MouseEvent(type, {bubbles: true, clientX: r.left + x, clientY: r.top + y,
                           button: 0, buttons}));
  const out = {b: [], c: []};

  // b1: press on the image, release off the canvas (e.g. over the mask
  // panel), then hover back with no button held at all.
  offsetX = 0; offsetY = 0; isPanning = false; pendingClick = null;
  fire(c, 'mousedown', 100, 100, 1);
  const b1press = {pendingClick: !!pendingClick, isPanning};
  fire(document.body, 'mouseup', 100, 100, 0);
  const b1release = {pendingClick: !!pendingClick, isPanning, offsetX, offsetY};
  fire(c, 'mousemove', 160, 140, 0);
  fire(c, 'mousemove', 260, 190, 0);
  out.b.push({case: 'click_release_off_canvas', press: b1press, release: b1release,
              hover: {isPanning, pendingClick: !!pendingClick, offsetX, offsetY}});

  // b2: a drag past the pan threshold, released off the canvas.
  isPanning = false; pendingClick = null; offsetX = 0; offsetY = 0;
  fire(c, 'mousedown', 100, 100, 1);
  fire(c, 'mousemove', 120, 100, 1);
  const b2drag = {isPanning, offsetX, offsetY};
  fire(document.body, 'mouseup', 120, 100, 0);
  const b2release = {isPanning, offsetX, offsetY};
  fire(c, 'mousemove', 300, 250, 0);
  out.b.push({case: 'drag_pan_release_off_canvas', drag: b2drag, release: b2release,
              hover: {isPanning, offsetX, offsetY}});

  // b3 control: press and release ON the canvas is still an ordinary click,
  // with no exception raised by the SELECT-mode click-on-nothing path.
  isPanning = false; pendingClick = null;
  let threw = null;
  try {
    fire(c, 'mousedown', 50, 50, 1);
    fire(c, 'mouseup', 50, 50, 0);
  } catch (e) { threw = String((e && e.message) || e); }
  out.b.push({case: 'on_canvas_click_control', threw,
              isPanning, pendingClick: !!pendingClick});

  // c: a press held, then the window loses focus before it is ever released.
  isPanning = false; pendingClick = null;
  fire(c, 'mousedown', 100, 100, 1);
  const cBefore = {pendingClick: !!pendingClick, isPanning};
  window.dispatchEvent(new Event('blur'));
  out.c.push({before: cBefore, after: {pendingClick: !!pendingClick, isPanning}});

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


def test_a_gesture_released_off_canvas_does_not_stick():
    """A press that ends off the canvas is closed out, not left open.

    Mirrors the drift investigation's own probe: dispatch real MouseEvents at
    the rendered canvas and read the template's own state variables back, so
    the assertion is about what the page actually does, not what its source
    text says it should do.
    """
    result = _run_probe(_GESTURE_PROBE_JS, width=1280)
    by_case = {row["case"]: row for row in result["b"]}

    click = by_case["click_release_off_canvas"]
    check(click["press"]["pendingClick"] is True,
          "a plain press arms a pending click")
    check(click["release"]["pendingClick"] is False,
          "releasing off the canvas clears the pending click instead of "
          "leaving it for the next hover to pick up")
    check(click["hover"]["isPanning"] is False,
          "a no-button hover after that release never starts a pan")
    check(click["hover"]["offsetX"] == 0 and click["hover"]["offsetY"] == 0,
          "and the view offset stays put: got (%s, %s)"
          % (click["hover"]["offsetX"], click["hover"]["offsetY"]))

    drag = by_case["drag_pan_release_off_canvas"]
    check(drag["drag"]["isPanning"] is True,
          "dragging past the threshold still starts a pan")
    check(drag["release"]["isPanning"] is False,
          "releasing off the canvas ends the pan instead of leaving it live")
    check(drag["hover"]["offsetX"] == drag["release"]["offsetX"]
          and drag["hover"]["offsetY"] == drag["release"]["offsetY"],
          "a no-button hover after that release moves the view no further: "
          "release=(%s, %s) hover=(%s, %s)"
          % (drag["release"]["offsetX"], drag["release"]["offsetY"],
             drag["hover"]["offsetX"], drag["hover"]["offsetY"]))

    control = by_case["on_canvas_click_control"]
    check(control["threw"] is None,
          "an ordinary on-canvas click still raises nothing: %s"
          % control["threw"])
    check(control["isPanning"] is False and control["pendingClick"] is False,
          "and it resolves as a click, not a stuck gesture")

    blurred = result["c"][0]
    check(blurred["before"]["pendingClick"] is True,
          "a held press arms a pending click before the window loses focus")
    check(blurred["after"]["pendingClick"] is False,
          "and losing focus mid-press clears it instead of leaving it for "
          "whenever focus returns")


_COMMIT_ONCE_PROBE_JS = """
<script>
(function() {
  showScreen('review');
  // Stand in for the network calls a real commit makes (submitDrawEdit is a
  // POST to /draw_edit, doBoxAdd to /add_box); counting calls instead of
  // hitting a route none of this page's stub context has behind it.
  let submitDrawEditCalls = 0, doBoxAddCalls = 0;
  submitDrawEdit = function(path) { submitDrawEditCalls++; return Promise.resolve(); };
  doBoxAdd = function(imgBox) { doBoxAddCalls++; return Promise.resolve(); };
  doBoxRedraw = function(imgBox) { return Promise.resolve(); };

  const c = document.getElementById('canvas');
  const r = c.getBoundingClientRect();
  const fire = (target, type, x, y, buttons) => target.dispatchEvent(
    new MouseEvent(type, {bubbles: true, clientX: r.left + x, clientY: r.top + y,
                           button: 0, buttons}));
  const out = {};

  // A freehand draw released off the canvas: cancelCanvasGesture (the window
  // mouseup listener) must commit it exactly once, and a stray hover
  // afterwards must not commit it again.
  interactionMode = 'draw'; selectedMask = 0; isPanning = false;
  pendingClick = null; isDrawing = false; drawPath = []; offsetX = 0; offsetY = 0;
  fire(c, 'mousedown', 100, 100, 1);
  fire(c, 'mousemove', 110, 100, 1);
  fire(c, 'mousemove', 120, 110, 1);
  fire(document.body, 'mouseup', 120, 110, 0);
  fire(c, 'mousemove', 200, 200, 0);
  out.drawOffCanvas = {calls: submitDrawEditCalls, isDrawing};

  // The same gesture released ON the canvas: only the canvas's own mouseup
  // handler runs (the window listener explicitly skips e.target === canvas),
  // so this is a regression lock against ever calling both for one release.
  submitDrawEditCalls = 0;
  interactionMode = 'draw'; selectedMask = 0; isDrawing = false; drawPath = [];
  fire(c, 'mousedown', 100, 100, 1);
  fire(c, 'mousemove', 110, 100, 1);
  fire(c, 'mousemove', 120, 110, 1);
  fire(c, 'mouseup', 120, 110, 0);
  out.drawOnCanvas = {calls: submitDrawEditCalls, isDrawing};

  // A box drag released off the canvas commits exactly once too.
  interactionMode = 'box'; isDrawingBox = false; boxForNewMask = false;
  isPanning = false; pendingClick = null;
  fire(c, 'mousedown', 100, 100, 1);
  fire(c, 'mousemove', 140, 140, 1);
  fire(document.body, 'mouseup', 140, 140, 0);
  fire(c, 'mousemove', 300, 300, 0);
  out.boxOffCanvas = {calls: doBoxAddCalls, isDrawingBox};

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


def test_a_draw_or_box_gesture_commits_exactly_once():
    """cancelCanvasGesture commits a real edit once, never zero or twice.

    Bug 2's fix reuses the canvas mouseup handler's own commit threshold, so
    a draw or box gesture that already has enough content to save is saved
    rather than thrown away on an off-canvas release. This proves the count:
    a real /draw_edit or /add_box call fires exactly once per gesture,
    whether the release happens on the canvas or off it, and a redundant
    mousemove afterwards (the buttons === 0 self-heal) never adds a second.
    """
    result = _run_probe(_COMMIT_ONCE_PROBE_JS, width=1280)

    off_canvas = result["drawOffCanvas"]
    check(off_canvas["calls"] == 1,
          "a freehand draw released off the canvas commits exactly once, "
          "got %d calls" % off_canvas["calls"])
    check(off_canvas["isDrawing"] is False,
          "and the draw gesture is closed out, not left armed")

    on_canvas = result["drawOnCanvas"]
    check(on_canvas["calls"] == 1,
          "the same draw released ON the canvas also commits exactly once "
          "(never twice), got %d calls" % on_canvas["calls"])
    check(on_canvas["isDrawing"] is False,
          "and is closed out there too")

    box_off_canvas = result["boxOffCanvas"]
    check(box_off_canvas["calls"] == 1,
          "a box drag released off the canvas commits exactly once, got %d "
          "calls" % box_off_canvas["calls"])
    check(box_off_canvas["isDrawingBox"] is False,
          "and the box gesture is closed out, not left armed")


def main():
    """Run every check; skip cleanly with no failure if Chrome is absent."""
    if not (os.path.exists(CHROME) and os.access(CHROME, os.X_OK)):
        print("SKIP: %s not found; layout/gesture probes need headless Chrome"
              % CHROME)
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
