"""The Edit Masks review screen holds its layout still while status messages
come and go, and its top bar renders no stray glyph.

Edit Masks (scripts/TCRMPclip_editMasks/src/templates/index.html) kept the
top bar combinedAnnotate replaced on 2026-09-27: one flex-wrap row holding
every control AND the status message. A long message (a refused save names
the mask, the reason and what to do next) widened that row past the window,
the row wrapped onto a second line, and the canvas below moved down, then
back up when the message cleared. The frame timer's markup also held a bare
":" until the first frame was selected, and its reading grew through a
session ("this 1m 12s (paused) . avg ... (n=NNN, target 15s)").

combinedAnnotate's fix, ported here: the status message gets its own
fixed-height row (ellipsis for a message too long to fit, the full text on
the row's title), the controls keep a row of their own, and the timer renders
nothing until a frame has a reading, then shows only the live reading with
the session average on its title. tests/test_annotator_layout_stability.py
covers combinedAnnotate; this suite covers Edit Masks.

What this suite proves, in headless Chrome at 946 and 1280 px:
  - the canvas area and the canvas itself keep the same top position through
    every status message (empty, short, the longest real refusals, 200
    characters) set as a flash or a persistent message and then cleared, with
    the timer empty, early and late;
  - a message cut off on screen is whole in the status row's title;
  - the processing badge, a real frame name, the batch line and the
    unlabeled count never move the canvas, and the Next unlabeled button
    keeps one width;
  - the timer is empty before a frame and after a render with no reading,
    no element in the top bar shows a lone ":", and a reading starts with
    "Frame" and keeps the session average in its title ("No session average
    yet." on the first frame); Pause is disabled, with no pointer, while the
    timer is empty;
  - a real frame name shows whole at both widths, and the batch slot fits
    its text without a wide empty gap;
  - combinedAnnotate's timer matches: it shows nothing until it has a
    reading, "No session average yet." on the first frame's hover text, and
    Pause disabled, with no pointer, until the timer has a reading.

Rendering and DOM probing only: a static jinja2 render of the template, no
server, no SAM3, no GPU. orchestrator_url is left empty so the render never
asks the live orchestrator on :5050 for anything. Chrome needs a short TMPDIR
on this box (its socket path is limited); run_all_tests.sh provides one.

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_editmasks_layout_stability.py
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
import subprocess
import sys
import tempfile

from jinja2 import Environment, FileSystemLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
_TPL_DIR = os.path.join(_SCRIPTS, "TCRMPclip_editMasks", "src", "templates")
_CA_TPL_DIR = os.path.join(_SCRIPTS, "TCRMPclip_combinedAnnotate", "src", "templates")

CHROME = "/usr/bin/google-chrome"
PROBE_TIMEOUT_S = 30
CHROME_ARGS = ["--headless", "--disable-gpu", "--hide-scrollbars", "--no-sandbox"]
WIDTHS = (946, 1280)
# Neither marker is a substring of the other, so the last start marker in the
# dumped DOM is the live result, never the probe's own script text.
_MARK_START = "@@EDITMASKS_PROBE_START@@"
_MARK_END = "@@EDITMASKS_PROBE_END@@"

# A real TCRMP clip name (TEN_COMMANDMENTS.md naming convention).
LONG_FILENAME = "TCRMP20251122_clip_CBS_T307.jpeg"
# The longest messages the page really writes: a refused frame save after a
# frame switch, carrying the expert-review lock's own two sentences, and a
# refused write with no answer in time.
LONGEST_REFUSAL = ("Mask statuses on " + LONG_FILENAME + " were not saved: the server "
                   "refused it (HTTP 409). Flagging for expert review is temporarily "
                   "unavailable. VICARIUS is accepting no new review tags.")
TIMEOUT_REFUSAL = ("Mask C on " + LONG_FILENAME + " was not saved: no answer within 15 "
                   "seconds. Move to another frame and back to see what the server kept.")
STATUS_MESSAGES = ["", "Saving...", "Mask C was not saved: it has no species. Pick a "
                   "species and accept again.", TIMEOUT_REFUSAL, LONGEST_REFUSAL, "x" * 200]
TIMER_STAGES = ["", "Frame 4.2s", "Frame 1m 12s (paused)"]
STRAY_GLYPHS = (":", "\u2014", "--")

FAILURES = []


def check(cond, name):
    """Record one assertion and print its verdict."""
    print("  %s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILURES.append(name)


def _render():
    """Render the Edit Masks template with a stub context, as app.index would."""
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
        review_tagging_enabled=False, session_mode="edit",
    )


def _render_combined():
    """Render the combinedAnnotate template with a stub context, as its app.index would."""
    labels = [{"code": "PA", "name": "Porites astreoides", "category": "Coral"}]
    env = Environment(loader=FileSystemLoader(_CA_TPL_DIR))
    return env.get_template("index.html").render(
        default_input="/tmp/in", default_export="/tmp/inprocess/run_demo/4test",
        default_categories=["Target species only"],
        all_categories=["Target species only", "Coral"],
        target_species=["PA"], target_labels=labels,
        default_review_dir="/tmp/review", default_contacts="a@b.c",
        target_species_only=True, reference_default=True,
        default_confidence=0.4, default_min_area=100, default_merge_dist=20,
        default_overlap="larger_wins", default_mask_size="generous",
        default_thin_ratio=0.12, default_simplify=1.5, default_exemplar=0.5,
        default_dev_tracker="cuda", default_dev_exemplar="cuda",
        orchestrator_url="", orchestrated=False, session_mode="configure",
        review_locked_msg="locked", review_tagging_enabled=False,
        default_session_name="26-09-26_LO_demo", session_label="", session_label_tooltip="",
    )


def _run_probe(probe_body, width, height=900, render=None):
    """Load the rendered page with `probe_body` run on load; return its JSON.

    The probe runs in a classic script appended before </body>, so it sees
    the page's own top-level functions and state as bare names. `render`
    picks the page (Edit Masks by default).
    """
    page = (render or _render)()
    if "</body>" not in page:
        raise RuntimeError("the rendered page has no </body> to put the probe before")
    probe = ("<script>window.addEventListener('load', () => { const out = (() => {%s})(); "
             "const pre = document.createElement('pre'); "
             "pre.textContent = '%s' + JSON.stringify(out) + '%s'; "
             "document.body.appendChild(pre); });</script>") % (probe_body, _MARK_START, _MARK_END)
    page = page.replace("</body>", probe + "\n</body>")
    fd, path = tempfile.mkstemp(suffix=".html", prefix="editmasks_probe_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(page)
        result = subprocess.run(
            [CHROME, *CHROME_ARGS, "--virtual-time-budget=2000",
             "--window-size=%d,%d" % (width, height), "--dump-dom", "file://" + path],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    finally:
        os.unlink(path)
    dom = result.stdout
    if _MARK_START not in dom or _MARK_END not in dom:
        raise RuntimeError("the probe never reported back (chrome rc=%s); stderr tail: %r"
                           % (result.returncode, result.stderr[-2000:]))
    start = dom.rindex(_MARK_START) + len(_MARK_START)
    return json.loads(dom[start:dom.index(_MARK_END, start)])


_MEASURE_JS = """
  const wrap = document.getElementById('canvas-wrap');
  const canvas = document.getElementById('canvas');
  const statusEl = document.getElementById('status-text');
  const measure = (label) => ({
    label,
    wrapTop: Math.round(wrap.getBoundingClientRect().top),
    canvasTop: Math.round(canvas.getBoundingClientRect().top),
    statusHeight: Math.round(statusEl.getBoundingClientRect().height),
    statusText: statusEl.textContent, statusTitle: statusEl.title,
  });
"""


def _status_probe():
    """Probe body: every timer stage times every message, flash and persistent, then cleared."""
    return """
  showScreen('review');
  %s
  const timerEl = document.getElementById('frame-timer');
  const out = [measure('start')];
  for (const t of %s) {
    timerEl.textContent = t;
    for (const msg of %s) {
      setStatus(msg); out.push(measure('flash ' + t + ' ' + msg.length));
      setConflictStatus(msg); out.push(measure('kept ' + t + ' ' + msg.length));
      setStatus(''); out.push(measure('cleared ' + t + ' ' + msg.length));
    }
  }
  return out;
""" % (_MEASURE_JS, json.dumps(TIMER_STAGES), json.dumps(STATUS_MESSAGES))


def test_status_messages_never_move_the_canvas():
    """The canvas holds still while every status message appears and clears."""
    for width in WIDTHS:
        rows = _run_probe(_status_probe(), width)
        wrap_tops = sorted({r["wrapTop"] for r in rows})
        canvas_tops = sorted({r["canvasTop"] for r in rows})
        heights = sorted({r["statusHeight"] for r in rows})
        check(len(wrap_tops) == 1,
              "%dpx: the canvas area's top is the same through every message and timer "
              "stage, saw %s" % (width, wrap_tops))
        check(len(canvas_tops) == 1,
              "%dpx: the canvas's own top is the same through every message, saw %s"
              % (width, canvas_tops))
        check(len(heights) == 1 and heights[0] > 0,
              "%dpx: the status row keeps one height, empty or full, saw %s" % (width, heights))
        longest = [r for r in rows if r["statusText"] == LONGEST_REFUSAL]
        check(bool(longest) and all(r["statusTitle"] == LONGEST_REFUSAL for r in longest),
              "%dpx: the longest refusal is whole in the status row's title" % width)


# Text the top bar's other items carry through a session: the batch line as
# the page loads and late in a long project, and the unlabeled count from none
# to three digits.
BATCH_LINES = ["", "Batch 1/3 (1-50 of 130)", "Batch 12/30 (551-600 of 1480)"]
UNLABELED_COUNTS = ["", "(3)", "(12)", "(999)"]


def _session_text_probe():
    """Probe body: badge, frame name, batch line and unlabeled count through their range."""
    return """
  showScreen('review');
  %s
  const badge = document.getElementById('bg-processing-badge');
  const nameEl = document.getElementById('image-filename');
  const batchEl = document.getElementById('batch-info');
  const countEl = document.getElementById('unlabeled-count');
  const jumpBtn = countEl.closest('button');
  const out = [];
  for (const batch of %s) {
    batchEl.textContent = batch;
    for (const count of %s) {
      countEl.textContent = count;
      for (const name of ['', %s]) {
        nameEl.textContent = name;
        for (const shown of [false, true]) {
          if (shown) { badge.style.display = ''; badge.style.visibility = 'visible'; }
          else { badge.style.display = 'none'; badge.style.visibility = 'hidden'; }
          const row = measure([batch.length, count, name.length, shown].join(' '));
          row.jumpWidth = Math.round(jumpBtn.getBoundingClientRect().width);
          out.push(row);
        }
      }
    }
  }
  return out;
""" % (_MEASURE_JS, json.dumps(BATCH_LINES), json.dumps(UNLABELED_COUNTS), json.dumps(LONG_FILENAME))


def test_session_text_never_moves_the_canvas():
    """The badge, frame name, batch line and unlabeled count leave the canvas where it is."""
    for width in WIDTHS:
        rows = _run_probe(_session_text_probe(), width)
        tops = sorted({r["wrapTop"] for r in rows})
        check(len(tops) == 1,
              "%dpx: the canvas top is the same with the badge shown or hidden, the frame "
              "name empty or full, every batch line and every unlabeled count, saw %s"
              % (width, tops))
        widths = sorted({r["jumpWidth"] for r in rows})
        check(len(widths) == 1,
              "%dpx: the Next unlabeled button keeps one width as its count changes, saw %s"
              % (width, widths))


_TIMER_PROBE = """
  showScreen('review');
  const timerEl = document.getElementById('frame-timer');
  const bar = document.querySelector('.review-topbar');
  const lone = () => Array.from(bar.querySelectorAll('*'))
    .filter(el => el.children.length === 0 && el.offsetParent !== null)
    .map(el => el.textContent.trim())
    .filter(t => %s.includes(t));
  const pause = document.getElementById('frame-timer-pause');
  const pauseState = () => ({disabled: pause.disabled, cursor: getComputedStyle(pause).cursor});
  const out = {asRendered: timerEl.textContent, loneAsRendered: lone()};
  _timerRender();
  out.noReading = timerEl.textContent;
  out.loneNoReading = lone();
  out.pauseIdle = pauseState();
  _frameStartMs = Date.now() - 4200;
  _frameTimesSec = [];
  _timerRender();
  out.reading = timerEl.textContent.trim();
  out.firstFrameTitle = timerEl.title;
  out.pauseLive = pauseState();
  _frameTimesSec = [12];
  _timerRender();
  out.laterTitle = timerEl.title;
  _frameStartMs = 0;
  _frameTimesSec = [];
  return out;
""" % json.dumps(list(STRAY_GLYPHS))


def test_the_timer_renders_nothing_until_it_has_a_reading():
    """No stray ':' (or other placeholder) before a frame; a reading when there is one."""
    for width in WIDTHS:
        r = _run_probe(_TIMER_PROBE, width)
        check(r["asRendered"].strip() == "",
              "%dpx: the timer is empty as the page renders (got %r)" % (width, r["asRendered"]))
        check(not r["loneAsRendered"],
              "%dpx: nothing in the top bar shows a lone placeholder glyph (found %r)"
              % (width, r["loneAsRendered"]))
        check(r["noReading"].strip() == "" and not r["loneNoReading"],
              "%dpx: a timer render with no frame yet still shows nothing (got %r)"
              % (width, r["noReading"]))
        check(r["reading"].startswith("Frame ") and any(ch.isdigit() for ch in r["reading"]),
              "%dpx: with a frame open the timer shows its reading (got %r)" % (width, r["reading"]))
        check("No session average yet." in r["firstFrameTitle"] and "--" not in r["firstFrameTitle"],
              "%dpx: on the first frame the timer's title says there is no session average yet "
              "(got %r)" % (width, r["firstFrameTitle"]))
        check("Session average: 12.0s over 1 frame." in r["laterTitle"],
              "%dpx: once a frame is timed, the session average is on the timer's title (got %r)"
              % (width, r["laterTitle"]))
        idle, live = r["pauseIdle"], r["pauseLive"]
        check(idle["disabled"] and idle["cursor"] not in ("pointer", "crosshair"),
              "%dpx: Pause is disabled, with no pointer, while the timer is empty (got %r)"
              % (width, idle))
        check(not live["disabled"],
              "%dpx: Pause is enabled once a frame has a reading (got %r)" % (width, live))


_FIT_PROBE = """
  showScreen('review');
  const nameEl = document.getElementById('image-filename');
  const batchEl = document.getElementById('batch-info');
  const textWidth = (el) => { const r = document.createRange(); r.selectNodeContents(el);
                              return Math.round(r.getBoundingClientRect().width); };
  nameEl.textContent = %s;
  batchEl.textContent = 'Batch 1/3 (1-50 of 130)';
  const shortGap = Math.round(batchEl.getBoundingClientRect().width) - textWidth(batchEl);
  batchEl.textContent = 'Batch 2/5 (51-100 of 230)';
  return {nameMinWidth: Math.round(parseFloat(getComputedStyle(nameEl).minWidth) || 0),
          nameTextWidth: textWidth(nameEl),
          nameClipped: nameEl.scrollWidth > nameEl.clientWidth + 1,
          nameWidth: Math.round(nameEl.getBoundingClientRect().width),
          shortGap,
          typicalClipped: batchEl.scrollWidth > batchEl.clientWidth + 1};
""" % json.dumps(LONG_FILENAME)
# The most room a fixed batch slot may leave empty after a short batch line.
MAX_BATCH_GAP_PX = 30


def test_frame_name_and_batch_line_fit():
    """A real frame name shows whole at both widths, and the batch slot fits its text."""
    for width in WIDTHS:
        r = _run_probe(_FIT_PROBE, width)
        check(r["nameMinWidth"] >= r["nameTextWidth"],
              "%dpx: the frame name's slot never shrinks below a real frame name, so it wraps "
              "with the timer instead of being cut (min %dpx, name %dpx)"
              % (width, r["nameMinWidth"], r["nameTextWidth"]))
        check(not r["nameClipped"],
              "%dpx: a real frame name shows whole, not cut off (width %dpx)" % (width, r["nameWidth"]))
        check(not r["typicalClipped"], "%dpx: a typical batch line shows whole" % width)
        check(r["shortGap"] <= MAX_BATCH_GAP_PX,
              "%dpx: a short batch line leaves at most %dpx empty in its slot (got %dpx)"
              % (width, MAX_BATCH_GAP_PX, r["shortGap"]))


_CA_TIMER_PROBE = """
  showScreen('review');
  const timerEl = document.getElementById('frame-timer');
  const pause = document.getElementById('frame-timer-pause');
  const pauseState = () => ({disabled: pause.disabled, cursor: getComputedStyle(pause).cursor});
  const out = {pauseAsRendered: pauseState()};
  _timerRender();
  out.pauseIdle = pauseState();
  out.noReading = timerEl.textContent;
  _frameStartMs = Date.now() - 4200;
  _frameTimesSec = [];
  _timerRender();
  out.firstFrameTitle = timerEl.title;
  out.pauseLive = pauseState();
  _frameTimesSec = [12];
  _timerRender();
  out.laterTitle = timerEl.title;
  _frameStartMs = 0;
  _frameTimesSec = [];
  return out;
"""


def test_combined_timer_matches_edit_masks():
    """combinedAnnotate: no session average yet on the first frame; Pause only with a reading."""
    for width in WIDTHS:
        r = _run_probe(_CA_TIMER_PROBE, width, render=_render_combined)
        check("No session average yet." in r["firstFrameTitle"] and "--" not in r["firstFrameTitle"],
              "combinedAnnotate %dpx: on the first frame the timer's title says there is no session "
              "average yet (got %r)" % (width, r["firstFrameTitle"]))
        check("Session average: 12.0s over 1 frame." in r["laterTitle"],
              "combinedAnnotate %dpx: once a frame is timed, the session average is on the timer's "
              "title (got %r)" % (width, r["laterTitle"]))
        for name in ("pauseAsRendered", "pauseIdle"):
            state = r[name]
            check(state["disabled"] and state["cursor"] not in ("pointer", "crosshair"),
                  "combinedAnnotate %dpx: Pause is disabled, with no pointer, while the timer has no "
                  "reading (%s: %r)" % (width, name, state))
        check(not r["pauseLive"]["disabled"],
              "combinedAnnotate %dpx: Pause is enabled once a frame has a reading (got %r)"
              % (width, r["pauseLive"]))
        check(r["noReading"].strip() == "",
              "combinedAnnotate %dpx: with no reading the timer shows nothing, no placeholder "
              "(got %r)" % (width, r["noReading"]))


def _run(case):
    """Run one case; a probe that never reports back fails that case only."""
    print(case.__name__)
    try:
        case()
    except Exception as exc:
        check(False, "%s raised %s: %s" % (case.__name__, type(exc).__name__, exc))


def main():
    """Run every case and return the process exit code."""
    for case in (test_status_messages_never_move_the_canvas,
                 test_session_text_never_moves_the_canvas,
                 test_the_timer_renders_nothing_until_it_has_a_reading,
                 test_frame_name_and_batch_line_fit,
                 test_combined_timer_matches_edit_masks):
        _run(case)
    print("PASS" if not FAILURES else "%d FAILED" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
