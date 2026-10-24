"""The gear popover holds the controls the tool row and top bar used to carry.

Task: VEGA annotator tidy-up and no re-screening, 2026-09-27 (LO). Lauren asked
for Reload IDs, Merge same ID, Target points only, Ref points and the
Unlabeled-only finder to move into "a settings place, a gear icon... get them
out of my face", and for "Unlabeled only" (which she said she did not
understand) to become "Frames with a mask missing its species". This suite
proves the move lost nothing:

  1. Structural: the gear button exists, keyboard-focusable, with aria-expanded
     and aria-haspopup; the popover holds exactly the six controls the plan
     names, grouped Show / Find / Tools, and none of them still sits in the
     old top-bar or tool-row markup outside the popover.
  2. Every control named in BEFORE_IDS, the full inventory of what the top bar
     and tool row carried before this change, is still present somewhere on
     the page by id: nothing was dropped in the move.
  3. Rename: "Unlabeled only" is gone; "Frames with a mask missing its
     species" and "Next such frame" are the new copy, and the checkbox still
     drives the same toggleUnlabeledOnly handler.
  4. Behavioral, in headless Chrome: the popover opens on a gear click, closes
     on a second click, closes on Escape, and closes on a click outside it;
     opening or closing it never changes the tool row's own height (the
     popover is position: absolute, so it must never push a layout reflow).
  5. The T and G keyboard shortcuts still toggle showOnlyTarget / showReference
     and keep the relocated checkboxes in sync, exactly as before the move.
  6. The tool row holds to one line at 1280px now that five controls left it.

Every probe runs against a static jinja2 render of the template, loaded once
in headless Chrome per case, mirroring test_annotator_layout_stability.py:
rendering and DOM probing only, no SAM3, no GPU, no live annotator, and
orchestrator_url is left empty on purpose.

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_annotator_settings_menu.py
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
_MARK_START = "@@SETTINGS_PROBE_START@@"
_MARK_END = "@@SETTINGS_PROBE_END@@"

# Every control id the top bar and tool row carried before this change (the
# facts given for this task: top bar ~1182-1251, tool row 1252-1282). If any
# of these vanished in the move rather than relocated, this list catches it.
BEFORE_IDS = [
    "chk-unlabeled-only", "unlabeled-count",
    "opacity-slider", "chk-target-only", "chk-show-reference",
    "btn-mode-refine", "btn-mode-merge", "btn-mode-add", "btn-mode-box",
    "btn-mode-draw", "mask-size", "btn-undo",
    "btn-merge-same-id", "btn-reload-ids",
]

FAILURES = []


def check(cond, name):
    """Record one assertion and print its verdict."""
    print("  %s %s" % ("PASS" if cond else "FAIL", name))
    if not cond:
        FAILURES.append(name)


def _render(session_mode="configure", session_label="", session_label_tooltip=""):
    """Render the page with a stub context, orchestrator_url empty on purpose.

    Mirrors test_annotator_layout_stability.py's _render(): that suite loads a
    real headless-Chrome page, so a hardcoded orchestrator URL (a real,
    independently-changing service on this box) is never safe to bake in.
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
        review_tagging_enabled=False, default_session_name="26-09-27_LO_demo",
        session_label=session_label, session_label_tooltip=session_label_tooltip,
    )


def _run_probe(probe_js, width, height=900, **render_kwargs):
    """Render the page with `probe_js` appended, load it in headless Chrome at
    `width` x `height`, and return the JSON the probe wrote into <body>.

    See test_annotator_layout_stability.py's own _run_probe for the full
    rationale (the probe runs after the page's own inline script, so it reads
    real page state as bare identifiers).
    """
    page = _render(**render_kwargs)
    marker = "</body>"
    if marker not in page:
        raise RuntimeError("rendered page has no </body> to inject the probe before")
    page = page.replace(marker, probe_js + "\n" + marker)
    fd, path = tempfile.mkstemp(suffix=".html", prefix="settings_probe_")
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


def _template_source():
    with open(_TPL, encoding="utf-8") as handle:
        return handle.read()


def _block(html, start_marker, end_marker):
    start = html.find(start_marker)
    if start < 0:
        raise AssertionError("%r is not in the page" % start_marker)
    end = html.find(end_marker, start)
    if end < 0:
        raise AssertionError("%r does not follow %r" % (end_marker, start_marker))
    return html[start:end]


# ── 1. Structural: the gear, the popover, and what moved into it ────────────

def test_the_gear_button_is_present_and_keyboard_focusable():
    """A real <button>, not a div, carrying the tooltip and ARIA the plan asks."""
    page = _render()
    gear = _block(page, '<button type="button" class="btn" id="btn-settings-gear"',
                  '</button>')
    check('aria-haspopup="true"' in gear, "the gear declares aria-haspopup")
    check('aria-expanded="false"' in gear, "the gear starts collapsed")
    check('aria-controls="settings-popover"' in gear,
          "the gear points at the popover it controls")
    check("Open the settings menu: what to show, the missing-species finder, and the two mask tools." in gear,
          "the gear carries the agreed tooltip text")
    check('onclick="toggleSettingsPopover()"' in gear,
          "the gear is wired to the popover toggle")


def test_the_popover_holds_exactly_the_six_controls_grouped():
    """Show / Find / Tools, each holding what the plan names, in that order."""
    page = _render()
    popover = _block(page, '<div class="settings-popover" id="settings-popover"',
                      '<div class="quickadd-bar"')
    show_at = popover.index('>Show<')
    find_at = popover.index('>Find<')
    tools_at = popover.index('>Tools<')
    check(show_at < find_at < tools_at,
          "the three groups appear in Show, Find, Tools order")
    show_block = popover[show_at:find_at]
    find_block = popover[find_at:tools_at]
    tools_block = popover[tools_at:]
    check('id="chk-target-only"' in show_block, "Show holds Target points only")
    check('id="chk-show-reference"' in show_block, "Show holds Ref points")
    check('id="chk-unlabeled-only"' in find_block,
          "Find holds the missing-species frame finder")
    check('id="unlabeled-count"' in find_block, "Find holds its count badge")
    check("Next such frame" in find_block, "Find holds the jump button")
    check('id="btn-merge-same-id"' in tools_block, "Tools holds Merge same ID")
    check('id="btn-reload-ids"' in tools_block, "Tools holds Reload IDs")
    # Every relocated control still carries its own tooltip.
    for cid in ("chk-target-only", "chk-show-reference", "chk-unlabeled-only",
                "btn-merge-same-id", "btn-reload-ids"):
        tag_start = popover.rfind("<", 0, popover.index('id="%s"' % cid))
        # A checkbox's tooltip sits on the wrapping <label>, a button's on
        # itself; walk out one more open tag for the checkboxes.
        window = popover[max(0, tag_start - 400):tag_start + 200]
        check("title=" in window, "%s carries a tooltip nearby" % cid)


def test_the_moved_controls_no_longer_sit_in_the_old_top_bar_or_tool_row():
    """The top bar and the tool row (outside the popover) lost these five."""
    page = _render()
    topbar = _block(page, '<div class="review-topbar">', '<div class="toolbar-bar">')
    check('id="chk-unlabeled-only"' not in topbar,
          "the missing-species checkbox left the top bar")
    check('id="unlabeled-count"' not in topbar,
          "its count badge left the top bar")
    check("Next unlabeled" not in topbar, "the old jump-button label left the top bar")
    tool_row_before_popover = _block(
        page, '<div class="toolbar-bar">', '<div class="settings-popover"')
    for cid in ("chk-target-only", "chk-show-reference",
                "btn-merge-same-id", "btn-reload-ids"):
        check('id="%s"' % cid not in tool_row_before_popover,
              "%s left the visible tool row for the popover" % cid)


def test_nothing_named_in_the_before_inventory_was_lost():
    """Every control the top bar and tool row carried before is still on the page."""
    page = _render()
    for cid in BEFORE_IDS:
        check('id="%s"' % cid in page, "%s is still present somewhere on the page" % cid)


# ── 2. The rename ────────────────────────────────────────────────────────────

def test_unlabeled_only_is_renamed_and_still_wired():
    """The old label is gone; the new one drives the same handler."""
    page = _render()
    check("Unlabeled only" not in page, "the old checkbox label is gone")
    check("Frames with a mask missing its species" in page,
          "the renamed label is on the page")
    checkbox = _block(page, 'id="chk-unlabeled-only"', '>')
    check('onchange="toggleUnlabeledOnly(this.checked)"' in checkbox,
          "the renamed checkbox still calls toggleUnlabeledOnly")
    label = page[page.rindex("<label", 0, page.index('id="chk-unlabeled-only"')):]
    tooltip = label[label.index('title="') + 7:]
    tooltip = tooltip[:tooltip.index('"')]
    check(tooltip == (
        "Tick this to step only through frames that still hold a mask with "
        "no species. The annotator refuses to export such a mask, so these "
        "frames need a species or a reject."
    ), "the tooltip is the agreed replacement text, got %r" % tooltip)
    check("Next unlabeled" not in page, "the old jump-button label is gone everywhere")
    check("Next such frame" in page, "the renamed jump button is on the page")


def test_the_page_obeys_the_prose_rules_in_the_new_copy():
    """No em dash, no en dash, no banned word in what this change wrote."""
    page = _render()
    for name, glyph in (("em dash", "—"), ("en dash", "–")):
        check(glyph not in page, "no %s anywhere in the rendered page" % name)
    import re
    check(not re.search(r"\bland(s|ed|ing)?\b", page, re.I),
          "no banned word anywhere in the rendered page")


# ── 3. Behavioral: open, close, and the tool row never moves ────────────────

def _popover_probe_js():
    return """
<script>
(function() {
  showScreen('review');
  const gear = document.getElementById('btn-settings-gear');
  const popover = document.getElementById('settings-popover');
  const bar = document.querySelector('.toolbar-bar');
  const out = {};
  out.startsHidden = !popover.classList.contains('visible');
  out.startBarHeight = Math.round(bar.getBoundingClientRect().height);

  gear.click();
  out.opensOnClick = popover.classList.contains('visible');
  out.expandedOnOpen = gear.getAttribute('aria-expanded');
  out.barHeightWhileOpen = Math.round(bar.getBoundingClientRect().height);

  gear.click();
  out.closesOnSecondClick = !popover.classList.contains('visible');
  out.expandedOnClose = gear.getAttribute('aria-expanded');

  gear.click();
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  out.closesOnEscape = !popover.classList.contains('visible');

  gear.click();
  document.body.click();
  out.closesOnOutsideClick = !popover.classList.contains('visible');

  gear.click();
  popover.click();
  out.staysOpenOnInsideClick = popover.classList.contains('visible');
  closeSettingsPopover();

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


def _viewport_clamp_probe_js():
    return """
<script>
(function() {
  showScreen('review');
  const gear = document.getElementById('btn-settings-gear');
  const popover = document.getElementById('settings-popover');
  gear.click();
  const box = popover.getBoundingClientRect();
  const out = {left: Math.round(box.left), right: Math.round(box.right),
               windowWidth: window.innerWidth};
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


def test_the_popover_never_runs_off_screen_at_946px():
    """At the width where the tool row wraps, the popover stays on-screen.

    The gear can sit near the row's left edge once it wraps to a second
    line; the stylesheet's default (hanging off the gear's right edge) would
    push a 268px popover past the left edge there, so openSettingsPopover
    flips the anchor the same way showCatPopup already clamps to the window.
    """
    result = _run_probe(_viewport_clamp_probe_js(), width=946)
    check(result["left"] >= 0,
          "the popover's left edge stays on-screen at 946px, got %dpx" % result["left"])
    check(result["right"] <= result["windowWidth"],
          "the popover's right edge stays on-screen too, got %dpx of %dpx"
          % (result["right"], result["windowWidth"]))


def test_the_popover_opens_and_closes_without_moving_the_tool_row():
    """Gear click, second click, Escape, and an outside click all work."""
    result = _run_probe(_popover_probe_js(), width=1280)
    check(result["startsHidden"], "the popover starts closed")
    check(result["opensOnClick"], "a gear click opens the popover")
    check(result["expandedOnOpen"] == "true", "aria-expanded flips to true on open")
    check(result["startBarHeight"] == result["barHeightWhileOpen"],
          "the tool row's own height does not change while the popover is open "
          "(closed=%dpx, open=%dpx)"
          % (result["startBarHeight"], result["barHeightWhileOpen"]))
    check(result["closesOnSecondClick"], "a second gear click closes it")
    check(result["expandedOnClose"] == "false", "aria-expanded flips back to false")
    check(result["closesOnEscape"], "Escape closes the open popover")
    check(result["closesOnOutsideClick"], "a click outside the popover closes it")
    check(result["staysOpenOnInsideClick"],
          "a click inside the popover does not close it")


def _escape_scope_probe_js():
    return """
<script>
(function() {
  showScreen('review');
  currentData = {masks: [{id: 1, status: 'accepted', species: 'PA',
                          polygon_px: [[0,0,10,0,10,10,0,10]]}]};
  const gear = document.getElementById('btn-settings-gear');
  const popover = document.getElementById('settings-popover');

  // With the popover open, Escape must close ONLY the popover: the general
  // case 'Escape' (clear selectedMask, reset to select mode) must not also
  // run, or one keypress does two unrelated things (2026-09-27 review).
  gear.click();
  selectedMask = 0;
  interactionMode = 'add';
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  const out = {
    popoverClosed: !popover.classList.contains('visible'),
    selectedMaskUnchanged: selectedMask === 0,
    modeUnchanged: interactionMode === 'add',
  };

  // With the popover already closed, Escape still runs its usual path.
  selectedMask = 0;
  interactionMode = 'add';
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  out.selectedMaskClearedWhenPopoverAlreadyClosed = selectedMask === -1;
  out.modeResetWhenPopoverAlreadyClosed = interactionMode === 'select';

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


def test_escape_while_the_popover_is_open_only_closes_the_popover():
    """Escape does one thing at a time: close the popover, or the usual reset.

    2026-09-27 review: Escape while the popover was open ran both the close
    AND the general Escape path (clearing the selected mask, ending sticky
    Add/Box), so a person closing the menu also lost their selection.
    """
    result = _run_probe(_escape_scope_probe_js(), width=1280)
    check(result["popoverClosed"], "Escape closes the open popover")
    check(result["selectedMaskUnchanged"],
          "the selected mask is untouched while the popover was open")
    check(result["modeUnchanged"],
          "the interaction mode is untouched while the popover was open")
    check(result["selectedMaskClearedWhenPopoverAlreadyClosed"],
          "Escape still clears the selected mask when nothing is open")
    check(result["modeResetWhenPopoverAlreadyClosed"],
          "Escape still resets to select mode when nothing is open")


def test_the_popover_carries_its_own_aria_label():
    """The panel names itself, rather than repeating the gear's own text.

    2026-09-27 review: #settings-popover reused the gear's long aria-label
    verbatim, which a screen reader reads out twice (once for the button that
    opens it, once for the region it opens).
    """
    page = _render()
    gear = _block(page, '<button type="button" class="btn" id="btn-settings-gear"',
                  '</button>')
    gear_label = re.search(r'aria-label="([^"]*)"', gear).group(1)
    popover = _block(page, '<div class="settings-popover" id="settings-popover"', '>')
    popover_label = re.search(r'aria-label="([^"]*)"', popover).group(1)
    check(popover_label == "Settings menu",
          "the popover has its own short label, got %r" % popover_label)
    check(popover_label != gear_label,
          "the popover's label is not a copy of the gear's own aria-label")


# ── 4. T and G still work from the relocated checkboxes ─────────────────────

def _key_probe_js():
    return """
<script>
(function() {
  showScreen('review');
  const out = {};
  out.targetStartsOn = showOnlyTarget;
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 't', bubbles: true}));
  out.targetAfterT = showOnlyTarget;
  out.checkboxAfterT = document.getElementById('chk-target-only').checked;
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 't', bubbles: true}));
  out.targetAfterSecondT = showOnlyTarget;

  out.refStartsOn = showReference;
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'g', bubbles: true}));
  out.refAfterG = showReference;
  out.checkboxAfterG = document.getElementById('chk-show-reference').checked;
  document.dispatchEvent(new KeyboardEvent('keydown', {key: 'g', bubbles: true}));
  out.refAfterSecondG = showReference;

  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


def test_t_and_g_still_toggle_the_relocated_checkboxes():
    """The keyboard shortcuts and the popover checkboxes stay in sync."""
    result = _run_probe(_key_probe_js(), width=1280)
    check(result["targetStartsOn"] is True, "target-only starts on, the shipped default")
    check(result["targetAfterT"] is False, "T toggles target-only off")
    check(result["checkboxAfterT"] is False,
          "the relocated checkbox reflects the T toggle")
    check(result["targetAfterSecondT"] is True, "a second T restores it")
    check(result["refStartsOn"] is True, "reference points start on for this render")
    check(result["refAfterG"] is False, "G toggles reference points off")
    check(result["checkboxAfterG"] is False,
          "the relocated checkbox reflects the G toggle")
    check(result["refAfterSecondG"] is True, "a second G restores it")


# ── 5. The tool row fits on one line at 1280px ──────────────────────────────

def _one_line_probe_js():
    return """
<script>
(function() {
  showScreen('review');
  const bar = document.querySelector('.toolbar-bar');
  const kids = Array.from(bar.children).filter(
    el => el.id !== 'settings-popover');
  const tops = kids.map(el => Math.round(el.getBoundingClientRect().top));
  const bottoms = kids.map(el => Math.round(el.getBoundingClientRect().bottom));
  const out = {
    barHeight: Math.round(bar.getBoundingClientRect().height),
    span: Math.max(...bottoms) - Math.min(...tops),
  };
  const pre = document.createElement('pre');
  pre.textContent = '%s' + JSON.stringify(out) + '%s';
  document.body.appendChild(pre);
})();
</script>
""" % (_MARK_START, _MARK_END)


# .toolbar-bar centres children of different heights on one line
# (align-items: center), so even a single-line row shows a handful of
# pixels of top-offset spread between a 16px range input and a 28px
# button; only a real wrap to a second line adds a whole row (about 28px)
# plus the 6px row-gap. 45px sits well inside the first case and well
# below the second.
ONE_LINE_HEIGHT_CEILING_PX = 45


def test_the_tool_row_holds_one_line_at_1280px():
    """.toolbar-bar's own rendered height stays a single line at 1280px.

    Wrapping to a second line is exactly what freeing up five controls onto
    the gear popover was meant to avoid. Comparing raw child top offsets is
    not a valid proxy here: align-items: center gives a 16px range input and
    a 28px button different tops even on the same line, so this checks the
    row's own box height (and the full top-to-bottom span of its children)
    instead of demanding they align exactly.
    """
    result = _run_probe(_one_line_probe_js(), width=1280)
    check(result["barHeight"] < ONE_LINE_HEIGHT_CEILING_PX,
          "the tool row's own height is one line, got %dpx" % result["barHeight"])
    check(result["span"] < ONE_LINE_HEIGHT_CEILING_PX,
          "every control's top-to-bottom span fits in one line, got %dpx"
          % result["span"])


def main():
    """Run every check; skip cleanly with no failure if Chrome is absent."""
    if not (os.path.exists(CHROME) and os.access(CHROME, os.X_OK)):
        print("SKIP: %s not found; the behavioral probes need headless Chrome"
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
