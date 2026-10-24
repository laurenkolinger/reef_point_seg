# CA/tests/test_review_lock_ui.py
"""Expert-review flagging is locked in the browser, and the read path is not.

Lauren Olinger, 2026-09-06: no new review tag may be added to a mask in any
usage of this annotator (the in-project Place plus Segment app on port 5080,
the Refine loop in resume mode, and the standalone Annotate Images app on port
5096). One template serves all three, so one change covers all three. The
REVIEW control stays on screen, visibly disabled, and explains itself in a
tooltip; the server refuses the request underneath (a sibling change).

Updated 2026-09-27 (LO review of the VEGA annotator tidy-up plan): the
quickadd bar's own REVIEW button was a control nobody could click while the
lock holds, sitting in the row Lauren wants decluttered, so it is left out of
that bar entirely instead of shown disabled; it renders again, live, the
moment `review_tagging_enabled` is true. The label popup's REVIEW chip keeps
the original disabled-not-removed treatment, since it sits among the species
codes an operator is actively choosing from and losing it there would read as
a missing option rather than a suspended one.

What this suite proves:
  - the quick-add REVIEW button is entirely absent from the markup while the
    lock holds, and reappears live (its key badge, its own tooltip, wired to
    toggleQuickReview) the moment `review_tagging_enabled` is true;
  - the label popup's REVIEW chip is still there, carries an id, is inert
    (no onclick, aria-disabled), and carries its own locked tooltip;
  - neither R key binding can arm REVIEW or leave the app armed;
  - `armedMaskFields`, which decides what a newly placed mask is born
    carrying, can never return a review tag while the lock holds;
  - no path that touches many masks at once can apply REVIEW;
  - the read path is untouched: a mask already flagged still paints amber,
    still survives the target-only filter, and is still counted;
  - none of the strings this change added breaks the platform prose rules.

Two idioms, both already used in this folder: parse the rendered page as text
(test_unlabeled_filter.py) and slice a function out of the template to run it
under node (test_target_only_shows_pending.py).

No pytest, GPU-free. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_review_lock_ui.py
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

import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_TPL = os.path.join(_SRC, "templates", "index.html")
sys.path.insert(0, _SRC)

# ── The copy under test, verbatim from the plan's section 4 ──────────────────
# The quickadd bar's own locked tooltip (review_locked_button_tooltip) was
# removed 2026-09-27 along with the button it was the only user of: the
# control does not render while locked, so it has no tooltip to check.
CHIP_TOOLTIP = (
    "VICARIUS has temporarily suspended expert-review flagging, so the REVIEW "
    "label cannot be applied to a mask. Pick a species code instead, or press "
    "Escape to leave the mask pending."
)
LOCKED_MSG = (
    "Flagging for expert review is temporarily unavailable. VICARIUS is "
    "accepting no new review tags."
)
# The expert-review folder tooltip, in two halves. The first sentence is the
# same in both states and carries the only fact the tooltip adds beyond its
# label: where the folder is. The second says what happens to the folder, and it
# is the half the lock rewrites, so the tooltip never says the export both runs
# and is suspended.
EXPORT_FOLDER_TOOLTIP_LEAD = (
    "Set the local folder holding the public expert-review site, a git clone "
    "of laurenkolinger/reefpointseg-review with GitHub Pages switched on; the "
    "default is /mnt/tear/REVIEW_reefpointseg."
)
EXPORT_FOLDER_TOOLTIP_RUNNING = (
    "VICARIUS writes every mask already flagged for expert review into that "
    "folder and pushes it so an expert can open them at the Pages URL."
)
EXPORT_FOLDER_TOOLTIP_SUSPENDED = (
    "VICARIUS has temporarily suspended the expert-review export, so no mask "
    "is written into that folder and nothing is pushed."
)
LOCKED_EXPORT_FOLDER_TOOLTIP = (
    EXPORT_FOLDER_TOOLTIP_LEAD + " " + EXPORT_FOLDER_TOOLTIP_SUSPENDED)
UNLOCKED_EXPORT_FOLDER_TOOLTIP = (
    EXPORT_FOLDER_TOOLTIP_LEAD + " " + EXPORT_FOLDER_TOOLTIP_RUNNING)
# Every string this change writes into the page, checked for banned prose.
ADDED_STRINGS = [CHIP_TOOLTIP, LOCKED_MSG, LOCKED_EXPORT_FOLDER_TOOLTIP]

# The amber a flagged mask must keep painting, from REVIEW_COLORS.review.
AMBER_AT_90_PERCENT_ALPHA = "hsla(40, 92%, 55%, 0.9)"

_fail = 0


def check(c, m):
    """Record one assertion. Prints PASS or FAIL and counts the failures."""
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _template_source():
    """Return the raw Jinja template text.

    Returns:
        str: the whole of src/templates/index.html.
    """
    with open(_TPL) as fh:
        return fh.read()


def _rendered_page(review_tagging_enabled=False):
    """Render the annotator page through the app's own test client.

    Rendering rather than reading the template proves the Jinja lock branch
    resolves, including the in-template fallback used while app.py has not yet
    started passing the lock variables.

    The route reads mask_ops' REVIEW_TAGGING_ENABLED out of the app module at
    request time, so setting it for the length of one request renders the page
    exactly as flipping the one constant would. It is always put back.

    Args:
        review_tagging_enabled: the state of the expert-review switch to render
            under. False, the shipped state, is the default.

    Returns:
        str: the rendered HTML of GET /.
    """
    import app as A
    A.app.config["TESTING"] = True
    previous = A.REVIEW_TAGGING_ENABLED
    A.REVIEW_TAGGING_ENABLED = review_tagging_enabled
    try:
        res = A.app.test_client().get("/")
    finally:
        A.REVIEW_TAGGING_ENABLED = previous
    if res.status_code != 200:
        raise AssertionError("GET / returned %s, not 200" % res.status_code)
    return res.get_data(as_text=True)


def _folder_tooltip(page):
    """Return the expert-review folder tooltip out of a rendered page.

    Args:
        page: the rendered HTML.

    Returns:
        str: the title attribute of the expert-review folder label.

    Raises:
        AssertionError: when the label or its tooltip is not on the page.
    """
    found = re.search(r'<label title="([^"]*)">\s*Expert-review folder', page)
    if not found:
        raise AssertionError("the expert-review folder label carries no tooltip")
    return found.group(1)


def _sentence_count(text):
    """Count the sentences in one tooltip.

    Every sentence in this copy ends in a full stop, and no full stop appears
    inside it (the folder path and the repository name carry none), so the full
    stops are the sentences.

    Args:
        text: the tooltip text.

    Returns:
        int: how many sentences it holds.
    """
    return text.count(".")


def _tag_carrying(html, needle):
    """Return the single HTML tag that contains `needle`.

    Args:
        html: the document to search.
        needle: a substring that appears inside exactly one opening tag,
            for example 'id="qa-btn-review"'.

    Returns:
        str: the opening tag text, angle brackets included.

    Raises:
        AssertionError: when the needle is absent from the document.
    """
    pos = html.find(needle)
    if pos < 0:
        raise AssertionError("%r is not in the page" % needle)
    start = html.rfind("<", 0, pos)
    end = html.find(">", pos)
    if start < 0 or end < 0:
        raise AssertionError("%r is not inside a tag" % needle)
    return html[start:end + 1]


def _extract_function(name, src=None):
    """Slice one top-level JavaScript function verbatim out of the template.

    Every function in this template closes with a brace in column one, so the
    first '\\n}\\n' after the declaration is its end.

    Args:
        name: the function name, without 'function'.
        src: template text to search; read from disk when omitted.

    Returns:
        str: the function source, declaration through closing brace.
    """
    src = src if src is not None else _template_source()
    marker = "\nfunction %s(" % name
    start = src.index(marker) + 1
    end = src.index("\n}\n", start) + len("\n}\n")
    return src[start:end]


def _top_level_functions(src=None):
    """Map every top-level JavaScript function name to its source text.

    A top-level function starts at column one with 'function' or 'async
    function' and ends at the next line that is exactly '}'.

    Args:
        src: template text to scan; read from disk when omitted.

    Returns:
        dict[str, str]: function name to function body text.
    """
    src = src if src is not None else _template_source()
    out = {}
    lines = src.splitlines()
    head = re.compile(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")
    i = 0
    while i < len(lines):
        m = head.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j] != "}":
            j += 1
        out[m.group(1)] = "\n".join(lines[i:j + 1])
        i = j + 1
    return out


# The spellings that actually put a review tag on a mask, as opposed to a
# comment or a read predicate that merely names the label.
_REVIEW_WRITE_SPELLINGS = (
    "species: 'REVIEW'", 'species: "REVIEW"',
    "species = 'REVIEW'", 'species = "REVIEW"',
    "review: true", "review = true",
)


def _strip_js_comments(text):
    """Drop // line comments so prose about REVIEW is not read as a write."""
    return "\n".join(re.sub(r"//.*$", "", line) for line in text.splitlines())


def _writes_a_review_tag(body):
    """Report whether a function body contains a review-tag write."""
    code = _strip_js_comments(body)
    return any(spell in code for spell in _REVIEW_WRITE_SPELLINGS)


def _is_behind_the_lock(body):
    """Report whether a function body consults the expert-review lock."""
    return "reviewLockGuard()" in body or "REVIEW_TAGGING_ENABLED" in body


def _find_node():
    """Return a usable node binary path, or None when node is unavailable."""
    for cand in ("node", "/usr/bin/node"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


def _run_node(harness, label):
    """Run one JavaScript harness under node and report its exit status.

    Args:
        harness: the complete script to run.
        label: the assertion message for the PASS or FAIL line.
    """
    node = _find_node()
    if node is None:
        print("    (node unavailable, skipping) " + label)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        res = subprocess.run([node, path], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            print("    " + line)
        if res.stderr.strip():
            print("    [stderr] " + res.stderr.strip())
        check(res.returncode == 0, label)
    finally:
        os.unlink(path)


# ── The quickadd control: absent while locked, live once unlocked ────────────
# 2026-09-27 (LO review): Lauren asked for the label row decluttered, and a
# control nobody can click ("Expert review (unavailable)") was exactly the
# kind of thing she meant. Unlike the label popup's REVIEW chip (kept on
# screen and disabled, since it sits among species codes an operator is
# actively choosing from), the quickadd bar's REVIEW button is left out of the
# markup entirely while the lock holds, and comes back live, unchanged from
# before, once review_tagging_enabled flips true.

def test_review_button_is_absent_while_locked():
    """The quick-add REVIEW button renders nowhere in the locked page."""
    page = _rendered_page()
    check('id="qa-btn-review"' not in page,
          "the quick-add REVIEW button is not in the markup while locked")
    check("Expert review (unavailable)" not in page,
          "no disabled-looking REVIEW control is left in its place")


def test_review_button_is_live_once_unlocked():
    """Flipping the flag brings the button back exactly as it worked before."""
    page = _rendered_page(review_tagging_enabled=True)
    check('id="qa-btn-review"' in page,
          "the quick-add REVIEW button renders once the feature is live")
    tag = _tag_carrying(page, 'id="qa-btn-review"')
    check("disabled" not in tag, "it carries no disabled attribute when live")
    check('onclick="toggleQuickReview()"' in tag, "and it is wired to arm REVIEW")
    check("Tap R to flag the next placed mask for expert REVIEW" in tag,
          "carrying its live tooltip")
    start = page.index('id="qa-btn-review"')
    block = page[start:page.index("</button>", start)]
    check('<span class="qa-key">R</span>' in block,
          "and its key badge, so the R shortcut reads as a live hint again")


def test_review_chip_is_present_disabled_and_has_an_id():
    """The label popup REVIEW chip stays, gains an id, and is inert."""
    page = _rendered_page()
    check('id="cat-btn-review"' in page, "label popup REVIEW chip carries an id")
    tag = _tag_carrying(page, 'id="cat-btn-review"')
    check("onclick" not in tag, "label popup REVIEW chip has no click handler left")
    check('aria-disabled="true"' in tag, "label popup REVIEW chip is marked disabled")
    check('tabindex="0"' in tag, "label popup REVIEW chip is keyboard reachable for its tooltip")
    check(CHIP_TOOLTIP in tag, "label popup REVIEW chip tooltip is the locked copy")
    check("cat-btn-locked" in tag, "label popup REVIEW chip carries the locked class")
    check("cat-btn-locked" in page.split("<script>")[0],
          "a locked rule for the popup chip exists so the state actually paints")
    check("hsl(40,92%,68%)" not in tag,
          "the locked chip drops the inline amber that would beat the disabled rule")


# ── The keyboard: neither R binding can arm REVIEW ───────────────────────────

def test_r_key_no_longer_arms_review():
    """Both R bindings refuse, say so, and leave nothing armed."""
    src = _template_source()
    check("armReview();" in src, "the main R key still routes through armReview")
    check("assignNewMaskReview();" in src,
          "the label popup r key still routes through assignNewMaskReview")
    harness = """
const REVIEW_TAGGING_ENABLED = false;
const REVIEW_LOCKED_MSG = %(msg)s;
let armedReview = false;
let numKeyDown = null;
let stickyAddMode = false;
let stickyBoxMode = false;
let interactionMode = 'select';
let autoAdvance = false;
let pendingScanCandidateIdx = -1;
let pendingAddMaskIdx = 0;
let currentData = {masks: [{id: 1, label: 'A', species: '', status: 'pending'}]};
let statusLines = [];
let popupHidden = false;
let persisted = [];
let labeledCandidates = [];
function setStatus(s) { statusLines.push(s); }
function setReviewBtnActive() {}
function highlightQuickBtn() {}
function _updateStickyAddIndicator() {}
function setInteractionMode(m) { interactionMode = m; }
function hideCatPopup() { popupHidden = true; }
function persistMaskReview(id) { persisted.push(id); }
function editBlockedByLoad() { return false; }
function labelScanCandidate(f) { labeledCandidates.push(f); }
function flashMode() {}
function resumeStickyMode() {}
function renderMaskList() {}
function render() {}
%(guard)s
%(armReview)s
%(assign)s
let fails = 0;
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
// Three presses of R, the way an operator who does not know yet would press it.
armReview(); armReview(); armReview();
want(armedReview === false, 'R never arms REVIEW');
want(interactionMode === 'select', 'R does not push the app into the sticky add loop');
want(stickyAddMode === false, 'R leaves the sticky add loop alone');
want(numKeyDown === null, 'R leaves no armed label behind');
want(statusLines.length === 3, 'R answers every press');
want(statusLines.every(s => s === REVIEW_LOCKED_MSG), 'R answers with the locked message only');
// Box mode is the second mode the key works in.
interactionMode = 'box'; stickyBoxMode = true;
armReview();
want(armedReview === false, 'R never arms REVIEW in box mode either');
want(interactionMode === 'box', 'R does not disturb box mode');
// The popup r key, against a mask and against a scan candidate.
statusLines = [];
assignNewMaskReview();
want(currentData.masks[0].species === '', 'the popup r key never writes species REVIEW');
want(currentData.masks[0].review === undefined, 'the popup r key never writes the review flag');
want(currentData.masks[0].status === 'pending', 'the popup r key never accepts the mask');
want(persisted.length === 0, 'the popup r key sends no relabel request');
want(popupHidden === false, 'the popup stays open so the operator can pick a species');
pendingScanCandidateIdx = 0;
assignNewMaskReview();
want(labeledCandidates.length === 0, 'the popup r key never labels a scan candidate REVIEW');
want(statusLines.every(s => s === REVIEW_LOCKED_MSG), 'the popup r key answers with the locked message only');
process.exit(fails ? 1 : 0);
""" % {
        "msg": '"%s"' % LOCKED_MSG,
        "guard": _extract_function("reviewLockGuard", src),
        "armReview": _extract_function("armReview", src),
        "assign": _extract_function("assignNewMaskReview", src),
    }
    _run_node(harness, "neither R binding can arm or apply REVIEW")


# ── The newly placed mask is never born flagged ──────────────────────────────

def test_armed_fields_never_carry_a_review_tag():
    """`armedMaskFields` cannot return a review tag while the lock holds."""
    src = _template_source()
    fn = _extract_function("armedMaskFields", src)
    harness = """
const TARGET_LABELS = [{code: 'OFAV', name: 'Orbicella faveolata', category: 'Coral'}];
let REVIEW_TAGGING_ENABLED = false;
let armedReview = true;      // a stale arm, the worst case
let numKeyDown = null;
%(fn)s
let fails = 0;
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
let f = armedMaskFields();
want(f.species !== 'REVIEW', 'a stale arm cannot make a mask born as REVIEW');
want(f.reviewFlag === false, 'a stale arm cannot set the review flag');
want(f.species === '', 'a stale arm leaves the new mask unlabelled');
numKeyDown = 0;
f = armedMaskFields();
want(f.species === 'OFAV', 'an armed species still labels the new mask');
want(f.reviewFlag === false, 'an armed species never carries a review flag');
numKeyDown = null;
// The lock is one flag, not a deletion: flipping it back restores the feature.
REVIEW_TAGGING_ENABLED = true;
f = armedMaskFields();
want(f.species === 'REVIEW' && f.reviewFlag === true,
     'the arming path is intact behind the flag');
process.exit(fails ? 1 : 0);
""" % {"fn": fn}
    _run_node(harness, "armedMaskFields never returns a review tag under the lock")


# ── No many-mask path can apply REVIEW ───────────────────────────────────────

def test_no_bulk_path_can_apply_review():
    """Nothing that touches a set of masks at once can write a review tag."""
    src = _template_source()
    funcs = _top_level_functions(src)
    multi = {n: b for n, b in funcs.items() if "checkedMaskIds" in b}
    check(len(multi) > 0, "the multi-select set is used somewhere, so this test has teeth")
    offenders = [n for n, b in multi.items() if _writes_a_review_tag(b)]
    check(not offenders,
          "no multi-select path mentions REVIEW (found: %s)" % (", ".join(offenders) or "none"))
    # The relabel field is the only way this page can retag an existing mask.
    # Since 2026-10-15 every label write (a species, a category chip, Unlabel,
    # REVIEW) travels with its status in one PUT built by persistMaskStatus and
    # reached through labelMask, so the review tag is guarded at the callers:
    # persistMaskReview is the only one that passes it.
    check(src.count("relabel: {") == 1, "the relabel request body is written in exactly one place")
    check("relabel: {" in funcs.get("persistMaskStatus", ""),
          "the only relabel sender is persistMaskStatus")
    check("labelMask(" in funcs.get("persistMaskReview", ""),
          "persistMaskReview stores the tag through labelMask")
    label_writers = [n for n, b in funcs.items()
                     if "labelMask(" in b and n not in ("labelMask", "persistMaskReview")]
    check(len(label_writers) > 0, "other label writers exist, so the next check has teeth")
    tagging = [n for n in label_writers if _writes_a_review_tag(funcs[n])]
    check(not tagging,
          "no label writer other than persistMaskReview writes a review tag (found: %s)"
          % (", ".join(tagging) or "none"))
    callers = [n for n, b in funcs.items()
               if "persistMaskReview(" in b and n != "persistMaskReview"]
    check(callers == ["assignNewMaskReview"],
          "persistMaskReview has one caller, the guarded popup chip (found: %s)"
          % (", ".join(callers) or "none"))
    check(_is_behind_the_lock(funcs.get("assignNewMaskReview", "")),
          "that one caller is behind the lock flag")
    # The standalone session's label list is the other many-mask entry point:
    # every frame of the session inherits it.
    sa = funcs.get("saStart", "")
    check("'REVIEW'" in sa and _is_behind_the_lock(sa),
          "the standalone session refuses REVIEW as a session label")


# ── The read path is untouched ───────────────────────────────────────────────

def test_flagged_mask_still_renders_amber():
    """A mask already flagged still paints the fixed REVIEW amber."""
    src = _template_source()
    colors_start = src.index("const REVIEW_COLORS = {")
    colors = src[colors_start:src.index("};", colors_start) + 2]
    harness = """
%(state)s
%(colors)s
%(color)s
let fails = 0;
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
const amber = '%(amber)s';
want(maskColor({id: 7, review: true, status: 'accepted'}, 0.9) === amber,
     'the review boolean still paints amber');
want(maskColor({id: 8, species: 'REVIEW', status: 'accepted'}, 0.9) === amber,
     'the REVIEW species still paints amber');
want(maskReviewState({id: 9, review: true}) === 'review',
     'the review state classifier is unchanged');
want(maskColor({id: 10, species: 'OFAV', status: 'accepted'}, 0.9) !== amber,
     'an ordinary mask keeps its per-id hue');
process.exit(fails ? 1 : 0);
""" % {
        "state": _extract_function("maskReviewState", src),
        "colors": colors,
        "color": _extract_function("maskColor", src),
        "amber": AMBER_AT_90_PERCENT_ALPHA,
    }
    _run_node(harness, "a flagged mask still renders amber")


def test_pending_filter_and_counter_still_see_a_flagged_mask():
    """Target-only view and the panel counter both still include a flagged mask."""
    src = _template_source()
    # The slice test_target_only_shows_pending.py depends on must stay intact.
    start = src.index("function isReviewMask")
    end = src.index("function toggleTargetOnly")
    check(start < end, "isReviewMask still precedes toggleTargetOnly")
    trio = src[start:end]
    for forbidden in ("document", "window", "fetch("):
        check(forbidden not in trio,
              "the visibility slice stays free of %s so it runs in a bare context" % forbidden)
    harness = """
let showOnlyTarget = true;
const TARGET_SPECIES = ['OFAV', 'MCAV'];
%(trio)s
let fails = 0;
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
const flaggedByBoolean = {id: 1, review: true, species: 'REVIEW', status: 'pending'};
const flaggedBySpecies = {id: 2, species: 'REVIEW', status: 'accepted'};
want(isReviewMask(flaggedByBoolean) === true, 'the review predicate still reads the boolean');
want(isReviewMask(flaggedBySpecies) === true, 'the review predicate still reads the species');
want(shouldDrawMask(flaggedByBoolean) === true, 'target-only still shows a flagged mask');
want(shouldDrawMask(flaggedBySpecies) === true, 'target-only still shows a REVIEW-species mask');
// The panel counter counts what the canvas draws (renderMaskList's `visible`).
const masks = [
  flaggedByBoolean, flaggedBySpecies,
  {id: 3, species: 'OFAV', status: 'accepted'},
  {id: 4, species: 'MACA', status: 'accepted'},   // non-target, filtered out
];
const hideRejected = true;
const visible = masks.filter(m => shouldDrawMask(m) && !(hideRejected && m.status === 'rejected'));
want(visible.length === 3, 'the counter still counts both flagged masks');
want(visible.filter(m => m.status === 'pending').length === 1,
     'a flagged pending mask is still counted as pending');
process.exit(fails ? 1 : 0);
""" % {"trio": trio}
    _run_node(harness, "the target-only filter and the panel counter still see a flagged mask")


def test_folder_tooltip_is_two_sentences_and_does_not_contradict_itself():
    """The expert-review folder tooltip states one story in two sentences.

    Under the lock it used to run to four sentences and disagree with itself:
    one said flagged masks were exported to the folder and pushed, a later one
    said nothing was written or pushed. The suspension now replaces the export
    sentence instead of following it, in both states, and the folder path stays
    because it is the only fact the tooltip adds beyond its own label.
    """
    locked = _folder_tooltip(_rendered_page())
    check(locked == LOCKED_EXPORT_FOLDER_TOOLTIP,
          "the locked tooltip is the agreed copy, got %r" % locked)
    check(_sentence_count(locked) == 2,
          "the locked tooltip is two sentences, counted %d" % _sentence_count(locked))
    check("/mnt/tear/REVIEW_reefpointseg" in locked,
          "the locked tooltip keeps the default folder path")
    for claim in ("exported here", "auto-pushed", "are exported"):
        check(claim not in locked,
              "the locked tooltip no longer claims the export runs: %r" % claim)
    check("nothing is pushed" in locked,
          "the locked tooltip says the push does not happen")

    unlocked = _folder_tooltip(_rendered_page(review_tagging_enabled=True))
    check(unlocked == UNLOCKED_EXPORT_FOLDER_TOOLTIP,
          "the unlocked tooltip is the agreed copy, got %r" % unlocked)
    check(_sentence_count(unlocked) == 2,
          "the unlocked tooltip is two sentences, counted %d" % _sentence_count(unlocked))
    check("/mnt/tear/REVIEW_reefpointseg" in unlocked,
          "the unlocked tooltip keeps the default folder path")
    check("suspended" not in unlocked,
          "the unlocked tooltip carries no suspension sentence")
    check(locked != unlocked,
          "the one constant is what decides which second sentence is shown")
    for text in (locked, unlocked):
        check(text.startswith("Set "),
              "the tooltip opens with an imperative, got %r" % text[:20])
        check(not text.startswith("You"), "the tooltip does not open with You")


def test_persist_mask_review_says_why_it_stays():
    """The unreachable REVIEW writer carries a comment naming it the reversal.

    persistMaskReview is the only path that writes species REVIEW and the
    review boolean back to the server from this page. No caller can reach it
    while the lock holds, so a reader needs to be told in the source why dead-
    looking code is being kept.
    """
    src = _template_source()
    declaration = src.index("async function persistMaskReview(")
    # The whole comment block is what sits between the blank line above it and
    # the declaration, so a sentence anywhere in the block is read, not only the
    # last line of it.
    comment = src[src.rindex("\n\n", 0, declaration):declaration]
    check("reversal" in comment.lower(),
          "the comment names the function as the reversal path, got %r" % comment[-200:])
    check("REVIEW_TAGGING_ENABLED" in comment,
          "the comment names the constant that brings it back")
    for banned in ("\u2014", "\u2013"):
        check(banned not in comment, "no dash glyph in the comment")


# ── Prose ─────────────────────────────────────────────────────────────────────────

def test_no_banned_prose_in_the_strings_i_added():
    """Every string this change adds obeys the platform prose rules."""
    page = _rendered_page()
    banned_chars = {"em dash": "—", "en dash": "–"}
    for text in ADDED_STRINGS:
        label = text[:40]
        check(text in page or text == LOCKED_MSG, "the copy is on the page: %s" % label)
        for name, ch in banned_chars.items():
            check(ch not in text, "no %s in: %s" % (name, label))
        check("UVI" not in text, "no UVI in: %s" % label)
        check(not re.search(r"\bland(s|ed|ing)?\b", text, re.I), "no banned word in: %s" % label)
        check(all(ord(c) < 0x2190 for c in text), "no emoji or symbol glyphs in: %s" % label)
        check(not text.startswith("You"), "does not open with You: %s" % label)
        check(text.count(". ") + text.count(".") >= 2, "two complete sentences: %s" % label)
        for screen_word in (" above", " below", " this tab", " on screen"):
            check(screen_word not in text.lower(), "no screen reference in: %s" % label)
    check("VICARIUS" in CHIP_TOOLTIP, "the chip tooltip names VICARIUS as the actor")


def _run(case):
    """Run one case and keep going when a missing element aborts it.

    A control that vanished raises out of the helpers rather than returning
    False, and one such case must not hide the verdict of the others.

    Args:
        case: the zero-argument test function to run.
    """
    global _fail
    print(case.__name__)
    try:
        case()
    except Exception as exc:
        print("  FAIL %s raised %s: %s" % (case.__name__, type(exc).__name__, exc))
        _fail += 1


if __name__ == "__main__":
    for _case in (
        test_review_button_is_absent_while_locked,
        test_review_button_is_live_once_unlocked,
        test_review_chip_is_present_disabled_and_has_an_id,
        test_r_key_no_longer_arms_review,
        test_armed_fields_never_carry_a_review_tag,
        test_no_bulk_path_can_apply_review,
        test_flagged_mask_still_renders_amber,
        test_pending_filter_and_counter_still_see_a_flagged_mask,
        test_folder_tooltip_is_two_sentences_and_does_not_contradict_itself,
        test_persist_mask_review_says_why_it_stays,
        test_no_banned_prose_in_the_strings_i_added,
    ):
        _run(_case)
    print("PASS" if _fail == 0 else "%d FAILED" % _fail)
    sys.exit(1 if _fail else 0)
