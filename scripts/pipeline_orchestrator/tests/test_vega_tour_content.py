"""
Module:  test_vega_tour_content.py
Purpose: House suite for the VEGA guided tour's content, stylesheet and wiring
         note (task T15 of the 2026-09-16 student-ready build, version 2 of
         2026-09-17). The content file is data (static/vega_tour_content.js);
         this suite loads it through node and checks every step against the
         spec.
Inputs:  static/vega_tour_content.js, static/vega_tour.css, static/vega_tour.js,
         templates/_vega_tour_wiring.md, and the templates of every surface the
         tour visits (read only).
Outputs: a PASS/FAIL line per check on stdout; exit 0 when every check passed.

House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_vega_tour_content.py

Spec: docs/superpowers/specs/2026-09-16-vega-tutorial-design.md, sections 2
(the step shape and the advance kinds), 5 (the tutorial set), 6 (coverage).
Copy rules: docs/superpowers/specs/2026-09-02-workflow-page-design.md,
sections 6 and 7.

Version 2 rules this suite pins (Lauren, 2026-09-17):
  - no step advances on a button press: there is no Next control at all;
  - every step names a real control and a real action on it;
  - every body is bite size and its last sentence names that action;
  - the anchor glows, and the pulse is wrapped in prefers-reduced-motion.

Version 3 rules (Lauren, 2026-09-17, walking the tour):
  - a step may only name a page the tour is actually wired into, proved by
    reading that page's own template for the tour's link and script tags. This
    is the check that would have caught the boxes track dead-ending on the
    Bounding Boxes page, which carries no tour;
  - a step whose action opens another window says so in its body and declares
    the surface it opens, and the engine carries the tour's place across;
  - plain words first: the jargon Lauren named is gone, and a word a student
    has not been taught is explained where it is first met.

Covers:
  SHAPE     - every step carries the spec's keys with valid values; ids are
              unique; the advance kind is one of the four real actions; an
              event step names its watcher; a navigate step names its target.
  ANCHORS   - every anchor selector exists in its surface's template (the
              orchestrator's index.html and its included partials, the
              annotator, the matrix, the labels page, the Bounding Boxes page,
              the desktop launcher), resolved by id, attribute or class.
  ACTIONS   - no step advances on a press of the tour's own; every body's last
              sentence opens with the action the student takes.
  COPY      - complete sentences; no sentence opens with "You"; no em or en
              dash, emoji, the banned word, "UVI", or a screen reference;
              only <em> and <code> markup; bodies bite size for a 340 px
              callout; every fallback is a sentence.
  TERMS     - each term of art is defined once per track; a species code is
              expanded before it stands alone; binomials are italic.
  COVERAGE  - both tracks reach every stage 1 to 10 of spec section 6; the
              training wait is an event step on step 6; stage 2 of each track
              opens that track's TUTORIAL_ project from the launcher; the
              first screen says boxes are imported, not drawn; the last step
              is about resuming.
  CSS       - every rule is scoped under .vega-tour; no font size under 12 px;
              colours are tokens, their fallback values, or alpha washes of
              them; the ring glows; the pulse is guarded by motion.
  ENGINE    - no Next control is built, the four watchers are there, the
              engine keeps to the prose rules and the storage prefix.
  SURFACES  - every surface a step names is a page whose template loads
              vega_tour.css, vega_tour_content.js and vega_tour.js.
  HANDOFF   - a step that opens another window declares the surface it opens
              and says so in its body; the engine carries the place across.
  JARGON    - no untaught word reaches a student: a banned list, and a term
              that must be explained in plain words where it is first met.
  WIRING    - the note names every page and both assets.
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
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
_REPO = os.path.dirname(_SCRIPTS)
_DRIVE = os.path.abspath(os.path.join(_REPO, "..", "..", "..", ".."))

CONTENT_JS = os.path.join(_ORCH, "static", "vega_tour_content.js")
ENGINE_JS = os.path.join(_ORCH, "static", "vega_tour.js")
TOUR_CSS = os.path.join(_ORCH, "static", "vega_tour.css")
WIRING_MD = os.path.join(_ORCH, "templates", "_vega_tour_wiring.md")
NODE = "/usr/bin/node"

# The template(s) each surface renders from. The orchestrator includes three
# partials, so an anchor may live in any of the four files.
SURFACE_TEMPLATES = {
    "orchestrator": [
        os.path.join(_ORCH, "templates", "index.html"),
        os.path.join(_ORCH, "templates", "_step6_runs.html"),
        os.path.join(_ORCH, "templates", "_step7_report.html"),
        os.path.join(_ORCH, "templates", "_step8_gallery.html"),
    ],
    "annotator": [os.path.join(_SCRIPTS, "TCRMPclip_combinedAnnotate", "src", "templates", "index.html")],
    "matrix": [os.path.join(_SCRIPTS, "_matrix", "templates", "matrix", "index.html")],
    "labels": [os.path.join(_SCRIPTS, "_labels", "templates", "labels", "index.html")],
    "boxes": [os.path.join(_SCRIPTS, "_boxes", "templates", "boxes", "index.html")],
    "launcher": [os.path.join(_DRIVE, "vicarius_ui_os", "templates", "module_launcher.html")],
}
VALID_SURFACES = set(SURFACE_TEMPLATES) | {"any"}
VALID_TRACKS = {"boxes", "outlines", "both"}
# Version 2: four kinds, every one of them a real action on the live page.
# "next" is gone, and a step that advances on a button press is a failure.
VALID_ADVANCE = {"click", "input", "navigate", "event"}
STAGES = set(range(1, 11))
STEP_KEYS = {"id", "stage", "track", "surface", "anchor", "title", "body", "advance", "fallback"}
OPTIONAL_KEYS = {"event", "target", "handoff"}
# The three files every page the tour visits must load, in an href or a src.
TOUR_ASSETS = ("vega_tour.css", "vega_tour_content.js", "vega_tour.js")

EM_DASH = "—"
EN_DASH = "–"
DASH_ENTITIES = ("&mdash;", "&ndash;", "&#8212;", "&#8211;", "&#x2014;", "&#x2013;")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
BANNED_WORD_RE = re.compile(r"\bland(s|ed|ing)?\b", re.IGNORECASE)
SCREEN_REFS = ("above", "below", "this tab", "this window")
ALLOWED_TAG_RE = re.compile(r"</?(em|code)>")
ANY_TAG_RE = re.compile(r"<[^>]+>")
SENTENCE_END_RE = re.compile(r"[.?!]\s*$")

# Bite size (Lauren, 2026-09-17: "break up the text ... more bite size").
# 40 words is the measured ceiling of the shipped copy plus a little headroom:
# at 310 px, the callout's width at a 946 px window, Inter 14 px over 1.5 line
# height fits about six words a line, so 40 words is seven lines and the whole
# card stays near 280 px tall, well under a third of that window.
MAX_BODY_WORDS = 40
MAX_BODY_SENTENCES = 3
MAX_TITLE_CHARS = 60
MIN_FALLBACK_WORDS = 6

# The last sentence of every body is the action, and an action opens with the
# verb the student performs. No other opening is allowed there.
ACTION_OPENERS = {
    "Press", "Type", "Click", "Choose", "Tick", "Untick", "Drag", "Tap", "Pick",
    "Keep", "Open", "Wait", "Watch", "Check", "Select", "Set", "Go", "Read",
    "Move", "Scroll", "Return", "Name", "Hover",
}

# A term of art is defined once per track, in this exact opening.
TERM_DEFINITIONS = ("A mask is", "A label is", "A project is", "A training set is",
                    "A training project is", "Precision is", "Recall is", "A batch is",
                    "An epoch is", "Confidence is")
# Species codes that must be expanded before they stand alone.
CODE_EXPANSIONS = {"PA": "Porites astreoides", "SS": "Siderastrea siderea"}

# Lauren, 2026-09-17, walking the tour: "the staged dataset, cards, these are
# all jargony terms. not super helpful". A word on this list teaches a student
# nothing, so the tour names what the thing is instead. Matching is on the
# lowercased visible text of a title, body or fallback.
JARGON_NEVER = (
    "staged dataset", "staged box dataset", "staging", "staged ",
    "the rail", "tools list", "class list",
    "holdout", "held out", "held-out",
    "composite", "segmenter", "detector", "weights",
    "the card", "cards", "chip",
)
# "frame" is the codebase's word for a picture. The one place a student meets
# it is the Model Inference source named "Unused survey frames" on screen.
FRAME_ALLOWED_IN = "survey frames"

# A word a student has not been taught is explained where it is first met:
# term -> the plain words that must be in that step, or in an earlier step of
# the same track. Checked in content order, per track.
TAUGHT_BEFORE_USED = {
    "annotator": "tracing",
    "mask": "outline",
    "epoch": "pass",
    "batch": "picture",
    "champion": "best",
    "confidence": "sure",
    "loss": "wrong",
    "map50": "score",
    "fine-tune": "onward",
    "split": "kept back",
}

# The stylesheet's tokens and the fallback values the design allows beside them.
TOKEN_VALUES = {
    "#E20074", "#B8005C", "#FFB3D9", "#FFE6F2", "#0a0a0f", "#050508", "#11111a", "#191926",
    "#15151f", "#F5EDF0", "#b0a7ab", "#00FFB3", "#FFB800", "#FF4D6D", "#2a2a38", "#1d1d26",
}
# The alpha washes the design allows: of --magenta and of --bg.
ALLOWED_RGBA = {"226, 0, 116", "10, 10, 15", "5, 5, 8"}
MIN_FONT_PX = 12

_pass = 0
_fail = 0
_failures = []


def check(cond, msg):
    """Record one assertion; a failure is printed and counted, never raised."""
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        _failures.append(msg)
        print("  FAIL:", msg)


def run_test(name, fn):
    """Run one named test and print its PASS or FAIL line."""
    before = _fail
    try:
        fn()
    except Exception:  # noqa: BLE001 - a crash is a failure, reported in place
        check(False, f"{name} raised:\n{traceback.format_exc()}")
    print(f"  {'PASS' if _fail == before else 'FAIL'} {name}")


def read(path):
    """Read a UTF-8 text file, failing loudly with the path when it is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing file: {path}")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def load_content():
    """Load the content module through node and return it as a dict."""
    if not os.path.isfile(NODE):
        raise FileNotFoundError(f"node is required to load the content file and is not at {NODE}")
    script = ("const c = require(process.argv[1]); process.stdout.write(JSON.stringify(c));")
    out = subprocess.run([NODE, "-e", script, CONTENT_JS], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"node could not load {CONTENT_JS}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def strip_tags(text):
    """Return the visible words of a body: markup removed, entities kept."""
    return ANY_TAG_RE.sub("", text or "")


def sentences(text):
    """Split visible text into sentences on terminal punctuation."""
    plain = strip_tags(text).strip()
    parts = re.split(r"(?<=[.?!])\s+", plain)
    return [p for p in parts if p]


def track_steps(content, track):
    """The steps of one track, in content order."""
    return [s for s in content["steps"] if s.get("track") in (track, "both")]


def anchor_chain(step):
    """A step's anchor as a list: one selector, or the chain it declares."""
    anchor = step.get("anchor")
    if anchor is None:
        return []
    return list(anchor) if isinstance(anchor, list) else [anchor]


def selector_parts(selector):
    """Break a selector into (tag, id, classes, [(attr, value)]) per compound part."""
    parts = []
    for compound in selector.strip().split():
        m = re.match(r"^([a-z]+)?(#[\w-]+)?((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$", compound, re.IGNORECASE)
        if not m:
            raise ValueError(f"unsupported selector syntax: {selector!r}")
        attrs = []
        for a in re.findall(r"\[[^\]]+\]", m.group(4) or ""):
            am = re.match(r'^\[([\w-]+)(?:="([^"]*)")?\]$', a)
            if not am:
                raise ValueError(f"unsupported attribute syntax in {selector!r}")
            attrs.append((am.group(1), am.group(2)))
        parts.append((m.group(1), (m.group(2) or "")[1:], [c for c in (m.group(3) or "").split(".") if c], attrs))
    return parts


def selector_in_templates(selector, texts):
    """Whether every compound part of a selector is present in one of the templates."""
    joined = "\n".join(texts)
    for tag, id_, classes, attrs in selector_parts(selector):
        if id_ and f'id="{id_}"' not in joined:
            return False
        for cls in classes:
            if not re.search(r'class="[^"]*\b' + re.escape(cls) + r'\b[^"]*"', joined):
                return False
        for attr, value in attrs:
            if value is None:
                if f"{attr}=" not in joined and f" {attr}" not in joined:
                    return False
                continue
            if f'{attr}="{value}"' in joined:
                continue
            # The launcher builds data-tutorial in script from the project it
            # is rendering, so the attribute and its value are literals in the
            # helper rather than one string in the markup.
            if attr == "data-tutorial" and f'{attr}="' in joined and f'"{value}"' in joined:
                continue
            return False
        if tag and not re.search(r"<" + re.escape(tag) + r"[\s>]", joined):
            return False
    return True


def tour_assets_in(surface):
    """Which of the tour's three files a surface's templates actually load.

    A file counts only when it is inside an href or a src on the same line, so
    a mention in a comment or in prose is not mistaken for wiring.

    @param surface one of SURFACE_TEMPLATES
    @returns dict of asset name -> the template that loads it, or None
    """
    found = {asset: None for asset in TOUR_ASSETS}
    for path in SURFACE_TEMPLATES[surface]:
        text = read(path)
        for line in text.splitlines():
            for asset in TOUR_ASSETS:
                if found[asset] or asset not in line:
                    continue
                if re.search(r'(?:src|href)\s*=\s*"[^"]*' + re.escape(asset), line):
                    found[asset] = path
    return found


def css_rules(css):
    """Yield (selector, declarations) pairs from a stylesheet, media blocks unwrapped."""
    text = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    text = re.sub(r"@media[^{]*\{((?:[^{}]*\{[^{}]*\})*)\s*\}", r"\1", text)
    text = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*\s*\}", "", text)
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", text):
        yield m.group(1).strip(), m.group(2)


def media_blocks(css, condition):
    """The bodies of every @media block whose condition carries the given text."""
    out = []
    for m in re.finditer(r"@media([^{]*)\{((?:[^{}]*\{[^{}]*\})*)\s*\}", css):
        if condition in m.group(1):
            out.append(m.group(2))
    return out


CONTENT = None


def test_content_loads():
    global CONTENT
    CONTENT = load_content()
    check(isinstance(CONTENT.get("steps"), list) and CONTENT["steps"], "the content carries a step list")
    check(set(CONTENT.get("tracks", {})) == {"boxes", "outlines"}, "two tracks are declared")
    check(isinstance(CONTENT.get("version"), str) and CONTENT["version"], "the content is versioned")


def test_step_shape():
    ids = [s.get("id") for s in CONTENT["steps"]]
    check(len(ids) == len(set(ids)), f"step ids are unique: {[i for i in ids if ids.count(i) > 1]}")
    for s in CONTENT["steps"]:
        sid = s.get("id", "?")
        keys = set(s)
        check(STEP_KEYS <= keys, f"{sid}: missing keys {sorted(STEP_KEYS - keys)}")
        check(keys <= STEP_KEYS | OPTIONAL_KEYS, f"{sid}: unknown keys {sorted(keys - STEP_KEYS - OPTIONAL_KEYS)}")
        check(re.match(r"^[a-z0-9-]+$", str(sid)) is not None, f"{sid}: id is a slug")
        check(s.get("stage") in STAGES, f"{sid}: stage {s.get('stage')!r} is 1 to 10")
        check(s.get("track") in VALID_TRACKS, f"{sid}: track {s.get('track')!r}")
        check(s.get("surface") in VALID_SURFACES, f"{sid}: surface {s.get('surface')!r}")
        check(s.get("advance") in VALID_ADVANCE, f"{sid}: advance {s.get('advance')!r}")
        if s["advance"] == "event":
            check(re.match(r"^step:[678](:start)?$", str(s.get("event"))) is not None, f"{sid}: event names a step watcher, got {s.get('event')!r}")
        else:
            check("event" not in s, f"{sid}: only an event step carries an event")
        if s["advance"] == "navigate":
            check(isinstance(s.get("target"), str) and s["target"], f"{sid}: a navigate step names its target")
        else:
            check("target" not in s, f"{sid}: only a navigate step carries a target")


def test_every_step_has_a_control_and_an_action():
    """No step advances on a press of the tour's own, and each names its control."""
    first_screens = [s for s in CONTENT["steps"] if s["surface"] == "any"]
    check(len(first_screens) == 1, f"one first screen, got {len(first_screens)}")
    for s in CONTENT["steps"]:
        sid = s["id"]
        check(s.get("advance") != "next", f"{sid}: advance 'next' is gone; a step completes on a real action")
        if s["surface"] == "any":
            # The one step whose control the tour itself draws: the track
            # radios in the callout, which the engine watches as an input.
            check(s["anchor"] is None, f"{sid}: the first screen has no page anchor")
            check(s["advance"] == "input", f"{sid}: the first screen completes on the track radio")
        else:
            chain = anchor_chain(s)
            check(bool(chain), f"{sid}: anchor names the real control")
            check(len(chain) <= 3, f"{sid}: an anchor chain is short, got {len(chain)}")
            check(all(isinstance(sel, str) and sel.strip() for sel in chain), f"{sid}: every selector in the chain is a selector")
    kinds = {s["advance"] for s in CONTENT["steps"]}
    check(kinds <= VALID_ADVANCE, f"advance kinds are the four real actions, got {sorted(kinds)}")
    check("click" in kinds and "input" in kinds, "both direct actions are used")


def test_every_body_ends_by_naming_the_action():
    """With no forward button, the last sentence is the only way on."""
    for s in CONTENT["steps"]:
        sid = s["id"]
        sents = sentences(s["body"])
        check(bool(sents), f"{sid}.body: at least one sentence")
        if not sents:
            continue
        last = sents[-1]
        opener = last.split()[0].strip(",:;") if last.split() else ""
        check(opener in ACTION_OPENERS,
              f"{sid}.body: the last sentence names the action, got {last[:60]!r}")
        check(len(last.split()) <= 18, f"{sid}.body: the action sentence is short, got {len(last.split())} words")


def test_anchors_exist_in_their_templates():
    texts = {surface: [read(p) for p in paths] for surface, paths in SURFACE_TEMPLATES.items()}
    for s in CONTENT["steps"]:
        if s["surface"] == "any":
            continue
        for sel in anchor_chain(s):
            try:
                found = selector_in_templates(sel, texts[s["surface"]])
            except ValueError as exc:
                found = False
                check(False, f"{s['id']}: {exc}")
            check(found, f"{s['id']}: anchor {sel!r} is not in the {s['surface']} template")
            check(not re.search(r":contains|text\(", sel), f"{s['id']}: anchors to ids, never to text")


def _copy_checks(sid, field, text, is_title=False):
    plain = strip_tags(text)
    check(EM_DASH not in text and EN_DASH not in text and not any(e in text for e in DASH_ENTITIES), f"{sid}.{field}: em or en dash")
    check(not EMOJI_RE.search(text), f"{sid}.{field}: emoji")
    check(not BANNED_WORD_RE.search(plain), f"{sid}.{field}: the banned word")
    check("UVI" not in plain, f"{sid}.{field}: UVI")
    low = plain.lower()
    for ref in SCREEN_REFS:
        check(re.search(r"\b" + re.escape(ref) + r"\b", low) is None, f"{sid}.{field}: screen reference {ref!r}")
    stray = [t for t in ANY_TAG_RE.findall(text) if not ALLOWED_TAG_RE.fullmatch(t)]
    check(not stray, f"{sid}.{field}: only <em> and <code> markup, found {stray}")
    check("’" not in text and "“" not in text and "”" not in text, f"{sid}.{field}: straight quotes only")
    if is_title:
        check(len(plain) <= MAX_TITLE_CHARS, f"{sid}.title: over {MAX_TITLE_CHARS} characters")
        check(not plain.endswith("."), f"{sid}.title: no full stop on a title")
        check(plain[:1].isupper(), f"{sid}.title: sentence case")
        check(plain != plain.upper() or len(plain) < 4, f"{sid}.title: not uppercase")
        return
    sents = sentences(text)
    check(sents and SENTENCE_END_RE.search(plain.strip()) is not None, f"{sid}.{field}: ends as a sentence: {plain[-40:]!r}")
    for sent in sents:
        check(not re.match(r"^You\b", sent), f"{sid}.{field}: a sentence opens with You: {sent[:60]!r}")
        check(sent[:1].isupper() or sent[:1] in "<(*", f"{sid}.{field}: sentence starts with a capital: {sent[:40]!r}")
    check("!" not in plain, f"{sid}.{field}: no exclamation marks")


def test_every_surface_is_a_surface_the_tour_is_wired_into():
    """The check that would have caught the boxes track dead end.

    2026-09-17 (LO): steps boxes-datasets and boxes-advanced named the Bounding
    Boxes page, which Lauren has protected and which therefore carries no tour
    script tag. A student pressed Bounding Boxes, arrived, and the tour was
    gone: step 5 could never be reached. A step may only name a page whose own
    template loads all three tour files, and this test reads those templates
    rather than trusting the wiring note.
    """
    used = sorted({s["surface"] for s in CONTENT["steps"]} - {"any"})
    check(bool(used), "the content names at least one surface")
    for surface in used:
        check(surface in SURFACE_TEMPLATES, f"surface {surface!r} names no template to read")
        if surface not in SURFACE_TEMPLATES:
            continue
        loaded = tour_assets_in(surface)
        missing = [asset for asset in TOUR_ASSETS if not loaded[asset]]
        who = ", ".join(sorted(st["id"] for st in CONTENT["steps"] if st["surface"] == surface))
        check(
            not missing,
            f"surface {surface!r} carries no tour: {SURFACE_TEMPLATES[surface][0]} does not load "
            f"{', '.join(missing)}, so steps ({who}) would strand a student. Wire the page per "
            f"templates/_vega_tour_wiring.md, or move those steps to a page that is wired.",
        )
    # Named on purpose: this page is the one Lauren has protected, and the
    # assertion flips to a plain wiring check on the day it is wired.
    boxes_loaded = tour_assets_in("boxes")
    if not all(boxes_loaded.values()):
        check("boxes" not in used,
              "the Bounding Boxes page is not wired for the tour, so no step may live on it")


def test_a_handoff_step_says_where_it_goes():
    """A new window is never a surprise, and the tour carries its place there."""
    surfaces = set(SURFACE_TEMPLATES)
    handoffs = [s for s in CONTENT["steps"] if s.get("handoff")]
    check(bool(handoffs), "at least one step opens another window and declares it")
    for s in handoffs:
        sid = s["id"]
        check(s["handoff"] in surfaces, f"{sid}: handoff {s['handoff']!r} is a real surface")
        body = strip_tags(s["body"]).lower()
        check("opens" in body, f"{sid}: the body says the action opens something")
        check("tab" in body or "window" in body,
              f"{sid}: the body names the tab or window that opens, got {body!r}")
        check(s["handoff"] != s["surface"], f"{sid}: a handoff names another page, not this one")
    # The two moves across addresses a student actually makes.
    by_id = {s["id"]: s for s in CONTENT["steps"]}
    for sid in ("open-tutorial-boxes", "open-tutorial-outlines"):
        check(by_id.get(sid, {}).get("handoff") == "orchestrator",
              f"{sid}: opening the tutorial project hands the tour to the pipeline window")
    engine = read(ENGINE_JS)
    check("vega_tour" in engine and "HANDOFF_PARAM" in engine, "the engine names the handoff parameter")
    for name in ("withHandoff", "stripHandoff", "parseHandoff", "installOpenWatch", "clearUrlHandoff"):
        check(name in engine, f"the engine carries {name}")
    check("KEYS.moving" in engine or "moving:" in engine, "the engine marks a move at the same address")


def test_no_untaught_word_reaches_a_student():
    """Lauren, 2026-09-17: the jargon sweep, pinned so it cannot creep back."""
    for s in CONTENT["steps"]:
        sid = s["id"]
        text = strip_tags(" ".join([s["title"], s["body"], s["fallback"]])).lower()
        for word in JARGON_NEVER:
            check(word not in text, f"{sid}: jargon {word!r} is in the copy; name what the thing is")
        for m in re.finditer(r"\bframes?\b", text):
            around = text[max(0, m.start() - 12):m.end() + 2]
            check(FRAME_ALLOWED_IN in around,
                  f"{sid}: {'frame'!r} is the codebase's word for a picture, got {around!r}")
    for track in ("boxes", "outlines"):
        seen = ""
        for s in track_steps(CONTENT, track):
            text = strip_tags(" ".join([s["title"], s["body"], s["fallback"]])).lower()
            for term, plain in TAUGHT_BEFORE_USED.items():
                if term not in text or term in seen:
                    continue
                check(plain in text or plain in seen,
                      f"{track}/{s['id']}: {term!r} is used before {plain!r} explains it")
            seen += " " + text
    joined = " ".join(strip_tags(s["body"]) for s in CONTENT["steps"]).lower()
    check("a folder of pictures" in joined or "folder of pictures" in joined,
          "the tour says in plain words what a training folder is")


def test_copy_rules():
    for s in CONTENT["steps"]:
        sid = s["id"]
        check(isinstance(s.get("title"), str) and s["title"].strip(), f"{sid}: title present")
        check(isinstance(s.get("body"), str) and s["body"].strip(), f"{sid}: body present")
        check(isinstance(s.get("fallback"), str) and s["fallback"].strip(), f"{sid}: fallback present")
        _copy_checks(sid, "title", s["title"], is_title=True)
        _copy_checks(sid, "body", s["body"])
        _copy_checks(sid, "fallback", s["fallback"])
        words = len(strip_tags(s["body"]).split())
        check(words <= MAX_BODY_WORDS, f"{sid}.body: {words} words, over the bite-size ceiling of {MAX_BODY_WORDS}")
        check(len(sentences(s["body"])) <= MAX_BODY_SENTENCES, f"{sid}.body: over {MAX_BODY_SENTENCES} sentences")
        check(len(strip_tags(s["fallback"]).split()) >= MIN_FALLBACK_WORDS, f"{sid}.fallback: a full sentence, not a fragment")
        if s["surface"] != "any":
            check(s["fallback"] != s["body"], f"{sid}: the fallback is its own sentence")


def test_terms_defined_once_per_track():
    for track in ("boxes", "outlines"):
        joined = "\n".join(strip_tags(s["body"]) for s in track_steps(CONTENT, track))
        for term in TERM_DEFINITIONS:
            n = joined.count(term)
            check(n <= 1, f"{track}: {term!r} defined {n} times, once at most")
        seen = ""
        for s in track_steps(CONTENT, track):
            text = strip_tags(s["title"] + " " + s["body"])
            for code, name in CODE_EXPANSIONS.items():
                if re.search(r"\b" + code + r"\b", text) and name not in seen and name not in text:
                    check(False, f"{track}/{s['id']}: code {code} stands alone before {name} is named")
            seen += " " + text
        for s in track_steps(CONTENT, track):
            for em in re.findall(r"<em>([^<]*)</em>", s["body"]):
                check(re.match(r"^[A-Z][a-z]+ [a-z]+$", em) is not None, f"{s['id']}: italic is for a binomial, got {em!r}")


def test_coverage_of_every_stage():
    for track in ("boxes", "outlines"):
        stages = {s["stage"] for s in track_steps(CONTENT, track)}
        check(stages == STAGES, f"{track}: stages reached {sorted(stages)}, want 1 to 10")
        steps = track_steps(CONTENT, track)
        check(steps and steps[0]["id"] == "welcome", f"{track}: opens on welcome")
        check(steps and steps[-1]["id"] == "resume", f"{track}: ends on resume")
        order = [s["stage"] for s in steps]
        check(order == sorted(order), f"{track}: stages run in order, got {order}")
        surfaces = {s["surface"] for s in steps}
        check("orchestrator" in surfaces, f"{track}: visits the pipeline window")
        check("matrix" in surfaces, f"{track}: visits the Coverage Matrix")
        check("labels" in surfaces, f"{track}: visits the Label Manager")
    boxes = {s["surface"] for s in track_steps(CONTENT, "boxes")}
    outlines = {s["surface"] for s in track_steps(CONTENT, "outlines")}
    check("annotator" in outlines, "the outlines track visits the annotator")
    check("annotator" not in boxes, "the boxes track never sends a student to draw boxes in the annotator")
    # The boxes track tells a student where a folder of boxed pictures is
    # prepared without putting a step on that page, which carries no tour.
    stage3 = [s for s in track_steps(CONTENT, "boxes") if s["stage"] == 3]
    check(len(stage3) == 1, f"boxes: one step covers where the boxes came from, got {[s['id'] for s in stage3]}")
    if stage3:
        check(stage3[0]["surface"] == "orchestrator", "boxes: that step stays in the pipeline window")
        body = strip_tags(stage3[0]["body"])
        check("Bounding Boxes" in body, "boxes: it names the Bounding Boxes page in words")
        check("own tab" in body, "boxes: it says the page opens in its own tab")


def test_stage_two_opens_the_tutorial_project():
    """Each track starts by opening its own TUTORIAL_ project from the launcher."""
    for track in ("boxes", "outlines"):
        stage2 = [s for s in track_steps(CONTENT, track) if s["stage"] == 2]
        check(len(stage2) == 1, f"{track}: one stage 2 step, got {[s['id'] for s in stage2]}")
        if not stage2:
            continue
        step = stage2[0]
        chain = anchor_chain(step)
        check(step["surface"] == "launcher", f"{track}: stage 2 happens on the launcher, got {step['surface']}")
        check(step["advance"] in ("click", "navigate"), f"{track}: stage 2 completes when the project is opened")
        check(chain[:1] == [f'[data-tutorial="{track}"]'], f"{track}: stage 2 glows that one tutorial card, got {chain[:1]}")
        check(chain[-1:] == ["#runs-list"], f"{track}: stage 2 falls back to the project list, got {chain[-1:]}")
        body = strip_tags(step["body"])
        check("TUTORIAL" in body, f"{track}: stage 2 names the TUTORIAL card")
        check(track in body, f"{track}: stage 2 says which family the card teaches")
        # The card's display name is "TUTORIAL boxes (detection)". Pinning an
        # exact string in the copy strands a student after any rename, so the
        # sentence names the card in words instead.
        check("TUTORIAL_" not in body, f"{track}: stage 2 does not pin an exact project name")


def test_the_training_wait_and_the_first_and_last_screens():
    by_id = {s["id"]: s for s in CONTENT["steps"]}
    wait = by_id.get("train-wait")
    check(wait is not None, "a train-wait step exists")
    if wait:
        check(wait["advance"] == "event" and wait.get("event") == "step:6", "the training wait is an event on step 6")
        check(wait["track"] == "both" and wait["stage"] == 7, "the wait is stage 7 on both tracks")
        check(wait["anchor"] == "#s6-progress", "the wait rings the panel's own progress bar")
        low = strip_tags(wait["body"]).lower()
        check("wrong" in low and "loss" in low, "while waiting the copy says what the trainer measures")
        check("kept back" in low, "and says the pictures kept back are never shown to it")
    start = by_id.get("train-start")
    check(start is not None and start["anchor"] == "#s6-run-btn", "the start step rings the Start Training button")
    welcome = by_id.get("welcome")
    check(welcome is not None and welcome["surface"] == "any" and welcome["stage"] == 1, "welcome is stage 1 on any surface")
    if welcome:
        low = strip_tags(welcome["body"]).lower()
        check("box" in low and "outline" in low, "the first screen names both kinds")
        check("never draws boxes" in low or "already drawn" in low,
              "the first screen says VEGA does not draw boxes, so nobody hunts for a tool that does")
    resume = by_id.get("resume")
    check(resume is not None and resume["stage"] == 10, "the last step is stage 10")
    if resume:
        check("{project_name}" in resume["body"] and "{stage_name}" in resume["body"], "the last step names the project and its parked stage")
    evaluate = [s for s in CONTENT["steps"] if s["stage"] == 8]
    joined = " ".join(strip_tags(s["body"]) for s in evaluate).lower()
    for word in ("precision", "recall", "map50", "metrics.json", "report.pdf"):
        check(word in joined, f"the Evaluate stage explains {word}")
    check("step7_evaluatemodel" in joined, "the Evaluate stage says where the files are on disk")
    for step in CONTENT["steps"]:
        if step["advance"] == "event":
            check(step["surface"] == "orchestrator", f"{step['id']}: watchers belong to the pipeline window")


def test_stylesheet_rules():
    css = read(TOUR_CSS)
    check(EM_DASH not in css and EN_DASH not in css and not EMOJI_RE.search(css), "no dash or emoji in the stylesheet")
    for selector, decls in css_rules(css):
        for part in selector.split(","):
            part = part.strip()
            if not part or part.startswith("@") or re.match(r"^\d+%$", part) or part in ("from", "to"):
                continue
            check(part.startswith(".vega-tour") or part.startswith(":root"), f"rule not scoped under .vega-tour: {part!r}")
        for size in re.findall(r"font-size\s*:\s*([\d.]+)px", decls):
            check(float(size) >= MIN_FONT_PX, f"{selector.strip()!r}: font-size {size}px is under the {MIN_FONT_PX} px floor")
        for hexval in re.findall(r"#[0-9a-fA-F]{3,8}\b", decls):
            check(hexval in TOKEN_VALUES or hexval.lower() in {t.lower() for t in TOKEN_VALUES}, f"{selector.strip()!r}: colour {hexval} is not a token value")
        for rgba in re.findall(r"rgba\(([^)]*)\)", decls):
            base = ",".join(rgba.split(",")[:3]).strip()
            base = re.sub(r"\s*,\s*", ", ", base)
            check(base in ALLOWED_RGBA, f"{selector.strip()!r}: rgba({rgba}) is not an alpha wash of a token")
    check("Syne" in css and "Inter" in css and "Space Mono" in css, "the three families are named")
    check("font-size: 17px" in css or "font-size:17px" in css, "the title is 17 px")
    check("font-size: 14px" in css or "font-size:14px" in css, "the body is 14 px")
    check("--magenta" in css, "the ring uses the magenta token")
    check(".vega-tour-callout" in css and ".vega-tour-ring" in css and ".vega-tour-chip" in css, "the three parts are styled")
    check(".vega-tour-where" in css, "the line that says where to go is styled")


def test_the_anchor_glows():
    """The ring is a glow: a magenta halo, a pulse, and the dimmed backdrop kept."""
    css = read(TOUR_CSS)
    ring = ""
    for selector, decls in css_rules(css):
        if selector.strip() == ".vega-tour-ring":
            ring = decls
    check(bool(ring), "the ring rule is there")
    shadows = re.findall(r"rgba\(([^)]*)\)", ring)
    magenta = [s for s in shadows if s.replace(" ", "").startswith("226,0,116")]
    check(len(magenta) >= 2, f"the ring draws a magenta halo and a bloom, found {len(magenta)}")
    check("200vmax" in ring, "the dimmed backdrop cutout is kept")
    check(re.search(r"border:\s*2px solid var\(--magenta", ring) is not None, "the ring itself is a 2 px magenta line")
    check("@keyframes vega-tour-pulse" in css, "a pulse is defined")
    guarded = media_blocks(css, "prefers-reduced-motion: no-preference")
    check(any("vega-tour-pulse" in b for b in guarded), "the pulse runs only under prefers-reduced-motion: no-preference")
    bare = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    outside = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*\s*\}", "", bare)
    outside = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*\s*\}", "", outside)
    check(".vega-tour-ring::after" in outside, "a strong static second ring is drawn for a reader who wants no motion")
    check("animation" not in outside, "no animation runs outside the motion guard")


def test_engine_has_no_forward_control():
    js = read(ENGINE_JS)
    check("'next'" not in js and '"next"' not in js, "the engine builds no control named next")
    check("'Finish'" not in js and '"Finish"' not in js, "no Finish button is built")
    check("press Next" not in js and "Next continues" not in js, "no copy tells a student to press Next")
    check("ArrowRight" not in js, "no key moves the tour forward")
    for name in ("installClick", "installInput", "installNavigate", "installEvent"):
        check(name in js, f"the engine watches the {name[7:].lower()} kind")
    check("addEventListener('click'" in js, "the click watcher listens on the live page")
    check("addEventListener('input'" in js and "addEventListener('change'" in js, "the input watcher listens for both events")
    check("SURFACE_NAMES" in js, "a stuck student is told which page to go to")


def test_engine_prose_and_prefix():
    js = read(ENGINE_JS)
    check(EM_DASH not in js and EN_DASH not in js and not EMOJI_RE.search(js), "no dash or emoji in the engine")
    check("UVI" not in js, "no UVI in the engine")
    check(not BANNED_WORD_RE.search(re.sub(r"https?://\S+", "", js)), "the banned word is not in the engine")
    check("'vega.tour.'" in js or '"vega.tour."' in js, "the storage prefix is the spec's")
    check("role" in js and "dialog" in js and "aria-labelledby" in js, "the callout is a labelled dialog")
    check("try {" in js and "catch" in js, "storage reads and writes are guarded")
    check("setInterval" not in js.replace("clearInterval", ""), "the engine adds no polling loop of its own")
    check("MutationObserver" in js, "the wait watches the panel the existing poll writes")
    content = read(CONTENT_JS)
    check(EM_DASH not in content and EN_DASH not in content, "no dash in the content file")
    check("module.exports" in content and "VegaTourContent" in content, "the content loads in node and in a browser")


def test_wiring_note():
    md = read(WIRING_MD)
    for needle in ("index.html", "module_launcher.html", "TCRMPclip_combinedAnnotate", "_matrix", "_labels", "_boxes",
                   "vega_tour.css", "vega_tour.js", "vega_tour_content.js", "<link", "<script"):
        check(needle in md, f"the wiring note names {needle}")
    check(EM_DASH not in md and EN_DASH not in md and not BANNED_WORD_RE.search(md), "the note keeps to the prose rules")



def test_the_callout_states_its_own_text_transform():
    """2026-09-17 (LO): the desktop launcher uppercases option text, so the tour
    hosted there rendered its track names in capitals, which the design spec
    reserves for the window title, tabs and buttons. The callout declares its
    own transform, and only its buttons are uppercase."""
    css = open(TOUR_CSS, encoding="utf-8").read()
    check(".vega-tour-callout, .vega-tour-callout * { text-transform: none; }" in css,
          "the callout resets any text-transform its host page applies")
    check(".vega-tour-callout .vega-tour-btn { text-transform: uppercase; }" in css,
          "only the tour's buttons are uppercase")

def main():
    """Run every check and exit non-zero when any failed."""
    print("VEGA TOUR CONTENT:")
    run_test("test_content_loads", test_content_loads)
    if CONTENT is None:
        print(f"\n{_pass} passed, {_fail} failed")
        sys.exit(1)
    for name, fn in [
        ("test_step_shape", test_step_shape),
        ("test_every_step_has_a_control_and_an_action", test_every_step_has_a_control_and_an_action),
        ("test_every_body_ends_by_naming_the_action", test_every_body_ends_by_naming_the_action),
        ("test_anchors_exist_in_their_templates", test_anchors_exist_in_their_templates),
        ("test_every_surface_is_a_surface_the_tour_is_wired_into", test_every_surface_is_a_surface_the_tour_is_wired_into),
        ("test_a_handoff_step_says_where_it_goes", test_a_handoff_step_says_where_it_goes),
        ("test_no_untaught_word_reaches_a_student", test_no_untaught_word_reaches_a_student),
        ("test_copy_rules", test_copy_rules),
        ("test_terms_defined_once_per_track", test_terms_defined_once_per_track),
        ("test_coverage_of_every_stage", test_coverage_of_every_stage),
        ("test_stage_two_opens_the_tutorial_project", test_stage_two_opens_the_tutorial_project),
        ("test_the_training_wait_and_the_first_and_last_screens", test_the_training_wait_and_the_first_and_last_screens),
        ("test_stylesheet_rules", test_stylesheet_rules),
        ("test_the_anchor_glows", test_the_anchor_glows),
        ("test_engine_has_no_forward_control", test_engine_has_no_forward_control),
        ("test_engine_prose_and_prefix", test_engine_prose_and_prefix),
        ("test_wiring_note", test_wiring_note),
        ("test_the_callout_states_its_own_text_transform", test_the_callout_states_its_own_text_transform),
    ]:
        run_test(name, fn)
    print(f"\n{_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
