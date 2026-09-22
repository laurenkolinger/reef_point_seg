"""
Task 1 of the 2026-09-01 quality sweep: orchestrator SPA linkage, relics,
copy and consistency.

House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_ui_sweep_20260901.py

Covers:
  ARCHIVED PANELS - #panel-4 / #panel-5 and their JS hydration are gone, the
                    orphan SAM3 mini-poll is gone, backend routes are untouched.
  COPY            - no em dash or en dash in index.html, orchestrator.js,
                    style.css, or in any client-facing string literal in app.py.
  STEP NUMBERING  - app.STEP_DISPLAY matches the rendered sidebar numbering and
                    the client-facing error strings use it.
  TOOLTIPS        - every control named in the task brief carries title=.
  LINKAGE         - the sidebar Tools cluster links the matrix, label manager
                    and annotator; the Step 6 source hint deep-links the matrix.
  RELICS          - .service-link CSS, showSetup() shim and the no-op nextDs
                    override are gone; the VICARIUS fallback port is 5090.
  CACHE-BUST      - ASSET_VERSION reaches both asset tags.
"""

import ast
import os
import re
import sys
import subprocess
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_TPL = os.path.join(_ORCH, "templates", "index.html")
_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_CSS = os.path.join(_ORCH, "static", "style.css")
_APP = os.path.join(_ORCH, "app.py")

sys.path.insert(0, _ORCH)

EM = "\u2014"
EN = "\u2013"

_fail = 0
_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    global _fail
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _fail += 1
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ── cached sources ──────────────────────────────────────────────────
_CACHE = {}


def _read(path):
    if path not in _CACHE:
        with open(path) as fh:
            _CACHE[path] = fh.read()
    return _CACHE[path]


def _html():
    if "html" not in _CACHE:
        import app as appmod
        client = appmod.create_app().test_client()
        resp = client.get("/")
        _CACHE["status"] = resp.status_code
        _CACHE["html"] = resp.get_data(as_text=True)
    return _CACHE["html"]


def _node():
    for cand in ("/usr/bin/node", "node"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


# ── archived panels 4 + 5 ───────────────────────────────────────────
def test_panels_4_and_5_removed():
    html = _html()
    check('id="panel-4"' not in html, "#panel-4 still rendered")
    check('id="panel-5"' not in html, "#panel-5 still rendered")
    check('id="nav-footer-4"' not in html, "nav-footer-4 left behind")
    check('id="nav-footer-5"' not in html, "nav-footer-5 left behind")
    # Every widget those panels owned goes with them.
    for orphan in ("s4-port", "s4-clip-dir", "s4-target-only", "s4-auto-sam3",
                   "s4-reference", "s4-shuffle", "s4-batch-size", "s4-sam3-mini",
                   "s4-sam3-phase", "s4-batch-readout", "s4-service", "s4-log",
                   "s5-port", "s5-confidence", "s5-min-area", "s5-overlap",
                   "s5-symlink", "s5-batch-size", "s5-sam3-panel", "s5-service",
                   "s5-batch-readout", "s5-log"):
        check(f'id="{orphan}"' not in html, f"orphan element {orphan} survives panel removal")
    check("Open Place Points UI" not in html, "legacy Step 4 launch button survives")
    check("Open SAM3 Review UI" not in html, "legacy Step 5 launch button survives")


def test_js_drops_panel_4_and_5_wiring():
    js = _read(_JS)
    m = re.search(r"const STEP_ACTION_BTN = \{(.*?)\};", js, re.S)
    check(m is not None, "STEP_ACTION_BTN block not found")
    check("s4-run-btn" not in m.group(1), "STEP_ACTION_BTN still maps archived step 4")
    check("s5-run-btn" not in m.group(1), "STEP_ACTION_BTN still maps archived step 5")
    m = re.search(r"const OPEN_UI_LABELS = \{(.*?)\};", js)
    check(m is not None, "OPEN_UI_LABELS block not found")
    check("'Place Points'" not in m.group(1), "OPEN_UI_LABELS still lists Place Points")
    check("'SAM3 Review'" not in m.group(1), "OPEN_UI_LABELS still lists SAM3 Review")
    check("step4test:" in m.group(1) and "editmasks:" in m.group(1),
          "OPEN_UI_LABELS lost a live entry")
    # populateConfigs must not hydrate the archived panels any more.
    i = js.index("function populateConfigs(")
    body = js[i:js.index("\n}\n", i)]
    check("state.steps['4']" not in body, "populateConfigs still hydrates archived step 4")
    check("state.steps['5']" not in body, "populateConfigs still hydrates archived step 5")
    check("s5-tracker-device" not in body, "populateConfigs still writes step 5 widgets")


def test_sam3_mini_poll_gone():
    js = _read(_JS)
    check("startSam3MiniPoll" not in js,
          "startSam3MiniPoll survives; it polled /api/step/5/sam3_status forever "
          "to paint deleted widgets")
    check("updateSam3MiniPanel" not in js, "updateSam3MiniPanel survives with no widgets to paint")
    check("s4-sam3-phase" not in js, "JS still reaches for the removed SAM3 mini widgets")
    check("/api/step/5/sam3_status" not in js,
          "no client surface should poll the archived step 5 SAM3 status")


def test_backend_routes_for_steps_4_and_5_untouched():
    src = _read(_APP)
    for route in ('"/api/step/5/sam3_status"', '"/api/step/5/kick"'):
        check(route in src, f"backend route {route} must stay (other code calls it)")
    check("def _sam3_drive(" in src, "the SAM3 driver must stay on the server side")


# ── copy: no em / en dashes ─────────────────────────────────────────
# HTML entity spellings of the same two characters. A literal-only scan misses
# these, and they render as real dashes in the browser.
DASH_ENTITIES = ("&ndash;", "&mdash;", "&#8211;", "&#8212;", "&#x2013;", "&#x2014;",
                 "&#X2013;", "&#X2014;")


def test_frontend_files_have_no_dashes():
    for path in (_TPL, _JS, _CSS):
        text = _read(path)
        name = os.path.basename(path)
        check(EM not in text, f"{name}: em dash present")
        check(EN not in text, f"{name}: en dash present")
        for ent in DASH_ENTITIES:
            check(ent.lower() not in text.lower(), f"{name}: dash entity {ent} present")
    # The rendered page is what the reader sees, so check it too: a dash could
    # arrive from a Jinja include rather than from index.html itself.
    html = _html()
    check(EM not in html and EN not in html, "rendered page carries an em/en dash")
    for ent in DASH_ENTITIES:
        check(ent.lower() not in html.lower(), f"rendered page carries dash entity {ent}")


def _app_client_strings():
    """Every string literal in app.py that is not a docstring."""
    tree = ast.parse(_read(_APP))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            first = node.body[0] if getattr(node, "body", None) else None
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docs.add(id(first.value))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs):
            out.append(node.value)
    return out


def test_app_client_strings_have_no_dashes():
    bad = [s for s in _app_client_strings() if EM in s or EN in s]
    check(not bad, f"app.py client strings still carry em/en dashes: {bad[:3]}")
    ent = [s for s in _app_client_strings()
           if any(e.lower() in s.lower() for e in DASH_ENTITIES)]
    check(not ent, f"app.py client strings still carry dash entities: {ent[:3]}")


# ── step numbering ──────────────────────────────────────────────────
def _sidebar_numbering(html):
    """{data-step: displayed circle number} for the numeric sidebar tiles."""
    nav = re.search(r'<ul class="step-nav" id="step-nav">(.*?)</ul>', html, re.DOTALL)
    check(nav is not None, "step-nav <ul> not found")
    out = {}
    for li in re.findall(r"<li\b(.*?)</li>", nav.group(1), re.DOTALL):
        ds = re.search(r'data-step="([^"]+)"', li)
        if not ds:
            continue
        circle = re.search(r'<div class="step-circle"([^>]*)>([^<]*)<', li)
        if not circle:
            continue
        num = re.search(r'data-num="([^"]+)"', circle.group(1))
        shown = num.group(1) if num else circle.group(2).strip()
        if shown.isdigit():
            out[ds.group(1)] = shown
    return out


def test_step_display_matches_sidebar():
    import app as appmod
    check(hasattr(appmod, "STEP_DISPLAY"), "app.py must expose STEP_DISPLAY")
    expected = {"1": "1", "2": "2", "3": "3", "step4test": "4",
                "6": "5", "7": "6", "8": "7"}
    check(appmod.STEP_DISPLAY == expected,
          f"app.STEP_DISPLAY is {appmod.STEP_DISPLAY}, expected {expected}")
    check(_sidebar_numbering(_html()) == expected,
          f"sidebar numbering {_sidebar_numbering(_html())} does not match STEP_DISPLAY")


def test_js_step_display_matches_app():
    js = _read(_JS)
    m = re.search(r"const STEP_DISPLAY = \{(.*?)\};", js, re.S)
    check(m is not None, "orchestrator.js must define STEP_DISPLAY")
    pairs = dict(re.findall(r"(\w+)\s*:\s*'([^']+)'", m.group(1)))
    check(pairs == {"1": "1", "2": "2", "3": "3", "step4test": "4",
                    "6": "5", "7": "6", "8": "7"},
          f"JS STEP_DISPLAY is {pairs}")


def test_app_errors_use_display_numbers():
    src = _read(_APP)
    check("def step_ref(" in src, "app.py must expose step_ref() for user-facing step names")
    strings = "\n".join(_app_client_strings())
    for legacy in ("Run Step 7 (Evaluate)", "has step 6 run?", "Step 7 env not found",
                   "Step 8 env not found", "Run Step 3 (Choose Images) first",
                   "Run Step 4 first", "run Step 4 first", "Re-run Step 8 (Inference)",
                   "train a model (Step 5)", "Step 1 outputs not linked",
                   "Reopen Step 6", "Re-annotate in Step 4"):
        check(legacy not in strings, f"internal step number leaks to the client: {legacy!r}")
    # step_ref() is what the promoted strings interpolate, so assert on it
    # directly plus on the call sites that were named in the brief.
    import app as appmod
    check(appmod.step_ref(7) == "Evaluate Model (step 6 in the sidebar)",
          f"step_ref(7) is {appmod.step_ref(7)!r}")
    check(appmod.step_ref(6) == "Train Model (step 5 in the sidebar)",
          f"step_ref(6) is {appmod.step_ref(6)!r}")
    check(appmod.step_ref("step4test") == "Place + Segment (step 4 in the sidebar)",
          f"step_ref('step4test') is {appmod.step_ref('step4test')!r}")
    check(appmod.step_ref(8) == "Model Inference (step 7 in the sidebar)",
          f"step_ref(8) is {appmod.step_ref(8)!r}")
    for call in ("{step_ref(7)} env not found", "Has {step_ref(6)} run?",
                 "Run {step_ref(7)} ", "{step_ref(8)} env not found",
                 "Run {step_ref(3)} first", "{step_ref(1)} outputs are not linked"):
        check(call in src, f"expected step_ref() call site missing: {call!r}")


# ── tooltips ────────────────────────────────────────────────────────
def _has_title(html, needle, window=420):
    """True when the tag containing `needle` carries a title= attribute."""
    i = html.find(needle)
    if i < 0:
        return False
    start = html.rfind("<", 0, i)
    end = html.find(">", i)
    if start < 0 or end < 0:
        return False
    return 'title="' in html[start:end + 1]


def test_listed_controls_have_tooltips():
    html = _html()
    for onclick in ("browseRemaps()", "refreshStep4loopRuns()", "refreshStep6Presets()",
                    "importStep6Preset()", "refreshStep7Presets()", "importStep7Preset()",
                    "applyStep7Defaults()", "refreshRunList()", "refreshStep7Rounds()",
                    "copyStep7Pdf()", "closeConfirmModal()"):
        check(_has_title(html, f'onclick="{onclick}"'),
              f"control onclick={onclick} has no title= tooltip")
    # The inline gallery block and the static Download PDF anchor were
    # replaced on 2026-09-05 by the two cards, whose own Refresh and Copy PDF
    # controls carry the tooltips; the PDF link is built in code from the run.
    for eid in ("s8-gallery-refresh", "s7-copy-pdf-btn", "confirm-ok-btn"):
        check(_has_title(html, f'id="{eid}"'), f"#{eid} has no title= tooltip")
    # refreshRunList() is wired to two Refresh buttons (Step 6 eval + Step 7 run
    # pickers); both need one.
    check(html.count('onclick="refreshRunList()"') == 2,
          "expected two refreshRunList() buttons (evaluate + inference pickers)")
    for m in re.finditer(r'<button[^>]*onclick="refreshRunList\(\)"[^>]*>', html):
        check('title="' in m.group(0), "a refreshRunList() Refresh button has no tooltip")
    # Topbar tagline explains the pipeline shorthand.
    m = re.search(r'<span[^>]*title="([^"]*)"[^>]*>\s*CVR -&gt; CLIP|CVR -> CLIP', html)
    tag = re.search(r'id="topbar-pipeline"[^>]*title="([^"]+)"', html)
    check(tag is not None, "topbar tagline needs id=topbar-pipeline plus a plain-words tooltip")
    tip = tag.group(1).lower()
    check("segment" in tip and "train" in tip,
          "topbar tagline tooltip must translate the shorthand into plain words")


def test_gallery_verdict_buttons_have_tooltips():
    # The tiles moved out of orchestrator.js into the gallery card module on
    # 2026-09-05, so the tooltips are read from there. The card's own suites
    # (test_step8_gallery_card.py, test_vega_gallery.js) drive the rendering.
    gallery_js = os.path.join(_ORCH, "static", "orchestrator_gallery.js")
    js = _read(gallery_js)
    i = js.index("function renderVerdictButtons(")
    body = js[i:js.index("\n  }\n", i)]
    check("title=" in body, "the verdict buttons must carry a title= tooltip")
    t = js.index("const VERDICT_TOOLTIPS")
    tips = js[t:js.index("\n  });", t)]
    for verdict in ("yes:", "no:", "skip:"):
        check(verdict in tips, f"gallery verdict tooltip {verdict} not found")


def test_modal_tooltips_name_the_keys():
    html = _html()
    cancel = re.search(r'<button[^>]*onclick="closeConfirmModal\(\)"[^>]*>', html)
    check(cancel is not None, "confirm modal Cancel button not found")
    check("Escape" in cancel.group(0), "Cancel tooltip must name the Escape key")
    ok = re.search(r'<button[^>]*id="confirm-ok-btn"[^>]*>', html)
    check(ok is not None, "confirm modal Confirm button not found")
    check("Enter" in ok.group(0), "Confirm tooltip must name the Enter key")


def test_keyboard_bindings_present():
    js = _read(_JS)
    check("'Escape'" in js, "no Escape key binding")
    i = js.index("'Escape'")
    win = js[max(0, i - 600):i + 900]
    check("closeConfirmModal()" in win, "Escape must close the confirm modal")
    check("hideOpenUiOverlay()" in win, "Escape must close the open-UI overlay")
    check("'Enter'" in win, "Enter must confirm the open modal")
    check("confirm-ok-btn" in win, "Enter must click the confirm button")


# ── linkage ─────────────────────────────────────────────────────────
def test_tools_cluster_present():
    html = _html()
    nav = re.search(r'<ul class="step-nav" id="step-nav">(.*?)</ul>', html, re.DOTALL)
    check(nav is not None, "step-nav <ul> not found")
    block = nav.group(1)
    check("Tools" in block, "sidebar Tools group heading missing")
    for href, label in (("/matrix", "Coverage Matrix"),
                        ("/labels", "Label Manager"),
                        ("/annotate", "Annotate Images")):
        m = re.search(r'<a[^>]*href="' + re.escape(href) + r'"[^>]*>', block)
        check(m is not None, f"Tools link to {href} missing")
        check('target="_blank"' in m.group(0), f"Tools link {href} must open in a new tab")
        check('title="' in m.group(0), f"Tools link {href} needs a tooltip")
        check(label in block, f"Tools link label {label!r} missing")
    # The Tools group sits below the Review group.
    check(block.index(">Review<") < block.index(">Tools<"),
          "Tools group must sit below the Review group")


def test_step6_source_hint_links_the_matrix():
    js = _read(_JS)
    i = js.index("async function refreshStep6Sources(")
    body = js[i:js.index("\n}\n", i)]
    check("/matrix#training" in body,
          "the 'no composite exports yet' hint must deep-link /matrix#training")
    check("target=\\\"_blank\\\"" in body or 'target="_blank"' in body,
          "the matrix hint link must open in a new tab")
    check("note.innerHTML" in body, "the hint must render as HTML so the link is clickable")


def test_step6_tooltip_names_training_sets_tab():
    # The coverage matrix's Stats tab lost the composite build form and
    # registry hygiene to a new Training Sets tab (2026-09-03 stage-rail
    # redesign, Task 9); the Step 6 source-select tooltip pointed a reader
    # at "(Stats tab)" for a form that no longer lives there.
    html = _html()
    check('(Stats tab)' not in html and 'Training Sets tab' in html,
          "Step 6 tooltip names the Training Sets tab")


# ── relics ──────────────────────────────────────────────────────────
def test_relics_removed():
    css = _read(_CSS)
    check(".service-link" not in css, ".service-link CSS relic survives")
    js = _read(_JS)
    check("function showSetup(" not in js, "showSetup() shim survives")
    check("if (ds === 'step4test') nextDs = '6';" not in js,
          "the no-op nextDs='6' override survives")


def test_bounce_fallback_port():
    js = _read(_JS)
    check("127.0.0.1:5090" in js, "VICARIUS fallback must point at port 5090")
    check("5077" not in js, "stale VICARIUS port 5077 still referenced")


def test_internal_codenames_gone():
    html = _html()
    check("Refine (4.loop)" not in html, "internal codename '4.loop' still shown")
    check("Refine (model loop)" in html, "Refine tile must read 'Refine (model loop)'")
    check("(combined annotator)" not in html, "step4test h2 still carries the internal codename")
    check("Open Combined Annotator" not in html, "launch button still says Combined Annotator")
    check("Open Annotator" in html, "launch button must read 'Open Annotator'")
    # The Expert Review I/O tile was inert from 2026-07-09 and was removed
    # entirely on 2026-09-16: a tile nobody can press is not left on the rail.
    # Its blueprint is still mounted in app.py and is reached directly.
    check("expertids" not in html, "dead Expert Review markup still on the page")
    check("Expert Review I/O" not in html, "the removed Expert Review tile label is back")


def test_step3_progress_and_flag_copy():
    js = _read(_JS)
    m = re.search(r"line\.match\((/Selected[^;]*?/i)\)", js)
    check(m is not None, "step 3 progress regex must match 'Selected N frames'")
    # select_images.py prints `Selected {n:,} frames`, so the count can carry
    # thousands separators; the regex has to survive them.
    check("," in m.group(1), "the step 3 regex must tolerate thousands separators")
    # The old pattern was /(\d+)\s*frames?\s*selected/i and never fired. Assert
    # against the whole file that no regex literal still spells it.
    check(re.search(r"/\([^/]*\)[^/]*frames\??\\s\*selected/", js) is None,
          "the old 'N frames selected' regex must be gone")
    check("frames?\\s*selected" not in js,
          "the old 'frames selected' regex body must be gone")
    # And prove the new regex actually matches what the script prints.
    node = _node()
    check(node is not None, "node not available")
    prog = (f"const re = {m.group(1)};"
            "const out = ['Selected 1,234 frames', 'Selected 7 frames',"
            " 'Selected 0 frames'].map(l => { const g = l.match(re);"
            " return g ? parseInt(g[1].replace(/,/g, ''), 10) : 'NO MATCH'; });"
            "console.log(out.join('|'));")
    r = subprocess.run([node, "-e", prog], capture_output=True, text=True)
    check(r.returncode == 0, f"node failed on the step 3 regex: {r.stderr}")
    check(r.stdout.strip() == "1234|7|0",
          f"step 3 regex does not parse select_images.py output: {r.stdout.strip()}")
    html = _html()
    m = re.search(r'id="s3-target"[^>]*title="([^"]+)"', html)
    check(m is not None, "#s3-target tooltip missing")
    check("--target-instances" not in m.group(1),
          "#s3-target tooltip cites a flag that does not exist (--target-instances)")
    check("--target" in m.group(1), "#s3-target tooltip must cite the real --target flag")
    i = html.index('id="panel-3"')
    panel = html[i:html.index('id="panel-', i + 20)]
    check("evenly across" not in panel,
          "Step 3 must not claim frames are balanced evenly across years/sites/transects")


def test_step2_vocabulary():
    html = _html()
    i = html.index('id="panel-2"')
    panel = html[i:html.index('id="panel-', i + 20)]
    check("Load Previous Remap Settings" not in panel, "old Step 2 h3 survives")
    check("Load a previous label mapping" in panel, "new Step 2 h3 missing")
    check("Open Recode UI" not in panel, "Open Recode UI button not renamed")
    check("Edit Mapping" in panel, "Edit Mapping button missing")
    check("Load Labels" in panel, "Load Labels button lost")
    check("remap_log.json" in panel, "remap jargon must survive inside the tooltips")


def test_dewordify():
    html = _html()
    i = html.index('id="panel-6"')
    panel6 = html[i:html.index('id="panel-', i + 20)]
    check("Distributed Data Parallel" in panel6, "DDP explainer content lost")
    ddp = re.search(r'<details class="advanced">\s*<summary[^>]*>[^<]*DDP[^<]*</summary>', panel6)
    check(ddp is not None, "the DDP essay must collapse into a <details class=advanced>")
    check("guaranteed 2" not in panel6, "the DDP essay was not trimmed")
    i = html.index('id="panel-step4test"')
    panel4 = html[i:html.index('id="panel-', i + 20)]
    # 2026-09-16: the panel's description block went out with every other
    # paragraph (workflow page design version 2, section 6), so the sentence
    # that used to state the export paths twice cannot be stated at all.
    check('class="step-description"' not in panel4,
          "the Place + Segment panel must carry no description paragraph")
    check("The SAM3 parameters below control segmentation." not in panel4,
          "the duplicated Export note survives")
    # A checkbox label carries its own words and nothing else; the explanation
    # is on the control's tooltip and its "?" glyph.
    for cid in ("s4test-reference",):
        m = re.search(r'<label[^>]*for="' + cid + r'"[^>]*>(.*?)</label>', panel4, re.S)
        check(m is not None, f"label for {cid} not found")
        check("<span" not in m.group(1),
              f"{cid} label still carries an inline explainer span (tooltip only)")
        check('title="' in m.group(1), f"{cid} must carry its explanation as a tooltip")


def test_star_legend_is_tooltips():
    html = _html()
    i = html.index('id="panel-6"')
    panel6 = html[i:html.index('id="panel-', i + 20)]
    check("Time-impact legend" not in panel6, "the Step 6 star legend prose survives")
    stars = re.findall(r'<span[^>]*>\s*&#9733;[^<]*</span>|<span[^>]*>\u2605[^<]*</span>', panel6)
    check(len(stars) >= 6, f"expected the per-field star badges to remain (found {len(stars)})")
    for s in stars:
        check('title="' in s, f"star badge without a tooltip: {s[:80]}")


# ── live status for the non-chain tiles ─────────────────────────────
def test_tool_tiles_have_live_status():
    html = _html()
    for sid in ("step4test-status-text", "step4loop-status-text", "editmasks-status-text"):
        check(f'id="{sid}"' in html, f"missing live status span {sid}")
    js = _read(_JS)
    check("function refreshToolTileStatuses(" in js,
          "refreshToolTileStatuses() must drive the non-chain tile captions")
    i = js.index("async function refreshToolTileStatuses(")
    body = js[i:js.index("\n}\n", i)]
    check("/api/step/step4test/route_status" in body,
          "tool tile status must read the routing phase for Place + Segment")
    check("'step4loop', 'editmasks'" in body,
          "tool tile status must cover the Refine and Edit Masks tiles")
    check("/api/step/${key}/status" in body,
          "tool tile status must read each tool's /status route")
    # The step4loop / editmasks /status payloads are StageRunner.poll_status:
    # {running, exit_code, log_tail} plus healthy/port. There is NO error key,
    # so an `s.error` test would leave the error caption unreachable and a
    # crashed sub-app would render the idle caption. The error state has to be
    # derived from a nonzero exit code on a process that stopped running.
    check("s.error" not in body,
          "the /status payload has no error field; deriving from s.error is dead code")
    check("toolExitedBadly(" in body,
          "the error caption must come from the exit-code derivation")
    d = js.index("function toolExitedBadly(")
    deriv = js[d:js.index("\n}\n", d)]
    check("exit_code" in deriv, "toolExitedBadly() must read exit_code")
    check("running === false" in deriv or "!s.running" in deriv,
          "toolExitedBadly() must require the process to have stopped")
    check("!== 0" in deriv, "toolExitedBadly() must require a NONZERO exit code")
    for word in ("running", "open", "error", "routing"):
        check(f"'{word}'" in body, f"tool tile status must be able to read {word!r}")
    # One route_status read per tick, shared by the banner and the tile.
    i = js.index("async function pollToolTiles(")
    tick = js[i:js.index("\n}\n", i)]
    check(tick.count("/api/step/step4test/route_status") == 1,
          "pollToolTiles must read route_status exactly once per tick")
    check("renderStep4testRouteBanner(rs)" in tick and "refreshToolTileStatuses(rs)" in tick,
          "the shared route_status must be handed to both consumers")
    check("_toolTilesTickBusy" in tick,
          "pollToolTiles needs an in-flight guard so slow ticks cannot stack")
    check("if (_toolTilesTickBusy) return;" in tick,
          "the in-flight guard must skip the tick, not queue it")
    # The banner keeps its own fetch for every other caller.
    b = js.index("async function renderStep4testRouteBanner(")
    banner = js[b:js.index("\n}\n", b)]
    check("routeStatus" in banner and "rs === undefined" in banner,
          "renderStep4testRouteBanner must accept a shared route_status and fall back to its own read")


def test_tool_tile_tooltips_match_emitted_states():
    """Each tile tooltip has to name the states the code can actually paint,
    and the idle caption must not just repeat the tile title."""
    html = _html()
    js = _read(_JS)
    i = js.index("const TOOL_TILE_IDLE = {")
    idle = dict(re.findall(r"(\w+):\s*'([^']+)'", js[i:js.index("};", i)]))
    check(set(idle) == {"step4test", "step4loop", "editmasks"},
          f"unexpected idle caption map: {idle}")
    nav = re.search(r'<ul class="step-nav" id="step-nav">(.*?)</ul>', html, re.DOTALL)
    for key, caption in idle.items():
        li = re.search(r'<div class="step-status" id="' + key + r'-status-text"\s*'
                       r'title="([^"]+)">([^<]*)</div>', nav.group(1))
        check(li is not None, f"{key} status span not found")
        tip, shown = li.group(1), li.group(2).strip()
        check(shown == caption,
              f"{key} renders {shown!r} but TOOL_TILE_IDLE says {caption!r}")
        title = re.search(r'<li[^>]*data-step="' + key + r'"[\s\S]*?'
                          r'<div class="step-title">([^<]*)</div>', nav.group(1))
        check(title is not None, f"{key} tile title not found")
        check(shown.lower() != title.group(1).strip().lower(),
              f"{key} idle caption just repeats the tile title ({shown!r})")
        # The tooltip must name every state this tile can show and no others.
        for state in ["running", "open", "error"] + (["routing"] if key == "step4test" else []):
            check(state in tip, f"{key} tooltip does not name the {state!r} state")
        check(caption in tip, f"{key} tooltip does not name its idle caption")
        for bogus in ("not started", "starting"):
            check(bogus not in tip, f"{key} tooltip names {bogus!r}, which the code never emits")


# ── cache-bust ──────────────────────────────────────────────────────
def test_asset_version_on_both_tags():
    import app as appmod
    check(getattr(appmod, "ASSET_VERSION", None) == "20260921-start",
          "app.ASSET_VERSION must be '20260921-start'")
    html = _html()
    v = appmod.ASSET_VERSION
    check(f"/static/style.css?v={v}" in html, "style.css tag missing the asset version")
    check(f"/static/orchestrator.js?v={v}" in html, "orchestrator.js tag missing the asset version")
    tpl = _read(_TPL)
    check(tpl.count("{{ asset_version }}") >= 2,
          "the template must inject asset_version into both asset URLs (single bump point)")


# ── smoke ───────────────────────────────────────────────────────────
def test_smoke_render_200():
    _html()
    check(_CACHE["status"] == 200, f"GET / != 200 (got {_CACHE['status']})")


def test_smoke_node_check():
    node = _node()
    check(node is not None, "node not available")
    r = subprocess.run([node, "--check", _JS], capture_output=True, text=True)
    check(r.returncode == 0, f"orchestrator.js node --check failed:\n{r.stderr}")


if __name__ == "__main__":
    print("Task 1 orchestrator SPA sweep (2026-09-01)\n")
    print("SMOKE:")
    run(test_smoke_render_200)
    run(test_smoke_node_check)
    print("ARCHIVED PANELS:")
    run(test_panels_4_and_5_removed)
    run(test_js_drops_panel_4_and_5_wiring)
    run(test_sam3_mini_poll_gone)
    run(test_backend_routes_for_steps_4_and_5_untouched)
    print("COPY:")
    run(test_frontend_files_have_no_dashes)
    run(test_app_client_strings_have_no_dashes)
    run(test_step2_vocabulary)
    run(test_dewordify)
    run(test_star_legend_is_tooltips)
    run(test_internal_codenames_gone)
    run(test_step3_progress_and_flag_copy)
    print("STEP NUMBERING:")
    run(test_step_display_matches_sidebar)
    run(test_js_step_display_matches_app)
    run(test_app_errors_use_display_numbers)
    print("TOOLTIPS + KEYBOARD:")
    run(test_listed_controls_have_tooltips)
    run(test_gallery_verdict_buttons_have_tooltips)
    run(test_modal_tooltips_name_the_keys)
    run(test_keyboard_bindings_present)
    print("LINKAGE:")
    run(test_tools_cluster_present)
    run(test_step6_source_hint_links_the_matrix)
    run(test_step6_tooltip_names_training_sets_tab)
    run(test_tool_tiles_have_live_status)
    run(test_tool_tile_tooltips_match_emitted_states)
    print("RELICS + CACHE-BUST:")
    run(test_relics_removed)
    run(test_bounce_fallback_port)
    run(test_asset_version_on_both_tags)

    print(f"\n{len(_RESULTS) - _fail}/{len(_RESULTS)} passed.")
    if _fail:
        print("\nFAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"--- {name} ---\n{detail}")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
