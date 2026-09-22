"""
Self-contained smoke + unit tests for the pipeline-orchestrator UI revamp
(Track A + Phase-2 shell).

No pytest dependency: run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_orchestrator_ui.py

Covers:
  SMOKE  - app imports, GET / renders 200, both JS sources pass `node --check`,
           inline template <script> blocks (Jinja-stripped) pass `node --check`.
  UNIT   - nav order is 1,2,3,step4test,6,7,8 (Place + Segment is Step 4; the
           legacy 4/5 panels were deleted 2026-09-01; expertids tile disabled
           2026-07-09 via data-step-disabled),
           every live step panel + non-chain panel has a nav-footer placeholder,
           the Place + Segment panel has a collapsed Advanced <details> and
           carries info-circle tooltips on its visible settings,
           the JS getNavOrder()/renderStepNavFooters() pick the right neighbours.
"""

import os
import re
import sys
import subprocess
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_TPL = os.path.join(_ORCH, "templates", "index.html")
_JS = os.path.join(_ORCH, "static", "orchestrator.js")

sys.path.insert(0, _ORCH)

# ── tiny test harness ───────────────────────────────────────────────
_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _node():
    for cand in ("/usr/bin/node", "node"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


# ── render helper (cached) ──────────────────────────────────────────
_RENDERED = {}


def _html():
    if "html" not in _RENDERED:
        import app as appmod
        client = appmod.create_app().test_client()
        resp = client.get("/")
        _RENDERED["status"] = resp.status_code
        _RENDERED["html"] = resp.get_data(as_text=True)
    return _RENDERED["html"]


def _nav_order(html):
    nav = re.search(r'<ul class="step-nav" id="step-nav">(.*?)</ul>', html, re.DOTALL)
    check(nav is not None, "step-nav <ul> not found")
    return re.findall(r'data-step="([^"]+)"', nav.group(1))


# ── SMOKE ───────────────────────────────────────────────────────────
def test_smoke_render_200():
    _html()
    check(_RENDERED["status"] == 200, f"GET / != 200 (got {_RENDERED['status']})")
    # The wordmark reads VEGA since the V-fleet naming ruling of 2026-09-02;
    # test_ui_sweep_20260905.py pins both wordmarks and the window title.
    check("VEGA" in _RENDERED["html"], "page wordmark missing")


def test_smoke_node_check_js():
    node = _node()
    check(node is not None, "node not available")
    r = subprocess.run([node, "--check", _JS], capture_output=True, text=True)
    check(r.returncode == 0, f"orchestrator.js node --check failed:\n{r.stderr}")


def test_smoke_node_check_inline_blocks():
    node = _node()
    check(node is not None, "node not available")
    html = open(_TPL).read()
    blocks = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, re.DOTALL | re.IGNORECASE)
    checked = 0
    import tempfile
    for attrs, body in blocks:
        if "src=" in attrs.lower():
            continue
        body = body.strip()
        if not body:
            continue
        body = re.sub(r"\{%.*?%\}", "", body, flags=re.DOTALL)
        body = re.sub(r"\{\{.*?\}\}", "0", body, flags=re.DOTALL)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(body)
            path = fh.name
        try:
            r = subprocess.run([node, "--check", path], capture_output=True, text=True)
            check(r.returncode == 0, f"inline block failed node --check:\n{r.stderr}")
            checked += 1
        finally:
            os.unlink(path)
    # The window carries no inline code any more: every block moved into a file
    # under static/ during the 2026-09-16 panel rewrite, so this checks that any
    # block that comes back still parses, and that each file the page loads
    # carries the cache-bust the release bumps.
    srcs = re.findall(r'<script\b[^>]*src="([^"]+)"', html)
    check(len(srcs) >= 1, "the page must load at least one script file")
    for src in srcs:
        check("?v={{ asset_version }}" in src,
              f"script {src} must carry the cache-bust query")


# ── UNIT: template structure ────────────────────────────────────────
def test_nav_order_phase2_reorder():
    order = _nav_order(_html())
    # 4.test is promoted to Step 4 (route key stays "step4test"); the legacy Place
    # Points (4) + Segment (5) tiles are archived. Train/eval/inference keep their
    # route keys 6/7/8 (displayed as 5/6/7 via the circle data-num). The expertids
    # tile is disabled 2026-07-09 (data-step-disabled) so it drops out of the chain.
    # step4loop (Refine loop) is a non-chain tile appended after Inference (8) and
    # before the Review divider (Wave 3, 2026-07-09). editmasks (Edit Masks) is a
    # non-chain tile in the Review group, after the divider (Task 10, 2026-07-10).
    check(
        order == ["1", "2", "3", "step4test", "6", "7", "8", "step4loop", "editmasks"],
        f"unexpected nav order: {order}",
    )


def test_nav_tool_glyphs():
    """Each tool tile keeps its own glyph, so the Tools group reads as four
    distinct destinations rather than four identical circles. The Expert Review
    I/O tile and its star went out with the 2026-09-16 relic sweep: nothing in
    the window reached it, so it is no longer drawn."""
    html = _html()
    for glyph, name in (("&#9638;", "Coverage Matrix"), ("&#9782;", "Label Manager"),
                        ("&#9998;", "Annotate Images"), ("&#9636;", "Bounding Boxes")):
        tile = re.search(r'<li class="nav-tool">.*?' + re.escape(glyph)
                         + r'.*?<div class="step-title">' + re.escape(name),
                         html, re.DOTALL)
        check(tile is not None, f"{name} tool tile lost its {glyph} glyph")
    check('data-step="s4review"' not in html, "s4review tile should be removed")


def test_expertids_entry_point_absent():
    """The Expert Review pathway was disabled on 2026-07-09 and its dead tile,
    panel and footer were removed on 2026-09-16: a control nobody can press and
    a panel nobody can open are not kept on screen. The /expertids blueprint is
    still mounted in app.py and is reached directly, not from this window."""
    html = _html()
    for marker in ('data-step="expertids"', 'data-step-disabled="expertids"',
                   'id="panel-expertids"', 'id="nav-footer-expertids"',
                   "Expert Review I/O"):
        check(marker not in html, f"dead Expert Review markup still on the page: {marker}")


def test_all_panels_have_nav_footers():
    html = _html()
    # expertids dropped 2026-09-16 with the rest of its dead markup.
    for ds in ("1", "2", "3", "step4test", "6", "7", "8", "step4loop", "editmasks"):
        check(f'id="nav-footer-{ds}"' in html, f"missing nav-footer for {ds}")


def test_s4review_shell_absent():
    html = _html()
    check('id="panel-s4review"' not in html, "panel-s4review should be removed")
    check('data-step="s4review"' not in html, "s4review nav tile should be removed")


def test_step4test_tile_present():
    # The combined-annotator tile + panel must be present. It is promoted to
    # Step 4 (route key stays "step4test"); the experimental copy is gone.
    html = _html()
    check('data-step="step4test"' in html, "step4test nav tile missing")
    check('id="panel-step4test"' in html, "panel-step4test missing")
    check("Place + Segment" in html, "combined-annotator tile label missing")
    check("experimental" not in html, "experimental copy should be removed (4.test promoted)")
    check('id="nav-footer-step4test"' in html, "nav-footer-step4test missing")
    src = open(_JS).read()
    check("function openStep4test" in src, "openStep4test JS launcher missing")


def test_legacy_step4_step5_panels_are_gone():
    """2026-09-01: the archived Place Points (4) and Segment (5) panels were
    deleted from the DOM. Their backend routes stay; their widgets must not."""
    html = _html()
    check('id="panel-4"' not in html, "panel-4 must be gone")
    check('id="panel-5"' not in html, "panel-5 must be gone")
    check('id="s4-port"' not in html, "s4-port survived the panel removal")
    check('id="s5-port"' not in html, "s5-port survived the panel removal")
    check(html.count('class="advanced"') >= 2, "expected >=2 Advanced details blocks")


def test_step4test_advanced_and_info_circles():
    """The SAM3 settings the archived Segment panel carried now live only on
    the Place + Segment panel, inside its Advanced block."""
    html = _html()
    for i in ("s4test-tracker-device", "s4test-exemplar-device", "s4test-symlink",
              "s4test-confidence", "s4test-overlap"):
        check(f'id="{i}"' in html, f"{i} input lost from the Place + Segment panel")
    # The info-circle span became a focusable .tip button on 2026-09-16, drawn
    # by the q() and sec() macros: "?" beside a field label, "i" beside a
    # section heading. Count them on this panel, not on the whole page.
    tpl = open(_TPL).read()
    i = tpl.index('id="panel-step4test"')
    j = tpl.index('id="panel-', i + 20)
    panel = tpl[i:j]
    glyphs = panel.count("{{ q(") + panel.count("{{ sec(")
    check(glyphs >= 6,
          f"expected at least 6 tooltip glyphs on the Place + Segment panel, found {glyphs}")
    check('class="info-circle"' not in html, "the retired info-circle span is back")


def test_tool_tile_status_spans_present():
    """The batch readouts belonged to the archived panels. The live captions on
    the non-chain tool tiles replaced them (2026-09-01)."""
    html = _html()
    for i in ("s4-batch-readout", "s5-batch-readout"):
        check(f'id="{i}"' not in html, f"archived batch readout {i} survives")
    for i in ("step4test-status-text", "step4loop-status-text", "editmasks-status-text"):
        check(f'id="{i}"' in html, f"missing tool tile status span {i}")


def test_open_ui_overlay_present():
    html = _html()
    check('id="open-ui-overlay"' in html, "open-ui-overlay missing")
    check('id="open-ui-modal"' in html, "open-ui-modal missing")
    check("hideOpenUiOverlay()" in html, "overlay close handler not wired")


def test_cache_bust_bumped():
    html = _html()
    m = re.search(r"orchestrator\.js\?v=([^\"']+)", html)
    check(m is not None, "orchestrator.js cache-bust query missing")
    import app as appmod
    check(m.group(1) == appmod.ASSET_VERSION,
          f"cache-bust query is {m.group(1)}, expected app.ASSET_VERSION ({appmod.ASSET_VERSION})")


def test_reroute_button_removed():
    """The "Re-route from Step 3" button + handler are gone (routing folded into
    Step 3's Open action)."""
    html = _html()
    check("Re-route from Step 3" not in html, "reroute button still present in HTML")
    check("rerouteStep4test" not in html, "rerouteStep4test handler still referenced in HTML")
    js = open(_JS).read()
    check("rerouteStep4test" not in js, "rerouteStep4test still defined in orchestrator.js")
    check("/api/step/step4test/reroute" not in js, "stale /reroute endpoint still called in JS")


# ── UNIT: JS logic executed under node ──────────────────────────────
def _run_node_snippet(js):
    node = _node()
    check(node is not None, "node not available")
    src = open(_JS).read()
    # Provide a minimal DOM shim so the pure-ish helpers run headless. We stub
    # only what the functions under test touch.
    harness = r"""
const __navOrder = ["1","2","3","step4test","6","7","8","step4loop","editmasks"];
const __titles = {
  "1":"Make All Points","2":"Recode Labels","3":"Choose Images",
  "step4test":"Place + Segment",
  "6":"Train Model","7":"Evaluate Model","8":"Model Inference",
  "step4loop":"Refine (model loop)",
  "editmasks":"Edit Masks"
};
const __footers = {};
const __batch = {};
global.document = {
  querySelectorAll(sel){
    if (sel === '#step-nav li[data-step]') {
      return __navOrder.map(ds => ({ getAttribute:(k)=> k==='data-step'?ds:null,
        querySelector:()=>({textContent:__titles[ds]}) }));
    }
    return [];
  },
  querySelector(sel){
    const m = sel.match(/data-step="([^"]+)"/);
    if (m) { const ds=m[1]; return { querySelector:()=>({textContent:__titles[ds]}) }; }
    return null;
  },
  getElementById(id){
    if (id.startsWith('nav-footer-')) {
      const ds = id.slice('nav-footer-'.length);
      const node = { _html:'', set innerHTML(v){ this._html=v; __footers[ds]=v; }, get innerHTML(){return this._html;} };
      return node;
    }
    if (id.endsWith('-batch-done')) { return { set textContent(v){__batch[id]=v;}, get textContent(){return __batch[id];} }; }
    if (id.endsWith('-batch-left')) { return { set textContent(v){__batch[id]=v;}, get textContent(){return __batch[id];} }; }
    if (id.endsWith('-batch-size')) { return { value: global.__BATCH_SIZE }; }
    return null;
  }
};
global.__footers = __footers;
global.__batch = __batch;
const STEP_DISPLAY = {1:"1",2:"2",3:"3",step4test:"4",6:"5",7:"6",8:"7"};
"""
    # Extract just the functions we need from the real source so we test the
    # shipped implementation rather than a copy.
    wanted = []
    for name in ("getNavOrder", "navStepArg", "navTitleFor",
                 "renderStepNavFooters", "escHtmlNav", "escAttr",
                 "stepDisplayNum"):
        m = re.search(r"\nfunction %s\b[\s\S]*?\n}\n" % re.escape(name), "\n" + src + "\n")
        check(m is not None, f"could not extract function {name} from orchestrator.js")
        wanted.append(m.group(0))
    program = harness + "\n" + "\n".join(wanted) + "\n" + js
    r = subprocess.run([_node(), "-e", program], capture_output=True, text=True)
    check(r.returncode == 0, f"node snippet errored:\n{r.stderr}")
    return r.stdout.strip()


def test_js_getNavOrder():
    out = _run_node_snippet("console.log(JSON.stringify(getNavOrder()));")
    check(out == '["1","2","3","step4test","6","7","8","step4loop","editmasks"]',
          f"getNavOrder() wrong: {out}")


def test_js_footers_neighbours():
    # Promoted topology: 1,2,3,step4test,6,7,8,step4loop,editmasks (legacy 4,5
    # tiles archived; expertids disabled 2026-07-09 so it left the chain).
    # step4test sits right after Choose Images; since it replaces Place Points
    # + Segment, its Next is OVERRIDDEN to Train Model (step 6). Train (6)
    # Prev -> step4test, Next -> 7. step4loop (Refine loop) is second-to-last;
    # its Next loops back to Train (6) rather than dead-ending, so step 8's
    # Next targets step4loop instead of being disabled. editmasks (Edit Masks,
    # Task 10) is now last: its Prev goes to step4loop, its Next is disabled
    # (nothing after it), same as any array-final tile.
    js = (
        "renderStepNavFooters();"
        "const f = global.__footers;"
        "const out = {"
        "  three: f['3'],"
        "  step4test: f['step4test'],"
        "  six: f['6'],"
        "  eight: f['8'],"
        "  step4loop: f['step4loop'],"
        "  editmasks: f['editmasks']"
        "};"
        "console.log(JSON.stringify(out));"
    )
    out = _run_node_snippet(js)
    import json
    f = json.loads(out)
    check('switchStep("step4test")' in f["three"], "step3 Next should go to step4test")
    check("switchStep(3)" in f["step4test"], "step4test Prev should go to step 3")
    check("switchStep(6)" in f["step4test"], "step4test Next should go to Train Model (step 6)")
    check('switchStep("step4test")' in f["six"], "train (6) Prev should go to step4test")
    check("switchStep(7)" in f["six"], "train (6) Next should go to step 7")
    check('switchStep("expertids")' not in f["eight"], "step8 Next must not target disabled expertids")
    check("switchStep(\"step4loop\")" in f["eight"], "step8 Next should go to Refine loop (step4loop)")
    check("switchStep(8)" in f["step4loop"], "step4loop Prev should go to step 8")
    check("switchStep(6)" in f["step4loop"], "step4loop Next should loop back to Train Model (step 6)")
    check("Loop to Train" in f["step4loop"], "step4loop Next label should read 'Loop to Train (5)'")
    check("switchStep(\"step4loop\")" in f["editmasks"], "editmasks Prev should go to step4loop")
    check("disabled" in f["editmasks"], "editmasks Next should be disabled (last tile in nav order)")


def test_js_step_display_num():
    """Displayed step numbers come from one map (2026-09-01), so a user-facing
    string can never print an internal chain slot again."""
    js = (
        "console.log([stepDisplayNum(3), stepDisplayNum(6), stepDisplayNum(7),"
        " stepDisplayNum(8), stepDisplayNum('step4test')].join('|'));"
    )
    out = _run_node_snippet(js)
    check(out == "3|5|6|7|4", f"stepDisplayNum wrong: {out}")


# ── ADVERSARIAL / structural skeptic cases ──────────────────────────
def _details_body(html, anchor_id):
    """Return the inner text of the <details class="advanced"> block that
    contains the given element id, or None if it's not nested inside one."""
    # Find every advanced details block and return the one wrapping anchor_id.
    for m in re.finditer(r'<details class="advanced">(.*?)</details>', html, re.DOTALL):
        if f'id="{anchor_id}"' in m.group(1):
            return m.group(1)
    return None


def test_adv_step4test_sam3_inputs_are_inside_advanced():
    # Skeptic: the inputs could exist on the page but NOT be collapsed. Assert
    # they are physically nested in the Place + Segment Advanced <details>.
    html = _html()
    body = _details_body(html, "s4test-tracker-device")
    check(body is not None, "s4test-tracker-device is not inside a <details class=advanced>")
    for needed in ("s4test-exemplar-device", "s4test-confidence", "s4test-min-area"):
        check(f'id="{needed}"' in body, f"{needed} not in the same Advanced block")


def test_adv_ddp_essay_is_collapsed():
    """The DDP explainer used to sit open beside the Device select. It is now a
    one-line hint plus a collapsed Advanced block (2026-09-01)."""
    html = _html()
    i = html.index('id="panel-6"')
    panel = html[i:html.index('id="panel-', i + 20)]
    check("Distributed Data Parallel" in panel, "DDP explainer content lost")
    m = re.search(r'<details class="advanced">\s*<summary[^>]*>[^<]*DDP[^<]*</summary>', panel)
    check(m is not None, "the DDP explainer must live in a collapsed Advanced block")


def test_adv_advanced_details_closed_by_default():
    # Collapsed = no `open` attribute on either advanced <details>.
    html = _html()
    opens = re.findall(r'<details class="advanced"([^>]*)>', html)
    check(len(opens) >= 2, "expected >=2 advanced details blocks")
    for attrs in opens:
        check("open" not in attrs, "advanced <details> must be closed by default")


def test_adv_expertids_onshow_hook_wired():
    # Phase-2 contract: from its NEW position the expertids panel must still
    # fire window.ExpertIDs.onShow() on show.
    src = open(_JS).read()
    check("window.ExpertIDs.onShow()" in src, "expertids onShow hook lost")
    check("step === 'expertids'" in src, "switchStep no longer special-cases expertids")


def test_adv_step4test_next_targets_train():
    # step4test is promoted to Step 4 and replaces Place Points + Segment, so its
    # Next jumps to Train Model (step 6, displayed as Step 5).
    js = (
        "renderStepNavFooters();"
        "console.log(JSON.stringify({s4test: global.__footers['step4test']}));"
    )
    out = _run_node_snippet(js)
    import json
    f = json.loads(out)
    check("switchStep(6)" in f["s4test"], "step4test Next must go to Train Model (step 6)")
    check('switchStep("s4review")' not in out, "must not reference the removed s4review")


def test_adv_first_panel_prev_disabled():
    # Step 1 (first in nav order) must have a disabled Prev so we never index
    # off the front of the order array.
    js = (
        "renderStepNavFooters();"
        "console.log(global.__footers['1']);"
    )
    out = _run_node_snippet(js)
    check("disabled" in out, "step 1 Prev should be disabled (first in order)")
    check("switchStep(2)" in out, "step 1 Next should target step 2")


# ── Task 4: surfaced routing clip-dir + symlink controls ────────────
def test_s4test_clip_dir_and_symlink_controls_present():
    html = _html()
    for cid in ("s4test-clip-dir", "s4test-symlink"):
        check(f'id="{cid}"' in html, f"{cid} input missing from #panel-step4test")
    # platform rule: every field carries a tooltip
    check(re.search(r'id="s4test-clip-dir"[^>]*title="[^"]+"', html) is not None,
          "s4test-clip-dir missing a title= tooltip")
    check(re.search(r'id="s4test-symlink"[^>]*title="[^"]+"', html) is not None,
          "s4test-symlink missing a title= tooltip")


def test_s4test_collects_clip_dir_and_symlink():
    js = open(_JS).read()
    check("cfg.clip_dir" in js, "collectStep4testConfig does not collect clip_dir")
    check("cfg.symlink_images" in js,
          "collectStep4testConfig does not collect symlink_images")


def test_step4test_popup_opens_on_user_gesture():
    """Regression guard for the "Open Combined Annotator does nothing" bug.

    The annotator popup MUST be opened synchronously inside the openStep4test
    click handler (openStep4testPlaceholder), then NAVIGATED once healthy - not
    opened with a fresh window.open() after the routing/SAM3 await chain, which
    the browser's popup blocker silently kills."""
    js = open(_JS).read()
    # The placeholder popup is opened up-front, inside the click handler, and
    # openStep4test bails with a "blocked" message when the browser refuses it.
    check("function openStep4testPlaceholder()" in js,
          "openStep4testPlaceholder() (synchronous popup) is missing")
    m = re.search(r"async function openStep4test\(\)\s*\{(.*?)\n\}", js, re.S)
    check(m is not None, "openStep4test() not found")
    # Strip // comment lines so prose mentioning "await" does not confuse the
    # ordering check below.
    body = "\n".join(ln for ln in m.group(1).splitlines()
                     if not ln.lstrip().startswith("//"))
    check("openStep4testPlaceholder()" in body,
          "openStep4test() must open the popup on the click, before any await")
    check(body.index("openStep4testPlaceholder()") < body.index("await "),
          "popup must be opened BEFORE the first await (or the popup blocker kills it)")
    # openStep4testWindow navigates the existing popup rather than opening a new
    # one from inside the async callback.
    w = re.search(r"async function openStep4testWindow\(uiPort, win\)\s*\{(.*?)\n\}", js, re.S)
    check(w is not None, "openStep4testWindow(uiPort, win) not found")
    check("win.location.href = url" in w.group(1),
          "openStep4testWindow must navigate the existing popup (win.location.href)")


# ── Wave 3 (2026-07-09): Refine loop (step4loop) tile + panel ───────────────
def test_step4loop_tile_present():
    html = _html()
    check('data-step="step4loop"' in html, "step4loop nav tile missing")
    check("Refine (model loop)" in html, "Refine (model loop) tile label missing")
    check("&#8635;" in html, "step4loop circle glyph (&#8635;) missing")
    check('id="panel-step4loop"' in html, "panel-step4loop missing")
    check('id="nav-footer-step4loop"' in html, "nav-footer-step4loop missing")
    # Tile must be positioned after the inference tile (data-step="8") and
    # before the Review section divider.
    eight_idx = html.index('data-step="8"')
    loop_idx = html.index('data-step="step4loop"')
    review_idx = html.index('Review</div>')
    check(eight_idx < loop_idx < review_idx,
          "step4loop tile must sit after the inference tile (8) and before the Review divider")


def test_step4loop_panel_inputs_have_tooltips():
    html = _html()
    for cid in ("s4loop-run-dir", "s4loop-conf", "s4loop-max", "s4loop-skip-empty"):
        check(f'id="{cid}"' in html, f"{cid} input missing from #panel-step4loop")
        check(re.search(rf'id="{cid}"[^>]*title="[^"]+"', html) is not None,
              f"{cid} missing a title= tooltip")
    for bid in ("s4loop-open-btn", "s4loop-stop-btn", "s4loop-folder-btn"):
        check(f'id="{bid}"' in html, f"{bid} button missing from #panel-step4loop")
    check('id="s4loop-next-card"' in html, "s4loop-next-card missing")
    # 2026-09-21: the two builtin-preset jumps were renamed to plain jumps
    # into the Train step's starting-model choice (see the VEGA
    # starting-model-and-training-data design).
    check('id="s4loop-goto-base"' in html, "s4loop-goto-base button missing")
    check('id="s4loop-goto-champion"' in html, "s4loop-goto-champion button missing")


def test_step4loop_reset_description_present():
    js = open(_JS).read()
    check("step4loop:" in js, "RESET_DESCRIPTIONS missing a step4loop key")
    m = re.search(r"const RESET_DESCRIPTIONS = \{(.*?)\};", js, re.S)
    check(m is not None, "RESET_DESCRIPTIONS block not found")
    check("step4loop:" in m.group(1), "step4loop key not inside RESET_DESCRIPTIONS")


def test_step4loop_js_functions_present():
    js = open(_JS).read()
    for fn in ("refreshStep4loopRuns", "collectStep4loopConfig",
               "openStep4loop", "finishStep4loopOpen", "stopStep4loop"):
        check(f"function {fn}(" in js, f"{fn}() missing from orchestrator.js")


def test_step4loop_open_ui_label_present():
    js = open(_JS).read()
    check("step4loop:" in js and "OPEN_UI_LABELS" in js, "OPEN_UI_LABELS missing a step4loop entry")
    m = re.search(r"const OPEN_UI_LABELS = \{(.*?)\};", js)
    check(m is not None, "OPEN_UI_LABELS block not found")
    check("step4loop:" in m.group(1), "step4loop key not inside OPEN_UI_LABELS")


# ── Task 10 (2026-07-10): Edit Masks (editmasks) tile + panel ───────────────
def test_editmasks_tile_present():
    html = _html()
    check('data-step="editmasks"' in html, "editmasks nav tile missing")
    check("Edit Masks" in html, "Edit Masks tile label missing")
    check("&#9998;" in html, "editmasks circle glyph (&#9998; pencil) missing")
    check('id="panel-editmasks"' in html, "panel-editmasks missing")
    check('id="nav-footer-editmasks"' in html, "nav-footer-editmasks missing")
    # The tile sits in the Review group, after that group's divider. The Expert
    # Review I/O tile it used to sit beside was removed on 2026-09-16, so the
    # divider and the Tools group that follows it are what bound the group now.
    review_idx = html.index('Review</div>')
    editmasks_idx = html.index('data-step="editmasks"')
    tools_idx = html.index('Tools</div>')
    check(review_idx < editmasks_idx, "editmasks tile must sit after the Review divider")
    check(editmasks_idx < tools_idx, "editmasks tile must sit before the Tools group")


def test_editmasks_panel_buttons_have_tooltips():
    html = _html()
    for bid in ("s4em-open-btn", "s4em-stop-btn", "s4em-folder-btn"):
        check(f'id="{bid}"' in html, f"{bid} button missing from #panel-editmasks")
        check(re.search(rf'id="{bid}"[^>]*title="[^"]+"', html) is not None,
              f"{bid} missing a title= tooltip")
    # The nav tile itself also carries a tooltip (platform rule: every field).
    check(re.search(r'data-step="editmasks"[^>]*title="[^"]+"', html) is not None,
          "editmasks nav tile missing a title= tooltip")


def test_editmasks_reset_description_present():
    js = open(_JS).read()
    check("editmasks:" in js, "RESET_DESCRIPTIONS missing an editmasks key")
    m = re.search(r"const RESET_DESCRIPTIONS = \{(.*?)\};", js, re.S)
    check(m is not None, "RESET_DESCRIPTIONS block not found")
    check("editmasks:" in m.group(1), "editmasks key not inside RESET_DESCRIPTIONS")


def test_editmasks_js_functions_present():
    js = open(_JS).read()
    for fn in ("openEditMasks", "openEditMasksPlaceholder", "closeEditMasksWin",
               "finishEditMasksOpen", "watchEditMasksWindow", "stopEditMasks",
               "openEditMasksFolder"):
        check(f"function {fn}(" in js, f"{fn}() missing from orchestrator.js")


def test_editmasks_open_ui_label_present():
    js = open(_JS).read()
    check("editmasks:" in js and "OPEN_UI_LABELS" in js, "OPEN_UI_LABELS missing an editmasks entry")
    m = re.search(r"const OPEN_UI_LABELS = \{(.*?)\};", js)
    check(m is not None, "OPEN_UI_LABELS block not found")
    check("editmasks:" in m.group(1), "editmasks key not inside OPEN_UI_LABELS")


def test_editmasks_popup_opens_on_user_gesture():
    """Same popup-blocker regression guard as step4test/step4loop: the popup
    must be opened synchronously inside the click handler, before any await."""
    js = open(_JS).read()
    check("function openEditMasksPlaceholder()" in js,
          "openEditMasksPlaceholder() (synchronous popup) is missing")
    m = re.search(r"async function openEditMasks\(\)\s*\{(.*?)\n\}", js, re.S)
    check(m is not None, "openEditMasks() not found")
    body = "\n".join(ln for ln in m.group(1).splitlines()
                     if not ln.lstrip().startswith("//"))
    check("openEditMasksPlaceholder()" in body,
          "openEditMasks() must open the popup on the click, before any await")
    check(body.index("openEditMasksPlaceholder()") < body.index("await "),
          "popup must be opened BEFORE the first await (or the popup blocker kills it)")
    w = re.search(r"async function finishEditMasksOpen\(r, win\)\s*\{(.*?)\n\}", js, re.S)
    check(w is not None, "finishEditMasksOpen(r, win) not found")
    check("win.location.href = url" in w.group(1),
          "finishEditMasksOpen must navigate the existing popup (win.location.href)")


# ── Wave 3: Step 6 pin-split + fine-tune presets (Tasks 7/8) ────────────────
def test_step6_pin_split_checkbox_present():
    html = _html()
    check('id="s6-pin-split"' in html, "s6-pin-split checkbox missing")
    m = re.search(r'<input type="checkbox" id="s6-pin-split"([^>]*)>', html)
    check(m is not None, "s6-pin-split input tag not found")
    check("checked" in m.group(1), "s6-pin-split should be checked by default")
    check('title="' in m.group(1), "s6-pin-split missing a title= tooltip")


# 2026-09-21: the "Continue from a previous run" optgroup folded into
# s6-model and the two built-in preset buttons are gone, replaced by the
# Starting model select (base models only) and the Continue from a VEGA
# model disclosure. See the VEGA starting-model-and-training-data design.
def test_step6_model_optgroup_and_presets_present():
    html = _html()
    check('id="s6-model-runs-optgroup"' not in html, "the previous-run optgroup must be gone from s6-model")
    check("Continue from a previous run" not in html, "the retired optgroup label must be gone")
    check('id="s6-preset-finetune"' not in html, "s6-preset-finetune button must be gone")
    check('id="s6-preset-retrain"' not in html, "s6-preset-retrain button must be gone")
    check('id="s6-continue-details"' in html, "the Continue from a VEGA model disclosure is missing")
    check('id="s6-continue-from"' in html, "the Continue from select is missing")
    check('id="s6-starts-from-details"' in html, "the What the model starts from disclosure is missing")
    check(re.search(r'id="s6-continue-from"[^>]*title="[^"]+"', html) is not None,
          "s6-continue-from missing a title= tooltip")


# 2026-09-21: the two built-in preset buttons and applyStep6BuiltinPreset()
# are retired; the Refine card's after-export buttons jump straight to the
# Train panel through gotoStep6Start('base'|'champion') instead, and the
# runs-optgroup reader refreshStep6ModelRunOptions() is replaced by
# refreshStep6ContinueOptions() for the new Continue from a VEGA model
# disclosure. See the VEGA starting-model-and-training-data design.
def test_step6_preset_apply_js_present():
    js = open(_JS).read()
    check("function applyStep6BuiltinPreset(" not in js, "applyStep6BuiltinPreset() must be gone")
    check("async function gotoStep6Start(" in js, "gotoStep6Start() missing")
    check("function refreshStep6ModelRunOptions(" not in js, "refreshStep6ModelRunOptions() must be gone")
    check("async function refreshStep6ContinueOptions(" in js, "refreshStep6ContinueOptions() missing")
    check("model_path:" in js, "STEP6_PRESET_APPLIERS missing a model_path applier")
    check("freeze:" in js or "'freeze'" in js, "STEP6_PRESET_APPLIERS missing a freeze applier")
    check("cfg.model_path" in js, "collectConfig(6) does not collect model_path")
    check("cfg.pin_split" in js, "collectConfig(6) does not collect pin_split")


# D1 (2026-09-02, final-fix wave): a preset must never strand a project's
# starting model out of sync with its training task. Before this fix, the
# preset entry point wrote STEP6_PRESET_APPLIERS[key](val) straight into the
# form and nothing reconciled the result, so the tuned_v1_regularized default
# preset's `model: yolo11m-seg.pt` survived unchanged on a detect-task
# box-dataset project and the server refused the run. The fix funnels the
# entry point through one applyStep6PresetParams() that calls
# onStep6TaskChange() LAST, after every params[key] has already written.
#
# 2026-09-21: the second entry point this reconciliation used to guard,
# applyStep6BuiltinPreset(), is retired along with the two built-in preset
# buttons; importStep6Preset() is now the only caller of
# applyStep6PresetParams(), and the Refine card's after-export buttons reach
# the Train panel through gotoStep6Start() instead, which never touches
# preset params at all (it clears or sets Continue from a VEGA model).
def test_step6_preset_params_reconciles_after_apply():
    """House pattern: assert the call ordering inside the extracted function
    body, and that the one remaining preset entry point routes through it
    instead of looping over STEP6_PRESET_APPLIERS directly."""
    js = open(_JS).read()
    check("function applyStep6PresetParams(" in js,
          "applyStep6PresetParams() (the shared preset-apply + reconcile step) is missing")
    body = _js_function_body(js, "applyStep6PresetParams")
    check("STEP6_PRESET_APPLIERS[key]" in body,
          "applyStep6PresetParams must still write every preset param into the form")
    check("onStep6TaskChange()" in body,
          "applyStep6PresetParams must reconcile the model against the task")
    # Ordering, not just presence: the reconciliation call must read from the
    # loop's own text ordering as coming strictly AFTER the last
    # STEP6_PRESET_APPLIERS[key] reference, so it always runs once every
    # param (including a preset's `model` key) has already been written.
    last_applier_use = body.rindex("STEP6_PRESET_APPLIERS[key]")
    reconcile_at = body.index("onStep6TaskChange()")
    check(reconcile_at > last_applier_use,
          f"onStep6TaskChange() must be called AFTER the params loop "
          f"(applier use at {last_applier_use}, reconcile at {reconcile_at})")

    importer = _js_function_body(js, "importStep6Preset")
    check("applyStep6PresetParams(params)" in importer,
          "importStep6Preset must call applyStep6PresetParams (not loop over STEP6_PRESET_APPLIERS itself)")
    check("for (const [key, val] of Object.entries(params))" not in importer,
          "importStep6Preset must not keep its own apply loop now that applyStep6PresetParams reconciles")

    check("function applyStep6BuiltinPreset(" not in js,
          "the retired second preset entry point must be gone")
    goto = _js_function_body(js, "gotoStep6Start")
    check("STEP6_PRESET_APPLIERS" not in goto,
          "gotoStep6Start must not touch preset params; it only sets or clears "
          "Continue from a VEGA model")
    check("onStep6ContinueChange()" in goto,
          "gotoStep6Start('base') must clear the continue-from pick through onStep6ContinueChange")


def test_step6_preset_reconciliation_behavioral():
    """Real node execution (not just source-text): a segmentation preset's
    model applied to a detect-task project must land on the DETECT sibling,
    and vice versa, exactly reproducing the finalverify D1 repro
    (task=detect, preset model=yolo11m-seg.pt -> must NOT stay yolo11m-seg.pt).
    A non-stock model with no yolo11 sibling (e.g. yolo26l-seg.pt) still
    reconciles to the task's catalog fallback when its own suffix disagrees
    with the task (2026-09-21: step6ModelValueMatchesTask added this general
    catch, because the YOLO26 optgroup is outlines-only, spec section 3.2, so
    a YOLO26 model left selected on a detect-task project would be exactly
    the "form shows one thing, the launch carries another" bug this
    reconciliation exists to prevent)."""
    node = _node()
    check(node is not None, "node not available")
    src = open(_JS).read()
    wanted = []
    # renderStep6RecipeOptions is pulled in because onStep6TaskChange calls it
    # (2026-09-15: the training task is half of what chooses a recipe, so the
    # recipe list follows it). With no #s6-preset-select in this harness it
    # returns at once, which is what makes it safe to include here.
    #
    # step6ModelValueMatchesTask is the inline fallback check onStep6TaskChange
    # already used (a bare name reference, so Node throws ReferenceError at
    # call time if it is left out, not at parse time, which is why this went
    # unnoticed until the function actually ran).
    #
    # applyStep6ContinueTaskFilter, onStep6ContinueChange and
    # step6ContinueOptionElements are the new (2026-09-21) chain
    # onStep6TaskChange calls last, to hide a Continue-from pick that no
    # longer matches the task: applyStep6ContinueTaskFilter reads
    # #s6-continue-from, which this harness's document stub does not define,
    # so it returns immediately without reaching the other two; they are
    # still extracted (not merely stubbed) so a future harness that does
    # define #s6-continue-from exercises the real chain rather than a no-op.
    for name in ("applyStep6PresetParams", "onStep6TaskChange",
                 "applyStep6TaskToModelPicker", "step6SiblingWeights",
                 "step6ModelValueMatchesTask", "applyStep6ContinueTaskFilter",
                 "onStep6ContinueChange", "step6ContinueOptionElements", "step6TaskValue",
                 "renderStep6RecipeOptions"):
        m = re.search(r"\nfunction %s\b[\s\S]*?\n}\n" % re.escape(name), "\n" + src + "\n")
        check(m is not None, f"could not extract function {name} from orchestrator.js")
        wanted.append(m.group(0))
    m = re.search(r"\nconst STEP6_PRESET_APPLIERS = \{[\s\S]*?\n};\n", "\n" + src + "\n")
    check(m is not None, "could not extract STEP6_PRESET_APPLIERS from orchestrator.js")
    wanted.append(m.group(0))
    for name in ("STEP6_SEG_OPTGROUPS", "STEP6_DETECT_OPTGROUP"):
        m = re.search(r"\nconst %s = .*;\n" % re.escape(name), "\n" + src + "\n")
        check(m is not None, f"could not extract const {name} from orchestrator.js")
        wanted.append(m.group(0))

    harness = r"""
function makeSelect(value, optionValues) {
  return { value, options: optionValues.map(v => ({ value: v })) };
}
const s6Task = { value: 'detect' };
const s6Model = makeSelect('', ['yolo11n.pt','yolo11n-seg.pt','yolo11m.pt','yolo11m-seg.pt',
                                 'yolo11x.pt','yolo11x-seg.pt','yolo26l-seg.pt']);
global.document = {
  getElementById(id) {
    if (id === 's6-task') return s6Task;
    if (id === 's6-model') return s6Model;
    return null; // optgroup lookups etc: applyStep6TaskToModelPicker null-guards these
  }
};
"""
    program = harness + "\n" + "\n".join(wanted) + "\n" + """
const results = {};
// D1 repro: detect task, segmentation preset model -> must reconcile to yolo11m.pt.
s6Task.value = 'detect'; s6Model.value = '';
applyStep6PresetParams({ model: 'yolo11m-seg.pt' });
results.detect_task_seg_preset = s6Model.value;

// Mirror case: segment task, detector preset model -> must reconcile to yolo11m-seg.pt.
s6Task.value = 'segment'; s6Model.value = '';
applyStep6PresetParams({ model: 'yolo11m.pt' });
results.segment_task_detect_preset = s6Model.value;

// Non-stock model (no yolo11 sibling): step6SiblingWeights offers nothing to
// convert TO, but step6ModelValueMatchesTask still catches the suffix
// mismatch and falls back to the task's catalog default, same as any other
// seg-suffixed value would on a detect-task project.
s6Task.value = 'detect'; s6Model.value = '';
applyStep6PresetParams({ model: 'yolo26l-seg.pt' });
results.detect_task_yolo26_preset = s6Model.value;

// The matching case: a YOLO26 model on a SEGMENT-task project matches its
// own suffix, so it is left exactly as the preset set it.
s6Task.value = 'segment'; s6Model.value = '';
applyStep6PresetParams({ model: 'yolo26l-seg.pt' });
results.segment_task_yolo26_preset = s6Model.value;

console.log(JSON.stringify(results));
"""
    r = subprocess.run([node, "-e", program], capture_output=True, text=True)
    check(r.returncode == 0, f"node snippet errored:\n{r.stderr}")
    import json
    results = json.loads(r.stdout.strip())
    check(results["detect_task_seg_preset"] == "yolo11m.pt",
          f"D1: a segmentation preset's model must reconcile to the detect sibling on a detect-task "
          f"project (got {results['detect_task_seg_preset']!r}, the finalverify repro left this at yolo11m-seg.pt)")
    check(results["segment_task_detect_preset"] == "yolo11m-seg.pt",
          f"a detector preset's model must reconcile to the seg sibling on a segment-task project "
          f"(got {results['segment_task_detect_preset']!r})")
    check(results["detect_task_yolo26_preset"] == "yolo11m.pt",
          f"a YOLO26 (outlines-only) preset model must still reconcile to the task's catalog "
          f"fallback on a detect-task project, since its own family offers nothing that task "
          f"could use (got {results['detect_task_yolo26_preset']!r})")
    check(results["segment_task_yolo26_preset"] == "yolo26l-seg.pt",
          f"a YOLO26 model whose suffix already matches the task is left exactly as the preset "
          f"set it (got {results['segment_task_yolo26_preset']!r})")


# ── Wave 3: Step 7 Rounds table + Promote + champion badge (Task 9) ─────────
def test_step7_rounds_table_present():
    html = _html()
    check('id="s7-rounds-table"' in html, "s7-rounds-table missing")
    check('id="s7-rounds-tbody"' in html, "s7-rounds-tbody missing")
    # The score columns lost their (M) suffix on 2026-09-05: a row is now read
    # in its own metric family (outlines for a segmenter, boxes for a
    # detector) and each cell carries a family chip beside its number.
    for col in ("Round", "Run", "Base", "mAP50-95", "Recall", "Gates", "Promote"):
        check(col in html, f"Rounds table missing column header: {col}")
    for gone in ("mAP50-95(M)", "Recall(M)"):
        check(gone not in html, f"Rounds table still carries the mask-only header {gone}")


def test_step7_rounds_js_present():
    js = open(_JS).read()
    check("function refreshStep7Rounds(" in js, "refreshStep7Rounds() missing")
    check("/api/step/7/rounds" in js, "refreshStep7Rounds must call GET /api/step/7/rounds")
    check("/api/step/7/promote" in js, "promoteRun must call POST /api/step/7/promote")
    check("function promoteRun(" in js, "promoteRun() missing")


def test_step8_champion_badge_present():
    html = _html()
    check('id="s8-champion-badge"' in html, "s8-champion-badge missing from the inference run picker")
    js = open(_JS).read()
    check("champion_run_dir" in js, "refreshRunList must read champion_run_dir to pre-select/badge the champion")


def _js_function_body(js, name):
    """Return the source of `function <name>(` up to the next top-level function."""
    m = re.search(r"\n(?:async )?function " + re.escape(name) + r"\(", js)
    check(m is not None, f"{name}() missing from orchestrator.js")
    rest = js[m.end():]
    nxt = re.search(r"\n(?:async )?function \w+\(", rest)
    return rest[: nxt.start()] if nxt else rest


def test_auto_project_name_js_wiring():
    """Auto project names (2026-08-26): the topbar is drawn by renderProjectName()
    everywhere `state` is replaced from the server, and the Step 3 picker asks
    the server for a name preview instead of formatting names client-side."""
    js = open(_JS).read()
    check("function renderProjectName(" in js, "renderProjectName() missing from orchestrator.js")
    check("function previewProjectName(" in js, "previewProjectName() missing from orchestrator.js")
    check("/api/project/name-preview" in js, "orchestrator.js must GET /api/project/name-preview")

    enter = _js_function_body(js, "enterApp")
    check("topbar-project-name" not in enter,
          "enterApp() must not assign #topbar-project-name inline (use renderProjectName())")
    check("renderProjectName()" in enter, "enterApp() must call renderProjectName()")

    # Every server-driven `state` replacement re-draws the topbar.
    for fn in ("runStep", "linkExistingStep1", "markDone", "doResetStep", "autoApplyRemap"):
        check("renderProjectName()" in _js_function_body(js, fn),
              f"{fn}() replaces state but never calls renderProjectName()")
    poll = _js_function_body(js, "startPolling")
    check("renderProjectName()" in poll,
          "startPolling() completion branch reloads state but never calls renderProjectName()")

    # runStep reads the refreshed name off the config PUT response.
    run_body = _js_function_body(js, "runStep")
    check("state.name = saved.name" in run_body,
          "runStep() must copy the config PUT response's name into state.name")

    # Picker paths that change the selection trigger the preview.
    for fn in ("selectAllStep3Labels", "removeStep3Chip", "renderStep3Labels"):
        check("previewProjectName()" in _js_function_body(js, fn),
              f"{fn}() changes the Step 3 selection but never calls previewProjectName()")

    # The preview never writes state.name (server owns the rename on save).
    preview = _js_function_body(js, "previewProjectName")
    check("state.name =" not in preview, "previewProjectName() must not mutate state.name")
    check("encodeURIComponent" in preview, "previewProjectName() must URL-encode the species query")
    check("_namePreviewSeq" in preview,
          "previewProjectName() must drop out-of-order replies (request sequence token)")
    # Review fixes (2026-08-26): a state refresh re-derives an unsaved picker
    # preview, and the tooltip tells legacy hand-named projects the truth.
    check("function step3PickerDirty(" in js, "step3PickerDirty() missing from orchestrator.js")
    render = _js_function_body(js, "renderProjectName")
    check("previewProjectName()" in render and "step3PickerDirty()" in render,
          "renderProjectName() must re-fire the preview when the Step 3 picker is dirty")
    check("el.title" in render and "MANUAL_NAME_TIP" in render,
          "renderProjectName() must set the topbar tooltip per state.name_auto")

    html = _html()
    m = re.search(r'id="topbar-project-name"[^>]*title="([^"]+)"', html)
    check(m is not None, "#topbar-project-name has no title tooltip")
    check("initials" in m.group(1) and "image count" in m.group(1),
          "#topbar-project-name tooltip must describe the auto name parts")


# ── Task 4 (2026-08-26): one loader for the annotator launches ──────────────
# The three annotator launches (Combined Annotator, Refine Review, Edit Masks)
# used to paint TWO progress surfaces at once: the SPA modal (#open-ui-overlay)
# and the popup they open on the click. The popup is now the single loader for
# those three; the modal stays for every other service step.
def test_annotator_launches_use_single_loader():
    js = open(_JS).read()

    def body(fn):
        i = js.index(f"async function {fn}(")
        j = js.index("\n}\n", i)
        return js[i:j]

    for fn in ("openStep4test", "openStep4loop", "openEditMasks"):
        b = body(fn)
        check("showOpenUiOverlay(" not in b, f"{fn} must not open the SPA modal")
        check("setOpenUiError(" not in b, f"{fn} must not use the modal error")
    check("function showOpenUiOverlay" in js, "modal helper kept for non-annotator steps")


def test_boot_painter_supports_error_and_retry():
    js = open(_JS).read()
    i = js.index("function paintStep4testBoot(")
    b = js[i:js.index("\n}\n", i)]
    check("error" in b and "retry" in b and "boot-retry" in b,
          "paintStep4testBoot paints error + retry link")


def test_cache_bust_is_single_bump_point():
    """2026-09-01: both asset URLs read one ASSET_VERSION from app.py rather
    than carrying hand-edited query strings."""
    tpl = open(_TPL).read()
    check("orchestrator.js?v={{ asset_version }}" in tpl, "JS tag must read asset_version")
    check("style.css?v={{ asset_version }}" in tpl, "CSS tag must read asset_version")
    import app as appmod
    check(appmod.ASSET_VERSION == "20260921-start",
          f"ASSET_VERSION is {appmod.ASSET_VERSION}")


def test_annotator_popup_paintable_check():
    """A relaunch while the tool window is already open gets THAT window back
    from window.open('', name). It is cross-origin (its own port), so the title
    write throws and every later paint no-ops, which used to leave a failing
    relaunch with no message anywhere. Each placeholder records whether the
    document took our writes, and every failure goes through the one helper that
    reads that flag before choosing the popup or the inline note."""
    js = open(_JS).read()
    for fn in ("openStep4testPlaceholder", "openStep4loopPlaceholder",
               "openEditMasksPlaceholder"):
        b = _js_function_body(js, fn)
        check("bootPaintable[" in b, f"{fn}() must record whether the popup document is paintable")
        check("= true" in b and "= false" in b, f"{fn}() must set the paintable flag both ways")
    fail = _js_function_body(js, "annotatorLaunchFailed")
    check("bootPaintable[" in fail, "annotatorLaunchFailed() must consult the paintable flag")
    check("typeof paintable === 'boolean'" in fail,
          "annotatorLaunchFailed() must let the caller's snapshotted flag override the shared map")
    # Each opener snapshots the flag for ITS popup and hands it on, so a second
    # launch of the same flow cannot redirect the first launch's error message.
    for fn, key in (("openStep4test", "step4test"), ("openStep4loop", "step4loop"),
                    ("openEditMasks", "editmasks")):
        b = _js_function_body(js, fn)
        check(f"const paintable = bootPaintable['{key}']" in b,
              f"{fn}() must snapshot bootPaintable['{key}'] right after opening its popup")
        check(b.count("paintable);") >= 2,
              f"{fn}() must pass its snapshotted flag into annotatorLaunchFailed()")
    check("setAnnotatorOpenError(" in fail, "annotatorLaunchFailed() must fall back to the inline note")
    for fn in ("openStep4test", "openStep4loop", "openEditMasks", "finishStep4testOpen",
               "openStep4testWindow", "finishStep4loopOpen", "finishEditMasksOpen"):
        b = _js_function_body(js, fn)
        check("annotatorLaunchFailed(" in b,
              f"{fn}() must report launch failures through annotatorLaunchFailed()")
        check("error: true" not in b,
              f"{fn}() must not paint an error straight into the popup (skips the paintable check)")


def test_boot_skeleton_extracted():
    """One skeleton builder owns the boot-* id contract paintStep4testBoot reads,
    instead of three hand-copied blocks that could drift apart."""
    js = open(_JS).read()
    check("function bootSkeletonHtml(" in js, "bootSkeletonHtml() missing from orchestrator.js")
    check(js.count('id="boot-title"') == 1,
          'the boot ids belong in one place (id="boot-title" is written more than once)')
    for fn in ("openStep4testPlaceholder", "openStep4loopPlaceholder",
               "openEditMasksPlaceholder"):
        check("bootSkeletonHtml(" in _js_function_body(js, fn),
              f"{fn}() must build its popup body from bootSkeletonHtml()")


def test_closing_the_popup_stops_the_launch():
    """Every launch popup tells the operator that closing it stops the start.
    The closed-window branch of each wait loop has to keep that promise, or a
    cold SAM3 load runs on holding the GPU with nobody watching it."""
    js = open(_JS).read()
    check("cancelStep4testRoute()" in _js_function_body(js, "finishStep4testOpen"),
          "finishStep4testOpen() must cancel the routing pass when the popup closes")
    for fn, endpoint in (("openStep4testWindow", "/api/step/step4test/stop"),
                         ("finishStep4loopOpen", "/api/step/step4loop/stop"),
                         ("finishEditMasksOpen", "/api/step/editmasks/stop")):
        check(endpoint in _js_function_body(js, fn),
              f"{fn}() must post {endpoint} when the popup closes")


def test_prelaunch_failure_keeps_a_live_service_row():
    """/run answers success:false for ordinary outcomes too ("Nothing new to
    review"), while a Refine Review from an EARLIER launch can still be up with
    its window open. Hiding the row then takes Stop away from a live service, so
    the hide only ever undoes what THIS launch revealed."""
    js = open(_JS).read()
    for fn in ("openStep4loop", "openEditMasks"):
        b = _js_function_body(js, fn)
        check("rowWasHidden =" in b, f"{fn}() must capture the service row's prior visibility")
        check(b.index("rowWasHidden =") < b.index("style.display = 'block'"),
              f"{fn}() must capture the row's visibility BEFORE it reveals the row")
        check("if (rowWasHidden)" in b,
              f"{fn}() must hide the row only when this launch is the one that revealed it")


def test_prelaunch_failure_hides_service_row():
    """A launch the server refused never started a process, so the service row
    and its Stop button must not stay up offering to stop one. The !port /
    !healthy timeouts keep them: there a half-started process really exists."""
    js = open(_JS).read()
    for fn in ("openStep4loop", "openEditMasks"):
        b = _js_function_body(js, fn)
        check("if (svc) svc.style.display = 'none';" in b,
              f"{fn}() must hide the service row when the launch POST reports failure")
        check("if (stopBtn) stopBtn.style.display = 'none';" in b,
              f"{fn}() must hide the Stop button when the launch POST reports failure")


# ── Matrix drill-down wiring (spec 2026-08-26, Owner O) ─────────────────
def test_step6_source_select_present():
    tpl = open(_TPL).read()
    m = re.search(r'<select id="s6-source"[^>]*>', tpl)
    check(m is not None, "Step 6 training-source select (#s6-source) missing")
    check('title="' in m.group(0), "#s6-source needs a title tooltip")
    check(tpl.index('id="panel-6"') < m.start() < tpl.index('id="s6-preset-select"'),
          "#s6-source must sit in the Step 6 panel above the preset loader")
    check("This project's Step 4 export" in tpl[m.end():m.end() + 400],
          "default option must be the project's own Step 4 export")
    check('id="s6-source-note"' in tpl, "source note line missing")


def test_step6_source_js_wiring():
    js = open(_JS).read()
    i = js.index("function collectConfig(")
    body = js[i:js.index("\n}\n", i)]
    check("cfg.source_dir" in body and "s6-source" in body, "collectConfig(6) must send source_dir from #s6-source")
    check("async function refreshStep6Sources(" in js, "refreshStep6Sources missing")
    j = js.index("async function refreshStep6Sources(")
    fn = js[j:js.index("\n}\n", j)]
    check("/api/composites" in fn, "refreshStep6Sources must read /api/composites")
    check("This project's Step 4 export" in fn, "rebuilt options must keep the default entry")
    # Called from the Step 6 panel-open hook, next to the other Step 6
    # refreshers. The window is measured to the next step's hook (900 covers
    # the full block with margin) rather than a bare guess: a fixed guess of
    # 700 cut the call off mid-name once the explanatory comment above these
    # refresher calls grew with the 2026-09-21 runs-card move, which is a
    # test-only defect (the call itself is present), not a script one.
    k = js.index("if (step === 6) {")
    hook = js[k:k + 900]
    check("refreshStep6Sources(" in hook, "refreshStep6Sources not called when the Step 6 panel opens")
    # And when a project loads, seeded with the saved source_dir.
    check("refreshStep6Sources(s6.source_dir" in js, "project load must seed the saved source_dir")


def test_matrix_registration_passes_open_folder_fn():
    src = open(os.path.join(_ORCH, "app.py")).read()
    check("import host_open" in src, "app.py must import host_open")
    check("open_folder_fn=host_open.open_in_file_manager" in src,
          "matrix registration must pass open_folder_fn=host_open.open_in_file_manager")
    check(os.path.isfile(os.path.join(_ORCH, "host_open.py")), "host_open.py missing")
    import host_open
    check(callable(host_open.gui_env) and callable(host_open.open_in_file_manager), "host_open API")
    i = src.index("def _gui_env():")
    check("host_open.gui_env()" in src[i:i + 400], "_gui_env must delegate to host_open")
    i = src.index("def _open_in_file_manager(path):")
    check("host_open.open_in_file_manager(path)" in src[i:i + 300], "_open_in_file_manager must delegate")
    check('@app.route("/api/composites")' in src, "/api/composites route missing")


def test_restart_script_markers():
    sh = open(os.path.join(os.path.dirname(_ORCH), "restart_reef_point_seg.sh")).read()
    check("orchestrator.js?v=20260921-start" in sh, "restart script must fetch the fresh JS")
    check("20260826-auto-project-name" not in sh and "20260826-annotator-notes" not in sh
          and "20260902-uniform-training" not in sh and "20260902-final-polish" not in sh,
          "stale cache-bust markers left in the restart script")
    check("assert_html_present '20260921-start'" in sh, "restart self-test must pin the new cache-bust")
    check("assert_html_present 'Refine (model loop)'" in sh, "restart self-test must pin the renamed Refine tile label")
    check("id=\"s6-source\"" in sh, "restart self-test must check the Step 6 source select")
    check("function refreshStep6Sources" in sh, "restart self-test must check the JS refresher")


def test_owner_o_copy_has_no_banned_strings():
    """Only the strings this change added: the source block in the template,
    refreshStep6Sources in JS, host_open.py, and the new orchestrator helpers."""
    tpl = open(_TPL).read()
    i = tpl.index("<!-- Training source (2026-08-26)")
    j = tpl.index('id="s6-source-note"', i)
    js = open(_JS).read()
    k = js.index("async function refreshStep6Sources(")
    src = open(os.path.join(_ORCH, "app.py")).read()
    a = src.index("def _reviewer_initials():")
    b = src.index("def _folder_response(path, project_dir):")
    for name, text in (("template", tpl[i:j]), ("js", js[k:js.index("\n}\n", k)]),
                       ("host_open", open(os.path.join(_ORCH, "host_open.py")).read()),
                       ("app helpers", src[a:b])):
        low = text.lower()
        check("\u2014" not in text, f"{name}: em dash")
        check(not re.search(r"\bland\b", low), f"{name}: banned word")
        check(not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u2604\u2606-\u27BF]", text), f"{name}: emoji (U+2605 star legend excepted)")


def test_step6_class_picker_follows_source():
    """H2 (2026-08-26): the class picker is loaded from the selected training
    source, re-fetched when #s6-source changes, and collectConfig(6) tells the
    server which source the ids belong to."""
    js = open(_JS).read()
    i = js.index("async function refreshStep6Classes(")
    fn = js[i:js.index("\n}\n", i)]
    check("s6-source" in fn and "source_dir=" in fn and "encodeURIComponent" in fn,
          "refreshStep6Classes must read #s6-source and pass ?source_dir=")
    check("host.dataset.source = src" in fn, "the picker must record the source it was loaded from")
    check("_s6ClassesReq" in fn, "stale responses must be discarded (request token)")
    i = js.index("function collectConfig(")
    body = js[i:js.index("\n}\n", i)]
    check("cfg.include_classes_source" in body and "dataset.source" in body,
          "collectConfig(6) must send include_classes_source from the picker's recorded source")
    # Since 2026-09-05 the same handler also re-applies box-dataset mode, so
    # picking a staged box dataset pins the task and hides the class picker.
    check("e.target.id === 's6-source') { applyStep6BoxDatasetMode(); refreshStep6Classes(); }" in js,
          "changing #s6-source must re-apply box-dataset mode and re-fetch the class picker")
    j = js.index("async function refreshStep6Sources(")
    src_fn = js[j:js.index("\n}\n", j)]
    check(src_fn.count("syncStep6ClassesToSource(sel)") == 2,
          "refreshStep6Sources must re-sync the picker on both exit paths")
    check("function syncStep6ClassesToSource(" in js, "syncStep6ClassesToSource missing")
    check("data.dropped_classes" in js, "runStep must surface classes the training source lacks")
    app_src = open(os.path.join(_ORCH, "app.py")).read()
    k = app_src.index("def step6_list_classes():")
    route = app_src[k:app_src.index("@app.route", k)]
    check('request.args.get("source_dir")' in route and "_validate_outline_source(" in route,
          "list_classes must accept ?source_dir= validated like the training source")
    check('"include_classes_source": picker_source' in route, "list_classes must echo the picker source")
    m = re.search(r"\n    def _run_step6\(.*?(?=\n    def _run_step7\()", app_src, re.DOTALL)
    check(m is not None and "_translate_class_ids(" in m.group(0) and "include_classes_source" in m.group(0),
          "_run_step6 must map picker ids into the training source by name")
    tpl = open(_TPL).read()
    for label in ("Refresh classes", "Select all", "Clear"):
        mm = re.search(r'<button[^>]*onclick="(refreshStep6Classes|selectAllStep6Classes)\([^"]*"[^>]*>' + label + "</button>", tpl)
        check(mm is not None and 'title="' in mm.group(0), f"class-picker button '{label}' needs a title tooltip")
    check("Refresh from Step 4" not in tpl, "old picker button label removed")


def _render_with_state(state):
    """Render the window for one project state, without a running server.

    The Flask test client's GET / renders with no project open, so it can only
    ever show the seven-stage rail. A detection project's rail and its stage
    numbers are resolved in the template from project_state, so the family-aware
    copy has to be rendered with a state to be seen at all.

    Args:
        state: the mapping /api/project/state returns under its "state" key.
    Returns:
        The rendered page as text.
    """
    import app as appmod
    application = appmod.create_app()
    with application.app_context():
        return application.jinja_env.get_template("index.html").render(
            asset_version="test",
            repo_root="",
            projects_dir="",
            supporting_data_dir="",
            all_points_csv="",
            master_codes_csv="",
            project_loaded=True,
            project_state=state,
        )


# A project minted from a staged box dataset: it trains a detector and has no
# annotation stages, so its rail is Train, Evaluate, Model Inference.
_DETECTOR_STATE = {
    "is_training_project": True,
    "project_type": "bbox_training",
    "training_source": {"kind": "bbox", "task_default": "detect"},
}


def _panel(html, panel_id):
    """The markup of one step panel, from its id to the next panel's id.

    Args:
        html: the rendered page or the raw template.
        panel_id: the panel's id, such as "panel-6".
    Returns:
        The panel's markup. The last panel on the page runs to the end.
    """
    i = html.index(f'id="{panel_id}"')
    j = html.find('id="panel-', i + 20)
    return html[i:] if j < 0 else html[i:j]


def test_step6_panel_carries_no_prose():
    """The Train panel explains itself in tooltips, not paragraphs (workflow
    page design version 2, section 6). The explainer paragraph and the
    step-description block the panel used to open with went out on 2026-09-16,
    and what the trainer reads is named on the training-source row and in that
    row's tooltip instead."""
    tpl = open(_TPL).read()
    panel = _panel(tpl, "panel-6")
    check("<p><strong>What happens:</strong>" not in panel,
          "the Train panel must carry no What-happens paragraph")
    check('class="step-description"' not in panel,
          "the Train panel must carry no step-description block")
    check("<p>" not in panel, "the Train panel must carry no paragraph at all")
    check(not re.search(r"\bland\b", panel.lower()), "banned word in the Train panel")
    check("\u2014" not in panel and "\u2013" not in panel,
          "em or en dash in the Train panel copy")
    check("Choose which set of labelled images the trainer reads" in panel,
          "the training-source tooltip must say what the trainer reads")
    # 2026-09-21: "selected above" (a tooltip screen reference, design section
    # 7 rule 6) was replaced with a plain name for the row it reads from.
    check("it reads the ids and counts from the training source chosen in the Training source row" in panel,
          "the class-list tooltip must name the training source")


def test_step6_panel_copy_is_task_aware():
    """A box-dataset project trains a detector, so the Train panel may not
    hardcode 'segmentation model' or 'this project's Step 4 export', which such
    a project has neither of. Since 2026-09-16 the panel heading also carries
    the number the project's own rail gives it, so the same panel reads Step 5
    in a project that annotates its own images and Step 1 in one that does
    not, and the heading never contradicts the rail beside it."""
    tpl = open(_TPL).read()
    panel = _panel(tpl, "panel-6")
    check('stage_h2(1 if short_rail else 5, "Train Model")' in panel,
          "the Train panel heading must take its number from the rail")
    check("Train Segmentation Model" not in panel,
          "the Train panel must not hardcode 'Train Segmentation Model'")
    detector = _panel(_render_with_state(_DETECTOR_STATE), "panel-6")
    check('<span class="stage-num">1</span>: Train Model' in detector,
          "a detection project's Train panel must read as its first stage")
    segmenter = _panel(_render_with_state({"project_type": "standard"}), "panel-6")
    check('<span class="stage-num">5</span>: Train Model' in segmenter,
          "an ordinary project's Train panel must still read as its fifth stage")


def test_detection_rail_paints_no_duplicate_numbers():
    """A detection project's rail is numbered 1, 2, 3 and nothing else. The
    four annotation tiles are hidden by the server, not only by
    applyCompositeTrainingMode() after /api/project/state answers, or the
    first paint of every load shows 1, 2, 3, 4 above a second 1, 2, 3."""
    html = _render_with_state(_DETECTOR_STATE)
    nav = re.search(r'<ul class="step-nav" id="step-nav">(.*?)</ul>', html, re.DOTALL)
    check(nav is not None, "step-nav <ul> not found")
    visible = []
    for li in re.findall(r"<li\b(.*?)</li>", nav.group(1), re.DOTALL):
        if "display:none" in li:
            continue
        circle = re.search(r'class="step-circle"[^>]*>([^<]*)<', li)
        if circle and circle.group(1).strip().isdigit():
            visible.append(circle.group(1).strip())
    check(visible == ["1", "2", "3"],
          f"a detection rail must paint 1, 2, 3 and no more; it painted {visible}")
    ordinary = _render_with_state({"project_type": "standard"})
    nav2 = re.search(r'<ul class="step-nav" id="step-nav">(.*?)</ul>', ordinary, re.DOTALL)
    nums = [m for m in re.findall(r'class="step-circle"[^>]*>([^<]*)<', nav2.group(1))
            if m.strip().isdigit()]
    check(nums == ["1", "2", "3", "4", "5", "6", "7"],
          f"an ordinary rail must still paint 1 to 7; it painted {nums}")


def test_panels_align_on_one_label_column():
    """Every choice on the Train, Evaluate and Model Inference panels is a
    definition row, so each panel has one label left edge. A settings band
    (.subcard) carries no side padding for the same reason: with it, the rows
    inside a band started 12 px right of the rows outside one and a panel
    showed two label columns."""
    css = open(os.path.join(_ORCH, "static", "style.css")).read()
    i = css.index(".subcard {")
    block = css[i:css.index("}", i)]
    check("padding: 10px 0;" in block,
          "a settings band must carry no horizontal padding, or its rows misalign")
    tpl = open(_TPL).read()
    for panel_id in ("panel-6", "panel-7", "panel-8"):
        panel = _panel(tpl, panel_id)
        check(panel.count('class="field-head"') >= 3,
              f"{panel_id} must lay its choices out as definition rows")


def test_evaluate_actions_sit_in_named_rows():
    """No button floats against an edge with no row and no name: the
    recommended-defaults button is a definition row, and Copy PDF sits in the
    evaluation report's own heading row beside the section name."""
    tpl = open(_TPL).read()
    panel = _panel(tpl, "panel-7")
    i = panel.index('onclick="applyStep7Defaults()"')
    row = panel[panel.rindex('<div class="form-group">', 0, i):i]
    check("Recommended settings" in row,
          "the recommended-defaults button must sit in a named definition row")
    j = panel.index('id="s7-copy-pdf-btn"')
    row = panel[panel.rindex('<div class="form-group">', 0, j):j]
    check("Report PDF" in row,
          "Copy PDF must sit in a named definition row of its own")
    # The row must not repeat the heading the report card already carries.
    check(row.count("Evaluation report") == 0,
          "the Copy PDF row must not repeat the report card's own heading")


# ── D2/D3/D4/D8 (2026-09-02, final-fix wave): remaining theme + copy defects
# from the final-verification report ─────────────────────────────────────

def test_preset_display_names_have_no_em_dash():
    """D2: both preset display names carried an em dash and render in the
    Load preset select on every Train panel, for every project type. Only the
    _meta.name/description fields actually reach the UI (STEP6_PRESET_APPLIERS
    reads opt.textContent from name and opt.dataset.description from
    description), so those are the only strings this defect - or this test -
    scopes to; the file's prose comments (which never render) are untouched,
    same as the constraint's "remove them from any line you edit" wording."""
    import yaml as _yaml
    gr = os.path.dirname(os.path.dirname(_ORCH))
    presets_dir = os.path.join(gr, "config", "presets")
    checked = 0
    for fname in ("tuned_v1_regularized.yaml", "yolo26l_heavy_seg_aug.yaml"):
        path = os.path.join(presets_dir, fname)
        check(os.path.isfile(path), f"{fname} missing at {path}")
        raw = open(path).read()
        data = _yaml.safe_load(raw)
        check(data is not None and "_meta" in data and "name" in data["_meta"],
              f"{fname}: must still parse to a _meta.name")
        check("—" not in data["_meta"]["name"], f"{fname}: em dash still in _meta.name")
        check("—" not in (data["_meta"].get("description") or ""),
              f"{fname}: em dash in _meta.description")
        check("params" in data and isinstance(data["params"], dict) and data["params"],
              f"{fname}: the params block must be untouched")
        checked += 1
    check(checked == 2, "both preset files were checked")


def test_no_off_palette_teal_in_index_html():
    """D3: the Expert Review I/O tile's #00c8dc (four occurrences) is the last
    live teal in the orchestrator template family. Retuned to --text-dim, the
    same muted token the sidebar's own default (unstyled) step-circle already
    uses, so a permanently-disabled tile reads as muted rather than carrying
    a stray accent hue."""
    tpl = open(_TPL).read()
    check("#00c8dc" not in tpl, "the old teal hex must be gone from index.html")
    check("#06222a" not in tpl, "the teal-black ink paired with the old teal must be gone too")
    # The tile that carried the teal was removed with the rest of the Expert
    # Review markup on 2026-09-16, so the hex cannot come back through it.
    check("expertids" not in tpl, "Expert Review markup is back in the template")


def test_boot_overlay_uses_canonical_palette():
    """D4: the sub-window boot overlay orchestrator.js injects into every
    launched sub-window (Combined Annotator, Refine Review, Edit Masks) still
    carried the banned purple #c14bd6 and the whole pre-theme grey palette
    after annotate.html was rethemed. Both must now agree: same magenta bar,
    surface, ground and text as templates/annotate.html."""
    js = open(_JS).read()
    for banned in ("#c14bd6", "#9aa0ae", "#252833", "#6b7280", "#e05555",
                   "#3a3d4a", "#e8e8ea", "#12141a"):
        check(banned not in js, f"pre-theme boot-overlay color {banned} must be gone from orchestrator.js")
    # _js_function_body() stops at the next FUNCTION declaration, but
    # bootSkeletonHtml is followed by const declarations (BOOT_BODY_CSS,
    # bootPaintable, ANNOTATOR_UI) before the next function - use the
    # tighter col-0-brace match (same pattern _run_node_snippet uses) so
    # "skeleton" is really just this one function's own body.
    m = re.search(r"\nfunction bootSkeletonHtml\b[\s\S]*?\n}\n", "\n" + js)
    check(m is not None, "bootSkeletonHtml() not found")
    skeleton = m.group(0)
    check("#E20074" in skeleton, "bootSkeletonHtml must paint the canonical magenta progress fill")
    check("#11111a" in skeleton, "bootSkeletonHtml must use the canonical surface color")
    check("#0a0a0f" not in skeleton,  # ground belongs to BOOT_BODY_CSS, not the card itself
          "the page ground token belongs on BOOT_BODY_CSS, not inside the skeleton card")
    i = js.index("const BOOT_BODY_CSS")
    body_css = js[i:i + 400]
    check("#0a0a0f" in body_css, "BOOT_BODY_CSS must use the canonical page ground #0a0a0f")
    check("#F5EDF0" in body_css, "BOOT_BODY_CSS must use the canonical text color #F5EDF0")
    paint = _js_function_body(js, "paintStep4testBoot")
    check("#E20074" in paint and "#FF4D6D" in paint,
          "paintStep4testBoot must paint the canonical magenta/error colors, not the pre-theme pair")


def test_style_css_declares_font_tokens():
    """D8: style.css is the theme contract's named reference implementation
    for the three font roles, but declared no --font-* custom properties -
    the families were hardcoded at several call sites instead. Zero visual
    change intended: the token values match the hardcodes they replace."""
    css_path = os.path.join(_ORCH, "static", "style.css")
    css = open(css_path).read()
    root = css[css.index(":root"):css.index("\n}\n", css.index(":root")) + 2]
    check("--font-body:" in root and "'Inter'" in root, "--font-body token missing from :root")
    check("--font-display:" in root and "'Syne'" in root, "--font-display token missing from :root")
    check("--font-mono:" in root and "'Space Mono'" in root, "--font-mono token missing from :root")
    check("var(--font-display)" in css, "at least one hardcoded Syne heading must now reference the token")
    check("var(--font-mono)" in css, "at least one hardcoded Space Mono declaration must now reference the token")
    topbar = css[css.index(".topbar h1"):css.index("}", css.index(".topbar h1"))]
    check("font-family: var(--font-display)" in topbar, ".topbar h1 must read the display token")
    cardh2 = css[css.index(".card h2"):css.index("}", css.index(".card h2"))]
    check("font-family: var(--font-display)" in cardh2, ".card h2 must read the display token")
    logo = css[css.index(".setup-container .logo h1"):css.index("}", css.index(".setup-container .logo h1"))]
    check("'Silkscreen'" in logo and "var(--font-mono)" in logo,
          "the wordmark keeps its Silkscreen literal but its monospace fallback reads the token")


# ── main ───────────────────────────────────────────────────────────

# 2026-09-17 (LO), decision record 0006: the training-source picker knows three kinds.
_js_0006 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "orchestrator.js"), encoding="utf-8").read()
check("(c.kind === 'standalone' ? 'standalone' : 'composite')" in _js_0006, "the picker keeps a session as its own kind")
check("Annotate Images session(s)" in _js_0006, "the source note counts sessions in their own words")
check("(Annotate Images session)" in _js_0006, "a session option says what it is")
check("composite export(s) under" not in _js_0006, "the note says training set, never composite (spec vocabulary)")

if __name__ == "__main__":
    print("orchestrator UI revamp tests (Track A + Phase 2 shell)\n")
    print("SMOKE:")
    run(test_smoke_render_200)
    run(test_smoke_node_check_js)
    run(test_smoke_node_check_inline_blocks)
    print("UNIT (template):")
    run(test_nav_order_phase2_reorder)
    run(test_nav_tool_glyphs)
    run(test_expertids_entry_point_absent)
    run(test_all_panels_have_nav_footers)
    run(test_s4review_shell_absent)
    run(test_step4test_tile_present)
    run(test_legacy_step4_step5_panels_are_gone)
    run(test_s4test_clip_dir_and_symlink_controls_present)
    run(test_step4test_advanced_and_info_circles)
    run(test_tool_tile_status_spans_present)
    run(test_open_ui_overlay_present)
    run(test_cache_bust_bumped)
    run(test_reroute_button_removed)
    print("UNIT (JS logic under node):")
    run(test_js_getNavOrder)
    run(test_js_footers_neighbours)
    run(test_js_step_display_num)
    run(test_s4test_collects_clip_dir_and_symlink)
    run(test_step4test_popup_opens_on_user_gesture)
    print("ADVERSARIAL (skeptic / structural):")
    run(test_adv_step4test_sam3_inputs_are_inside_advanced)
    run(test_adv_ddp_essay_is_collapsed)
    run(test_adv_advanced_details_closed_by_default)
    run(test_adv_expertids_onshow_hook_wired)
    run(test_adv_step4test_next_targets_train)
    run(test_adv_first_panel_prev_disabled)
    print("WAVE 3 (Refine loop tile/panel + Step 6/7 additions):")
    run(test_step4loop_tile_present)
    run(test_step4loop_panel_inputs_have_tooltips)
    run(test_step4loop_reset_description_present)
    run(test_step4loop_js_functions_present)
    run(test_step4loop_open_ui_label_present)
    print("TASK 10 (Edit Masks tile/panel):")
    run(test_editmasks_tile_present)
    run(test_editmasks_panel_buttons_have_tooltips)
    run(test_editmasks_reset_description_present)
    run(test_editmasks_js_functions_present)
    run(test_editmasks_open_ui_label_present)
    run(test_editmasks_popup_opens_on_user_gesture)
    run(test_step6_pin_split_checkbox_present)
    run(test_step6_model_optgroup_and_presets_present)
    run(test_step6_preset_apply_js_present)
    run(test_step6_preset_params_reconciles_after_apply)
    run(test_step6_preset_reconciliation_behavioral)
    run(test_step7_rounds_table_present)
    run(test_step7_rounds_js_present)
    run(test_step8_champion_badge_present)
    print("AUTO PROJECT NAME (2026-08-26):")
    run(test_auto_project_name_js_wiring)
    print("TASK 4 (2026-08-26): one loader for annotator launches:")
    run(test_annotator_launches_use_single_loader)
    run(test_boot_painter_supports_error_and_retry)
    run(test_cache_bust_is_single_bump_point)
    run(test_annotator_popup_paintable_check)
    run(test_boot_skeleton_extracted)
    run(test_closing_the_popup_stops_the_launch)
    run(test_prelaunch_failure_hides_service_row)
    run(test_prelaunch_failure_keeps_a_live_service_row)
    print("MATRIX DRILL-DOWN WIRING (2026-08-26):")
    run(test_step6_source_select_present)
    run(test_step6_source_js_wiring)
    run(test_matrix_registration_passes_open_folder_fn)
    run(test_restart_script_markers)
    run(test_owner_o_copy_has_no_banned_strings)
    print("H2 + L9 (2026-08-26): class picker follows the training source; Step 6 copy:")
    run(test_step6_class_picker_follows_source)
    run(test_step6_panel_carries_no_prose)
    run(test_step6_panel_copy_is_task_aware)
    print("PANEL REWRITE (2026-09-16): rail, rows, actions:")
    run(test_detection_rail_paints_no_duplicate_numbers)
    run(test_panels_align_on_one_label_column)
    run(test_evaluate_actions_sit_in_named_rows)
    print("FINAL-FIX WAVE (2026-09-02): D2/D3/D4/D8:")
    run(test_preset_display_names_have_no_em_dash)
    run(test_no_off_palette_teal_in_index_html)
    run(test_boot_overlay_uses_canonical_palette)
    run(test_style_css_declares_font_tokens)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
