"""
Self-contained smoke + unit tests for the pipeline-orchestrator UI revamp
(Track A + Phase-2 shell).

No pytest dependency: run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_orchestrator_ui.py

Covers:
  SMOKE  - app imports, GET / renders 200, both JS sources pass `node --check`,
           inline template <script> blocks (Jinja-stripped) pass `node --check`.
  UNIT   - nav order is 1,2,3,step4test,6,7,8 (4.test promoted to Step 4; legacy 4/5
           archived; expertids tile disabled 2026-07-09 via data-step-disabled),
           every step panel + non-chain panel has a nav-footer placeholder,
           Step 4 / Step 5 each have a collapsed Advanced <details>,
           the moved inputs (port / clip-dir / devices / symlink) still exist,
           Step 5 visible settings carry info-circle tooltips,
           the JS getNavOrder()/renderStepNavFooters() pick the right neighbours,
           the JS updateBatchReadout() batch math is correct.
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
    check("REEF POINT SEG" in _RENDERED["html"], "page title missing")


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
    # not fatal if there are zero inline code blocks, but we expect at least one
    check(checked >= 1, "expected >=1 inline <script> code block")


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


def test_nav_star_and_diamond_glyphs():
    html = _html()
    exp = re.search(r'data-step-disabled="expertids".*?step-circle[^>]*>([^<]*)<', html, re.DOTALL)
    check(exp and exp.group(1).strip() == "&#9733;", "expertids circle is not a star (&#9733;)")
    check('data-step="s4review"' not in html, "s4review tile should be removed")


def test_expertids_tile_disabled():
    # Disabled 2026-07-09: the tile stays visible (restart self-test greps the
    # 'Expert Review I/O' text) but is inert - data-step renamed to
    # data-step-disabled so getNavOrder()/switchStep() skip it. The panel and
    # the /expertids blueprint stay live for testing.
    html = _html()
    check('data-step-disabled="expertids"' in html, "disabled expertids tile missing")
    check('data-step="expertids"' not in html, "expertids must not carry an active data-step")
    tile = re.search(r'<li[^>]*data-step-disabled="expertids"[^>]*>', html)
    check(tile is not None, "expertids <li> tag not found")
    check('class="step-disabled"' in tile.group(0), "expertids tile missing step-disabled class")
    check("onclick" not in tile.group(0), "expertids tile must not have an onclick")
    check("Temporarily disabled" in tile.group(0), "expertids tile missing disabled tooltip")
    check("Expert Review I/O" in html, "Expert Review I/O label must stay (restart self-test)")
    check("(disabled)" in html, "expertids subtitle missing the (disabled) hint")
    check('id="panel-expertids"' in html, "panel-expertids must remain in the page")
    check(".step-nav li.step-disabled" in html, "step-disabled CSS rule missing")


def test_all_panels_have_nav_footers():
    html = _html()
    for ds in ("1", "2", "3", "step4test", "4", "5", "6", "7", "8", "expertids", "step4loop", "editmasks"):
        check(f'id="nav-footer-{ds}"' in html, f"missing nav-footer for {ds}")


def test_s4review_shell_absent():
    html = _html()
    check('id="panel-s4review"' not in html, "panel-s4review should be removed")
    check('data-step="s4review"' not in html, "s4review nav tile should be removed")
    check("Expert Review I/O" in html, "expert tile not relabeled to Expert Review I/O")


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


def test_step4_advanced_holds_port_and_clipdir():
    html = _html()
    # Both Step4 advanced inputs still present (so collectConfig won't break).
    check('id="s4-port"' in html, "s4-port input lost")
    check('id="s4-clip-dir"' in html, "s4-clip-dir input lost")
    # They must live inside a <details class="advanced"> ... and that block must
    # appear before the Step-5 panel.
    check(html.count('class="advanced"') >= 2, "expected >=2 Advanced details (step4+step5)")


def test_step5_advanced_and_info_circles():
    html = _html()
    for i in ("s5-port", "s5-tracker-device", "s5-exemplar-device", "s5-symlink"):
        check(f'id="{i}"' in html, f"{i} input lost from Step 5")
    # Visible Step-5 settings carry info-circle tooltips.
    check(html.count('class="info-circle"') >= 6, "expected info-circles on Step-5 visible settings")


def test_batch_readouts_present():
    html = _html()
    for i in ("s4-batch-readout", "s4-batch-done", "s4-batch-left",
              "s5-batch-readout", "s5-batch-done", "s5-batch-left"):
        check(f'id="{i}"' in html, f"missing batch readout element {i}")


def test_open_ui_overlay_present():
    html = _html()
    check('id="open-ui-overlay"' in html, "open-ui-overlay missing")
    check('id="open-ui-modal"' in html, "open-ui-modal missing")
    check("hideOpenUiOverlay()" in html, "overlay close handler not wired")


def test_cache_bust_bumped():
    html = _html()
    m = re.search(r"orchestrator\.js\?v=([^\"']+)", html)
    check(m is not None, "orchestrator.js cache-bust query missing")
    check(m.group(1) == "20260710-edit-masks", "cache-bust query not bumped to 20260710-edit-masks")


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
  "step4loop":"Refine (4.loop)",
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
"""
    # Extract just the functions we need from the real source so we test the
    # shipped implementation rather than a copy.
    wanted = []
    for name in ("getNavOrder", "navStepArg", "navTitleFor",
                 "renderStepNavFooters", "escHtmlNav", "escAttr",
                 "updateBatchReadout"):
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


def test_js_batch_math():
    # batch size 10: total 25 frames -> ceil(25/10)=3 batches; processed 12 ->
    # floor(12/10)=1 done -> 2 left.
    js = (
        "global.__BATCH_SIZE='10';"
        "updateBatchReadout('s5', 12, 25);"
        "console.log(global.__batch['s5-batch-done'] + '|' + global.__batch['s5-batch-left']);"
    )
    out = _run_node_snippet(js)
    check(out == "1|2", f"batch math (size10) wrong: {out}")

    # 'all' / non-numeric -> per-frame (size 1): processed 7 of 9 -> 7 done, 2 left.
    js2 = (
        "global.__BATCH_SIZE='all';"
        "updateBatchReadout('s4', 7, 9);"
        "console.log(global.__batch['s4-batch-done'] + '|' + global.__batch['s4-batch-left']);"
    )
    out2 = _run_node_snippet(js2)
    check(out2 == "7|2", f"batch math (all) wrong: {out2}")

    # zero total -> 0 done / 0 left (no NaN).
    js3 = (
        "global.__BATCH_SIZE='10';"
        "updateBatchReadout('s5', 0, 0);"
        "console.log(global.__batch['s5-batch-done'] + '|' + global.__batch['s5-batch-left']);"
    )
    out3 = _run_node_snippet(js3)
    check(out3 == "0|0", f"batch math (empty) wrong: {out3}")


# ── ADVERSARIAL / structural skeptic cases ──────────────────────────
def _details_body(html, anchor_id):
    """Return the inner text of the <details class="advanced"> block that
    contains the given element id, or None if it's not nested inside one."""
    # Find every advanced details block and return the one wrapping anchor_id.
    for m in re.finditer(r'<details class="advanced">(.*?)</details>', html, re.DOTALL):
        if f'id="{anchor_id}"' in m.group(1):
            return m.group(1)
    return None


def test_adv_step4_inputs_are_inside_advanced():
    # Skeptic: the inputs could exist on the page but NOT be collapsed. Assert
    # they are physically nested in the Step-4 Advanced <details>.
    html = _html()
    body = _details_body(html, "s4-port")
    check(body is not None, "s4-port is not inside a <details class=advanced>")
    check('id="s4-clip-dir"' in body, "s4-clip-dir not in the same Step-4 Advanced block")


def test_adv_step5_inputs_are_inside_advanced():
    html = _html()
    body = _details_body(html, "s5-tracker-device")
    check(body is not None, "s5-tracker-device not inside a <details class=advanced>")
    for needed in ("s5-port", "s5-exemplar-device", "s5-symlink"):
        check(f'id="{needed}"' in body, f"{needed} not in the Step-5 Advanced block")


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
    check("Refine (4.loop)" in html, "Refine (4.loop) tile label missing")
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
    check('id="s4loop-goto-finetune"' in html, "s4loop-goto-finetune button missing")
    check('id="s4loop-goto-retrain"' in html, "s4loop-goto-retrain button missing")


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
    # Tile must sit in the Review group: after the Review section divider and
    # near the Expert Review I/O tile.
    review_idx = html.index('Review</div>')
    editmasks_idx = html.index('data-step="editmasks"')
    expertids_idx = html.index('data-step-disabled="expertids"')
    check(review_idx < editmasks_idx, "editmasks tile must sit after the Review divider")
    check(editmasks_idx < expertids_idx,
          "editmasks tile must sit before Expert Review I/O (near it in the Review group)")


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


def test_step6_model_optgroup_and_presets_present():
    html = _html()
    check('id="s6-model-runs-optgroup"' in html, "s6-model optgroup for previous runs missing")
    check("Continue from a previous run" in html, "optgroup label missing")
    check('id="s6-preset-finetune"' in html, "s6-preset-finetune button missing")
    check('id="s6-preset-retrain"' in html, "s6-preset-retrain button missing")
    check(re.search(r'id="s6-preset-finetune"[^>]*title="[^"]+"', html) is not None,
          "s6-preset-finetune missing a title= tooltip")
    check(re.search(r'id="s6-preset-retrain"[^>]*title="[^"]+"', html) is not None,
          "s6-preset-retrain missing a title= tooltip")


def test_step6_preset_apply_js_present():
    js = open(_JS).read()
    check("function applyStep6BuiltinPreset(" in js, "applyStep6BuiltinPreset() missing")
    check("function refreshStep6ModelRunOptions(" in js, "refreshStep6ModelRunOptions() missing")
    check("model_path:" in js, "STEP6_PRESET_APPLIERS missing a model_path applier")
    check("freeze:" in js or "'freeze'" in js, "STEP6_PRESET_APPLIERS missing a freeze applier")
    check("cfg.model_path" in js, "collectConfig(6) does not collect model_path")
    check("cfg.pin_split" in js, "collectConfig(6) does not collect pin_split")


# ── Wave 3: Step 7 Rounds table + Promote + champion badge (Task 9) ─────────
def test_step7_rounds_table_present():
    html = _html()
    check('id="s7-rounds-table"' in html, "s7-rounds-table missing")
    check('id="s7-rounds-tbody"' in html, "s7-rounds-tbody missing")
    for col in ("Round", "Run", "Base", "mAP50-95(M)", "Recall(M)", "Gates", "Promote"):
        check(col in html, f"Rounds table missing column header: {col}")


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


# ── main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("orchestrator UI revamp tests (Track A + Phase 2 shell)\n")
    print("SMOKE:")
    run(test_smoke_render_200)
    run(test_smoke_node_check_js)
    run(test_smoke_node_check_inline_blocks)
    print("UNIT (template):")
    run(test_nav_order_phase2_reorder)
    run(test_nav_star_and_diamond_glyphs)
    run(test_expertids_tile_disabled)
    run(test_all_panels_have_nav_footers)
    run(test_s4review_shell_absent)
    run(test_step4test_tile_present)
    run(test_step4_advanced_holds_port_and_clipdir)
    run(test_s4test_clip_dir_and_symlink_controls_present)
    run(test_step5_advanced_and_info_circles)
    run(test_batch_readouts_present)
    run(test_open_ui_overlay_present)
    run(test_cache_bust_bumped)
    run(test_reroute_button_removed)
    print("UNIT (JS logic under node):")
    run(test_js_getNavOrder)
    run(test_js_footers_neighbours)
    run(test_js_batch_math)
    run(test_s4test_collects_clip_dir_and_symlink)
    run(test_step4test_popup_opens_on_user_gesture)
    print("ADVERSARIAL (skeptic / structural):")
    run(test_adv_step4_inputs_are_inside_advanced)
    run(test_adv_step5_inputs_are_inside_advanced)
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
    run(test_step7_rounds_table_present)
    run(test_step7_rounds_js_present)
    run(test_step8_champion_badge_present)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
