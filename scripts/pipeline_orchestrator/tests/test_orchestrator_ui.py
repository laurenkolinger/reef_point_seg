"""
Self-contained smoke + unit tests for the pipeline-orchestrator UI revamp
(Track A + Phase-2 shell).

No pytest dependency: run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_orchestrator_ui.py

Covers:
  SMOKE  - app imports, GET / renders 200, both JS sources pass `node --check`,
           inline template <script> blocks (Jinja-stripped) pass `node --check`.
  UNIT   - nav order is 1,2,3,4,5,6,7,8,expertids (Review group below inference),
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
    check(
        order == ["1", "2", "3", "4", "5", "6", "7", "8", "expertids"],
        f"unexpected nav order: {order}",
    )


def test_nav_star_and_diamond_glyphs():
    html = _html()
    exp = re.search(r'data-step="expertids".*?step-circle[^>]*>([^<]*)<', html, re.DOTALL)
    check(exp and exp.group(1).strip() == "&#9733;", "expertids circle is not a star (&#9733;)")
    check('data-step="s4review"' not in html, "s4review tile should be removed")


def test_all_panels_have_nav_footers():
    html = _html()
    for ds in ("1", "2", "3", "4", "5", "6", "7", "8", "expertids"):
        check(f'id="nav-footer-{ds}"' in html, f"missing nav-footer for {ds}")


def test_s4review_shell_absent():
    html = _html()
    check('id="panel-s4review"' not in html, "panel-s4review should be removed")
    check('data-step="s4review"' not in html, "s4review nav tile should be removed")
    check("Expert Review I/O" in html, "expert tile not relabeled to Expert Review I/O")


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
    check(m.group(1) == "20260624-review-io", "cache-bust query not bumped to 20260624-review-io")


# ── UNIT: JS logic executed under node ──────────────────────────────
def _run_node_snippet(js):
    node = _node()
    check(node is not None, "node not available")
    src = open(_JS).read()
    # Provide a minimal DOM shim so the pure-ish helpers run headless. We stub
    # only what the functions under test touch.
    harness = r"""
const __navOrder = ["1","2","3","4","5","6","7","8","expertids"];
const __titles = {
  "1":"Make All Points","2":"Recode Labels","3":"Choose Images","4":"Place Points",
  "5":"Segment (SAM3)","6":"Train Model","7":"Evaluate Model","8":"Model Inference",
  "expertids":"Expert Review I/O"
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
    check(out == '["1","2","3","4","5","6","7","8","expertids"]',
          f"getNavOrder() wrong: {out}")


def test_js_footers_neighbours():
    # New topology: 1..8 then expertids last. Step 4 Next goes straight to step 5
    # (the s4review stub is gone); step 8 Next goes to expertids; expertids is
    # last, so its Next is disabled and its Prev targets step 8.
    js = (
        "renderStepNavFooters();"
        "const f = global.__footers;"
        "const out = {"
        "  four: f['4'],"
        "  eight: f['8'],"
        "  expertids: f['expertids']"
        "};"
        "console.log(JSON.stringify(out));"
    )
    out = _run_node_snippet(js)
    import json
    f = json.loads(out)
    check("switchStep(5)" in f["four"], "step4 Next should go to step 5")
    check('switchStep("expertids")' in f["eight"], "step8 Next should go to expertids")
    check("switchStep(8)" in f["expertids"], "expertids Prev should go to step 8")
    check("disabled" in f["expertids"], "expertids (last) should have a disabled Next")


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


def test_adv_step4_next_targets_step5():
    # The s4review stub is removed, so step 4 Next now goes straight to step 5.
    js = (
        "renderStepNavFooters();"
        "console.log(global.__footers['4']);"
    )
    out = _run_node_snippet(js)
    check("switchStep(5)" in out, "step4 Next must go to step 5")
    check('switchStep("s4review")' not in out, "step4 Next must not reference the removed s4review")


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
    run(test_all_panels_have_nav_footers)
    run(test_s4review_shell_absent)
    run(test_step4_advanced_holds_port_and_clipdir)
    run(test_step5_advanced_and_info_circles)
    run(test_batch_readouts_present)
    run(test_open_ui_overlay_present)
    run(test_cache_bust_bumped)
    print("UNIT (JS logic under node):")
    run(test_js_getNavOrder)
    run(test_js_footers_neighbours)
    run(test_js_batch_math)
    print("ADVERSARIAL (skeptic / structural):")
    run(test_adv_step4_inputs_are_inside_advanced)
    run(test_adv_step5_inputs_are_inside_advanced)
    run(test_adv_advanced_details_closed_by_default)
    run(test_adv_expertids_onshow_hook_wired)
    run(test_adv_step4_next_targets_step5)
    run(test_adv_first_panel_prev_disabled)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
