"""Tests for the editMasks app booting in 'edit' mode against a project export.

Where this fits: TCRMPclip_editMasks is a standalone fork of the combined
annotator dedicated to editing an EXISTING project export (relabel / delete /
redraw / add masks). It boots via the SAME POST /api/resume path a resume
session uses, but with config.SESSION_MODE == 'edit' (default) so provenance
rows are tagged 'edit'. This test proves the edit-mode boot loads the export's
segmentations.json editable and serves the mask data back to the client.

House harness style: standalone script, no pytest. Env vars are set BEFORE
`import app as A` (mirrors test_loop_resume.py). SAM3 loads only under
`__main__` in app.py, so the test client never touches the GPU; the routes
exercised here (resume, image GET) are SAM-free.

Run: env/bin/python scripts/TCRMPclip_editMasks/tests/test_edit_boot.py
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
from datetime import datetime

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

# -- Build the temp seeded export BEFORE import (env vars must be set first) --
_ROOT = tempfile.mkdtemp(prefix='edit_boot_test_')
_EXPORT_DIR = os.path.join(_ROOT, 'export')
_SEG_DIR = os.path.join(_EXPORT_DIR, 'segmentations')
os.makedirs(_SEG_DIR, exist_ok=True)

_RAW_DIR = os.path.join(_ROOT, 'raw')
os.makedirs(_RAW_DIR, exist_ok=True)

_FRAME_NAME = 'TCRMPX_clip_AAA_T118.jpeg'


def _make_image(path, w=200, h=100):
    im = Image.new('RGB', (w, h), (10, 20, 30))
    im.save(path, 'JPEG', quality=90)


_make_image(os.path.join(_RAW_DIR, _FRAME_NAME))

# Set env BEFORE importing app/mask_ops/config so cfg.* picks these up.
os.environ['TCRMP_EXPORT_DIR'] = _EXPORT_DIR
os.environ['TCRMP_SESSION_MODE'] = 'edit'
os.environ['TCRMP_PROVENANCE_SOURCE'] = 'edit'
os.environ['TCRMP_MANUAL_ANNOTATE'] = '1'
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'AA')

import config as cfg  # noqa: E402
import mask_ops  # noqa: E402

# Build one committed PA (present/accepted) mask on the frame using the same
# helper the annotator uses, so the fixture matches real export shape exactly.
_binary_mask = np.zeros((100, 200), dtype=bool)
_binary_mask[20:80, 20:180] = True  # well above MIN_MASK_AREA_PX default guard
_point_info = {'label': 'M', 'species': 'AA', 'name': 'Acropora agaricites',
               'category': 'Coral', 'x': 100, 'y': 50}
_seeded_mask = mask_ops.build_mask_dict(
    mask_id=0, binary_mask=_binary_mask, score=0.9,
    point_info=_point_info, source_type='auto',
)
assert _seeded_mask is not None, 'fixture mask must build (check MIN_MASK_AREA_PX)'
_seeded_mask['status'] = 'accepted'

_seg_dict = {
    _FRAME_NAME: {
        'image_path': _FRAME_NAME,
        'image_path_abs': os.path.join(_RAW_DIR, _FRAME_NAME),
        'image_width': 200,
        'image_height': 100,
        'masks': [_seeded_mask],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': False,
    },
}
with open(os.path.join(_SEG_DIR, 'segmentations.json'), 'w') as f:
    json.dump(_seg_dict, f)

import app as A  # noqa: E402

_fail = 0
def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print("FAIL:", msg)
    else:
        print("PASS:", msg)


# -- config reflects the edit-mode defaults / env set above ------------------
check(cfg.SESSION_MODE == 'edit', f"cfg.SESSION_MODE == 'edit' (got {cfg.SESSION_MODE!r})")
check(cfg.PROVENANCE_SOURCE == 'edit',
      f"cfg.PROVENANCE_SOURCE == 'edit' (got {cfg.PROVENANCE_SOURCE!r})")
check(cfg.PORT == 5083, f"cfg.PORT default == 5083 (got {cfg.PORT})")

A.app.config['TESTING'] = True

with A.app.test_client() as c:
    # -- The index page routes an edit session through the resume boot --------
    r_idx = c.get('/')
    check(r_idx.status_code == 200, f"GET / -> 200 (got {r_idx.status_code})")
    idx_html = r_idx.data.decode('utf-8', 'replace')
    check("const SESSION_MODE = \"resume\"" in idx_html,
          "edit session hands the template SESSION_MODE 'resume' so it boots via /api/resume")
    check("const ORCHESTRATED = true" in idx_html,
          "edit session is orchestrated (auto-boots headlessly from TCRMP_EXPORT_DIR)")

    # -- POST /api/resume reads export_dir from the POST BODY -> 200 ----------
    r = c.post('/api/resume', json={'export_dir': _EXPORT_DIR})
    check(r.status_code == 200, f"POST /api/resume -> 200 (got {r.status_code}, body={r.data!r})")
    resume_body = json.loads(r.data)
    check(resume_body.get('total_processed') == 1,
          f"resume loaded 1 processed frame (got {resume_body.get('total_processed')})")

    # -- GET /api/image/<f> returns the export's masks editable --------------
    r_img = c.get(f'/api/image/{_FRAME_NAME}')
    check(r_img.status_code == 200, f"GET /api/image/<frame> -> 200 (got {r_img.status_code})")
    img_data = json.loads(r_img.data)
    masks = img_data.get('masks', [])
    check(len(masks) == 1, f"frame has exactly 1 editable mask (got {len(masks)})")
    if masks:
        check(masks[0].get('species') == 'AA',
              f"masks[0].species == 'AA' (got {masks[0].get('species')})")
        check(masks[0].get('status') == 'accepted',
              f"masks[0].status == 'accepted' (got {masks[0].get('status')})")
    check(img_data.get('image_width') == 200,
          f"served image_width == 200 (got {img_data.get('image_width')})")


# ── STRUCTURAL: right-panel cleanup + font purge (parity with combinedAnnotate) ──
# Same assertions as combinedAnnotate's tests/test_segment_app.py
# test_panel_cleanup_and_fonts, against this fork's template.
def test_panel_cleanup_and_fonts():
    tpl = open(os.path.join(SRC, 'templates', 'index.html')).read()
    for bad in ('Silkscreen', 'Scrap Frame', 'btn-done-return', 'class="refine-hint"',
                'Fit View (F)', 'id="scan-panel"'):
        check(bad not in tpl, f"removed: {bad}")
    for good in ('id="btn-undo"', 'id="exemplar-panel"', 'id="btn-scan-this"', 'id="btn-scan-all"',
                 'id="scan-threshold"', 'Similarity', 'Press <kbd>K</kbd> for keyboard shortcuts'):
        check(good in tpl, f"present: {good}")
    check('function scrapFrame' not in tpl, 'scrapFrame JS removed')
    # editMasks-only: combinedAnnotate's standalone "Annotate Images" session
    # is NOT carried here, not even Jinja-gated. An edit session can only ever
    # hand the template session_mode 'resume', so a standalone block could
    # never render; carrying one would leave markup referencing functions and
    # routes this app does not have (handleImportFiles, /api/standalone/*).
    # The startup card is therefore unwrapped and the Import control is gone.
    for bad in ("{% if session_mode == 'standalone' %}",
                "{% if session_mode != 'standalone' %}",
                "SESSION_MODE === 'standalone'",
                'id="standalone-card"', 'id="btn-import-images"',
                'id="import-file-input"', 'handleImportFiles',
                'saInit', 'saStart', 'sa-drop', 'id="sa-label-list"'):
        check(bad not in tpl, f"standalone relic absent: {bad}")


# ── STRUCTURAL: Box mode, arrow keys, mask-size slider (parity) ─────
def test_box_mode_arrows_mask_size():
    tpl = open(os.path.join(SRC, 'templates', 'index.html')).read()
    for good in ('id="btn-mode-box"', 'async function doBoxAdd(', "case 'ArrowLeft':", "case 'ArrowRight':",
                 'id="mask-size"', 'function setMaskSize(', "/merge_many`", "/add_box`", 'BOX MODE'):
        check(good in tpl, f"present: {good}")
    kd = tpl[tpl.index("case 'b': case 'B':"):]
    kd = kd[:kd.index('break;')]
    check("prevImage" not in kd and "'box'" in kd, "B toggles box mode, not previous frame")
    kh = tpl[tpl.index('id="kbd-help"'):tpl.index('id="export-confirm-overlay"')]
    check('<kbd>B</kbd>' in kh and 'box' in kh.lower() and '&larr;' in kh and '&rarr;' in kh,
          "K sheet lists Box and both arrows")
    check('previous frame</span>' in kh and '<kbd>B</kbd></span><span class="kh-desc">previous' not in kh,
          "K sheet no longer says B = previous")
    check("Mask size" in kh, "K sheet mentions the mask-size slider")
    check('>Loading the annotator<' in tpl, "veil copy continues the popup loader")
    check('Orchestrator-driven view' not in tpl, "old veil hint removed")


# ── STRUCTURAL: sticky-loop exclusivity + reset clip (parity) ───────
def test_sticky_flags_and_reset_clip():
    tpl = open(os.path.join(SRC, 'templates', 'index.html')).read()

    # 1. Leaving a mode through the toolbar (setInteractionMode('select')) must
    #    end BOTH sticky loops, or the next accept/frame change silently
    #    re-enters Box. Entering box arms the box loop and clears sticky Add.
    sim = tpl[tpl.index('function setInteractionMode('):tpl.index('function updateModeButtons(')]
    reset = sim[sim.index("if (mode === 'select')"):]
    check('stickyBoxMode = false' in reset and 'stickyAddMode = false' in reset,
          'select hard-reset ends both sticky loops')
    check("if (mode === 'box')" in sim and 'stickyBoxMode = true' in sim
          and 'stickyAddMode = false' in sim,
          'entering box arms the box loop and clears sticky Add')

    # 2. B must stop the loop from the refine sub-step too (mode is 'refine'
    #    right after a box add, which is exactly when the status says B to stop).
    kd = tpl[tpl.index("case 'b': case 'B':"):]
    kd = kd[:kd.index('break;')]
    check("interactionMode === 'box' || stickyBoxMode" in kd,
          'B stops the box loop from the refine sub-step')

    # 3. The mask-size label follows the POST (not the drag), a refused switch
    #    restores the slider, and the success copy waits on the re-run result.
    sms = tpl[tpl.index('async function setMaskSize('):tpl.index('async function rerunPrompt(')]
    check('valEl.textContent = size' in sms
          and sms.index("'/api/mask_size'") < sms.index('valEl.textContent = size'),
          'mask-size label is written only after the POST succeeds')
    check('restore()' in sms, 'a refused mask size restores the slider position')
    check('const ok = await doBoxRedraw(' in sms and 'const ok = await rerunPrompt(' in sms
          and 'if (ok) setStatus(' in sms,
          'setMaskSize claims a re-run only when the re-run succeeded')
    dbr = tpl[tpl.index('async function doBoxRedraw('):tpl.index('// MASK SIZE')]
    check('return true;' in dbr and 'return false;' in dbr, 'doBoxRedraw reports success')
    rp = tpl[tpl.index('async function rerunPrompt('):tpl.index('// MERGE')]
    check('return true;' in rp and 'return false;' in rp, 'rerunPrompt reports success')

    # 4. A rejected mask must leave the checked set, or a later merge blocks on
    #    a row the list is hiding.
    # The whole function, not a fixed window: guards added at its top (the
    # frame-load gate, 2026-10-16) push the line further down.
    start = tpl.index('function rejectMask')
    rm = tpl[start:tpl.index('\n}\n', start)]
    check('checkedMaskIds.delete(m.id)' in rm, 'rejectMask unchecks the rejected mask')

    # 5. The resetMask CLIENT function is gone (a relic: no caller since Ctrl+Z
    #    reset was removed). The /reset_mask ROUTE stays, and still clips to live
    #    neighbours and answers blocked_by, so the contract is checked in app.py.
    check('resetMask' not in tpl, 'the dead resetMask client function is gone')
    src = open(os.path.join(SRC, 'app.py')).read()
    rs = src[src.index('def reset_mask('):src.index('def reset_mask(') + 3000]
    check('rejected_ids' in rs, '/reset_mask still honours the client rejected set')
    check('blocked_by' in rs, '/reset_mask still answers blocked_by')

    # editMasks-only: the lock guard sits in front of every one of these client
    # paths, and doBoxRedraw's guard must report the same false the rest of the
    # function does (the mask-size re-run reads its return value).
    check('return false;' in tpl[tpl.index('async function doBoxRedraw('):
                                 tpl.index('async function doBoxRedraw(') + 400],
          'the expert-lock guard in doBoxRedraw returns false, not undefined')


# ── SMOKE: the template's inline JS parses (parity with combinedAnnotate) ──
# ── STRUCTURAL: the review header wraps instead of clipping ──────────────────
# Parity with combinedAnnotate's tests/test_segment_app.py
# test_header_wrap_and_label_escaping: both templates carry the same rule.
def test_header_wraps():
    # 2026-10-16: Edit Masks took combinedAnnotate's two stacked rows
    # (.review-topbar is flex-direction: column, holding .topbar-row1 and
    # #status-text), so the wrap-instead-of-clip rule lives on .topbar-row1,
    # and the timer reaches the row's end inside .timer-pause-group with
    # Pause. Its own box is a fixed 22ch so a longer reading never changes how
    # the row wraps (tests in TCRMPclip_combinedAnnotate/tests/
    # test_editmasks_layout_stability.py).
    tpl = open(os.path.join(SRC, 'templates', 'index.html')).read()
    rule = tpl[tpl.index('.topbar-row1 {'):]
    rule = rule[:rule.index('}') + 1]
    check('flex-wrap: wrap' in rule, f'.topbar-row1 wraps instead of clipping: {rule!r}')
    check('min-height: 28px' in rule and 'row-gap: 6px' in rule,
          'a min-height with a wrap row-gap, not a hard height')
    check('.review-topbar #image-filename' in tpl, 'the filename is the only shrinkable cell')
    group_rule = tpl[tpl.index('.timer-pause-group {'):]
    group_rule = group_rule[:group_rule.index('}') + 1]
    check('margin-left: auto' in group_rule,
          "the timer-and-Pause group holds the row's right edge")


# Same check as combinedAnnotate's tests/test_segment_app.py
# test_smoke_template_js_node_check: extract every inline <script>, strip Jinja,
# hand it to `node --check`. A template edit that breaks the JS is otherwise
# invisible to a Flask-only suite, since the page still renders 200.
def test_smoke_template_js_node_check():
    node = None
    for cand in ('node', '/usr/bin/node'):
        try:
            subprocess.run([cand, '--version'], capture_output=True, check=True)
            node = cand
            break
        except Exception:
            continue
    if node is None:
        print('    (node unavailable, skipping JS node --check)')
        return
    tpl = os.path.join(SRC, 'templates', 'index.html')
    with open(tpl) as f:
        src = f.read()
    blocks = re.findall(r'<script>(.*?)</script>', src, flags=re.DOTALL)
    js = '\n'.join(blocks)
    js = re.sub(r'{%.*?%}', '', js, flags=re.DOTALL)
    js = re.sub(r'{{.*?}}', '0', js, flags=re.DOTALL)
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(js)
        path = f.name
    try:
        res = subprocess.run([node, '--check', path], capture_output=True, text=True)
        check(res.returncode == 0, f'template inline JS passes node --check: {res.stderr}')
    finally:
        os.unlink(path)


test_header_wraps()
test_panel_cleanup_and_fonts()
test_box_mode_arrows_mask_size()
test_sticky_flags_and_reset_clip()
test_smoke_template_js_node_check()

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
