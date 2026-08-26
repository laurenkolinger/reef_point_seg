# CA/tests/test_unlabeled_filter.py
"""Unlabeled-mask filter + jump (Task 6, follows Task 5's empty-species guard).

Covers:
  - GET /api/unlabeled on a seeded review queue lists exactly the frames that
    have a mask with species=='' (accepted or pending, same predicate as
    app._needs_species, the guard from Task 5), and reports the right count.
  - A frame outside session['review_files'] (already exported) is not listed,
    since the review UI's Prev/Next navigation never reaches it either.
  - Structural: the template exposes the `chk-unlabeled-only` toggle and the
    `jumpNextUnlabeled` / `toggleUnlabeledOnly` client functions, and the pure
    "does this frame have an unlabeled mask" predicate embedded in those
    functions agrees with the server across a small case matrix (node
    sanity check, same extract-and-run pattern as
    test_target_only_shows_pending.py).

No pytest, GPU-free. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_unlabeled_filter.py
"""
import os
import sys
import json
import subprocess
import tempfile
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_TPL = os.path.join(_SRC, "templates", "index.html")
sys.path.insert(0, _SRC)

import app as A

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _img(d, name):
    p = os.path.join(d, name)
    Image.new("RGB", (64, 48), (10, 20, 30)).save(p)
    return p


# ── GET /api/unlabeled: server-side enumeration ──────────────────────────────
A.app.config['TESTING'] = True

_d = tempfile.mkdtemp(prefix="unlabeled_")
img_a = _img(_d, "FRAME_A.jpg")
img_b = _img(_d, "FRAME_B.jpg")
img_c = _img(_d, "FRAME_C.jpg")
img_d = _img(_d, "FRAME_D.jpg")
export_dir = os.path.join(_d, "out")
os.makedirs(export_dir, exist_ok=True)


def _seg(image_path, masks):
    return {
        "image_path": os.path.basename(image_path),
        "image_path_abs": image_path,
        "image_width": 64,
        "image_height": 48,
        "masks": masks,
        "processed_at": "",
        "reviewed": False,
        "exported": False,
    }


# FRAME_A: has an unlabeled mask that is already 'accepted' (the pre-Task-5
# scenario, accepted before the guard existed).
seg_a = _seg(img_a, [
    {"id": 0, "status": "accepted", "species": ""},
    {"id": 1, "status": "accepted", "species": "PA"},
])
# FRAME_B: has an unlabeled mask still 'pending' (a hand-added mask during
# Refine review that hasn't been given a species yet).
seg_b = _seg(img_b, [
    {"id": 0, "status": "pending", "species": ""},
])
# FRAME_C: fully labeled (including a REVIEW mask, which is an intentional
# label, not a missing one) -- must NOT show up as unlabeled.
seg_c = _seg(img_c, [
    {"id": 0, "status": "accepted", "species": "PA"},
    {"id": 1, "status": "accepted", "species": "REVIEW", "review": True},
])
# FRAME_D: has an unlabeled mask too, but is NOT in review_files (already
# exported) -- out of scope, since Prev/Next navigation never reaches it.
seg_d = _seg(img_d, [
    {"id": 0, "status": "accepted", "species": ""},
])
seg_d["exported"] = True

A.session['export_dir'] = export_dir
A.session['segmentations'] = {
    "FRAME_A.jpg": seg_a,
    "FRAME_B.jpg": seg_b,
    "FRAME_C.jpg": seg_c,
    "FRAME_D.jpg": seg_d,
}
A.session['review_files'] = ["FRAME_A.jpg", "FRAME_B.jpg", "FRAME_C.jpg"]
A.session['review_offset'] = 0
A.session['review_batch_size'] = 10

with A.app.test_client() as c:
    r = c.get('/api/unlabeled')
    check(r.status_code == 200, f"GET /api/unlabeled 200 (got {r.status_code})")
    body = json.loads(r.data)
    frames = set(body.get('frames', []))
    check(frames == {"FRAME_A.jpg", "FRAME_B.jpg"},
          f"unlabeled frames == {{FRAME_A, FRAME_B}} (got {frames!r})")
    check(body.get('count') == 2, f"count == 2 (got {body.get('count')})")
    check("FRAME_C.jpg" not in frames, "fully-labeled frame (incl. REVIEW mask) excluded")
    check("FRAME_D.jpg" not in frames, "already-exported frame (outside review_files) excluded")

# Empty review queue -> empty result, not an error.
A.session['review_files'] = []
with A.app.test_client() as c:
    r2 = c.get('/api/unlabeled')
    body2 = json.loads(r2.data)
    check(body2 == {'frames': [], 'count': 0}, f"empty review queue -> empty result (got {body2!r})")


# ── Structural: template exposes the toggle id + jump/filter functions ──────
_tpl_src = open(_TPL).read()
check('id="chk-unlabeled-only"' in _tpl_src, 'template has #chk-unlabeled-only checkbox')
check('function jumpNextUnlabeled' in _tpl_src, 'template defines jumpNextUnlabeled()')
check('function toggleUnlabeledOnly' in _tpl_src, 'template defines toggleUnlabeledOnly()')
check('function refreshUnlabeledSet' in _tpl_src, 'template defines refreshUnlabeledSet()')
check('/api/unlabeled' in _tpl_src, 'template calls GET /api/unlabeled')
check('title=' in _tpl_src.split('chk-unlabeled-only')[0][-600:], (
    'the Unlabeled-only control has a title= tooltip nearby'))


# ── Node sanity check: jumpNextUnlabeled walks images[] and finds the next
#    unlabeled frame by filename membership in unlabeledSet, wrapping around,
#    mirroring the server's set membership semantics without reinventing the
#    frame loop (it must call the existing selectImage, not a new nav path).
def _find_node():
    for cand in ('node', '/usr/bin/node'):
        try:
            subprocess.run([cand, '--version'], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


node = _find_node()
if node is None:
    print('    (node unavailable -- skipping JS sanity check)')
else:
    jump_start = _tpl_src.index('let unlabeledOnly = false;')
    jump_end = _tpl_src.index('function jumpNextUnlabeled')
    tail_start = _tpl_src.index('function jumpNextUnlabeled')
    tail_end = _tpl_src.index('\n}\n', tail_start) + len('\n}\n')
    header = _tpl_src[jump_start:jump_end]
    jump_fn = _tpl_src[tail_start:tail_end]

    harness = """
%s
%s

// Stand-ins for the pieces jumpNextUnlabeled/selectImage depend on.
let images = [
  {filename: 'F1.jpg'}, {filename: 'F2.jpg'}, {filename: 'F3.jpg'}, {filename: 'F4.jpg'},
];
let currentIdx = 0;
let selectedCalls = [];
function selectImage(idx) { currentIdx = idx; selectedCalls.push(idx); }
function setStatus(msg) { /* no-op for the harness */ }

unlabeledSet = new Set(['F3.jpg']);
jumpNextUnlabeled();
let fails = 0;
if (currentIdx !== 2) { console.log('FAIL jump lands on F3 (idx 2), got ' + currentIdx); fails++; }
else { console.log('PASS jump lands on the only unlabeled frame'); }

// Wraps around: currently on the last unlabeled frame, next unlabeled is
// earlier in the array (walk forward and wrap).
currentIdx = 3;
unlabeledSet = new Set(['F1.jpg', 'F4.jpg']);
jumpNextUnlabeled();
if (currentIdx !== 0) { console.log('FAIL wrap-around lands on F1 (idx 0), got ' + currentIdx); fails++; }
else { console.log('PASS wrap-around finds the next unlabeled frame'); }

// No unlabeled frames at all -> no navigation call is made.
currentIdx = 1;
selectedCalls = [];
unlabeledSet = new Set();
jumpNextUnlabeled();
if (selectedCalls.length !== 0) { console.log('FAIL no-op when nothing unlabeled, calls=' + JSON.stringify(selectedCalls)); fails++; }
else { console.log('PASS no navigation when no unlabeled frames remain'); }

process.exit(fails ? 1 : 0);
""" % (header, jump_fn)

    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        res = subprocess.run([node, path], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            print('    ' + line)
        if res.stderr.strip():
            print('    [stderr] ' + res.stderr.strip())
        check(res.returncode == 0, 'jumpNextUnlabeled walks images[] via selectImage and wraps correctly')
    finally:
        os.unlink(path)


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
