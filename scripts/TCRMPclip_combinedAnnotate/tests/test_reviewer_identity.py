"""Reviewer identity (D2 / spec 4.2) for the annotator twins.

Covers, against combinedAnnotate in-process and then editMasks in a child
process (the two app.py files are separate copies):
  - GET/POST /api/reviewer: normalized (stripped, uppercased, and only when
    the value is 1 to 4 letters); anything else, blank included, falls back to
    the TCRMP_REVIEWER env default.
  - PUT /api/image/<f>/masks stamps mask['reviewer'] on a status change and
    on a species relabel; a refused (unlabeled) accept is not stamped; a
    mask that already carries initials keeps them.
  - accept_all and commit_mask stamp the accepted masks.
  - label_outcomes / label_provenance.csv carry the reviewer.
  - export_batch(reviewer=...) stamps accepted masks lacking initials and
    the registry rows carry reviewer (mask value first, session fallback).
  - editMasks only: the edit-time registry upsert carries the reviewer.
  - source markers: do_export passes reviewer=_current_reviewer().
  - boot survival: /api/configure {} and /api/resume {export_dir} (the bodies
    the page boot sends, no initials) keep a pushed POST /api/reviewer value;
    initials in the body still replace it; an empty session falls back to env.

subprocess.run is stubbed at module level (standing rule for annotator
tests); the real reference is kept only to re-run this file for editMasks.
TCRMP_MASK_REGISTRY_DIR points at a temp dir that is removed at the end.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_reviewer_identity.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

_real_run = subprocess.run


def _stubbed_run(*a, **k):
    raise RuntimeError("subprocess.run is stubbed in test_reviewer_identity")


subprocess.run = _stubbed_run

WHICH = sys.argv[1] if len(sys.argv) > 1 else 'combinedAnnotate'
assert WHICH in ('combinedAnnotate', 'editMasks'), WHICH

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_SCRIPTS, 'TCRMPclip_' + WHICH, 'src')

_ROOT = tempfile.mkdtemp(prefix='reviewer_identity_')
_REG = os.path.join(_ROOT, 'registry')
_EXPORT = os.path.join(_ROOT, 'run_20260101_revid', 'step4test_combinedAnnotate')
os.makedirs(_EXPORT, exist_ok=True)

os.environ['TCRMP_MASK_REGISTRY_DIR'] = _REG
os.environ['TCRMP_REVIEWER'] = 'env'
os.environ['TCRMP_PROJECT_ID'] = '20260101_revid'
os.environ['TCRMP_EXPORT_DIR'] = _EXPORT
# No Step 4 output under this path: /api/configure {} answers 404 after it has
# applied the boot-time reviewer rule, which is the part under test.
os.environ['TCRMP_INPUT_DIR'] = os.path.join(_ROOT, 'routed_input')
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA,OFAV,MCAV')

sys.path.insert(0, _SRC)
sys.path.insert(0, _SCRIPTS)

from PIL import Image  # noqa: E402

import app as A  # noqa: E402
import export_yolo as EY  # noqa: E402
from _reefreview.mask_registry import MaskRegistry, build_registry_record  # noqa: E402

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + f"[{WHICH}] {m}")
    if not c:
        _fail += 1


def _mask(mid, status, species, sx, sy, **extra):
    m = {"id": mid, "status": status, "species": species, "category": "Coral",
         "source_type": "manual_click", "source_x": sx, "source_y": sy, "review": False,
         "rle": {"counts": f"c{mid}{sx}{sy}", "size": [48, 64]},
         "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]], "area": 100}
    m.update(extra)
    return m


A.app.config['TESTING'] = True
FN = "TCRMP20201022_clip_SCP_T101.jpeg"
FN2 = "TCRMP20201022_clip_SCP_T102.jpeg"
img = os.path.join(_ROOT, FN)
Image.new("RGB", (64, 48), (1, 2, 3)).save(img)
img2 = os.path.join(_ROOT, FN2)
Image.new("RGB", (64, 48), (1, 2, 3)).save(img2)

seg = {"image_path": FN, "image_path_abs": img, "image_width": 64, "image_height": 48,
       "masks": [_mask(0, "pending", "PA", 10, 10),
                 _mask(1, "pending", "", 20, 10),
                 _mask(2, "accepted", "PA", 30, 10, reviewer="AB"),
                 _mask(3, "pending", "OFAV", 40, 10)],
       "reviewed": False, "exported": False}
seg2 = {"image_path": FN2, "image_path_abs": img2, "image_width": 64, "image_height": 48,
        "masks": [_mask(0, "pending", "PA", 10, 20), _mask(1, "pending", "", 20, 20)],
        "reviewed": False, "exported": False}
A.session['export_dir'] = _EXPORT
A.session['segmentations'] = {FN: seg, FN2: seg2}

with A.app.test_client() as c:
    # ── /api/reviewer ────────────────────────────────────────────────
    r = c.get('/api/reviewer')
    check(r.status_code == 200 and json.loads(r.data).get('reviewer') == 'ENV',
          f"GET /api/reviewer -> env default, normalized (got {r.data!r})")
    r = c.post('/api/reviewer', json={'reviewer': '  lo '})
    check(json.loads(r.data).get('reviewer') == 'LO', "POST /api/reviewer strips + uppercases")
    check(A.session['reviewer'] == 'LO', "POST /api/reviewer sets session['reviewer']")
    # Initials are 1 to 4 letters (the launcher's rule). Anything else is not an
    # identity and normalizes to '' -- which then falls back to the env default.
    r = c.post('/api/reviewer', json={'reviewer': 'abcdefghijk'})
    check(json.loads(r.data).get('reviewer') == 'ENV',
          "over-long initials are not an identity, so the env default stands")
    r = c.post('/api/reviewer', json={'reviewer': 'ab1'})
    check(json.loads(r.data).get('reviewer') == 'ENV',
          "initials with a digit are not an identity either")
    r = c.post('/api/reviewer', json={'reviewer': ''})
    check(json.loads(r.data).get('reviewer') == 'ENV', "blank POST falls back to the env default")
    r = c.post('/api/reviewer', json={'reviewer': 'lo'})
    check(json.loads(r.data).get('reviewer') == 'LO', "reviewer set back to LO")

    # ── masks PUT: status change + relabel stamp, refused accept does not ──
    r = c.put(f'/api/image/{FN}/masks', json={
        'updates': {'0': 'accepted', '1': 'accepted'},
        'relabel': {'3': {'species': 'MCAV'}},
    })
    check(r.status_code == 200 and json.loads(r.data).get('ok') is True,
          f"PUT masks -> 200 ok (got {r.status_code} {r.data!r})")
    by_id = {m['id']: m for m in seg['masks']}
    check(by_id[0]['status'] == 'accepted' and by_id[0].get('reviewer') == 'LO',
          f"accepted mask stamped reviewer LO (got {by_id[0].get('reviewer')!r})")
    check(by_id[1]['status'] == 'pending' and 'reviewer' not in by_id[1],
          "refused unlabeled accept is not stamped")
    check(by_id[2].get('reviewer') == 'AB', "untouched mask keeps its earlier initials")
    check(by_id[3]['species'] == 'MCAV' and by_id[3].get('reviewer') == 'LO',
          "relabel stamps reviewer on the relabeled mask")
    lo = seg.get('label_outcomes') or {}
    check(lo.get('PA', {}).get('reviewer') == 'LO',
          f"label_outcomes carry the reviewer (got {lo.get('PA')})")
    prov = os.path.join(_EXPORT, 'label_provenance.csv')
    rows = []
    if os.path.exists(prov):
        with open(prov, newline='') as f:
            rows = list(csv.DictReader(f))
    check(rows and all(rw.get('reviewer') == 'LO' for rw in rows),
          f"label_provenance.csv rows carry reviewer LO (got {[rw.get('reviewer') for rw in rows]})")

    if WHICH == 'editMasks':
        reg = MaskRegistry(root=_REG)
        uid0 = build_registry_record(FN, by_id[0], 'x', 'x')['uid']
        row0 = reg.lookup(uid0)
        check(row0 is not None and row0.get('reviewer') == 'LO',
              f"edit-time registry upsert carries reviewer LO (got {row0 and row0.get('reviewer')!r})")

    # ── accept_all ───────────────────────────────────────────────────
    r = c.post('/api/reviewer', json={'reviewer': 'cd'})
    r = c.post(f'/api/image/{FN2}/accept_all')
    body = json.loads(r.data)
    check(r.status_code == 200 and body.get('accepted') == 1, f"accept_all accepts 1 (got {body})")
    b2 = {m['id']: m for m in seg2['masks']}
    check(b2[0].get('reviewer') == 'CD', "accept_all stamps the current reviewer")
    check('reviewer' not in b2[1], "accept_all leaves the skipped unlabeled mask unstamped")

    # ── commit_mask ──────────────────────────────────────────────────
    cand = _mask(99, "pending", "PA", 50, 30)
    r = c.post(f'/api/image/{FN}/commit_mask', json={'mask': cand})
    body = json.loads(r.data)
    check(body.get('ok') is True and body.get('mask', {}).get('reviewer') == 'CD',
          f"commit_mask stamps the reviewer on the committed mask (got {body.get('mask', {}).get('reviewer')!r})")
    committed = seg['masks'][-1]
    check(committed.get('reviewer') == 'CD' and committed['status'] == 'accepted',
          "committed mask persisted with reviewer + accepted")

    # ── export: accepted mask lacking initials gets the session reviewer ──
    seg['masks'].append(_mask(7, "accepted", "PA", 60, 40))    # no reviewer yet
    stats = EY.export_batch({FN: seg}, _EXPORT, {}, symlink=False, reviewer=A._current_reviewer())
    check(stats.get('exported_masks', 0) >= 1, f"export wrote masks (got {stats})")
    m7 = seg['masks'][-1]
    check(m7.get('reviewer') == 'CD', f"export stamps CD on the accepted mask lacking one (got {m7.get('reviewer')!r})")
    check(by_id[2].get('reviewer') == 'AB', "export leaves earlier initials alone")
    reg = MaskRegistry(root=_REG)
    row7 = reg.lookup(build_registry_record(FN, m7, 'x', 'x')['uid'])
    row2 = reg.lookup(build_registry_record(FN, by_id[2], 'x', 'x')['uid'])
    row0 = reg.lookup(build_registry_record(FN, by_id[0], 'x', 'x')['uid'])
    check(row7 is not None and row7.get('reviewer') == 'CD', "registry row: session fallback CD")
    check(row2 is not None and row2.get('reviewer') == 'AB', "registry row: mask value AB wins")
    check(row0 is not None and row0.get('reviewer') == 'LO', "registry row: earlier stamp LO kept")
    check(all(r_.get('project_id') == '20260101_revid' for r_ in reg.rows()),
          "registry rows keyed by TCRMP_PROJECT_ID")

    # ── boot survival: the page boot POSTs no initials ───────────────
    # The orchestrator pushes the latest opener to a reused process, then the
    # browser window boots with /api/configure {} or /api/resume {export_dir}.
    # Neither body carries initials, so neither may reset the pushed value to
    # the launch-time TCRMP_REVIEWER.
    r = c.post('/api/reviewer', json={'reviewer': 'ab'})
    check(json.loads(r.data).get('reviewer') == 'AB', "pushed reviewer AB before the boot")
    r = c.post('/api/configure', json={})
    check(r.status_code in (200, 404), f"/api/configure {{}} answers (got {r.status_code})")
    r = c.get('/api/reviewer')
    check(json.loads(r.data).get('reviewer') == 'AB',
          f"/api/configure with no initials keeps the pushed AB (got {r.data!r})")
    A.flush_all()
    r = c.post('/api/resume', json={'export_dir': _EXPORT})
    check(r.status_code == 200, f"/api/resume {{export_dir}} -> 200 (got {r.status_code} {r.data!r})")
    r = c.get('/api/reviewer')
    check(json.loads(r.data).get('reviewer') == 'AB',
          f"/api/resume with no initials keeps the pushed AB (got {r.data!r})")
    c.post('/api/resume', json={'export_dir': _EXPORT, 'reviewer': ' xy '})
    check(json.loads(c.get('/api/reviewer').data).get('reviewer') == 'XY',
          "initials in the boot body still replace the session value")
    A.session['reviewer'] = ''
    c.post('/api/configure', json={})
    check(json.loads(c.get('/api/reviewer').data).get('reviewer') == 'ENV',
          "an empty session falls back to the TCRMP_REVIEWER default at boot")

# ── source markers ───────────────────────────────────────────────────
with open(os.path.join(_SRC, 'app.py')) as f:
    src = f.read()
check("@app.route('/api/reviewer', methods=['GET', 'POST'])" in src, "app.py declares /api/reviewer")
check("reviewer=_current_reviewer()," in src, "do_export passes reviewer=_current_reviewer() to export_batch")
check(src.count("_stamp_reviewer(mask)") >= 4, "PUT (status + relabel), accept_all, commit_mask all stamp")
check("session['reviewer'] = (data.get('reviewer') or cfg.REVIEWER)" not in src
      and "form.get('reviewer') or cfg.REVIEWER" not in src
      and src.count("_seed_reviewer(") >= 3,
      "configure, resume and any standalone start path seed the reviewer through _seed_reviewer")
for banned in ('—', 'land '):
    check(banned not in src[src.find('Reviewer identity (D2)'):src.find('def update_masks')],
          f"no banned string {banned!r} in the identity block")

# ── the editMasks twin, in its own process (separate app/config modules) ──
if WHICH == 'combinedAnnotate':
    env = dict(os.environ)
    env['TCRMP_MASK_REGISTRY_DIR'] = os.path.join(_ROOT, 'registry_em')
    env['TCRMP_SESSION_MODE'] = 'edit'
    child = _real_run([sys.executable, os.path.abspath(__file__), 'editMasks'],
                      env=env, capture_output=True, text=True)
    sys.stdout.write(child.stdout)
    if child.returncode != 0:
        sys.stdout.write(child.stderr[-3000:])
    check(child.returncode == 0, f"editMasks twin suite green (exit {child.returncode})")

shutil.rmtree(_ROOT, ignore_errors=True)
check(not os.path.exists(_ROOT), "temp root removed")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
