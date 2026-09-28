"""Accepting a mask requires a reviewer identity (both annotator twins).

Unattributed work is the defect this closes: an annotator process launched
without initials used to accept masks that nobody could be credited for. Every
ACCEPT path now refuses while the session has no reviewer:

  * PUT /api/image/<f>/masks   - only when an update would flip a mask from
                                 not-accepted to accepted. Rejects, relabels
                                 and an idempotent re-save of an already
                                 accepted mask still go through, so a frame
                                 save is never lost and the operator is never
                                 trapped.
  * POST /api/image/<f>/accept_all
  * POST /api/image/<f>/commit_mask

A refusal returns HTTP 200 with {ok: false, needs_reviewer: true, error: ...}
and mutates nothing, so the client can point the operator at the initials box
in the top bar and retry. Setting initials through POST /api/reviewer clears
the block immediately.

Template side (markers only, no browser): the review top bar carries the
initials input wired to /api/reviewer, the client handles a needs_reviewer
answer, and combinedAnnotate's standalone start card requires initials before
it will start a session.

subprocess.run is stubbed at module level (standing rule for annotator tests).
TCRMP_MASK_REGISTRY_DIR points at a temp dir that is removed at the end, and
TCRMP_REVIEWER is deliberately UNSET: an anonymous session is the case under
test.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_reviewer_required.py
     env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_reviewer_required.py editMasks
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_real_run = subprocess.run


def _stubbed_run(*a, **k):
    raise RuntimeError("subprocess.run is stubbed in test_reviewer_required")


subprocess.run = _stubbed_run

WHICH = sys.argv[1] if len(sys.argv) > 1 else 'combinedAnnotate'
assert WHICH in ('combinedAnnotate', 'editMasks'), WHICH

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_SCRIPTS, 'TCRMPclip_' + WHICH, 'src')

_ROOT = tempfile.mkdtemp(prefix='reviewer_required_')
_REG = os.path.join(_ROOT, 'registry')
_EXPORT = os.path.join(_ROOT, 'run_20260101_revreq', 'step4test_combinedAnnotate')
os.makedirs(_EXPORT, exist_ok=True)

os.environ['TCRMP_MASK_REGISTRY_DIR'] = _REG
os.environ['TCRMP_CLIP_INDEX_DIR'] = os.path.join(_ROOT, 'clip_index')
os.environ.pop('TCRMP_REVIEWER', None)     # the whole point: nobody is named
os.environ['TCRMP_PROJECT_ID'] = '20260101_revreq'
os.environ['TCRMP_EXPORT_DIR'] = _EXPORT
os.environ['TCRMP_INPUT_DIR'] = os.path.join(_ROOT, 'routed_input')
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA,OFAV,MCAV')

sys.path.insert(0, _SRC)
sys.path.insert(0, _SCRIPTS)

from PIL import Image  # noqa: E402

import app as A  # noqa: E402

_fail = 0

MSG = "Set your initials first (top bar) so this work is credited."


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + f"[{WHICH}] {m}")
    if not c:
        _fail += 1


def _mask(mid, status, species, sx, sy, **extra):
    m = {"id": mid, "status": status, "species": species, "category": "Coral",
         "source_type": "manual_click", "source_x": sx, "source_y": sy,
         "review": False,
         "rle": {"counts": f"c{mid}{sx}{sy}", "size": [48, 64]},
         "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]], "area": 100}
    m.update(extra)
    return m


def _real_mask(mid, status, species, x0, y0, w=12, h=10, **extra):
    """A mask whose RLE actually decodes, so the merge routes can run it.
    The plain ``_mask`` fixture above carries a stub RLE that only the accept
    routes ever look at; merging decodes geometry for real."""
    import numpy as np
    from mask_ops import rle_encode
    binary = np.zeros((48, 64), dtype=bool)
    binary[y0:y0 + h, x0:x0 + w] = True
    m = _mask(mid, status, species, x0, y0, **extra)
    m['rle'] = rle_encode(binary)
    m['area'] = int(binary.sum())
    m['bbox'] = [x0, y0, w, h]
    return m


def _fresh_seg(fn, path):
    return {"image_path": fn, "image_path_abs": path,
            "image_width": 64, "image_height": 48,
            "masks": [_mask(0, "pending", "PA", 10, 10),
                      _mask(1, "accepted", "PA", 20, 10, reviewer="AB"),
                      _mask(2, "pending", "OFAV", 30, 10)],
            "reviewed": False, "exported": False}


A.app.config['TESTING'] = True
FN = "TCRMP20201022_clip_SCP_T101.jpeg"
img = os.path.join(_ROOT, FN)
Image.new("RGB", (64, 48), (1, 2, 3)).save(img)
seg = _fresh_seg(FN, img)
A.session['export_dir'] = _EXPORT
A.session['segmentations'] = {FN: seg}

try:
    with A.app.test_client() as c:
        # ── nobody is named ──────────────────────────────────────────────
        r = c.get('/api/reviewer')
        check(json.loads(r.data).get('reviewer') == '',
              f"GET /api/reviewer is blank with no env default (got {r.data!r})")

        # ── masks PUT: an accept is refused, nothing is mutated ──────────
        r = c.put(f'/api/image/{FN}/masks', json={'updates': {'0': 'accepted'}})
        body = json.loads(r.data)
        check(r.status_code == 200, f"refused accept answers 200 (got {r.status_code})")
        check(body.get('ok') is False, f"refused accept reports ok false (got {body})")
        check(body.get('needs_reviewer') is True, f"needs_reviewer flag set (got {body})")
        check(body.get('error') == MSG, f"exact operator message (got {body.get('error')!r})")
        by_id = {m['id']: m for m in seg['masks']}
        check(by_id[0]['status'] == 'pending', "the refused mask stays pending")
        check('reviewer' not in by_id[0], "the refused mask is not stamped")

        # ── a reject still works while unattributed ──────────────────────
        r = c.put(f'/api/image/{FN}/masks', json={'updates': {'2': 'rejected'}})
        check(json.loads(r.data).get('ok') is True,
              f"reject is not blocked by the reviewer guard (got {r.data!r})")
        check(by_id[2]['status'] == 'rejected', "the reject was applied")

        # ── an idempotent re-save of an accepted mask is not an accept ───
        r = c.put(f'/api/image/{FN}/masks',
                  json={'updates': {'1': 'accepted', '2': 'rejected'}})
        body = json.loads(r.data)
        check(body.get('ok') is True and 'needs_reviewer' not in body,
              f"re-saving an already accepted mask is allowed (got {body})")
        check(by_id[1]['status'] == 'accepted' and by_id[1]['reviewer'] == 'AB',
              "the already accepted mask keeps its earlier initials")

        # ── a relabel of a PENDING mask is allowed ───────────────────────
        r = c.put(f'/api/image/{FN}/masks', json={'relabel': {'0': {'species': 'MCAV'}}})
        check(json.loads(r.data).get('ok') is True,
              f"a pending relabel is not blocked (got {r.data!r})")
        check(by_id[0]['species'] == 'MCAV', "the relabel was applied")

        # ── a relabel of an ACCEPTED mask is refused ─────────────────────
        # Changing the species of accepted work is a judgement on accepted
        # work, so it carries the same identity requirement as making it.
        r = c.put(f'/api/image/{FN}/masks', json={'relabel': {'1': {'species': 'OFAV'}}})
        body = json.loads(r.data)
        check(body.get('needs_reviewer') is True and body.get('ok') is False,
              f"relabelling an accepted mask is refused (got {body})")
        check(by_id[1]['species'] == 'PA', "the accepted mask keeps its species")
        check(by_id[1].get('reviewer') == 'AB', "and keeps its earlier initials")

        # ── accept_all is refused whole ──────────────────────────────────
        r = c.post(f'/api/image/{FN}/accept_all')
        body = json.loads(r.data)
        check(body.get('needs_reviewer') is True and body.get('ok') is False,
              f"accept_all refused (got {body})")
        check(body.get('error') == MSG, "accept_all carries the same message")
        check(by_id[0]['status'] == 'pending', "accept_all accepted nothing")

        # ── commit_mask is refused ───────────────────────────────────────
        before = len(seg['masks'])
        r = c.post(f'/api/image/{FN}/commit_mask',
                   json={'mask': _mask(99, 'pending', 'PA', 50, 30)})
        body = json.loads(r.data)
        check(body.get('needs_reviewer') is True and body.get('ok') is False,
              f"commit_mask refused (got {body})")
        check(body.get('error') == MSG, "commit_mask carries the same message")
        check(len(seg['masks']) == before, "commit_mask appended nothing")

        # ── naming yourself clears the block ─────────────────────────────
        r = c.post('/api/reviewer', json={'reviewer': 'lo'})
        check(json.loads(r.data).get('reviewer') == 'LO', "reviewer set to LO")

        r = c.put(f'/api/image/{FN}/masks', json={'updates': {'0': 'accepted'}})
        body = json.loads(r.data)
        check(body.get('ok') is True and 'needs_reviewer' not in body,
              f"the same accept goes through once named (got {body})")
        check(by_id[0]['status'] == 'accepted' and by_id[0].get('reviewer') == 'LO',
              f"the accepted mask is credited to LO (got {by_id[0].get('reviewer')!r})")

        r = c.post(f'/api/image/{FN}/commit_mask',
                   json={'mask': _mask(99, 'pending', 'PA', 50, 30)})
        body = json.loads(r.data)
        check(body.get('ok') is True and body.get('mask', {}).get('reviewer') == 'LO',
              f"commit_mask goes through once named (got {body})")

        # ── merge is an accept path too ──────────────────────────────────
        # _merge_ids promotes the survivor to accepted as soon as ANY member
        # is, so a pending low-id mask merged with an accepted higher-id one
        # used to come out accepted with no initials on it at all.
        r = c.post('/api/reviewer', json={'reviewer': ''})
        check(json.loads(r.data).get('reviewer') == '', "reviewer cleared for the merge checks")
        seg['masks'] = [_real_mask(10, "pending", "PA", 4, 4),
                        _real_mask(11, "accepted", "PA", 10, 4, reviewer="AB"),
                        _real_mask(12, "pending", "PA", 16, 4),
                        _real_mask(13, "pending", "SS", 30, 4)]
        r = c.post(f'/api/image/{FN}/merge', json={'mask_id_a': 10, 'mask_id_b': 11})
        body = json.loads(r.data)
        check(body.get('needs_reviewer') is True and body.get('ok') is False,
              f"/merge refused when the survivor would be accepted (got {body})")
        ids = [m['id'] for m in seg['masks']]
        check(ids == [10, 11, 12, 13], f"/merge changed nothing (got {ids})")
        check(seg['masks'][0]['status'] == 'pending',
              "the pending mask was not promoted by the refused merge")

        r = c.post(f'/api/image/{FN}/merge_many', json={'mask_ids': [10, 11, 12]})
        body = json.loads(r.data)
        check(body.get('needs_reviewer') is True and body.get('ok') is False,
              f"/merge_many refused when the survivor would be accepted (got {body})")
        check([m['id'] for m in seg['masks']] == [10, 11, 12, 13],
              "/merge_many changed nothing")

        r = c.post(f'/api/image/{FN}/merge_same_id')
        body = json.loads(r.data)
        check(body.get('needs_reviewer') is True and body.get('ok') is False,
              f"/merge_same_id refused while an accepted mask is on the frame (got {body})")

        # A merge of PENDING masks only creates no accepted work, so it runs.
        seg['masks'] = [_real_mask(20, "pending", "PA", 4, 20),
                        _real_mask(21, "pending", "PA", 10, 20)]
        r = c.post(f'/api/image/{FN}/merge', json={'mask_id_a': 20, 'mask_id_b': 21})
        body = json.loads(r.data)
        check(body.get('ok') is True and 'needs_reviewer' not in body,
              f"a pending-only merge stays free (got {body})")
        check(body.get('mask', {}).get('status') == 'pending',
              "the pending merge survivor is still pending")

        # Named, the same accepted merge goes through AND is credited.
        c.post('/api/reviewer', json={'reviewer': 'zz'})
        seg['masks'] = [_real_mask(30, "pending", "PA", 4, 30),
                        _real_mask(31, "accepted", "PA", 10, 30, reviewer="AB")]
        r = c.post(f'/api/image/{FN}/merge', json={'mask_id_a': 30, 'mask_id_b': 31})
        body = json.loads(r.data)
        surv = body.get('mask') or {}
        check(body.get('ok') is True, f"the same merge goes through once named (got {body})")
        check(surv.get('status') == 'accepted', "the survivor is accepted")
        check(surv.get('reviewer') == 'ZZ',
              f"the accepted survivor is credited to the merger (got {surv.get('reviewer')!r})")

        # ── initials must look like initials ─────────────────────────────
        for junk in ('1', '!!', 'A1', 'ABCDE', 'l o'):
            r = c.post('/api/reviewer', json={'reviewer': junk})
            got = json.loads(r.data).get('reviewer')
            check(got == '', f"{junk!r} is not an identity (got {got!r})")
        r = c.post('/api/reviewer', json={'reviewer': 'abcd'})
        check(json.loads(r.data).get('reviewer') == 'ABCD', "4 letters are accepted")

        # ── an accept_all with nothing pending is a no-op, not a refusal ─
        r = c.post('/api/reviewer', json={'reviewer': ''})
        check(json.loads(r.data).get('reviewer') == '', "reviewer cleared again")
        seg['masks'] = [_real_mask(40, "rejected", "PA", 4, 4),
                        _real_mask(41, "accepted", "PA", 20, 4, reviewer="AB")]
        r = c.post(f'/api/image/{FN}/accept_all')
        body = json.loads(r.data)
        check(body.get('ok') is True and body.get('accepted') == 0,
              f"accept_all with nothing pending is a plain no-op (got {body})")

    # ── template markers ─────────────────────────────────────────────────
    tpl = os.path.join(_SRC, 'templates', 'index.html')
    html = open(tpl, encoding='utf-8').read()
    check('id="reviewer-input"' in html, "review top bar carries the initials input")
    idx = html.index('id="reviewer-input"')
    check('title=' in html[max(0, idx - 400):idx + 400],
          "the initials input carries a tooltip")
    check('reviewer-input' in html and "'/api/reviewer'" in html,
          "the initials input is wired to /api/reviewer")
    check('needs_reviewer' in html, "the client handles a needs_reviewer answer")
    check(html.count('needs_reviewer') >= 3,
          f"every accept path handles it (found {html.count('needs_reviewer')})")
    check('setReviewer' in html and 'loadReviewer' in html,
          "initials load on boot and POST on change")
    # No em dashes or en dashes in what was added.
    around = html[max(0, idx - 600):idx + 600]
    check('\u2014' not in around and '\u2013' not in around,
          "no em or en dashes around the new control")
    if WHICH == 'combinedAnnotate':
        check('id="sa-initials"' in html, "standalone start card has an initials field")
        sa = html.index('id="sa-initials"')
        check('title=' in html[max(0, sa - 400):sa + 400],
              "the standalone initials field carries a tooltip")
        check(re.search(r"sa-initials'\)\.value|sa-initials\"\)\.value", html) is not None,
              "the standalone start reads the initials field")

    # ── standalone start refuses an unnamed session (combinedAnnotate) ────
    if WHICH == 'combinedAnnotate':
        A.cfg.SESSION_MODE = 'standalone'
        A.cfg.STANDALONE_ROOT = os.path.join(_ROOT, '_standalone')
        with A.app.test_client() as c:
            c.post('/api/reviewer', json={'reviewer': ''})
            r = c.post('/api/standalone/start',
                       data={"session": "26-08-26_XX", "labels": "PA"},
                       content_type='multipart/form-data')
            body = json.loads(r.data)
            check(r.status_code == 400 and 'initials' in (body.get('error') or '').lower(),
                  f"standalone start without initials is refused (got {r.status_code} {body})")
            r = c.post('/api/standalone/start',
                       data={"session": "26-08-26_XX", "labels": "PA", "reviewer": "zz"},
                       content_type='multipart/form-data')
            body = json.loads(r.data)
            check(body.get('ok') is True, f"standalone start with initials works (got {body})")
            sj = os.path.join(A.cfg.STANDALONE_ROOT, '26-08-26_XX', 'session.json')
            check(os.path.exists(sj) and json.load(open(sj)).get('reviewer') == 'ZZ',
                  "session.json records the initials")
finally:
    shutil.rmtree(_ROOT, ignore_errors=True)
    subprocess.run = _real_run

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
