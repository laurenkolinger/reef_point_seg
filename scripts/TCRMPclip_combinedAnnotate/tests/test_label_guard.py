# CA/tests/test_label_guard.py
"""Guard: an accepted mask must always have a species (or be a REVIEW mask).

Covers:
  - PUT /api/image/<f>/masks refuses to accept a mask with empty species:
    the mask stays 'pending' and the response reports needs_species.
  - A mask with species 'PA' accepts fine.
  - A REVIEW mask (species == 'REVIEW') accepts fine, not caught by the guard.
  - POST /api/image/<f>/commit_mask refuses to commit an empty-species scan
    candidate (e.g. seeded from an unlabeled source mask via
    exemplar_scan(mode='this')): the candidate is not appended/accepted and
    the response reports needs_species instead of a mask.
  - export_yolo.export_batch hard-skips an accepted-but-empty-species mask:
    it never reaches the exported label lines, and blocked_unlabeled == 1.

No pytest, GPU-free. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_label_guard.py
"""
import os
import sys
import json
import tempfile
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

import app as A
import export_yolo as EY

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


# ── PUT /api/image/<f>/masks: server-side accept guard ──────────────────────
A.app.config['TESTING'] = True

_d = tempfile.mkdtemp(prefix="label_guard_")
img_path = _img(_d, "FRAME.jpg")
export_dir = os.path.join(_d, "out")
os.makedirs(export_dir, exist_ok=True)

seg = {
    "image_path": "FRAME.jpg",
    "image_path_abs": img_path,
    "image_width": 64,
    "image_height": 48,
    "masks": [
        {"id": 0, "status": "pending", "species": ""},        # unlabeled
        {"id": 1, "status": "pending", "species": "PA"},       # labeled target
        {"id": 2, "status": "pending", "species": "REVIEW", "review": True},  # intentional label
    ],
    "processed_at": "",
    "reviewed": False,
    "exported": False,
}

A.session['export_dir'] = export_dir
A.session['segmentations'] = {"FRAME.jpg": seg}
A.session['reviewer'] = "LO"

with A.app.test_client() as c:
    # Try to accept all three masks in one PUT.
    r = c.put(
        '/api/image/FRAME.jpg/masks',
        data=json.dumps({"updates": {"0": "accepted", "1": "accepted", "2": "accepted"}}),
        content_type='application/json',
    )
    check(r.status_code == 200, f"PUT /masks 200 (got {r.status_code})")
    body = json.loads(r.data)
    check(body.get('ok') is True, f"PUT /masks ok=True (got {body!r})")

    masks_by_id = {m['id']: m for m in seg['masks']}
    check(masks_by_id[0]['status'] == 'pending',
          f"empty-species mask #0 stays pending (got {masks_by_id[0]['status']!r})")
    check(masks_by_id[1]['status'] == 'accepted',
          f"PA mask #1 accepts (got {masks_by_id[1]['status']!r})")
    check(masks_by_id[2]['status'] == 'accepted',
          f"REVIEW mask #2 accepts, not caught by guard (got {masks_by_id[2]['status']!r})")

    needs = body.get('needs_species', [])
    check(0 in needs, f"response reports mask #0 needs_species (got {needs!r})")
    check(1 not in needs, "labeled mask #1 not reported in needs_species")
    check(2 not in needs, "REVIEW mask #2 not reported in needs_species")

# ── accept_all: same guard applies ───────────────────────────────────────────
seg2 = {
    "image_path": "FRAME2.jpg",
    "image_path_abs": img_path,
    "image_width": 64,
    "image_height": 48,
    "masks": [
        {"id": 0, "status": "pending", "species": ""},
        {"id": 1, "status": "pending", "species": "PA"},
        {"id": 2, "status": "pending", "species": "REVIEW", "review": True},
    ],
    "processed_at": "",
    "reviewed": False,
    "exported": False,
}
A.session['segmentations']["FRAME2.jpg"] = seg2

with A.app.test_client() as c:
    r2 = c.post('/api/image/FRAME2.jpg/accept_all')
    check(r2.status_code == 200, f"POST /accept_all 200 (got {r2.status_code})")
    body2 = json.loads(r2.data)
    check(body2.get('accepted') == 2, f"accept_all accepts only the 2 labeled masks (got {body2.get('accepted')})")
    needs2 = body2.get('needs_species', [])
    check(needs2 == [0], f"accept_all reports mask #0 needs_species (got {needs2!r})")

    masks2_by_id = {m['id']: m for m in seg2['masks']}
    check(masks2_by_id[0]['status'] == 'pending', "accept_all leaves unlabeled mask #0 pending")
    check(masks2_by_id[1]['status'] == 'accepted', "accept_all accepts PA mask #1")
    check(masks2_by_id[2]['status'] == 'accepted', "accept_all accepts REVIEW mask #2")


# ── commit_mask: the third accept path must not bypass the guard ────────────
# A scan candidate seeded from an unlabeled source mask (exemplar_scan(mode='this'))
# carries species=='' straight through. Posting it to /commit_mask must NOT
# append it to seg['masks'] as an accepted mask; it must be refused the same
# way update_masks/accept_all refuse it, via the shared _needs_species(...).
seg3 = {
    "image_path": "FRAME3.jpg",
    "image_path_abs": img_path,
    "image_width": 64,
    "image_height": 48,
    "masks": [
        {"id": 0, "status": "accepted", "species": "PA"},
    ],
    "processed_at": "",
    "reviewed": False,
    "exported": False,
}
A.session['segmentations']["FRAME3.jpg"] = seg3

with A.app.test_client() as c:
    unlabeled_cand = {
        "species": "", "polygon_norm": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]],
        "rle": None, "area": 10, "score": 0.9, "source_type": "exemplar",
    }
    r3 = c.post(
        '/api/image/FRAME3.jpg/commit_mask',
        data=json.dumps({"mask": unlabeled_cand}),
        content_type='application/json',
    )
    check(r3.status_code == 200, f"POST /commit_mask 200 (got {r3.status_code})")
    body3 = json.loads(r3.data)
    check(body3.get('ok') is not True,
          f"commit_mask refuses an empty-species candidate: ok is not True (got {body3!r})")
    check(bool(body3.get('needs_species')),
          f"commit_mask response signals needs_species (got {body3!r})")
    check('mask' not in body3,
          f"commit_mask does not return a committed mask for an empty-species candidate (got {body3!r})")
    check(len(seg3['masks']) == 1,
          f"empty-species candidate is NOT appended to seg['masks'] (got {len(seg3['masks'])} masks)")
    check(all(m['status'] == 'accepted' and m['species'] == 'PA' for m in seg3['masks']),
          "the only mask remains the original labeled+accepted mask, untouched")

    # A labeled candidate still commits fine through the same route.
    labeled_cand = {
        "species": "PA", "polygon_norm": [[0.3, 0.3, 0.4, 0.3, 0.4, 0.4]],
        "rle": None, "area": 10, "score": 0.9, "source_type": "exemplar",
    }
    r3b = c.post(
        '/api/image/FRAME3.jpg/commit_mask',
        data=json.dumps({"mask": labeled_cand}),
        content_type='application/json',
    )
    body3b = json.loads(r3b.data)
    check(body3b.get('ok') is True, f"commit_mask still accepts a labeled candidate (got {body3b!r})")
    check(body3b.get('mask', {}).get('status') == 'accepted',
          "labeled candidate commits as accepted")
    check(len(seg3['masks']) == 2, "labeled candidate IS appended to seg['masks']")


# ── export_yolo.export_batch: hard backstop on an accepted empty-species mask ──
# Even if a mask somehow reaches 'accepted' status with no species (bypassing
# the server guard above, e.g. via a direct commit path), export_batch must
# never write a YOLO line for it and must report it via blocked_unlabeled.
d2 = tempfile.mkdtemp(prefix="label_guard_export_")
img2 = _img(d2, "BLOCKED.jpg")
export_dir2 = os.path.join(d2, "out")

segs = {
    "BLOCKED.jpg": {
        "image_path_abs": img2,
        "image_width": 64,
        "image_height": 48,
        "masks": [
            {"status": "accepted", "species": "",
             "polygon_norm": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]]},
        ],
    },
}
stats = EY.export_batch(segs, export_dir2, {}, symlink=False)

check(stats.get('blocked_unlabeled') == 1,
      f"blocked_unlabeled == 1 (got {stats.get('blocked_unlabeled')})")
check(stats.get('exported_masks', 0) == 0,
      f"no mask lines exported for the blocked mask (got {stats.get('exported_masks')})")

label_path = os.path.join(export_dir2, "all_labels", "BLOCKED.txt")
if os.path.exists(label_path):
    label_content = open(label_path).read().strip()
    check(label_content == "", "blocked mask excluded from the label file (empty label, treated as negative)")
else:
    check(False, "label file should exist (frame exported as a negative once its only mask is blocked)")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
