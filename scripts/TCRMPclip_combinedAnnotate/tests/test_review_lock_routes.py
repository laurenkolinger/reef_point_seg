"""test_review_lock_routes.py: the expert-review flagging lock, route by route.

Module:  reef_point_seg (display name VEGA), Place plus Segment annotator.
Purpose: prove that no route of this app can add an expert-review tag to a mask
         while the lock is on, and that every read path over the review tags
         already on disk behaves exactly as it did before the lock.
Inputs:  none. Every fixture is built in a temporary directory and the
         segmentation engine is a fake, so the suite is GPU-free and offline.
Outputs: one PASS or FAIL line per assertion on standard output; exit code 0
         when every assertion passed, 1 otherwise.

Lauren Olinger asked on 2026-09-06 for expert-review flagging to be locked in
every usage of the annotator. This app serves three of the four usages: the
in-project Place plus Segment annotator, the Refine loop, and the standalone
Annotate Images app on port 5096.

No pytest. Run with the unified environment python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_review_lock_routes.py
"""
import atexit
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

# Nothing in this suite launches a process. Stub the two spawners at module
# level anyway, before app is imported: a stray launch route would otherwise
# sweep the annotator port range and kill whatever session is live on this box.
_SPAWNS = []


class _NoSpawn:
    """Stand-in for a finished subprocess: records the call, spawns nothing."""

    returncode = 0
    stdout = b""
    stderr = b""

    def __init__(self, *args, **kwargs):
        _SPAWNS.append((args, kwargs))

    def communicate(self, *args, **kwargs):
        return (b"", b"")

    def wait(self, *args, **kwargs):
        return 0


def _record_run(*args, **kwargs):
    """subprocess.run replacement: records the call and returns a finished stub."""
    return _NoSpawn(*args, **kwargs)


subprocess.run = _record_run
subprocess.Popen = _NoSpawn
subprocess.check_call = _record_run
subprocess.check_output = lambda *a, **k: _NoSpawn(*a, **k).stdout

# Keep every fixture write inside temporary directories: the mask registry and
# the clip index both default to paths under inprocess/.
_REGISTRY_DIR = tempfile.mkdtemp(prefix="review_lock_registry_")
_INDEX_DIR = tempfile.mkdtemp(prefix="review_lock_index_")
# Remove both on the way out, however the run ends. Without this every board run
# leaves two directories behind in the system temp folder (84 of them had piled
# up by 2026-09-06), which the house rule on test artifacts forbids.
atexit.register(shutil.rmtree, _REGISTRY_DIR, ignore_errors=True)
atexit.register(shutil.rmtree, _INDEX_DIR, ignore_errors=True)
os.environ.setdefault("TCRMP_MASK_REGISTRY_DIR", _REGISTRY_DIR)
os.environ.setdefault("TCRMP_CLIP_INDEX_DIR", _INDEX_DIR)
os.environ.setdefault("TCRMP_TARGET_SPECIES", "PA,OFRA")
os.environ.setdefault("TCRMP_MANUAL_ANNOTATE", "1")
os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")
os.environ.setdefault("VICARIUS_HOST_OPEN", "0")

import app as A  # noqa: E402
from mask_ops import (  # noqa: E402
    REVIEW_LOCKED_MSG, REVIEW_TAGGING_ENABLED, review_tag_requested,
    review_tags_dropped, rle_encode,
)

# Frame size and the size of a fake mask. MIN_MASK_AREA_PX is 500, so every
# fake mask is comfortably above it and survives build_mask_dict.
IMG_H, IMG_W = 200, 260
HALF_EDGE_PX = 20

# Every spelling of a review tag a request body can carry, as the map's
# "exact spellings the refusal must catch" table lists them.
REVIEW_SPELLINGS = (
    ("review boolean True", {"review": True}),
    ("review integer 1", {"review": 1}),
    ("review string 'no' (truthy under bool())", {"review": "no"}),
    ("species REVIEW", {"species": "REVIEW"}),
    ("species ' review ' (case and whitespace)", {"species": " review "}),
    ("species_code REVIEW with an empty species",
     {"species": "", "species_code": "REVIEW"}),
    ("category Review with no species and no boolean", {"category": "Review"}),
)

_fail = 0


def check(cond, msg):
    """Record one assertion: print PASS or FAIL and count the failures."""
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


class FakeEngine:
    """Segmentation engine stand-in: a square mask around whatever it is asked for."""

    _current_image_path = "x"

    def set_image(self, path):
        """Report the fixture frame size; no pixels are read."""
        return (IMG_W, IMG_H)

    def segment_point(self, x, y, label=1):
        """A square mask centred on the click, always above the minimum area."""
        m = np.zeros((IMG_H, IMG_W), dtype=bool)
        y0, y1 = max(0, int(y) - HALF_EDGE_PX), min(IMG_H, int(y) + HALF_EDGE_PX)
        x0, x1 = max(0, int(x) - HALF_EDGE_PX), min(IMG_W, int(x) + HALF_EDGE_PX)
        m[y0:y1, x0:x1] = True
        return {"mask": m, "score": 0.9, "bbox": None}

    def segment_box(self, x0, y0, x1, y1):
        """A rectangular mask filling the requested box."""
        m = np.zeros((IMG_H, IMG_W), dtype=bool)
        m[int(y0):int(y1), int(x0):int(x1)] = True
        return {"mask": m, "score": 0.9, "bbox": None}


_TMP = tempfile.mkdtemp(prefix="review_lock_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_IMG_PATH = os.path.join(_TMP, "FRAME.jpg")
Image.new("RGB", (IMG_W, IMG_H), (10, 20, 30)).save(_IMG_PATH)
_EXPORT_DIR = os.path.join(_TMP, "out")
os.makedirs(_EXPORT_DIR, exist_ok=True)

A.app.config["TESTING"] = True
A.session["sam_engine"] = FakeEngine()
A.session["export_dir"] = _EXPORT_DIR
A.session["reviewer"] = "LO"
A.session["segmentations"] = {}
A._ensure_image_embedded = lambda fn: True
# The overlay renderer writes PNGs beside the export; the lock has nothing to
# do with pixels, so it is stubbed out to keep the suite fast and file-free.
A.render_segmentation_overlays = lambda *a, **k: None
# The frame read route kicks a background embedding warm-up on the real engine;
# the fake one holds no prompt lock, so the warm-up is stubbed out.
A._prewarm_embeddings_async = lambda fn: None

_frame_seq = 0


def _frame(masks=None):
    """Register a fresh fixture frame and return (filename, seg dict).

    Each call mints its own filename so one case can never see another's masks.
    """
    global _frame_seq
    _frame_seq += 1
    filename = f"F{_frame_seq}.jpg"
    seg = {
        "image_path": filename,
        "image_path_abs": _IMG_PATH,
        "image_width": IMG_W,
        "image_height": IMG_H,
        "masks": list(masks or []),
        "processed_at": "",
        "reviewed": False,
        "exported": False,
    }
    A.session["segmentations"][filename] = seg
    return filename, seg


def _flagged_mask(mask_id=0, status="accepted"):
    """A mask already carrying an expert-review tag, as the 130 on disk do."""
    binary = np.zeros((IMG_H, IMG_W), dtype=bool)
    binary[10:50, 10:50] = True
    return {
        "id": mask_id,
        "status": status,
        "species": "REVIEW",
        "name": "REVIEW",
        "category": "Review",
        "review": True,
        "rle": rle_encode(binary),
        "polygon_norm": [[0, 0, 0, 0, 0, 0]],
        "area": 1600,
    }


def _is_locked_refusal(response):
    """True when a response is exactly the shared refusal shape."""
    if response.status_code != 409:
        return False
    body = response.get_json() or {}
    return (body.get("ok") is False
            and body.get("review_locked") is True
            and body.get("error") == REVIEW_LOCKED_MSG)


# ── Refusals, one test per writer ───────────────────────────────────────────

def test_add_mask_refuses_every_review_spelling():
    """POST /add refuses each spelling and leaves the frame without a mask."""
    print("test_add_mask_refuses_every_review_spelling")
    client = A.app.test_client()
    for label, extra in REVIEW_SPELLINGS:
        filename, seg = _frame()
        body = {"click_x": 100, "click_y": 100, "label": "M"}
        body.update(extra)
        r = client.post(f"/api/image/{filename}/add", json=body)
        check(_is_locked_refusal(r),
              f"/add refuses {label} with the shared 409 (got {r.status_code} {r.get_json()!r})")
        check(seg["masks"] == [],
              f"/add created no mask for {label} (got {len(seg['masks'])})")


def test_add_mask_box_refuses_every_review_spelling():
    """POST /add_box refuses each spelling and leaves the frame without a mask."""
    print("test_add_mask_box_refuses_every_review_spelling")
    client = A.app.test_client()
    for label, extra in REVIEW_SPELLINGS:
        filename, seg = _frame()
        body = {"box": [60, 60, 120, 110], "label": "B"}
        body.update(extra)
        r = client.post(f"/api/image/{filename}/add_box", json=body)
        check(_is_locked_refusal(r),
              f"/add_box refuses {label} with the shared 409 (got {r.status_code} {r.get_json()!r})")
        check(seg["masks"] == [],
              f"/add_box created no mask for {label} (got {len(seg['masks'])})")


def test_relabel_refuses_adding_a_tag():
    """PUT /masks refuses a relabel that asks for a review tag, in any spelling."""
    print("test_relabel_refuses_adding_a_tag")
    client = A.app.test_client()
    for label, extra in REVIEW_SPELLINGS:
        filename, seg = _frame([{"id": 0, "status": "pending", "species": "PA"}])
        r = client.put(f"/api/image/{filename}/masks",
                       json={"relabel": {"0": dict(extra)}})
        check(_is_locked_refusal(r),
              f"PUT /masks refuses {label} with the shared 409 (got {r.status_code} {r.get_json()!r})")
        mask = seg["masks"][0]
        check(mask["species"] == "PA" and not mask.get("review"),
              f"the mask kept its species and gained no flag for {label} (got {mask!r})")


def test_relabel_still_clears_an_existing_tag():
    """An explicit review false is a clear, not an add, and still goes through."""
    print("test_relabel_still_clears_an_existing_tag")
    client = A.app.test_client()
    filename, seg = _frame([_flagged_mask(0, status="pending")])
    r = client.put(f"/api/image/{filename}/masks",
                   json={"relabel": {"0": {"review": False, "species": "PA",
                                           "category": "Coral", "name": "Porites astreoides"}}})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"PUT /masks clearing a tag answers 200 ok (got {r.status_code} {body!r})")
    mask = seg["masks"][0]
    check(mask.get("review") is False,
          f"the review flag is cleared (got {mask.get('review')!r})")
    check(mask.get("species") == "PA",
          f"the species is replaced by the real code (got {mask.get('species')!r})")
    check(not A._is_review(mask),
          "the cleared mask no longer reads as a review mask")


def test_relabel_refuses_the_whole_body_when_one_entry_asks():
    """One offending entry refuses the whole body; nothing in it is applied."""
    print("test_relabel_refuses_the_whole_body_when_one_entry_asks")
    client = A.app.test_client()
    filename, seg = _frame([
        {"id": 0, "status": "pending", "species": "PA"},
        {"id": 1, "status": "pending", "species": "OFRA"},
        {"id": 2, "status": "pending", "species": "PA"},
    ])
    r = client.put(f"/api/image/{filename}/masks", json={
        "updates": {"0": "accepted"},
        "relabel": {"0": {"species": "OFRA"},
                    "1": {"review": True},
                    "2": {"species": "review"}},
    })
    body = r.get_json() or {}
    check(_is_locked_refusal(r),
          f"the whole body is refused with the shared 409 (got {r.status_code} {body!r})")
    check(body.get("review_locked_ids") == [1, 2],
          f"the refusal names the offending mask ids (got {body.get('review_locked_ids')!r})")
    check("2" in str(body.get("detail", "")),
          f"the refusal says how many entries asked for a tag (got {body.get('detail')!r})")
    by_id = {m["id"]: m for m in seg["masks"]}
    check(by_id[0]["species"] == "PA" and by_id[0]["status"] == "pending",
          f"the innocent entry was not applied either (got {by_id[0]!r})")
    check(not any(m.get("review") for m in seg["masks"]),
          "no mask in the frame gained a review flag")


def test_add_refuses_a_category_only_payload():
    """A body whose only review spelling is the category creates nothing.

    Both add routes read the category straight out of the body into point_info,
    so before the category joined the spellings the refusal catches, a body of
    {"category": "Review"} passed both guards and the category rode into the
    stored mask verbatim: the same tag, spelled the third way.
    """
    print("test_add_refuses_a_category_only_payload")
    client = A.app.test_client()
    for spelling in ("Review", "REVIEW", " review ", "ReViEw"):
        fn_add, seg_add = _frame()
        r = client.post(f"/api/image/{fn_add}/add",
                        json={"click_x": 100, "click_y": 100, "label": "M",
                              "category": spelling})
        check(_is_locked_refusal(r),
              f"/add refuses category={spelling!r} with the shared 409 "
              f"(got {r.status_code} {r.get_json()!r})")
        check(seg_add["masks"] == [],
              f"/add created no mask for category={spelling!r} "
              f"(got {len(seg_add['masks'])})")

        fn_box, seg_box = _frame()
        r = client.post(f"/api/image/{fn_box}/add_box",
                        json={"box": [60, 60, 120, 110], "label": "B",
                              "category": spelling})
        check(_is_locked_refusal(r),
              f"/add_box refuses category={spelling!r} with the shared 409 "
              f"(got {r.status_code} {r.get_json()!r})")
        check(seg_box["masks"] == [],
              f"/add_box created no mask for category={spelling!r} "
              f"(got {len(seg_box['masks'])})")

    fn_ok, seg_ok = _frame()
    r = client.post(f"/api/image/{fn_ok}/add",
                    json={"click_x": 100, "click_y": 100, "label": "M",
                          "species": "PA", "category": "Coral"})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"a real category still creates a mask (got {r.status_code} {body!r})")
    check(len(seg_ok["masks"]) == 1 and seg_ok["masks"][0]["category"] == "Coral",
          f"and it keeps that category (got {seg_ok['masks']!r})")


def test_relabel_refuses_a_category_only_payload():
    """PUT /masks refuses an entry whose only review spelling is the category.

    The relabel route wrote the category unconditionally, so this was the one
    spelling that reached an existing mask. The refusal runs before any field
    is applied, so a species named in the same entry is not written either.
    """
    print("test_relabel_refuses_a_category_only_payload")
    client = A.app.test_client()
    for spelling in ("Review", "REVIEW", " review ", "ReViEw"):
        filename, seg = _frame([{"id": 0, "status": "pending", "species": "PA",
                                 "category": "Coral"}])
        r = client.put(f"/api/image/{filename}/masks",
                       json={"relabel": {"0": {"category": spelling}}})
        body = r.get_json() or {}
        check(_is_locked_refusal(r),
              f"PUT /masks refuses category={spelling!r} with the shared 409 "
              f"(got {r.status_code} {body!r})")
        check(body.get("review_locked_ids") == [0],
              f"the refusal names the offending mask (got "
              f"{body.get('review_locked_ids')!r})")
        mask = seg["masks"][0]
        check(mask["category"] == "Coral",
              f"the mask kept its own category (got {mask['category']!r})")
        check(mask["species"] == "PA" and not mask.get("review"),
              f"and gained nothing else (got {mask!r})")

    filename, seg = _frame([{"id": 0, "status": "pending", "species": "PA",
                             "category": "Coral"}])
    r = client.put(f"/api/image/{filename}/masks",
                   json={"relabel": {"0": {"species": "OFRA", "category": "Review"}}})
    check(_is_locked_refusal(r),
          f"a species beside a Review category is refused whole (got {r.status_code})")
    check(seg["masks"][0]["species"] == "PA",
          f"the species in the same entry was not applied either (got "
          f"{seg['masks'][0]['species']!r})")


def test_scan_candidate_commit_strips_the_tag():
    """A scan candidate posted back with a tag is stored without one.

    Two shapes: a candidate whose only label was the tag has nothing left once
    the tag is stripped, so the existing empty-species label guard asks for a
    code; a candidate carrying a real species keeps it and loses the tag.
    """
    print("test_scan_candidate_commit_strips_the_tag")
    client = A.app.test_client()
    filename, seg = _frame([{"id": 0, "status": "accepted", "species": "PA"}])
    binary = np.zeros((IMG_H, IMG_W), dtype=bool)
    binary[100:140, 100:140] = True

    tag_only = {
        "species": "REVIEW", "name": "REVIEW", "category": "Review", "review": True,
        "rle": rle_encode(binary), "polygon_norm": [[0, 0, 0, 0, 0, 0]], "area": 1600,
    }
    before = json.dumps(tag_only, sort_keys=True)
    r = client.post(f"/api/image/{filename}/commit_mask", json={"mask": tag_only})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("needs_species") is True,
          f"a tag-only candidate falls into the label guard (got {r.status_code} {body!r})")
    check(len(seg["masks"]) == 1,
          f"the tag-only candidate was not stored (got {len(seg['masks'])} masks)")
    check(json.dumps(tag_only, sort_keys=True) == before,
          "the posted candidate itself was not mutated")

    counted_before = A.REVIEW_LOCK_COUNTS["candidates_stripped"]
    labelled = dict(tag_only, species="PA", name="Porites astreoides", category="Coral")
    r = client.post(f"/api/image/{filename}/commit_mask", json={"mask": labelled})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"a labelled candidate is still committed (got {r.status_code} {body!r})")
    stored = [m for m in seg["masks"] if m["id"] == 1]
    check(len(stored) == 1, f"exactly one new mask was appended (got {len(stored)})")
    if stored:
        mask = stored[0]
        check(mask.get("species") == "PA",
              f"the real species survives the strip (got {mask.get('species')!r})")
        check(not review_tag_requested(mask),
              f"the stored mask carries no review tag in any spelling (got "
              f"review={mask.get('review')!r})")
        check(not A._is_review(mask),
              "the stored mask does not read as a review mask")
    check(A.REVIEW_LOCK_COUNTS["candidates_stripped"] == counted_before + 1,
          f"the strip was counted (got "
          f"{A.REVIEW_LOCK_COUNTS['candidates_stripped'] - counted_before})")


def test_scan_candidate_commit_drops_client_supplied_expert_identifiers():
    """A candidate cannot bring the expert round trip's own identifiers with it.

    review_uid is the key the expert importer joins on and the key that makes a
    mask read-only in Edit Masks, so a freshly scanned candidate claiming one it
    never earned would corrupt that round trip.
    """
    print("test_scan_candidate_commit_drops_client_supplied_expert_identifiers")
    client = A.app.test_client()
    filename, seg = _frame([{"id": 0, "status": "accepted", "species": "PA"}])
    binary = np.zeros((IMG_H, IMG_W), dtype=bool)
    binary[60:100, 60:100] = True
    candidate = {
        "species": "PA", "name": "Porites astreoides", "category": "Coral",
        "review_uid": "RU-forged", "reviews": [{"code": "OFRA"}],
        "expert_id": {"mode": "EXPERT", "code": "OFRA"},
        "rle": rle_encode(binary), "polygon_norm": [[0, 0, 0, 0, 0, 0]], "area": 1600,
    }
    r = client.post(f"/api/image/{filename}/commit_mask", json={"mask": candidate})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"the candidate is still committed (got {r.status_code} {body!r})")
    stored = [m for m in seg["masks"] if m["id"] == 1]
    check(len(stored) == 1, f"exactly one new mask was appended (got {len(stored)})")
    if stored:
        mask = stored[0]
        check(mask.get("species") == "PA",
              f"the real species survives (got {mask.get('species')!r})")
        for key in A.CLIENT_REVIEW_PROVENANCE_KEYS:
            check(key not in mask,
                  f"the stored mask carries no client-supplied {key} (got {mask.get(key)!r})")


class _SlowReadCounts(dict):
    """The lock tally with a pause on every read, to force an interleave.

    A read-modify-write race is otherwise a matter of luck: CPython rarely
    switches threads between the read and the write of one dict entry. Widening
    that window turns the race into a certainty, so an unguarded increment
    always loses one of two concurrent commits and a guarded one never does.
    """

    PAUSE_SECONDS = 0.05

    def __getitem__(self, key):
        """Return the tally after a pause long enough for a thread switch."""
        value = super().__getitem__(key)
        time.sleep(self.PAUSE_SECONDS)
        return value


def test_candidate_counter_survives_concurrent_commits():
    """Two commits at once both reach the stripped-candidate tally.

    The annotator serves requests on several threads, so two scan candidates
    can be kept at the same moment: the operator in the browser and the Refine
    loop, or two browsers on the same run. The commit helper is where a kept
    candidate is stripped and counted, so it is called here directly, on its
    own scratch frame, with no provenance write in the way. That the route
    reaches this helper at all is proved by
    test_scan_candidate_commit_strips_the_tag above.
    """
    print("test_candidate_counter_survives_concurrent_commits")
    binary = np.zeros((IMG_H, IMG_W), dtype=bool)
    binary[100:140, 100:140] = True
    seg = {"masks": []}
    candidate = {
        "species": "PA", "name": "Porites astreoides", "category": "Review",
        "rle": rle_encode(binary), "polygon_norm": [[0, 0, 0, 0, 0, 0]],
        "area": 1600,
    }

    real_counts = A.REVIEW_LOCK_COUNTS
    A.REVIEW_LOCK_COUNTS = _SlowReadCounts(candidates_stripped=0)
    errors = []

    def commit_one():
        """Commit one tagged candidate, recording any exception for the caller."""
        try:
            A._commit_candidate_to_seg(seg, dict(candidate))
        except Exception as exc:                       # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=commit_one) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    counted = dict.__getitem__(A.REVIEW_LOCK_COUNTS, "candidates_stripped")
    A.REVIEW_LOCK_COUNTS = real_counts

    check(not errors, f"neither commit raised (got {errors!r})")
    check(not any(t.is_alive() for t in threads),
          "both commits finished: the tally lock is not held across a commit")
    check(counted == 2,
          f"both concurrent commits were counted, none lost to the other "
          f"(expected 2, got {counted})")
    check(len(seg["masks"]) == 2,
          f"both candidates were committed (got {len(seg['masks'])})")
    for mask in seg["masks"]:
        check(mask.get("species") == "PA",
              f"each committed mask keeps its real species (got "
              f"{mask.get('species')!r})")
        check(not review_tag_requested(mask),
              f"and carries no review tag in any spelling (got "
              f"category={mask.get('category')!r})")


def test_batch_segmentation_never_writes_the_flag():
    """The batch pass drops a review-flagged prompt's tag and counts the drop.

    The count comes from mask_ops, which is where a tag dropped while a mask is
    built is recorded, so the batch pass reports the same number every other
    caller of that counter reports.
    """
    print("test_batch_segmentation_never_writes_the_flag")
    before = review_tags_dropped()
    item = {
        "filename": "BATCH.jpg",
        "image_path": _IMG_PATH,
        "points": [
            {"x": 60, "y": 60, "species": "REVIEW", "review": True, "label": "R"},
            {"x": 200, "y": 160, "species": "PA", "label": "M"},
        ],
    }
    filename, n_masks, n_points = A._process_single_image(
        FakeEngine(), item, _EXPORT_DIR)
    seg = A.session["segmentations"]["BATCH.jpg"]
    check(n_points == 2, f"both prompts were read (got {n_points})")
    check(n_masks >= 1, f"the pass still produced masks (got {n_masks})")
    check(not any(m.get("review") for m in seg["masks"]),
          f"no mask was born with the review boolean (got {[m.get('review') for m in seg['masks']]})")
    check(not any(str(m.get("species", "")).strip().upper() == "REVIEW"
                  for m in seg["masks"]),
          f"no mask was born with the REVIEW species (got {[m.get('species') for m in seg['masks']]})")
    check(review_tags_dropped() == before + 1,
          f"the shared counter recorded the one dropped tag (got "
          f"{review_tags_dropped() - before})")


def test_refusal_answers_409_with_the_shared_message():
    """Every refusing route answers one shape the browser can special-case once."""
    print("test_refusal_answers_409_with_the_shared_message")
    client = A.app.test_client()
    fn_add, _ = _frame()
    fn_box, _ = _frame()
    fn_rel, _ = _frame([{"id": 0, "status": "pending", "species": "PA"}])
    responses = [
        ("/add", client.post(f"/api/image/{fn_add}/add",
                             json={"click_x": 100, "click_y": 100, "review": True})),
        ("/add_box", client.post(f"/api/image/{fn_box}/add_box",
                                 json={"box": [60, 60, 120, 110], "species": "REVIEW"})),
        ("PUT /masks", client.put(f"/api/image/{fn_rel}/masks",
                                  json={"relabel": {"0": {"review": True}}})),
    ]
    for name, r in responses:
        body = r.get_json() or {}
        check(r.status_code == 409, f"{name} answers HTTP 409 (got {r.status_code})")
        check(body.get("ok") is False, f"{name} answers ok false (got {body.get('ok')!r})")
        check(body.get("review_locked") is True,
              f"{name} answers review_locked true (got {body.get('review_locked')!r})")
        check(body.get("error") == REVIEW_LOCKED_MSG,
              f"{name} carries the shared message (got {body.get('error')!r})")


def test_ordinary_labelling_is_untouched():
    """A normal species code still creates, relabels and accepts a mask."""
    print("test_ordinary_labelling_is_untouched")
    client = A.app.test_client()
    filename, seg = _frame()
    r = client.post(f"/api/image/{filename}/add",
                    json={"click_x": 100, "click_y": 100, "species": "PA",
                          "category": "Coral", "label": "M"})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"/add still creates an ordinary mask (got {r.status_code} {body!r})")
    check(body.get("mask", {}).get("species") == "PA",
          f"the new mask carries its species (got {body.get('mask', {}).get('species')!r})")

    r = client.post(f"/api/image/{filename}/add_box",
                    json={"box": [150, 120, 210, 180], "species": "OFRA", "label": "B",
                          "rejected_ids": []})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("mask", {}).get("species") == "OFRA",
          f"/add_box still creates an ordinary mask (got {r.status_code} {body!r})")

    r = client.put(f"/api/image/{filename}/masks",
                   json={"relabel": {"0": {"species": "OFRA", "category": "Coral"}},
                         "updates": {"0": "accepted"}})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"PUT /masks still relabels and accepts (got {r.status_code} {body!r})")
    by_id = {m["id"]: m for m in seg["masks"]}
    check(by_id[0]["species"] == "OFRA" and by_id[0]["status"] == "accepted",
          f"the relabel and the accept both applied (got {by_id[0]!r})")


# ── Preservation, proving the tags already on disk still work ───────────────

def test_existing_flagged_mask_still_reads_back():
    """A mask flagged before the lock still reads, colours and counts as one."""
    print("test_existing_flagged_mask_still_reads_back")
    client = A.app.test_client()
    filename, seg = _frame([_flagged_mask(0, status="pending"),
                            {"id": 1, "status": "pending", "species": "PA"}])
    flagged = seg["masks"][0]

    check(A._is_review(flagged), "the read predicate still sees the flagged mask")
    check(A._needs_species(flagged) is False,
          "the flagged mask still counts as labelled, not as missing a species")

    r = client.get(f"/api/image/{filename}")
    wire = (r.get_json() or {}).get("masks", [])
    wire_flagged = [m for m in wire if m.get("id") == 0]
    check(len(wire_flagged) == 1, f"the flagged mask still reaches the page (got {wire!r})")
    if wire_flagged:
        check(wire_flagged[0].get("species") == "REVIEW"
              and wire_flagged[0].get("review") is True,
              f"the amber colour state travels unchanged (got {wire_flagged[0]!r})")

    r = client.put(f"/api/image/{filename}/masks", json={"updates": {"0": "accepted"}})
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"the flagged mask still accepts (got {r.status_code} {body!r})")
    check(seg["masks"][0]["status"] == "accepted",
          f"the flagged mask is accepted (got {seg['masks'][0]['status']!r})")
    check(0 not in body.get("needs_species", []),
          f"the flagged mask is absent from needs_species (got {body.get('needs_species')!r})")

    counted = A._count_review_masks({filename: seg}, [filename])
    check(counted == 1, f"the review counter still sees exactly one flagged mask (got {counted})")


def test_disk_read_back_still_restores_an_existing_tag():
    """The deliberate exemption: a read-back from disk still restores a tag."""
    print("test_disk_read_back_still_restores_an_existing_tag")
    in_memory = [{"id": 0, "status": "accepted", "species": "PA"},
                 {"id": 1, "status": "pending", "species": "", "review_uid": "RU-1"}]
    on_disk = [
        {"id": 0, "status": "accepted", "species": "PA"},
        {"id": 1, "status": "pending", "species": "REVIEW", "review": True,
         "review_uid": "RU-1", "reviews": [{"code": "PA", "by": "expert"}]},
    ]
    merged, expert = A._merge_imported_review_fields(in_memory, on_disk)
    check(merged >= 1, f"the read-back merged at least one mask (got {merged})")
    check(in_memory[1].get("review") is True,
          f"the existing tag is restored from disk (got {in_memory[1].get('review')!r})")
    check(in_memory[1].get("reviews"),
          f"the review history is restored too (got {in_memory[1].get('reviews')!r})")

    expert_memory = [{"id": 5, "status": "accepted", "species": "REVIEW",
                      "review": True, "review_uid": "RU-2"}]
    expert_disk = [{"id": 5, "status": "accepted", "species": "PA", "name": "Porites astreoides",
                    "category": "Coral", "review": False, "review_uid": "RU-2",
                    "expert_id": {"mode": "EXPERT", "code": "PA"}}]
    merged2, expert2 = A._merge_imported_review_fields(expert_memory, expert_disk)
    check(expert2 == 1, f"an expert identification is still counted (got {expert2})")
    check(expert_memory[0].get("species") == "PA"
          and expert_memory[0].get("review") is False,
          f"an expert identification still clears the flag (got {expert_memory[0]!r})")


def test_export_bundle_is_suspended_and_reports_a_count():
    """The outward-publishing export returns early with the flagged-mask count."""
    print("test_export_bundle_is_suspended_and_reports_a_count")
    fn_a, seg_a = _frame([_flagged_mask(0), {"id": 1, "status": "accepted", "species": "PA"}])
    fn_b, seg_b = _frame([_flagged_mask(0), _flagged_mask(1)])

    fired = []

    class _Tripwire:
        """Stands where the real exporter stands: firing means the lock leaked."""

        @staticmethod
        def export_flagged_masks(*args, **kwargs):
            fired.append(True)
            return {}

    prior_ok, prior_export = A._REVIEW_OK, A._review_export
    A._REVIEW_OK, A._review_export = True, _Tripwire
    try:
        stats = A._export_review_bundle({fn_a: seg_a, fn_b: seg_b}, [fn_a, fn_b])
    finally:
        A._REVIEW_OK, A._review_export = prior_ok, prior_export

    check(stats.get("disabled") is True,
          f"the export reports itself disabled (got {stats!r})")
    check(stats.get("review_masks_skipped") == 3,
          f"the export counts the three flagged masks it did not publish (got {stats!r})")
    check(not fired, "nothing was published: the real exporter was never called")


# ── the standalone label list and the target-species environment ────────────
# A standalone session hands its label list to every frame it opens, and the
# label list is what the quick-add row is drawn from. A label of REVIEW there
# put a live, armable REVIEW button beside the greyed locked one, so the page
# offered a control the server would only answer with a 409.
_STANDALONE_ROOT = tempfile.mkdtemp(prefix="review_lock_standalone_")
atexit.register(shutil.rmtree, _STANDALONE_ROOT, ignore_errors=True)
_MASTER_CODES = os.path.join(_STANDALONE_ROOT, "master_codes.csv")
with open(_MASTER_CODES, "w") as _fh:
    _fh.write("species_code,species_name,category\n"
              "PA,Porites astreoides,Coral\n"
              "MC,Montastraea cavernosa,Coral\n")


def _standalone_start(labels, session_name):
    """POST /api/standalone/start with these labels, leaving the suite intact.

    The route rewrites the module-level session dict wholesale (export dir,
    label list, queue, segmentations), so the dict is snapshotted and put back:
    every other case in this suite reads the fixture frames it holds. The three
    config values the route reads are swapped for temporary ones the same way.

    Args:
        labels: the label list to submit, as the browser sends it.
        session_name: the standalone session directory name.

    Returns:
        The Flask test-client response.
    """
    snapshot = dict(A.session)
    prior = (A.cfg.SESSION_MODE, A.cfg.STANDALONE_ROOT, A.cfg.MASTER_CODES_CSV)
    A.cfg.SESSION_MODE = "standalone"
    A.cfg.STANDALONE_ROOT = _STANDALONE_ROOT
    A.cfg.MASTER_CODES_CSV = _MASTER_CODES
    try:
        return A.app.test_client().post(
            "/api/standalone/start",
            data={"session": session_name, "reviewer": "LO",
                  "labels": ",".join(labels)},
            content_type="multipart/form-data")
    finally:
        (A.cfg.SESSION_MODE, A.cfg.STANDALONE_ROOT,
         A.cfg.MASTER_CODES_CSV) = prior
        A.session.clear()
        A.session.update(snapshot)


def test_standalone_start_refuses_a_review_label():
    """A standalone session cannot start with REVIEW anywhere in its labels."""
    print("test_standalone_start_refuses_a_review_label")
    for i, spelling in enumerate(("REVIEW", "review", "Review", " ReViEw ")):
        name = f"lock-refuse-{i}"
        r = _standalone_start(["PA", spelling], name)
        body = r.get_json() or {}
        check(_is_locked_refusal(r),
              f"labels ['PA', {spelling!r}] answer the shared 409 refusal "
              f"(got {r.status_code} {body!r})")
        check(body.get("review_locked_labels") == [spelling.strip()],
              f"the refusal names the label it refused (got "
              f"{body.get('review_locked_labels')!r})")
        check(not os.path.isdir(os.path.join(_STANDALONE_ROOT, name)),
              f"the refused start created no session directory for {name}")


def test_standalone_start_accepts_ordinary_labels():
    """An ordinary label list still starts a standalone session."""
    print("test_standalone_start_accepts_ordinary_labels")
    r = _standalone_start(["PA", "MC"], "lock-accept")
    body = r.get_json() or {}
    check(r.status_code == 200 and body.get("ok") is True,
          f"an ordinary label list still starts a session (got "
          f"{r.status_code} {body!r})")
    check(body.get("labels") == ["PA", "MC"],
          f"both labels are kept (got {body.get('labels')!r})")
    check(os.path.isdir(os.path.join(_STANDALONE_ROOT, "lock-accept")),
          "the session directory was created")


def test_target_species_env_drops_a_review_entry():
    """A REVIEW entry in TCRMP_TARGET_SPECIES is dropped, the rest survive."""
    print("test_target_species_env_drops_a_review_entry")
    prior = os.environ.get("TCRMP_TARGET_SPECIES")
    said = io.StringIO()
    try:
        os.environ["TCRMP_TARGET_SPECIES"] = "PA, review ,OFRA,Review"
        with contextlib.redirect_stdout(said):
            codes = A._load_target_species()
    finally:
        if prior is None:
            os.environ.pop("TCRMP_TARGET_SPECIES", None)
        else:
            os.environ["TCRMP_TARGET_SPECIES"] = prior

    check(codes == ["PA", "OFRA"],
          f"both REVIEW spellings are dropped and the real codes survive "
          f"(got {codes!r})")
    logged = said.getvalue()
    check("[review-lock]" in logged,
          f"the drop is logged under the lock's own tag (got {logged!r})")
    check("review" in logged and "Review" in logged,
          f"the log names both dropped entries (got {logged!r})")

    # An environment with no REVIEW entry is passed through untouched and
    # silently: the drop must not become a line on every launch.
    quiet = io.StringIO()
    try:
        os.environ["TCRMP_TARGET_SPECIES"] = "PA,OFRA"
        with contextlib.redirect_stdout(quiet):
            plain = A._load_target_species()
    finally:
        if prior is None:
            os.environ.pop("TCRMP_TARGET_SPECIES", None)
        else:
            os.environ["TCRMP_TARGET_SPECIES"] = prior
    check(plain == ["PA", "OFRA"],
          f"an ordinary environment value is unchanged (got {plain!r})")
    check(quiet.getvalue() == "",
          f"and nothing is logged for it (got {quiet.getvalue()!r})")


def test_hostile_bodies_are_refused_without_raising():
    """A malformed body refuses or answers normally, never with a 500.

    These routes are served on 0.0.0.0 with no authentication, so a request can
    carry anything at all: a missing body, a relabel map that is not a map, an
    entry that is not a map, and a key that is not a mask id.
    """
    print("test_hostile_bodies_are_refused_without_raising")
    client = A.app.test_client()

    filename, seg = _frame([{"id": 0, "status": "pending", "species": "PA"}])
    r = client.put(f"/api/image/{filename}/masks", json={})
    check(r.status_code == 200, f"an empty body answers 200 (got {r.status_code})")

    r = client.put(f"/api/image/{filename}/masks", json={"relabel": "REVIEW"})
    check(r.status_code == 200,
          f"a relabel map that is a string is ignored, not a 500 (got {r.status_code})")

    r = client.put(f"/api/image/{filename}/masks", json={"relabel": {"0": "REVIEW"}})
    check(r.status_code == 200,
          f"a relabel entry that is a string is ignored, not a 500 (got {r.status_code})")
    check(seg["masks"][0]["species"] == "PA",
          f"and it changed nothing (got {seg['masks'][0]['species']!r})")

    r = client.put(f"/api/image/{filename}/masks",
                   json={"relabel": {"not-an-id": {"review": True}}})
    body = r.get_json() or {}
    check(_is_locked_refusal(r),
          f"a non-numeric key still refuses (got {r.status_code} {body!r})")
    check(body.get("review_locked_ids") == ["not-an-id"],
          f"the refusal reports the key it could not read as an id (got "
          f"{body.get('review_locked_ids')!r})")

    fn_add, seg_add = _frame()
    r = client.post(f"/api/image/{fn_add}/add", json=None,
                    content_type="application/json")
    check(r.status_code in (200, 400, 500) and r.status_code != 409,
          f"a bodyless /add is not mistaken for a review request (got {r.status_code})")
    check(seg_add["masks"] == [], "a bodyless /add created no mask")


def test_the_refusal_shape_cannot_be_overridden():
    """A caller cannot hand the browser a second shape to handle."""
    print("test_the_refusal_shape_cannot_be_overridden")
    raised = None
    try:
        A._review_locked_response(error="something else")
    except ValueError as exc:
        raised = exc
    check(isinstance(raised, ValueError),
          f"overriding a fixed field raises ValueError (got {raised!r})")
    check("error" in str(raised),
          f"the error names the field it refused to overwrite (got {str(raised)!r})")


def test_no_subprocess_was_spawned():
    """No case in this suite launched a process, so no live annotator was killed."""
    print("test_no_subprocess_was_spawned")
    check(_SPAWNS == [], f"no subprocess was spawned (got {_SPAWNS!r})")


if __name__ == "__main__":
    check(REVIEW_TAGGING_ENABLED is False,
          f"the lock is on: REVIEW_TAGGING_ENABLED is False (got {REVIEW_TAGGING_ENABLED!r})")
    test_add_mask_refuses_every_review_spelling()
    test_add_mask_box_refuses_every_review_spelling()
    test_add_refuses_a_category_only_payload()
    test_relabel_refuses_adding_a_tag()
    test_relabel_refuses_a_category_only_payload()
    test_relabel_still_clears_an_existing_tag()
    test_relabel_refuses_the_whole_body_when_one_entry_asks()
    test_scan_candidate_commit_strips_the_tag()
    test_scan_candidate_commit_drops_client_supplied_expert_identifiers()
    test_candidate_counter_survives_concurrent_commits()
    test_batch_segmentation_never_writes_the_flag()
    test_refusal_answers_409_with_the_shared_message()
    test_ordinary_labelling_is_untouched()
    test_existing_flagged_mask_still_reads_back()
    test_disk_read_back_still_restores_an_existing_tag()
    test_export_bundle_is_suspended_and_reports_a_count()
    test_standalone_start_refuses_a_review_label()
    test_standalone_start_accepts_ordinary_labels()
    test_target_species_env_drops_a_review_entry()
    test_hostile_bodies_are_refused_without_raising()
    test_the_refusal_shape_cannot_be_overridden()
    test_no_subprocess_was_spawned()
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
