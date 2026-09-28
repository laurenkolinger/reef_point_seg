"""
Annotate Images (combinedAnnotate)

SAM3 segmentation plus review, in one app. Takes sam_click_prompts.json and
raw images from TCRMPclip_placePoints, runs SAM3 segmentation on each point,
and serves the annotation UI where the operator accepts, rejects, refines,
adds and labels masks before exporting to YOLO Segmentation format. The UI
calls this tool "Annotate Images" everywhere the operator can see it.

Usage:
    python app.py [--port 5070]
"""

import atexit
import functools
import hashlib
import os
import re
import signal
import sys
import json
import argparse
import subprocess
import tempfile
import threading
import numpy as np
import cv2
from datetime import datetime, timedelta, timezone

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from mask_ops import (
    rle_encode, rle_decode, mask_to_polygon, mask_bbox, mask_centroid,
    resolve_overlaps, merge_close_masks, check_no_overlap, union_many,
    build_mask_dict, update_mask_geometry, merge_overlapping_same_id,
    REVIEW_TAGGING_ENABLED, REVIEW_LOCKED_MSG, review_tag_requested,
    strip_review_tag,
)
from export_yolo import load_class_map, save_class_map, export_batch, update_data_yaml
from render_overlay import render_segmentation_overlays
from provenance import compute_label_outcomes, write_provenance_csv
import custom_imports as ci

app = Flask(__name__, template_folder='templates')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)         # this is actually the scripts/ dir

# Shared expert-review helpers live in scripts/_reefreview/. Import is guarded
# so a problem there can never stop the core segmentation app from running.
sys.path.insert(0, REPO_DIR)
try:
    from _reefreview import review_export as _review_export
    _REVIEW_OK = True
except Exception as _re_err:  # pragma: no cover
    _review_export = None
    _REVIEW_OK = False
    print(f"[review] _reefreview unavailable, expert-review export disabled: {_re_err}")

# Where the label list is cleaned of the expert-review tag (Lauren Olinger,
# 2026-09-06). A target label of REVIEW draws a live, armable REVIEW button in
# the quick-add row beside the greyed locked one, which offers the operator a
# control the server answers only with a 409. The environment value is DROPPED
# rather than refused, because TCRMP_TARGET_SPECIES comes from a launcher and
# refusing a whole launch over one entry would cost the operator the session;
# the standalone form, where a person types the list, refuses instead and says
# why. Logged with print(): log() is not defined yet when this runs at import.
def _drop_review_labels(labels, source):
    """Remove every label that spells out the expert-review tag.

    Parameters:
        labels: the label codes read from one source, in their original order.
        source: plain words naming where they came from, for the log line.

    Returns:
        list: the labels that are not the review tag, in their original order.
        The list comes back unchanged when review tagging is enabled again, so
        flipping REVIEW_TAGGING_ENABLED reverses this with no other edit.

    Example:
        _drop_review_labels(['PA', 'review'], 'TCRMP_TARGET_SPECIES') returns
        ['PA'] and prints one line naming 'review'.
    """
    if REVIEW_TAGGING_ENABLED:
        return list(labels)
    kept, dropped = [], []
    for label in labels:
        if review_tag_requested({'species': label}):
            dropped.append(label)
        else:
            kept.append(label)
    if dropped:
        print(f"[review-lock] dropped {len(dropped)} target label(s) spelling "
              f"out the expert-review tag from {source}: {', '.join(dropped)}. "
              f"{REVIEW_LOCKED_MSG}")
    return kept


def _load_target_species():
    """The target species codes this annotator offers, in priority order.

    Reads the orchestrator override TCRMP_TARGET_SPECIES first, then the
    TARGET_SPECIES line of TCRMPcvr_chooseImages' config.py, then a built-in
    default list. Every source is passed through _drop_review_labels, so no
    source can arm the suspended expert-review flag.

    Returns:
        list: the species codes, in the order the source gave them.
    """
    import re
    # Pipeline orchestrator override
    env_sp = os.environ.get('TCRMP_TARGET_SPECIES', '')
    if env_sp:
        return _drop_review_labels(
            [s.strip() for s in env_sp.split(',') if s.strip()],
            'TCRMP_TARGET_SPECIES')
    config_path = os.path.join(REPO_DIR, 'TCRMPcvr_chooseImages', 'src', 'config.py')
    try:
        with open(config_path) as f:
            for line in f:
                if line.strip().startswith('TARGET_SPECIES'):
                    match = re.search(r'\[(.+)\]', line)
                    if match:
                        return _drop_review_labels(
                            [s.strip().strip('"').strip("'")
                             for s in match.group(1).split(',')],
                            config_path)
    except FileNotFoundError:
        pass
    return _drop_review_labels(
        ['OFRA', 'PA', 'OA', 'OFAV', 'AL', 'MC', 'AA'],
        'the built-in default target list')

TARGET_SPECIES = _load_target_species()


def _is_review(p):
    """A REVIEW-flagged prompt/mask: segmented as its OWN positive mask (unlike
    a non-target '?' point, which is a negative refinement click), shipped to the
    expert-review site, and never trained."""
    return bool(p.get('review')) or p.get('species', p.get('species_code', '')) == 'REVIEW'


# ── Expert-review flagging lock ──────────────────────────────────
# Lauren Olinger, 2026-09-06: no review tag may be added to a mask in any
# annotator. The policy itself (the switch, the message, the spellings a
# refusal has to catch) lives in mask_ops.py so every annotator app shares one
# copy, and flipping REVIEW_TAGGING_ENABLED to True there reverses the lock
# everywhere. What lives here is the HTTP shape the routes answer with and the
# tally of what the lock dropped.
#
# The refusal is deliberately broader than _is_review above, and the gap widened
# on 2026-09-06 when the category joined the spellings it catches. The refusal
# recognises FOUR spellings, each normalised for case and whitespace and each
# tested on its own: the review key, species, species_code and category. The
# read predicate recognises TWO: a truthy review key or a species of exactly
# REVIEW. That asymmetry is deliberate. What may be WRITTEN is judged as widely
# as the tag can be spelled; what the masks already on disk MEAN is judged by
# _is_review, which drives the expert export, the training filter, the coverage
# matrix and the provenance ledger, so widening it would change their meaning.

# HTTP 409 Conflict: the request is well formed and the server refuses it by
# policy. One status and one body shape, so the browser special-cases it once.
REVIEW_LOCKED_STATUS = 409

# What the lock has dropped that mask_ops cannot see. A tag dropped while a
# mask is BUILT is counted by mask_ops.review_tags_dropped(), which is the one
# number every caller reports; a scan candidate arrives already built, so its
# strip is counted here. Read by the log lines and by the tests; never
# persisted, so a restart starts the tally again.
REVIEW_LOCK_COUNTS = {'candidates_stripped': 0}

# The tally is written from request threads (keeping a scan candidate is a
# route), and this app serves several at once, so the increment runs under this
# lock rather than as a bare read-modify-write. Two commits arriving together
# would otherwise both read the same value and one increment would be lost.
# mask_ops guards its own dropped-tag counter the same way.
_review_lock_counts_lock = threading.Lock()


def _record_candidate_stripped():
    """Add one to the stripped-candidate tally, safely across threads."""
    with _review_lock_counts_lock:
        REVIEW_LOCK_COUNTS['candidates_stripped'] += 1

# Expert round-trip keys a scan candidate posted by the browser may not bring
# with it under the lock. The shared strip_review_tag covers the tag spellings;
# these three are the round trip's own identifiers, which a candidate minted by
# a scan has no way to have earned: review_uid is the join key the expert
# importer matches on and the key that makes a mask read-only in Edit Masks.
CLIENT_REVIEW_PROVENANCE_KEYS = ('review_uid', 'reviews', 'expert_id')


def _review_tag_refused(payload):
    """True when a request body asks to add an expert-review tag under the lock.

    Parameters:
        payload: a request body or mask dict. Anything that is not a dict is
            not a request for a tag and answers False rather than raising:
            this app binds 0.0.0.0 with no authentication, so a route reads
            whatever a caller sends.

    Returns:
        True when review tagging is locked AND the payload asks for a tag in
        any of its spellings. False when the lock is off, so flipping
        REVIEW_TAGGING_ENABLED restores the previous behaviour exactly.

    Example:
        _review_tag_refused({'species': ' review '})  ->  True
    """
    if REVIEW_TAGGING_ENABLED:
        return False
    if not isinstance(payload, dict):
        return False
    return review_tag_requested(payload)


def _review_locked_response(detail='', **extra):
    """The one refusal every locked route answers with.

    Parameters:
        detail: an optional second sentence naming what was refused (a count,
            for a request carrying several entries). Left out of the body when
            empty.
        extra: further JSON fields for the caller, such as the mask ids that
            asked for a tag.

    Returns:
        A (response, status) pair Flask returns as HTTP 409 carrying
        {'ok': False, 'review_locked': True, 'error': REVIEW_LOCKED_MSG}.

    Raises:
        ValueError when extra tries to overwrite one of those three fixed
        fields, which would leave the browser with two shapes to handle.
    """
    body = {'ok': False, 'review_locked': True, 'error': REVIEW_LOCKED_MSG}
    clash = sorted(set(extra) & set(body))
    if clash:
        raise ValueError(
            "_review_locked_response: " + ", ".join(clash) + " is fixed by the "
            "shared refusal shape and cannot be overridden by a caller")
    if detail:
        body['detail'] = str(detail)
    body.update(extra)
    return jsonify(body), REVIEW_LOCKED_STATUS


def _review_locked_candidate(cand):
    """A scan candidate as it will actually be stored under the lock.

    Parameters:
        cand: the mask dict the browser posted back for a kept scan candidate.

    Returns:
        The candidate itself when it carries no expert-review tag, otherwise a
        tag-free copy of it. The candidate is never modified. Used so the label
        guard judges what will be stored rather than what was posted: a
        candidate kept while REVIEW was armed is an unlabelled mask once the
        tag is stripped, and the operator is asked for a species code.
    """
    if _review_tag_refused(cand):
        return strip_review_tag(cand)
    return cand


def _review_locked_relabel_ids(relabel):
    """Mask ids in a relabel body whose entry asks to ADD an expert-review tag.

    Parameters:
        relabel: the relabel map from a PUT /masks body, {mask id: {field:
            value}}. A non-dict map, a non-dict entry and a key that is not a
            mask id are all tolerated rather than raising.

    Returns:
        A list of the offending ids, integers in numeric order first, then any
        key that is not an integer as a string. Empty when nothing asks for a
        tag, which is the ordinary case.
    """
    if not isinstance(relabel, dict):
        return []
    flagged = [key for key, info in relabel.items() if _review_tag_refused(info)]
    numeric, other = [], []
    for key in flagged:
        try:
            numeric.append(int(key))
        except (TypeError, ValueError):
            other.append(str(key))
    return sorted(numeric) + sorted(other)


def _load_target_label_info():
    """Return [{code, name, category}, ...] for each target label.

    Parallels the step 4 helper: prefers master_codes_recoded.csv from the
    project dir, falls back to scanning the recoded all_points.csv. Always
    emits one entry per code so the UI has a stable index for 1..N keys.
    """
    import csv as _csv
    lookup = {}

    input_dir = os.environ.get('TCRMP_INPUT_DIR', '') or cfg.INPUT_DIR
    project_dir = os.path.dirname(input_dir) if input_dir else ''
    candidate_dirs = []
    if project_dir:
        candidate_dirs.append(project_dir)
        candidate_dirs.append(os.path.join(project_dir, 'step2_recodeSpecies'))
    candidate_dirs.append(os.path.join(REPO_DIR, 'TCRMPcvr_recodeSpecies', 'output'))
    candidate_dirs.append(os.path.join(REPO_DIR, 'output'))

    master_candidates = []
    for d in candidate_dirs:
        master_candidates.append(os.path.join(d, 'master_codes_recoded.csv'))
        master_candidates.append(os.path.join(d, 'master_codes.csv'))

    for mc in master_candidates:
        if not os.path.exists(mc):
            continue
        try:
            with open(mc, 'r') as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    code = (row.get('species_code') or row.get('code') or '').strip()
                    if not code:
                        continue
                    lookup[code] = {
                        'name': (row.get('species_name') or row.get('name') or '').strip(),
                        'category': (row.get('category') or '').strip(),
                    }
            if lookup:
                print(f"[target-labels] Loaded {len(lookup)} codes from {os.path.basename(mc)}")
                break
        except Exception as e:
            print(f"[target-labels] Failed to read {mc}: {e}")

    return [{'code': c,
             'name': lookup.get(c, {}).get('name', ''),
             'category': lookup.get(c, {}).get('category', '')}
            for c in TARGET_SPECIES]


ALL_TARGET_LABELS = _load_target_label_info()


def _target_labels():
    """Per-session label list (standalone picks them at start) else the env/CSV list."""
    return session.get('target_labels') or ALL_TARGET_LABELS


def _target_species():
    """Per-session target codes (standalone picks them at start) else TARGET_SPECIES."""
    return [l['code'] for l in session['target_labels']] if session.get('target_labels') else TARGET_SPECIES


# ── Session state ────────────────────────────────────────────────

session = {
    'input_dir': '',
    'export_dir': '',
    'categories': [],
    # Per-session label list, set by a standalone start. Empty means "use the
    # env/CSV list" (see _target_labels / _target_species).
    'target_labels': [],
    'configured': False,
    'phase': 'startup',          # startup | processing | review
    'sam_engine': None,

    # Per-image segmentation data, flat dict
    'segmentations': {},  # flat {image_filename: seg_dict}

    # Processing queue
    'processing_queue': [],      # list of {image_path, filename, prompts}
    'processing_idx': 0,

    # Review state
    'review_files': [],
    'review_offset': 0,
    'review_batch_size': cfg.REVIEW_BATCH_SIZE,

    # Class registry for YOLO export
    'class_map': {},

    # Cross-project no-rescreening data (VEGA annotator tidy-up, 2026-09-27):
    # {stem: {'screened': {label: {...}}, 'masks': [{...}, ...]}} loaded from
    # <export_dir>/prior_screening.json by load_prior_screening(), called from
    # configure()/resume()/standalone_start(). Read-only: never written back
    # into session['segmentations'], never exported, never touches the dirty
    # counter. See _prior_screening_for and get_image_data.
    'prior_screening': {},

    # Currently-embedded image for refinement
    'current_review_image': None,

    # Operator initials for provenance attribution (best-effort).
    'reviewer': cfg.REVIEWER,
}


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)


# ── Persistence (flat JSON) ──────────────────────────────────────

def _seg_path(export_dir):
    return os.path.join(export_dir, 'segmentations', 'segmentations.json')


def _manifest_path(export_dir):
    return os.path.join(export_dir, 'segmentations', 'processed_manifest.json')


# Atlantic Standard Time, fixed UTC-4, no daylight saving (matches
# provenance.AST; kept separately here so this file has no import-order
# dependency on provenance.py for a filename stamp).
_AST = timezone(timedelta(hours=-4))

# Group-readable, matching every other file a run writes: tempfile.mkstemp
# creates its temp file 0o600 (owner-only), which would otherwise make a
# freshly-saved segmentations.json unreadable by a teammate's process running
# under a different primary group.
_SAVED_FILE_MODE = 0o664


def _ast_stamp():
    """Compact AST timestamp for filenames: YYYYMMDD-HHMMSS."""
    return datetime.now(_AST).strftime('%Y%m%d-%H%M%S')


# The (mtime_ns, size) of each session's segmentations.json as THIS process
# last read or wrote it, keyed by the file's path so opening a different
# project (a fresh fixture in a test, or a real operator switching projects
# without restarting the annotator) never compares against a stale identity
# left by a different file. Populated by load_segmentations() and refreshed
# by save_segmentations() on every successful write of the real file. A
# stat-only mismatch is not by itself proof of a real conflict (a touch, or a
# rewrite that reproduced the same bytes, moves mtime without changing
# content), so _seg_disk_hash below carries the content check that decides.
_seg_disk_identity = {}
_seg_disk_hash = {}


def _seg_file_identity(path):
    """(mtime_ns, size) for `path`, or None when it does not exist."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _hash_bytes(data_bytes):
    """sha256 hex digest of `data_bytes`."""
    return hashlib.sha256(data_bytes).hexdigest()


def _hash_file(path):
    """sha256 hex digest of the file at `path`, or None when it does not
    exist (a plain read, not the tracked identity: this always reflects the
    file's CURRENT bytes, checked against the identity save_segmentations()
    already believes it holds)."""
    try:
        with open(path, 'rb') as f:
            return _hash_bytes(f.read())
    except FileNotFoundError:
        return None


# Monotonically increasing counter, incremented by _mark_dirty() under
# _dirty_lock. A session is dirty (_is_dirty()) while this is greater than
# _segmentations_saved_version. This replaces a plain boolean flag, which
# cleared itself unconditionally at the end of every save: an edit that
# lands WHILE a save is still serializing (a click ~2s after the last, in
# the middle of, say, a 24MB write) would set the flag, and the finishing
# save's `dirty = False` would then stomp that back out with no way to tell
# a captured edit from a missed one. save_segmentations() instead snapshots
# this counter BEFORE it starts writing and only ever advances
# _segmentations_saved_version to that snapshot once the write actually
# lands, so a save can only clear the edits it demonstrably captured; an
# increment that happens after the snapshot leaves the session dirty
# regardless of how that in-flight save turns out.
_dirty_lock = threading.Lock()
_segmentations_dirty_counter = 0
_segmentations_saved_version = 0


def _mark_dirty():
    """Record one more unsaved change to session['segmentations'].

    Central call site for every mutation: applied to a mutating route via the
    _dirty_on_success decorator below, and called directly from the one
    mutation point that runs outside any route's request cycle (the
    background segmentation loop's per-image write in
    _process_single_image).
    """
    global _segmentations_dirty_counter
    with _dirty_lock:
        _segmentations_dirty_counter += 1


def _is_dirty():
    """True while session['segmentations'] holds a change no save has
    captured yet."""
    return _segmentations_dirty_counter > _segmentations_saved_version


def _dirty_on_success(view):
    """Decorator: mark the segmentations store dirty after `view` answers
    with a non-error status.

    Applied to every route that can change session['segmentations'] content,
    so flush_before_exit() and the pre-kill POST /api/flush know there is
    unsaved work without re-deriving it from each route's internals, and a
    validation failure (400/404/409, a mask left exactly as it was) never
    marks a request dirty that changed nothing. functools.wraps keeps the
    wrapped function's __name__, which Flask uses as the endpoint name, so
    stacking this under @app.route registers no duplicate endpoint.
    """
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        result = view(*args, **kwargs)
        if isinstance(result, tuple):
            status = result[1] if len(result) > 1 else 200
        else:
            status = getattr(result, 'status_code', 200)
        if isinstance(status, int) and status < 400:
            _mark_dirty()
        return result
    return wrapper


def load_segmentations(export_dir):
    path = _seg_path(export_dir)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            raw = f.read()
        _seg_disk_identity[path] = _seg_file_identity(path)
        _seg_disk_hash[path] = _hash_bytes(raw)
        return json.loads(raw)
    _seg_disk_identity.pop(path, None)
    _seg_disk_hash.pop(path, None)
    return {}


# Version this file's own reader expects; bump alongside a documented shape
# change to the contract in docs/superpowers/plans/
# 2026-09-27-vega-annotator-tidy-and-no-rescreening.md.
_PRIOR_SCREENING_VERSION = 1


def load_prior_screening(export_dir):
    """Load <export_dir>/prior_screening.json: what OTHER reef_point_seg
    projects already screened on this project's images (no-rescreening rule,
    VEGA annotator tidy-up, 2026-09-27). The orchestrator regenerates this
    file, keyed by file stem, before every annotator launch, covering every
    frame of this project that carries at least one label screened
    elsewhere, partly or fully. A new Step 3 selection never includes a
    fully screened frame, but a frame selected before that rule existed, or
    screened elsewhere after selection, still needs its previews. See the
    plan's Contract section for the exact upstream shape.

    Args:
        export_dir: the project's export directory (session['export_dir']).

    Returns:
        {stem: {'screened': {label_code: {...}}, 'masks': [{...}, ...]}},
        keyed by file stem WITHOUT an extension (session['segmentations'] and
        session['prior_screening'] use different keys for the same frame:
        see _prior_screening_for). Never raises. A missing file, an unreadable
        file, non-JSON content, or a top level that is not a {"version": 1,
        "frames": {...}} object all become {} with exactly one log line, so a
        broken or not-yet-deployed prior-screening file can never stop the
        annotator from opening (W2's orchestrator half of this plan ships
        separately; until it does, every project sees an absent file here).
        A single malformed FRAME inside an otherwise valid file degrades to
        {'screened': {}, 'masks': []} for that frame alone, without an extra
        log line, so one bad entry never costs the rest of a good file.
    """
    path = os.path.join(export_dir, 'prior_screening.json')
    if not os.path.exists(path):
        log(f"[prior-screening] no prior_screening.json at {path}; every "
            f"frame is treated as unscreened elsewhere")
        return {}

    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        log(f"[prior-screening] could not read {path} ({e}); treating "
            f"every frame as unscreened elsewhere")
        return {}

    valid_envelope = (isinstance(payload, dict)
                       and payload.get('version') == _PRIOR_SCREENING_VERSION
                       and isinstance(payload.get('frames'), dict))
    if not valid_envelope:
        log(f"[prior-screening] {path} is not a version "
            f"{_PRIOR_SCREENING_VERSION} prior-screening file "
            f"({{'version': 1, 'frames': {{...}}}} expected); ignoring it")
        return {}

    frames = {}
    for stem, entry in payload['frames'].items():
        if not isinstance(entry, dict):
            continue
        screened = entry.get('screened')
        masks = entry.get('masks')
        frames[stem] = {
            'screened': screened if isinstance(screened, dict) else {},
            'masks': ([m for m in masks if isinstance(m, dict)]
                      if isinstance(masks, list) else []),
        }
    return frames


# Every writer to the session's on-disk stores serializes here. Three callers
# now race: POST /api/flush (the client arms a trailing flush 2s after any
# mask-status PUT), the export path, and the background segmentation loop.
# Reentrant, so flush_all() can hold it across all three of its writes while
# the individual savers take it again when called directly.
_store_write_lock = threading.RLock()


def _atomic_write_json(path, payload, indent):
    """Write JSON to `path` so a reader only ever sees a complete file.

    json.dump streams, so writing straight into the destination leaves it
    truncated when anything fails partway through. The bytes go to a temp file
    in the SAME directory (os.replace is only atomic within one filesystem),
    chmod'd group-readable to match the rest of a run's output (mkstemp
    creates it 0o600, owner-only, by default), and are then renamed over the
    destination. A failed write removes the temp file and leaves the previous
    file exactly as it was.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                               prefix='.' + os.path.basename(path) + '.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, _SAVED_FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_segmentations(export_dir, data):
    """Write segmentations.json, unless another process changed it first.

    The expert-ID importer (MOD/scripts/_expertids/importer.py) can rewrite
    this exact file while an annotator sits open on the same project. A
    mismatched (mtime_ns, size) against what THIS process last read or wrote
    (_seg_disk_identity) is a first, cheap signal, not a verdict on its own:
    a touch, or a rewrite that reproduced the exact same bytes, moves the
    stat without changing content, so a stat mismatch is followed by a
    sha256 comparison against _seg_disk_hash before anything is refused.
    Equal content adopts the current stat as ours and the save proceeds
    normally; different content diverts the in-memory copy to a sibling
    segmentations.conflict-<AST stamp>.json instead of overwriting the real
    file, with one line logged naming both paths, and keeps diverting on
    every later save while the real file stays unreconciled, so this process
    is never told it saved when it did not. A normal save (no external
    change, or the first save of a session that never loaded an existing
    file) proceeds and refreshes the tracked identity and hash to the file
    it just wrote, so this process's own next save is never mistaken for a
    conflict with itself.

    The dirty counter is snapshotted before the write starts, not after it
    finishes: see _segmentations_dirty_counter for why.

    Returns:
        (status, conflict_path): status is 'saved' when segmentations.json
        itself was written, or 'conflict' when the write was diverted to a
        sibling file instead; conflict_path is that sibling's absolute path,
        or None when status is 'saved'.
    """
    global _segmentations_saved_version
    path = _seg_path(export_dir)
    with _store_write_lock:
        version_being_saved = _segmentations_dirty_counter

        on_disk_stat = _seg_file_identity(path)
        known_stat = _seg_disk_identity.get(path)
        if known_stat is not None and on_disk_stat != known_stat:
            disk_hash = _hash_file(path)
            known_hash = _seg_disk_hash.get(path)
            if disk_hash is not None and known_hash is not None and disk_hash == known_hash:
                # Stat moved (a touch, a metadata-only rewrite) but the bytes
                # this process already has ARE the bytes on disk: not a real
                # conflict, just an updated stat to trust from here on.
                _seg_disk_identity[path] = on_disk_stat
            else:
                conflict_path = os.path.join(
                    os.path.dirname(path), f'segmentations.conflict-{_ast_stamp()}.json')
                _atomic_write_json(conflict_path, data, 1)
                log(f"conflict: {path} changed on disk since this process last "
                    f"saved it (an outside writer, most likely the expert-ID "
                    f"importer); wrote the in-memory copy to {conflict_path} "
                    f"instead of overwriting it")
                return 'conflict', conflict_path

        _atomic_write_json(path, data, 1)
        with open(path, 'rb') as f:
            written = f.read()
        _seg_disk_identity[path] = _seg_file_identity(path)
        _seg_disk_hash[path] = _hash_bytes(written)
        _segmentations_saved_version = version_being_saved
        return 'saved', None


def save_manifest(export_dir, filenames):
    with _store_write_lock:
        _atomic_write_json(_manifest_path(export_dir), {
            'processed': sorted(filenames),
            'last_updated': datetime.now().isoformat(),
            'count': len(filenames),
        }, 2)


EXPORT_MANIFEST_FIELDS = ['basename', 'n_masks', 'label_file', 'image_file',
                          'outcome', 'preview_file', 'batch_name', 'image_path']


def _read_export_manifest(export_dir):
    """Read export_manifest.csv into {basename: full_row_dict}, preserving ALL
    columns (base fields + dynamic per-species counts). {} if absent."""
    import csv as _csv
    path = os.path.join(export_dir, 'export_manifest.csv')
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        return {r['basename']: dict(r) for r in _csv.DictReader(f)}


def _next_batch_name(export_dir):
    """Sequential per-export label batch_<NNN>_<YYYYMMDD-HHMMSS>; <NNN> persists
    across batches in .export_batch_seq."""
    seq_path = os.path.join(export_dir, '.export_batch_seq')
    n = 0
    if os.path.exists(seq_path):
        try:
            n = int((open(seq_path).read().strip() or '0'))
        except ValueError:
            n = 0
    n += 1
    with open(seq_path, 'w') as f:
        f.write(str(n))
    return f"batch_{n:03d}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _publish_preview(export_dir, stem):
    """Copy the reviewed overlay for one frame into the flat previews/ folder.
    Returns the export-root-relative path, or "" when no source overlay exists.
    """
    import shutil as _shutil
    src = os.path.join(export_dir, 'segmentations', 'overlays_reviewed', f'{stem}_seg.jpg')
    if not os.path.exists(src):
        return ''
    previews_dir = os.path.join(export_dir, 'previews')
    os.makedirs(previews_dir, exist_ok=True)
    dest_name = f'{stem}_seg.jpg'
    _shutil.copyfile(src, os.path.join(previews_dir, dest_name))
    return os.path.join('previews', dest_name)


def write_export_manifest(export_dir, entries):
    """Cumulative manifest with dynamic per-species columns. Base columns are
    EXPORT_MANIFEST_FIELDS; every later column is a per-image accepted-mask count
    for one species (bare code header). Merge by basename (re-export overwrites),
    union all species columns, backfill missing species with 0, sort by basename.
    """
    import csv as _csv
    os.makedirs(export_dir, exist_ok=True)
    merged = _read_export_manifest(export_dir)
    for e in entries:
        merged[e['basename']] = dict(e)
    species = set()
    for row in merged.values():
        species.update(k for k in row if k not in EXPORT_MANIFEST_FIELDS)
    species_cols = sorted(species)
    fieldnames = EXPORT_MANIFEST_FIELDS + species_cols
    path = os.path.join(export_dir, 'export_manifest.csv')
    with open(path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for basename in sorted(merged):
            row = merged[basename]
            out = {k: row.get(k, '') for k in EXPORT_MANIFEST_FIELDS}
            for sc in species_cols:
                out[sc] = row.get(sc, 0)
            w.writerow(out)


def reconcile_export(export_dir, manifest_entries):
    """Reconcile the on-disk export against the CUMULATIVE manifest.

    Because all_images/ accumulates across batches, this-batch entries alone
    will never match the on-disk image count from batch 2 on. Reconcile against
    the full export_manifest.csv on disk (written cumulatively just before this
    call). If no manifest exists on disk (e.g. a direct reconcile_export call in
    a unit test), fall back to the passed-in this-batch entries — a single
    batch's cumulative is exactly that batch.
    """
    on_disk = _read_export_manifest(export_dir)
    rows = list(on_disk.values()) if on_disk else list(manifest_entries)

    imgs = os.listdir(os.path.join(export_dir, 'all_images')) \
        if os.path.isdir(os.path.join(export_dir, 'all_images')) else []
    exported = [e for e in rows
                if e['outcome'] in ('exported_with_masks', 'exported_empty')]
    for e in exported:
        lp = os.path.join(export_dir, 'all_labels', e['label_file'])
        assert os.path.exists(lp), f"routed frame {e['basename']} missing label file {lp}"
    assert len(imgs) == len(exported), (
        f"all_images={len(imgs)} != exported={len(exported)} "
        f"(every routed non-review/non-scrap image must be on disk)")
    return {"on_disk_images": len(imgs), "exported": len(exported)}


def flush_all():
    """Write every session store to disk: segmentations, the processed
    manifest, and the class map.

    Returns:
        (status, conflict_path): save_segmentations()'s own return, passed
        straight through, so a caller that flushed on the operator's behalf
        (POST /api/flush, scrap/unscrap) can tell them their edit was
        diverted rather than answering ok when it was not. ('saved', None)
        when no project is open, since there is nothing to conflict with.
    """
    export_dir = session['export_dir']
    if not export_dir:
        return 'saved', None
    # One lock across all three writes: a timer-driven /api/flush, an export
    # and the background segmentation loop all arrive here, and a reader must
    # never catch the set half-updated. save_class_map lives in the shared
    # export_yolo.py (parity-guarded, not editable from this task), so it is
    # serialized here but is not itself an atomic replace.
    with _store_write_lock:
        result, conflict_path = save_segmentations(export_dir, session['segmentations'])
        save_manifest(export_dir, list(session['segmentations'].keys()))
        if session['class_map']:
            save_class_map(export_dir, session['class_map'])
    return result, conflict_path


# ── Save on quit ─────────────────────────────────────────────────
# VEGA annotation safety net, 2026-09-27: a stopped or killed annotator must
# not drop an operator's un-flushed edits. Wired into SIGTERM, SIGINT and a
# normal interpreter exit from the __main__ block only, never at import time,
# so importing this module (every test in tests/ does) installs nothing.

# How long the quit-time flush waits for the session write lock before giving
# up. The GPU right-of-way sweep sends SIGTERM and the pipeline orchestrator's
# runner.kill follows with SIGKILL 5s later (which cannot be caught), so this
# must land comfortably inside that window.
_QUIT_FLUSH_LOCK_TIMEOUT_S = 4.0

# Set once flush_before_exit() has completed one attempt (saved, diverted to
# a conflict file, or found nothing to save), so a signal handler and the
# atexit hook its sys.exit() triggers on the way out never both act on the
# same quit: a lock that stays busy past the timeout is the one case this
# does NOT set, so a later attempt (e.g. atexit, if the signal handler's own
# attempt could not get the lock) still gets to try.
_quit_flushed = False


def flush_before_exit():
    """Flush the active session to disk once before the process exits, if it
    has an unsaved change.

    Idempotent per process: once a call has completed one attempt, later
    calls (a signal handler's sys.exit() re-running the atexit hook it just
    triggered) are a no-op, so a session is never written twice on the way
    out.

    An unconfigured session (session['export_dir'] is empty, i.e. no project
    was ever opened) writes nothing, matching flush_all()'s own guard: a
    quit before Open never creates an empty segmentations.json over a real
    one. An idle, unmutated session (_is_dirty() is False) also writes
    nothing: before this guard existed, an idle annotator never wrote on its
    own, and the expert-ID importer (MOD/scripts/_expertids/importer.py) can
    rewrite the same project's segmentations.json while the annotator sits
    open; skipping a no-op quit-time save keeps that true. The
    save_segmentations() conflict check underneath this guard covers the
    remaining case, a genuinely dirty session whose file changed on disk
    since this process last saw it.

    The session write lock is acquired with a bounded timeout rather than
    blocked on forever, because a request thread already mid-write can hold
    it when the signal arrives. When the lock is not free within
    _QUIT_FLUSH_LOCK_TIMEOUT_S seconds, the flush is abandoned: one line
    naming the export directory is logged, and the process is left to exit
    anyway rather than hang on a save that cannot complete.
    """
    global _quit_flushed
    if _quit_flushed:
        return
    export_dir = session.get('export_dir')
    if not export_dir:
        _quit_flushed = True
        return
    if not _is_dirty():
        _quit_flushed = True
        return
    if not _store_write_lock.acquire(timeout=_QUIT_FLUSH_LOCK_TIMEOUT_S):
        log(f"quit: session lock still busy after {_QUIT_FLUSH_LOCK_TIMEOUT_S}s, "
            f"{export_dir} was NOT saved on exit")
        return
    try:
        flush_all()
    finally:
        _store_write_lock.release()
    _quit_flushed = True


def _handle_quit_signal(signum, frame):
    """SIGTERM/SIGINT handler: flush the session once, then exit.

    Parameters:
        signum: the signal number Python's signal module delivers.
        frame: the interrupted stack frame; unused, required by the
            signal-handler calling convention.

    Returns:
        None. sys.exit(0) raises SystemExit before this function would
        otherwise return, so control never comes back to the caller.
    """
    flush_before_exit()
    sys.exit(0)


# ── Review helpers ───────────────────────────────────────────────

def _build_review_list():
    """Build the review queue.

    Mirrors the step 4 policy: once a frame is exported or scrapped it is
    considered done and does not re-enter the queue. Done-is-done.

    If `session['shuffle']` is set, the unexported list is shuffled so the
    user sees a random sample instead of alphabetical ordering.
    """
    import random as _random
    target_only = getattr(cfg, 'TARGET_SPECIES_ONLY', 0)
    target_codes = _target_species()
    unexported = []
    for fn in sorted(session['segmentations']):
        seg = session['segmentations'][fn]
        if seg.get('scrapped', False):
            continue
        if seg.get('exported', False):
            continue
        if target_only and not getattr(cfg, 'MANUAL_ANNOTATE', False):
            # Custom imports start maskless by definition (no routed prompts),
            # so the no-target-mask skip must not hide them from review.
            if not seg.get('custom_import', False):
                masks = seg.get('masks', [])
                if not any(m.get('species', '') in target_codes for m in masks):
                    continue
        unexported.append(fn)
    if session.get('shuffle'):
        _random.shuffle(unexported)
    session['review_files'] = unexported
    session['review_offset'] = 0


def _merge_custom_imports():
    """Graft previously imported custom images into the live session.

    Reads custom_imports/imports_manifest.json under the session export dir
    and adds a seg_dict for every entry not already present (the normal case:
    segmentations.json already carries them from the flush at import time, so
    this only backfills after a segmentations store reset). Returns the number
    added. Never raises: a missing delivered file is logged and skipped."""
    export_dir = session.get('export_dir')
    if not export_dir:
        return 0
    added = 0
    for entry in ci.load_manifest(export_dir):
        fn = entry.get('filename')
        if not fn or fn in session['segmentations']:
            continue
        seg = ci.seg_dict_for(export_dir, entry)
        if not os.path.exists(seg['image_path_abs']):
            log(f"[import] WARNING: delivered file missing for {fn}, skipping")
            continue
        session['segmentations'][fn] = seg
        added += 1
    if added:
        log(f"[import] restored {added} custom import(s) from the manifest")
    return added


def _find_segmentation(filename):
    """Return the seg_dict for a filename, or None."""
    return session['segmentations'].get(filename)


# ── Wire serialization ───────────────────────────────────────────
# A stored mask carries two heavy fields the browser never reads: the RLE blob
# (`rle`, used by render_overlay / export / merge math on the server) and the
# normalized polygon (`polygon_norm`, written to the YOLO label file at export).
# The canvas draws from `polygon_px` only. On a dense frame those two fields are
# most of the JSON, so every client-bound copy drops them. The disk store
# (segmentations.json) keeps both: wire_mask copies, it never mutates.
#
# One deliberate exception: exemplar scan candidates are ephemeral on the client
# and are POSTed back verbatim to /commit_mask, so those keep both fields.
_WIRE_DROP_FIELDS = ('rle', 'polygon_norm')


def wire_mask(mask):
    """A client-bound copy of one mask, without the fields no client reads."""
    if not isinstance(mask, dict):
        return mask
    return {k: v for k, v in mask.items() if k not in _WIRE_DROP_FIELDS}


def wire_masks(masks):
    """wire_mask over a list of masks."""
    return [wire_mask(m) for m in (masks or [])]


# ── Prior screening (cross-project no-rescreening) ─────────────────
# load_prior_screening() (above, near load_segmentations) fills
# session['prior_screening'] at configure/resume/standalone_start. The
# helpers below turn that per-stem record into what one frame's GET
# /api/image response needs: prior_masks (this project's OWN pixel space,
# since the source polygons are normalized 0..1) and prior_screened
# (pass-through). Neither ever writes into session['segmentations'].
_PRIOR_POLYGON_MIN_LEN = 6  # at least 3 (x, y) pairs, matching mask_to_polygon


def _prior_screening_for(filename):
    """This frame's prior-screening entry, or {} when there is none.

    prior_screening.json keys its frames by file STEM (no extension);
    session['segmentations'] keys by the full filename (extension included,
    e.g. '.jpeg'). Every lookup here goes through os.path.splitext so the two
    stores can disagree on that one point without either caller noticing.
    {} covers both "no prior_screening.json was ever loaded" (session key
    absent) and "this particular frame is fully unscreened elsewhere" (stem
    not in the loaded map): both mean the same thing to every caller here.
    """
    stem = os.path.splitext(filename)[0]
    return session.get('prior_screening', {}).get(stem, {})


def _scale_prior_ring(ring_norm, image_width, image_height):
    """One normalized polygon ring ([x0, y0, x1, y1, ...], 0..1) scaled to
    this frame's own pixel dimensions and rounded to the nearest whole pixel.

    Returns None, dropping the ring, when it is not a flat list of at least
    3 numeric (x, y) pairs (an odd length, a non-numeric coordinate, or a
    boolean masquerading as one): a hostile or truncated prior_screening.json
    must never raise here, only lose the one ring that cannot be drawn.
    """
    if (not isinstance(ring_norm, list)
            or len(ring_norm) < _PRIOR_POLYGON_MIN_LEN
            or len(ring_norm) % 2 != 0):
        return None
    scaled = []
    for coord, dim in zip(ring_norm, [image_width, image_height] * (len(ring_norm) // 2)):
        if isinstance(coord, bool) or not isinstance(coord, (int, float)):
            return None
        scaled.append(round(coord * dim))
    return scaled


def _scale_prior_masks(masks, image_width, image_height):
    """prior_screening.json's masks (normalized polygons from ANOTHER
    project) scaled to THIS frame's own pixel dimensions, for the annotator
    to draw as read-only, non-selectable previews. Ring-level validation
    only: a mask with at least one drawable ring survives with just that
    ring; a mask where every ring is malformed is dropped rather than sent
    with an empty polygon list. These entries are never written into
    seg['masks'] by any caller, so they can never reach segmentations.json,
    an export, the cross-project mask registry, or the undo stack.
    """
    out = []
    for m in (masks or []):
        if not isinstance(m, dict):
            continue
        rings = m.get('polygon_norm')
        if not isinstance(rings, list) or not rings:
            continue
        scaled_rings = [r for r in
                        (_scale_prior_ring(ring, image_width, image_height)
                         for ring in rings)
                        if r is not None]
        if not scaled_rings:
            continue
        out.append({
            'label': m.get('label', ''),
            'name': m.get('name', ''),
            'polygon_px': scaled_rings,
            'project_id': m.get('project_id', ''),
            'project_name': m.get('project_name', ''),
            'reviewer': m.get('reviewer', ''),
            'at': m.get('at', ''),
            'outcome': m.get('outcome', ''),
        })
    return out


# ── Embedding warm-up ────────────────────────────────────────────
# Opening a frame used to load only the pixels; the tracker's vision encoder
# then ran on the operator's FIRST click, so that click paid encoder + decoder.
# Opening a frame now kicks the encoder in the background, so the first click
# pays only the decoder. The set below keeps a second GET for the same frame
# from stacking a second thread behind the first.
_prewarm_lock = threading.Lock()
_prewarming = set()


def _prewarm_embeddings_async(filename):
    """Run the tracker vision encoder for the just-opened frame off-thread.

    Takes the engine's own _prompt_lock, the same lock every prompt takes, so
    the warm-up can never observe a half-switched frame: whichever frame is
    current when the lock is granted is the one whose embeddings get cached,
    and a prompt that arrives meanwhile simply waits and then finds them warm.
    Never raises: a failed warm-up only costs the old first-click latency.
    """
    engine = session.get('sam_engine')
    if engine is None:
        return
    with _prewarm_lock:
        if filename in _prewarming:
            return
        _prewarming.add(filename)

    def _run():
        try:
            with engine._prompt_lock:
                # Best-effort staleness check: if the operator has already
                # opened a different frame, skip the encoder pass rather than
                # spend it on a frame nobody is looking at. This is not a
                # guarantee (a switch can arrive right after the check), only
                # a cheap way to drop the common wasted case.
                if (getattr(engine, '_current_image', None) is not None
                        and session.get('current_review_image') == filename):
                    engine._ensure_embeddings()
        except Exception as e:
            log(f"[prewarm] embedding warm-up failed for {filename} (non-fatal): {e}")
        finally:
            with _prewarm_lock:
                _prewarming.discard(filename)

    threading.Thread(target=_run, name='prewarm-embeddings', daemon=True).start()


def _ensure_image_embedded(filename):
    """Ensure the SAM3 engine has the given image embedded for refinement."""
    engine = session.get('sam_engine')
    if engine is None:
        return False

    seg = _find_segmentation(filename)
    if seg is None:
        return False

    image_path = seg.get('image_path_abs')
    if not image_path or not os.path.exists(image_path):
        return False

    # Always verify engine has the right image loaded (don't trust session cache)
    if session.get('current_review_image') != filename or engine._current_image_path != image_path:
        engine.clear_image()  # force reload, and free the old frame's embeddings
        engine.set_image(image_path)
        session['current_review_image'] = filename

    return True


# ── Standalone session helpers ───────────────────────────────────

# A standalone session name becomes a directory under cfg.STANDALONE_ROOT, so it
# is restricted to a flat, path-safe token (no separators, no traversal).
_SESSION_NAME_RE = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


def _default_session_name():
    """Suggested standalone session name: AST date plus the operator initials,
    e.g. 26-08-26_LO. All timestamps in this app are AST (house rule), so the
    date comes from custom_imports' single AST clock, never a second one."""
    return ci.now_ast().strftime('%y-%m-%d') + '_' + (cfg.REVIEWER or 'LO')


def _read_master_codes():
    """The species dictionary the standalone label picker offers.
    [{code, name, category}, ...], empty when no master_codes CSV is resolvable."""
    import csv as _csv
    out = []
    path = cfg.MASTER_CODES_CSV
    if not path or not os.path.isfile(path):
        return out
    with open(path) as f:
        for row in _csv.DictReader(f):
            code = (row.get('species_code') or row.get('code') or '').strip()
            if code:
                out.append({'code': code,
                            'name': (row.get('species_name') or row.get('name') or '').strip(),
                            'category': (row.get('category') or '').strip()})
    return out


# ── Routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           default_input=cfg.INPUT_DIR,
                           default_export=cfg.EXPORT_DIR,
                           default_categories=cfg.DEFAULT_CATEGORIES,
                           all_categories=[
                               "Target species only",
                               "Coral", "Sponge", "Macroalgae", "Gorgonian",
                               "Dca", "Turf", "Non-living",
                           ],
                           target_species=_target_species(),
                           target_labels=_target_labels(),
                           default_review_dir=cfg.REVIEW_DIR,
                           default_contacts=cfg.REVIEW_CONTACTS,
                           target_species_only=bool(getattr(cfg, 'TARGET_SPECIES_ONLY', 1)),
                           reference_default=bool(getattr(cfg, 'REFERENCE_DEFAULT', 1)),
                           default_confidence=cfg.CONFIDENCE_THRESHOLD,
                           default_min_area=cfg.MIN_MASK_AREA_PX,
                           default_merge_dist=cfg.MERGE_DISTANCE_PX,
                           default_overlap=cfg.OVERLAP_STRATEGY,
                           default_mask_size=cfg.SAM3_MASK_SIZE,
                           default_thin_ratio=cfg.THIN_MASK_RATIO,
                           default_simplify=cfg.POLYGON_SIMPLIFY_EPSILON,
                           default_exemplar=cfg.EXEMPLAR_THRESHOLD,
                           default_dev_tracker=cfg.SAM3_DEVICE_TRACKER,
                           default_dev_exemplar=cfg.SAM3_DEVICE_EXEMPLAR,
                           orchestrator_url=_orchestrator_url(),
                           orchestrated=(bool(os.environ.get('TCRMP_INPUT_DIR'))
                                         or request.args.get('orchestrated') == '1'),
                           session_mode=getattr(cfg, 'SESSION_MODE', 'configure'),
                           # Expert-review lock (2026-09-06): the page renders
                           # its tooltip from this one string, so the wording an
                           # operator reads and the wording a refused request
                           # carries can never drift apart.
                           review_locked_msg=REVIEW_LOCKED_MSG,
                           review_tagging_enabled=REVIEW_TAGGING_ENABLED,
                           default_session_name=_default_session_name())


@app.route('/api/configure', methods=['POST'])
@_dirty_on_success
def configure():
    """Load sam_click_prompts.json for selected categories, build queue."""
    data = request.json or {}
    input_dir = data.get('input_dir', cfg.INPUT_DIR)
    export_dir = data.get('export_dir', cfg.EXPORT_DIR)
    categories = data.get('categories', cfg.DEFAULT_CATEGORIES)
    batch_size = data.get('review_batch_size', cfg.REVIEW_BATCH_SIZE)

    # Honor the target-labels-only flag EXPLICITLY (cfg.TARGET_SPECIES_ONLY, set
    # from env by the orchestrator panel), not only via the "Target species only"
    # Categories string. The orchestrated launch POSTs an empty body, so the
    # default must come from cfg/env here. A request may still override it.
    if data.get('target_species_only') is not None:
        target_only_flag = bool(data.get('target_species_only'))
    else:
        target_only_flag = bool(getattr(cfg, 'TARGET_SPECIES_ONLY', 1))
    if target_only_flag and 'Target species only' not in categories:
        categories = ['Target species only'] + list(categories)
    elif not target_only_flag and 'Target species only' in categories:
        categories = [c for c in categories if c != 'Target species only']

    session['input_dir'] = os.path.abspath(input_dir)
    session['export_dir'] = os.path.abspath(export_dir)
    session['categories'] = categories
    session['review_batch_size'] = batch_size
    session['shuffle'] = bool(data.get('shuffle', False))
    _seed_reviewer(data.get('reviewer'))

    # Expert-review settings: where to push the queue, and who gets the CSV.
    session['review_dir'] = (data.get('review_dir') or cfg.REVIEW_DIR)
    _raw_contacts = data.get('contacts', cfg.REVIEW_CONTACTS)
    if isinstance(_raw_contacts, str):
        _raw_contacts = _raw_contacts.split(',')
    session['contacts'] = [c.strip() for c in (_raw_contacts or []) if c and c.strip()]

    os.makedirs(session['export_dir'], exist_ok=True)

    # Load existing class map
    session['class_map'] = load_class_map(session['export_dir'])

    # No-rescreening cross-project preview (2026-09-27): read-only, never
    # feeds segmentations/exports/the registry/undo/the dirty counter.
    session['prior_screening'] = load_prior_screening(session['export_dir'])

    # Flat input: routed_input/ids/sam_click_prompts.json + routed_input/raw/
    input_dir_abs = session['input_dir']
    ids_path = os.path.join(input_dir_abs, 'ids', 'sam_click_prompts.json')
    raw_dir = os.path.join(input_dir_abs, 'raw')
    if not os.path.exists(ids_path):
        return jsonify({'error': (
            'No Step 4 output found for this project (no '
            'ids/sam_click_prompts.json under ' + input_dir_abs + '). Run Step 4 '
            '(routing) first, or open a project that has completed Step 4.'
        )}), 404

    # Load existing segmentations (single flat read)
    existing = load_segmentations(session['export_dir'])
    session['segmentations'] = {}
    session['segmentations'].update(existing)
    already_processed = set(existing.keys())

    with open(ids_path) as f:
        prompts = json.load(f)

    # Build processing queue
    queue = []
    skipped = 0

    for img_filename, img_data in prompts.items():
        if img_filename in already_processed:
            skipped += 1
            continue

        # Filter points by selected categories / target species
        target_sp_only = 'Target species only' in categories
        real_categories = [c for c in categories if c != 'Target species only']

        # Non-target prompt points (right-click from step 4): always
        # included in segmentation regardless of the target-species filter.
        # These are points the reviewer flagged as "segment this too" without
        # assigning a target label; SAM3 still produces a mask for each.
        def _is_non_target(p):
            return p.get('non_target') is True or \
                p.get('species', p.get('species_code', '')) == '?'

        target_codes = _target_species()
        if target_sp_only:
            # Filter directly by target species codes
            # sam_click_prompts uses 'species' field, not 'species_code'
            selected_points = [
                p for p in img_data.get('points', [])
                if p.get('species', p.get('species_code', '')) in target_codes
                or _is_non_target(p) or _is_review(p)
            ]
        elif real_categories:
            selected_points = [
                p for p in img_data.get('points', [])
                if p.get('category', '') in real_categories
                or _is_non_target(p) or _is_review(p)
            ]
            # If TARGET_SPECIES_ONLY config flag, further filter (but keep non-targets + review)
            if getattr(cfg, 'TARGET_SPECIES_ONLY', 0):
                selected_points = [
                    p for p in selected_points
                    if p.get('species', p.get('species_code', '')) in target_codes
                    or _is_non_target(p) or _is_review(p)
                ]
        else:
            selected_points = list(img_data.get('points', []))

        if not selected_points:
            continue

        # Resolve raw image path under the flat raw dir
        raw_rel = img_data.get('raw', img_data.get('raw_image', ''))
        raw_path = os.path.join(raw_dir, os.path.basename(raw_rel)) if raw_rel else None
        if not raw_path or not os.path.exists(raw_path):
            raw_path = os.path.join(raw_dir, img_filename)
        if not os.path.exists(raw_path):
            log(f"WARNING: Raw image not found for {img_filename}")
            continue

        if getattr(cfg, 'MANUAL_ANNOTATE', False):
            # Combined annotator: no SAM3 pass. Build an empty-mask seg_dict
            # carrying the FULL prompt-point set as read-only reference. The
            # `already_processed` skip at the top of this loop handles resume
            # (existing seg_dicts with hand-drawn masks are never overwritten).
            img_h, img_w = 0, 0
            _img = cv2.imread(raw_path)
            if _img is not None:
                img_h, img_w = _img.shape[:2]
            seg_dict = {
                'image_path': raw_rel or img_filename,
                'image_path_abs': raw_path,
                'image_width': img_w,
                'image_height': img_h,
                'masks': [],
                'reference_points': img_data.get('points', []),
                'processed_at': datetime.now().isoformat(),
                'reviewed': False,
                'exported': False,
            }
            session['segmentations'][img_filename] = seg_dict
            continue

        queue.append({
            'filename': img_filename,
            'image_path': raw_path,
            'points': selected_points,
        })

    session['processing_queue'] = queue
    session['processing_idx'] = 0

    # Custom imports (operator-supplied images) ride alongside routed frames:
    # backfill any manifest entry the segmentations store does not carry yet.
    _merge_custom_imports()

    if getattr(cfg, 'MANUAL_ANNOTATE', False) or not queue:
        session['phase'] = 'review'
        _build_review_list()
    else:
        session['phase'] = 'processing'

    # Log config
    config_log = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': input_dir_abs,
        'export_dir': session['export_dir'],
        'categories': categories,
        'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
        'min_mask_area': cfg.MIN_MASK_AREA_PX,
        'merge_distance': cfg.MERGE_DISTANCE_PX,
        'overlap_strategy': cfg.OVERLAP_STRATEGY,
    }
    config_path = os.path.join(session['export_dir'], 'config_log.json')
    with open(config_path, 'w') as f:
        json.dump(config_log, f, indent=2)

    session['configured'] = True

    log(f"Configured: {len(prompts)} images total, {len(queue)} to process, "
        f"{skipped} already done, categories={categories}")

    return jsonify({
        'images': len(prompts),
        'to_process': len(queue),
        'already_processed': skipped,
        'phase': session['phase'],
        'review_count': len(session.get('review_files', [])),
        'categories': categories,
    })


@app.route('/api/resume', methods=['POST'])
@_dirty_on_success
def resume():
    """Resume from existing segmentation output."""
    data = request.json
    session['export_dir'] = os.path.abspath(data['export_dir'])
    session['review_batch_size'] = data.get('review_batch_size', cfg.REVIEW_BATCH_SIZE)
    session['shuffle'] = bool(data.get('shuffle', False))
    _seed_reviewer(data.get('reviewer'))
    session['review_dir'] = (data.get('review_dir') or cfg.REVIEW_DIR)
    _rc = data.get('contacts', cfg.REVIEW_CONTACTS)
    if isinstance(_rc, str):
        _rc = _rc.split(',')
    session['contacts'] = [c.strip() for c in (_rc or []) if c and c.strip()]

    export_dir = session['export_dir']
    if not os.path.isdir(export_dir):
        return jsonify({'error': 'Export directory not found'}), 404

    # Load class map
    session['class_map'] = load_class_map(export_dir)

    # No-rescreening cross-project preview (2026-09-27): read-only, never
    # feeds segmentations/exports/the registry/undo/the dirty counter.
    session['prior_screening'] = load_prior_screening(export_dir)

    # Load flat segmentations (Task 2 will extend this to handle year-nested
    # legacy exports; for now load from the flat path only)
    if not os.path.exists(_seg_path(export_dir)):
        return jsonify({'error': 'No segmentation data found'}), 404

    session['segmentations'] = load_segmentations(export_dir)
    total = len(session['segmentations'])

    # Backfill custom imports missing from the loaded store (normally none:
    # imports are flushed into segmentations.json the moment they land).
    _merge_custom_imports()

    session['phase'] = 'review'
    session['processing_queue'] = []
    _build_review_list()
    session['configured'] = True

    return jsonify({
        'total_processed': total,
        'phase': 'review',
        'review_count': len(session['review_files']),
        'total_classes': len(session['class_map']),
    })


@app.route('/api/import_images', methods=['POST'])
@_dirty_on_success
def import_images():
    """Bring ANY custom image(s) into the active run's annotation queue.

    Accepts multipart uploads (field 'images', repeatable) or JSON
    {'paths': [server paths]}. Each image is copied verbatim into
    custom_imports/originals/, delivered (lores-downscaled when its long edge
    exceeds 1920px, same rule as pipeline frames) into custom_imports/raw/,
    manifested, added to the segmentations store + review queue, and recorded
    in the run's project.json (access_log + custom_imports). See
    custom_imports.py for the full contract."""
    export_dir = session.get('export_dir')
    if not export_dir:
        return jsonify({'error': 'No active session yet: configure or resume '
                                 'a project before importing images.'}), 400

    items = []
    if request.files:
        for fs in request.files.getlist('images'):
            if fs and fs.filename:
                items.append((fs.filename, fs.read()))
    else:
        data = request.get_json(silent=True) or {}
        for p in data.get('paths', []):
            p = str(p)
            items.append((os.path.basename(p), p))
    if not items:
        return jsonify({'error': 'No images supplied. Send multipart field '
                                 '"images" or JSON {"paths": [...]}.'}), 400

    initials = (session.get('reviewer') or cfg.REVIEWER or 'LO')
    entries, skipped = ci.ingest_files(
        export_dir, items, taken_names=set(session['segmentations'].keys()),
        reviewer=initials, log=log)

    recorded = False
    if entries:
        for e in entries:
            session['segmentations'][e['filename']] = ci.seg_dict_for(export_dir, e)
        flush_all()
        _build_review_list()
        recorded = ci.record_in_project(export_dir, entries, initials=initials)
        log(f"[import] {len(entries)} custom image(s) added to the review "
            f"queue ({len(skipped)} skipped); project.json "
            f"{'updated' if recorded else 'not found (standalone session)'}")

    return jsonify({
        'ok': bool(entries),
        'imported': [e['filename'] for e in entries],
        'skipped': [{'name': n, 'reason': r} for n, r in skipped],
        'project_recorded': recorded,
        'review_count': len(session.get('review_files', [])),
    })


@app.route('/api/standalone/master_codes')
def standalone_master_codes():
    """Species dictionary for the standalone label picker."""
    return jsonify({'codes': _read_master_codes()})


@app.route('/api/standalone/start', methods=['POST'])
@_dirty_on_success
def standalone_start():
    """Create a standalone session (no project): session dir, labels, images, queue.

    Accepts multipart (fields `session`, `labels` as a comma list, repeatable
    `images` file field, `reviewer` initials) or JSON {session, labels[],
    paths[], reviewer}. Everything the session owns lives under
    cfg.STANDALONE_ROOT/<session>: custom_imports/ for the frames,
    segmentations/ for the masks, session.json for the header.

    Initials are required: a standalone session has no project to inherit an
    identity from, so without them every mask it produces would be
    unattributed. They are refused with a 400 rather than defaulted.

    A label that spells out the expert-review tag is refused with the shared
    409 (Lauren Olinger, 2026-09-06): every frame of the session inherits this
    list, so one REVIEW entry would arm the suspended flag for the whole
    session.
    """
    if getattr(cfg, 'SESSION_MODE', '') != 'standalone':
        return jsonify({'error': 'not a standalone session'}), 400

    form = request.form if request.form else (request.get_json(silent=True) or {})
    if not _seed_reviewer(form.get('reviewer')):
        return jsonify({'error': 'your initials are required so this work is credited'}), 400
    name = (form.get('session') or '').strip()
    # name.strip('.') == '' rejects a name made only of dots ('.', '...'): the
    # regex allows a dot, and '.' would resolve the session dir to the root.
    if not _SESSION_NAME_RE.match(name) or '..' in name or name.strip('.') == '':
        return jsonify({'error': 'session name: letters, digits, dot, dash, underscore only'}), 400

    raw_labels = form.get('labels') or []
    if isinstance(raw_labels, str):
        raw_labels = [s.strip() for s in raw_labels.split(',')]
    labels = [l for l in raw_labels if l]
    if not labels:
        return jsonify({'error': 'pick at least one label'}), 400
    # The quick-add row is drawn from this list, so a label of REVIEW would put
    # a live, armable REVIEW button beside the greyed locked one and offer the
    # operator a control every add route answers with a 409. A person typed
    # this list, so it is refused and named rather than silently edited. The
    # template guards it ahead of the POST; this is what makes that guard true.
    refused_labels = [l for l in labels if _review_tag_refused({'species': l})]
    if refused_labels:
        log(f"[review-lock] standalone session '{name}' refused: "
            f"{len(refused_labels)} label(s) spell out the expert-review tag "
            f"({', '.join(refused_labels)}). {REVIEW_LOCKED_MSG}")
        return _review_locked_response(
            detail=f"Remove {', '.join(refused_labels)} from the label list "
                   f"to start this session.",
            review_locked_labels=refused_labels)
    lookup = {c['code']: c for c in _read_master_codes()}

    session_dir = os.path.join(os.path.abspath(cfg.STANDALONE_ROOT), name)
    os.makedirs(os.path.join(session_dir, 'segmentations'), exist_ok=True)
    session['export_dir'] = session_dir
    session['input_dir'] = session_dir
    session['target_labels'] = [{'code': c,
                                 'name': lookup.get(c, {}).get('name', ''),
                                 'category': lookup.get(c, {}).get('category', 'Coral')}
                                for c in labels]
    session['categories'] = ['Target species only']
    # Standalone sessions are not batched: one queue, exported when the operator
    # says so. A single huge batch size keeps the batch machinery out of the way.
    session['review_batch_size'] = 10 ** 9
    session['shuffle'] = False
    # session['reviewer'] is already in force: the required-initials check at
    # the top of this route seeded it before anything was written.
    session['review_dir'] = cfg.REVIEW_DIR
    session['contacts'] = []
    session['class_map'] = load_class_map(session_dir)
    session['segmentations'] = load_segmentations(session_dir)
    # A standalone session has no orchestrator-launched project to cross-check
    # against, so it never carries a prior_screening.json of its own; loading
    # it anyway (almost always {}) also clears out whatever a PREVIOUSLY open
    # routed project left in this process-global session, so a standalone run
    # started right after closing one never shows that project's previews.
    session['prior_screening'] = load_prior_screening(session_dir)
    # Reopening a session name that already holds imports: backfill anything the
    # segmentations store lost (same rule configure() uses for routed projects).
    _merge_custom_imports()

    items = []
    for fs in request.files.getlist('images'):
        if fs and fs.filename:
            items.append((fs.filename, fs.read()))
    raw_paths = form.get('paths') or []
    if isinstance(raw_paths, str):
        raw_paths = raw_paths.split(',')
    for p in raw_paths:
        p = str(p).strip()
        if p:
            items.append((os.path.basename(p), p))

    entries, skipped = ([], [])
    if items:
        entries, skipped = ci.ingest_files(
            session_dir, items, taken_names=set(session['segmentations'].keys()),
            reviewer=session['reviewer'], log=log)
        for e in entries:
            session['segmentations'][e['filename']] = ci.seg_dict_for(session_dir, e)
        flush_all()

    with open(os.path.join(session_dir, 'session.json'), 'w') as f:
        json.dump({'session': name,
                   'labels': labels,
                   'created': ci.now_ast().strftime('%Y-%m-%d %H:%M AST'),
                   'reviewer': session['reviewer']}, f, indent=2)

    session['phase'] = 'review'
    session['processing_queue'] = []
    session['processing_idx'] = 0
    _build_review_list()
    session['configured'] = True
    log(f"[standalone] session '{name}' ready at {session_dir}: "
        f"{len(entries)} image(s) imported, {len(skipped)} skipped, "
        f"labels {','.join(labels)}")
    return jsonify({'ok': True,
                    'session_dir': session_dir,
                    'imported': [e['filename'] for e in entries],
                    'skipped': [{'name': n, 'reason': r} for n, r in skipped],
                    'review_count': len(session.get('review_files', [])),
                    'labels': labels})


def _process_single_image(engine, item, export_dir):
    """Segment one image. Returns (filename, n_masks, n_points).

    Prompt policy:
      - Each target point (species ∈ TARGET_SPECIES) -> positive SAM3 click
        that anchors a mask for that species.
      - Non-target points (right-click reviewer markers from step 4 —
        species='?' or non_target=True) -> negative SAM3 clicks that refine
        the target masks by telling SAM3 "don't include these regions".
        They DO NOT produce their own masks.
    """
    filename = item['filename']
    image_path = item['image_path']
    points = item['points']

    img_w, img_h = engine.set_image(image_path)

    def _is_negative(p):
        # REVIEW always wins: a point flagged for expert review must produce its
        # OWN positive mask, even if it also carries non_target/'?' (precedence).
        if _is_review(p):
            return False
        return p.get('non_target') is True \
            or p.get('species', p.get('species_code', '')) == '?'

    negatives = [p for p in points if _is_negative(p)]
    targets = [p for p in points if not _is_negative(p)]
    neg_clicks = [(p['x'], p['y']) for p in negatives]

    raw_masks = []
    # Lock (2026-09-06): build_mask_dict drops a review tag rather than raising
    # (an exception in this loop would abandon the whole frame) and counts the
    # drop on its own process-wide tally. This frame's own figure is counted
    # here instead of read as a difference of that tally, because the tally is
    # not this frame's alone: the exemplar scan route derives its point_info
    # from an existing mask, so a scan seeded from one of the masks flagged
    # before the lock builds a tagged point_info on its own request thread,
    # outside _processing_lock, while this loop runs. Counting per call keeps
    # the logged number exact whatever else the process is doing; the running
    # total stays with mask_ops.review_tags_dropped().
    review_tags_dropped_here = 0
    for pt in targets:
        if neg_clicks:
            # Multi-point prompt: [positive, *negatives] with labels [1, 0…0].
            clicks = [(pt['x'], pt['y'])] + neg_clicks
            labels = [1] + [0] * len(neg_clicks)
            result = engine.refine_mask(clicks, labels)
        else:
            result = engine.segment_point(pt['x'], pt['y'], label=1)
        if result is not None:
            mask_dict = build_mask_dict(
                mask_id=len(raw_masks),
                binary_mask=result['mask'],
                score=result['score'],
                point_info=pt,
                source_type='auto',
                simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON,
            )
            if mask_dict is not None:
                # REVIEW prompts produce a real mask, but it is flagged so YOLO
                # export skips it and the batch export ships it to the expert.
                # Lock (2026-09-06): under the lock this stamp is skipped
                # whole. build_mask_dict has already dropped the tag, so the
                # mask arrives with an empty species, which the existing
                # empty-species accept guard handles, and the operator labels
                # it. Flipping REVIEW_TAGGING_ENABLED restores the stamp.
                if _is_review(pt) and REVIEW_TAGGING_ENABLED:
                    mask_dict['review'] = True
                    mask_dict['species'] = 'REVIEW'
                    mask_dict['category'] = 'Review'
                elif not REVIEW_TAGGING_ENABLED and review_tag_requested(pt):
                    # The same condition build_mask_dict counted its drop on,
                    # applied to the same prompt and only where a mask was
                    # actually built, so this frame's figure matches what the
                    # lock dropped for this frame exactly.
                    review_tags_dropped_here += 1
                mask_dict['mask'] = result['mask']
                raw_masks.append(mask_dict)

    if review_tags_dropped_here > 0:
        log(f"[review-lock] {filename}: {review_tags_dropped_here} review-flagged "
            f"prompt(s) produced masks with no review tag. {REVIEW_LOCKED_MSG}")

    if raw_masks:
        raw_masks = resolve_overlaps(
            raw_masks, strategy=cfg.OVERLAP_STRATEGY,
            min_area=cfg.MIN_MASK_AREA_PX, thin_ratio=cfg.THIN_MASK_RATIO,
        )
        raw_masks = merge_close_masks(
            raw_masks, distance_px=cfg.MERGE_DISTANCE_PX, same_species_only=True,
        )

    final_masks = []
    for i, m in enumerate(raw_masks):
        binary = m.pop('mask', None)
        if binary is not None:
            m = update_mask_geometry(m, binary, cfg.POLYGON_SIMPLIFY_EPSILON)
        if m is not None:
            m['id'] = i
            final_masks.append(m)

    seg_dict = {
        'image_path': item.get('raw_rel', filename),
        'image_path_abs': image_path,
        'image_width': img_w,
        'image_height': img_h,
        'masks': final_masks,
        'processed_at': datetime.now().isoformat(),
        'reviewed': False,
        'exported': False,
    }

    session['segmentations'][filename] = seg_dict
    # Runs on the background processing thread, outside any route's request
    # cycle, so the @_dirty_on_success decorator never sees it; mark it here
    # directly (the one sprinkled call site, everywhere else routes through
    # the decorator).
    _mark_dirty()
    render_segmentation_overlays(seg_dict, filename, export_dir, stage="auto")

    return filename, len(final_masks), len(points)


# Background processing thread
_processing_lock = threading.Lock()

def _background_process_all():
    """Process entire queue in background. Browser can disconnect and reconnect."""
    engine = session.get('sam_engine')
    if engine is None:
        log("ERROR: SAM3 engine not loaded, cannot process")
        return

    queue = session['processing_queue']
    export_dir = session['export_dir']
    total = len(queue)

    log(f"Background processing started: {total} images")

    for idx in range(session['processing_idx'], total):
        if session.get('_stop_processing'):
            log(f"Processing stopped at {idx}/{total}")
            break

        item = queue[idx]
        with _processing_lock:
            try:
                fn, n_masks, n_pts = _process_single_image(engine, item, export_dir)
                log(f"SAM3 {idx + 1}/{total}: {fn} ({n_pts} pts -> {n_masks} masks)")
                # Persist segmentations after EVERY image so an orchestrator
                # restart (or crash, or ctrl+C) can't rewind progress.
                # Cost: one ~KB JSON rewrite per image — cheap.
                try:
                    save_segmentations(export_dir, session['segmentations'])
                except Exception as flush_err:
                    log(f"flush failed after {fn}: {flush_err}")
            except Exception as e:
                log(f"ERROR processing {item['filename']}: {e}")

        session['processing_idx'] = idx + 1

    flush_all()

    # Clear engine's cached image so review forces a fresh set_image. The engine
    # owns this: clearing the two private attributes here would leave its
    # embedding caches pointing at the last processed frame.
    if engine is not None:
        engine.clear_image()
    session['current_review_image'] = None

    session['phase'] = 'review'
    _build_review_list()
    log(f"Background processing complete: {session['processing_idx']}/{total}")


@app.route('/api/process', methods=['POST'])
def process_start():
    """Start background processing of all images. Returns immediately."""
    if session.get('_processing_thread') and session['_processing_thread'].is_alive():
        return jsonify({
            'already_running': True,
            'processed': session['processing_idx'],
            'total': len(session['processing_queue']),
        })

    session['_stop_processing'] = False
    t = threading.Thread(target=_background_process_all, daemon=True)
    session['_processing_thread'] = t
    t.start()

    return jsonify({
        'started': True,
        'total': len(session['processing_queue']),
    })


@app.route('/api/process_status')
def process_status():
    """Poll processing progress."""
    queue = session['processing_queue']
    total = len(queue)
    done = session['processing_idx']
    is_running = (session.get('_processing_thread') and
                  session['_processing_thread'].is_alive())

    # Allow review even while processing is ongoing
    if done > 0 and not session.get('review_files'):
        _build_review_list()

    return jsonify({
        'processed': done,
        'total': total,
        'running': is_running,
        'done': done >= total and not is_running,
        'phase': session['phase'],
        'review_count': len(session.get('review_files', [])),
    })


def _flush_result_fields(result, conflict_path):
    """{'conflict': True, 'conflict_file': <basename>} when a save diverted
    to a sibling file, else {}. Merged into a save route's own response dict
    (which sets 'ok' itself, since a route may have real work to report even
    when the flush underneath it diverted, e.g. an export that still wrote
    its files) so the client always learns when its edit did not reach
    segmentations.json."""
    if result == 'conflict':
        return {'conflict': True, 'conflict_file': os.path.basename(conflict_path)}
    return {}


@app.route('/api/flush', methods=['POST'])
def force_flush():
    """Write the session to disk now: the client's 2s trailing debounce, an
    explicit Ctrl+S, a tab's pagehide beacon, and the orchestrator's pre-kill
    save before it stops this process all arrive here.

    Writes only when something changed since the last successful save
    (_is_dirty()); an idle session with nothing new answers ok with no disk
    write, so a quiet pre-kill or trailing-timer call on a session nobody
    touched can never race the expert-ID importer for segmentations.json.

    Answers {ok: false, conflict: true, conflict_file: <name>} rather than
    ok: true when save_segmentations() diverted the write to a conflict
    sibling (the file changed on disk since this process last saw it, most
    likely the expert-ID importer): the operator is never told their edit
    saved when it did not.
    """
    if not _is_dirty():
        return jsonify({'ok': True})
    result, conflict_path = flush_all()
    out = {'ok': result != 'conflict'}
    out.update(_flush_result_fields(result, conflict_path))
    return jsonify(out)


@app.route('/api/review_batch')
def review_batch():
    batch_size = session['review_batch_size']
    offset = session['review_offset']
    files = session['review_files']

    batch_files = files[offset:offset + batch_size]
    images = []
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if seg:
            n_masks = len(seg.get('masks', []))
            n_pending = sum(1 for m in seg.get('masks', []) if m.get('status') == 'pending')
            n_accepted = sum(1 for m in seg.get('masks', []) if m.get('status') == 'accepted')
            images.append({
                'filename': fn,
                'n_masks': n_masks,
                'n_pending': n_pending,
                'n_accepted': n_accepted,
                'reviewed': seg.get('reviewed', False),
                'exported': seg.get('exported', False),
                'custom_import': seg.get('custom_import', False),
            })

    total_batches = (len(files) + batch_size - 1) // batch_size if files else 0
    current_batch = (offset // batch_size) + 1 if files else 0

    return jsonify({
        'images': images,
        'batch_number': current_batch,
        'total_batches': total_batches,
        'total_images': len(files),
        'offset': offset,
        'batch_size': batch_size,
    })


@app.route('/api/unlabeled')
def unlabeled_frames():
    """List every frame in the review queue that has a mask with no species
    assigned yet (species=='', accepted or pending, see _needs_species).

    Scoped to session['review_files'], the same frame set the review UI's
    Prev/Next navigation walks. A frame already exported or scrapped is not
    reachable from that navigation, so it is out of scope here too. This
    powers the client-side "Unlabeled only" filter and "next unlabeled" jump,
    which only ever move within the current review queue's frame list.
    """
    frames = []
    for fn in session.get('review_files', []):
        seg = _find_segmentation(fn)
        if not seg:
            continue
        if any(_needs_species(m) for m in seg.get('masks', [])):
            frames.append(fn)
    return jsonify({'frames': frames, 'count': len(frames)})


@app.route('/api/next_review_batch', methods=['POST'])
def next_review_batch():
    # Removed-frames model: do_export() drops exported frames from review_files
    # and resets review_offset to 0 (see _build_review_list). There is no
    # "advance past the current batch" any more; the rebuilt list IS the next
    # batch at offset 0. This route now just reports whether any frames remain;
    # `done` is true only when review_files is empty. The frontend's
    # nextReviewBatch() no longer relies on it (it reloads /api/review_batch
    # directly), but the route is kept as a non-contradictory status probe.
    files = session['review_files']
    return jsonify({
        'offset': 0,
        'total_images': len(files),
        'done': len(files) == 0,
    })


@app.route('/api/image/<path:filename>')
def get_image_data(filename):
    """Get all mask data for a specific image."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    # Pre-embed this image for fast refinement. set_image only opens the pixels;
    # the warm-up below runs the vision encoder so the first click is decoder-only.
    if _ensure_image_embedded(filename):
        _prewarm_embeddings_async(filename)

    image_width = seg.get('image_width', 0)
    image_height = seg.get('image_height', 0)
    prior = _prior_screening_for(filename)

    return jsonify({
        'masks': wire_masks(seg.get('masks', [])),
        'image_width': image_width,
        'image_height': image_height,
        'reviewed': seg.get('reviewed', False),
        'exported': seg.get('exported', False),
        'custom_import': seg.get('custom_import', False),
        'reference_points': [
            {'x': p['x'], 'y': p['y'], 'label': p.get('label', ''),
             'species_code': p.get('species', '')}
            for p in seg.get('reference_points', [])
        ],
        # Cross-project no-rescreening previews (2026-09-27): read-only masks
        # and screened labels from OTHER reef_point_seg projects that already
        # covered this same image, so an operator can see what is already
        # done instead of re-screening it. Never fed back into this seg dict.
        'prior_masks': _scale_prior_masks(prior.get('masks'), image_width, image_height),
        'prior_screened': prior.get('screened') or {},
    })


def _needs_species(mask):
    """True when a mask has no real species classification yet.

    An empty species ('' or missing) means the operator hasn't labeled the
    mask. A REVIEW mask (species=='REVIEW') is an intentional label, not a
    missing one, so it is never caught here.
    """
    return (mask.get('species', '') or '') == ''


# ── Reviewer identity (D2) ───────────────────────────────────────
# Who reviewed a mask is recorded as initials on the mask itself
# (mask['reviewer']), in label_outcomes/label_provenance.csv, and in the
# cross-project registry. The orchestrator seeds TCRMP_REVIEWER at launch and
# POSTs /api/reviewer on every reuse of a live annotator, so the session value
# tracks whoever opened the project last. Blank stays blank: nothing is
# inferred into the data.

# Initials are 1 to 4 letters, the same rule the VICARIUS launcher enforces on
# the project it hands us. Nothing else counts as an identity: "1" or "!!"
# would otherwise satisfy the accept guard while crediting nobody.
_REVIEWER_RE = re.compile(r'^[A-Za-z]{1,4}$')


def _norm_reviewer(value):
    """Initials as stored everywhere: stripped and uppercased, and only when
    they are 1 to 4 letters. Anything else normalizes to '' (unknown), which
    the accept guard then refuses."""
    raw = str(value or '').strip()
    return raw.upper() if _REVIEWER_RE.match(raw) else ''


def _current_reviewer():
    """Session reviewer, else the TCRMP_REVIEWER env default; '' when unknown."""
    return _norm_reviewer(session.get('reviewer') or cfg.REVIEWER)


def _seed_reviewer(value):
    """Boot-time rule for /api/configure, /api/resume and the standalone form.
    Initials in the request replace the session value. A request without
    them keeps whatever POST /api/reviewer set, because the orchestrator
    pushes the latest opener to a reused process before the page boots and
    the boot body carries no initials. Only an empty session falls back to
    the TCRMP_REVIEWER launch default. Returns the value now in force."""
    rv = _norm_reviewer(value)
    if rv:
        session['reviewer'] = rv
    elif not session.get('reviewer'):
        session['reviewer'] = _norm_reviewer(cfg.REVIEWER)
    return session['reviewer']


def _stamp_reviewer(mask):
    """Record the current reviewer on a mask whose status or label just
    changed. A blank reviewer leaves any earlier initials in place."""
    rv = _current_reviewer()
    if rv:
        mask['reviewer'] = rv
    return rv


# An accepted mask is somebody's judgement, and a mask nobody can be credited
# for is worthless downstream: it cannot be audited, questioned or counted per
# reviewer. So every ACCEPT path refuses while the session has no initials.
# Rejects, relabels and re-saves are untouched, so an operator who has not
# named themselves is told what to do rather than trapped mid-frame.
NEEDS_REVIEWER_MSG = 'Set your initials first (top bar) so this work is credited.'


def _needs_reviewer_response():
    """Refusal body for an accept attempted by an unnamed session. HTTP 200
    with ok:false, the shape the client already reads for a refused accept."""
    return jsonify({'ok': False, 'needs_reviewer': True,
                    'error': NEEDS_REVIEWER_MSG})


def _accepts_new_masks(seg, updates):
    """True when this /masks PUT would flip at least one mask from
    not-accepted to accepted. An update that re-states the status of an
    already accepted mask (what a frame save sends) is not an accept."""
    by_id = {str(m.get('id')): m for m in seg.get('masks', [])}
    for mid, new_status in (updates or {}).items():
        if new_status != 'accepted':
            continue
        if by_id.get(str(mid), {}).get('status') != 'accepted':
            return True
    return False


def _relabels_accepted(seg, relabel):
    """True when this /masks PUT would relabel a mask that is ALREADY accepted.

    Changing the species of accepted work is a judgement on accepted work, so
    it carries the same identity requirement as making it. Relabelling a
    pending mask decides nothing yet and stays free."""
    by_id = {str(m.get('id')): m for m in seg.get('masks', [])}
    return any(by_id.get(str(mid), {}).get('status') == 'accepted'
               for mid in (relabel or {}))


@app.route('/api/reviewer', methods=['GET', 'POST'])
def api_reviewer():
    """GET -> {'reviewer'}; POST {'reviewer'} sets session['reviewer']
    (normalized) and echoes it back."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        session['reviewer'] = _norm_reviewer(data.get('reviewer'))
    return jsonify({'reviewer': _current_reviewer()})


def _merge_yields_accepted(seg, ids):
    """True when merging ``ids`` would leave an ACCEPTED survivor.

    ``_merge_ids`` promotes the survivor to accepted as soon as ANY member is
    accepted, so merging is a fourth way to create accepted work: a pending
    mask merged with an accepted one comes out accepted, credited to nobody.
    A merge whose result is still pending creates nothing and stays free.
    """
    by_id = {m.get('id'): m for m in seg.get('masks', [])}
    return any((by_id.get(i) or {}).get('status') == 'accepted' for i in ids)


def _frame_has_accepted(seg):
    """True when the frame holds an accepted mask that a same-species merge
    could absorb into or rebuild. Used by /merge_same_id, which decides what to
    union inside mask_ops and mutates survivors in place, so its guard has to
    be a cheap pre-check rather than an inspection of the result."""
    return any(m.get('status') == 'accepted' for m in seg.get('masks', []))


@app.route('/api/image/<path:filename>/masks', methods=['PUT'])
@_dirty_on_success
def update_masks(filename):
    """Bulk update mask statuses (accept/reject).

    Guard: a mask cannot be set to 'accepted' while it has no species
    assigned (empty string). REVIEW masks (species=='REVIEW') are exempt,
    since that's an intentional label, not a missing one. A refused mask stays at
    its prior status (or 'pending' if it had none) and its id is reported
    back in `needs_species` so the client can prompt for a label instead of
    silently dropping the accept.

    Reviewer guard: an accept, or a relabel of an ALREADY accepted mask, by a
    session with no initials is refused whole (nothing in the request is
    applied) with needs_reviewer. Rejects, relabels of pending masks and a
    re-save of an already accepted mask are none of those and go through, so a
    frame save is never lost.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    data = request.json or {}
    updates = data.get('updates', {})  # {mask_id: status}
    relabel = data.get('relabel', {})  # {mask_id: {species, review, category, name}} (optional)

    # Expert-review lock (2026-09-06): an entry that asks to ADD a review tag
    # refuses the WHOLE request, before the reviewer guard and before any field
    # is applied, so a body holding one offending entry is never half applied.
    # An explicit review false is a CLEAR, not an add, and goes through below.
    review_locked_ids = _review_locked_relabel_ids(relabel)
    if review_locked_ids:
        return _review_locked_response(
            detail=(f"{len(review_locked_ids)} of {len(relabel)} relabel entries "
                    f"asked for a review tag, so nothing in this request was applied."),
            review_locked_ids=review_locked_ids)

    if not _current_reviewer() and (_accepts_new_masks(seg, updates)
                                    or _relabels_accepted(seg, relabel)):
        return _needs_reviewer_response()

    needs_species = []
    for mask in seg.get('masks', []):
        mid = str(mask['id'])
        if mid in relabel:
            info = relabel[mid] or {}
            if 'species' in info:
                mask['species'] = info['species']
                _stamp_reviewer(mask)
            if 'review' in info:
                mask['review'] = bool(info['review'])
            if 'category' in info:
                mask['category'] = info['category']
            if 'name' in info:
                mask['name'] = info['name']
        if mid in updates:
            new_status = updates[mid]
            if new_status == 'accepted' and _needs_species(mask):
                needs_species.append(mask['id'])
                # Refused: leave status as-is (do not accept an unlabeled mask).
            else:
                mask['status'] = new_status
                _stamp_reviewer(mask)

    seg['reviewed'] = True
    _stamp_provenance(seg, filename)
    out = {'ok': True}
    if needs_species:
        out['needs_species'] = needs_species
    return jsonify(out)


@app.route('/api/image/<path:filename>/accept_all', methods=['POST'])
@_dirty_on_success
def accept_all_masks(filename):
    """Accept all pending masks for an image.

    Guard: pending masks with no species assigned are skipped (left
    'pending') instead of being accepted unlabeled. Their ids are reported
    in `needs_species`. REVIEW masks are exempt (intentional label).

    Reviewer guard: refused with needs_reviewer when the session has no
    initials and there is anything pending to accept. With nothing pending
    the call is a plain no-op and answers normally.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    if not _current_reviewer() and any(
            m.get('status') == 'pending' for m in seg.get('masks', [])):
        return _needs_reviewer_response()

    count = 0
    needs_species = []
    for mask in seg.get('masks', []):
        if mask.get('status') == 'pending':
            if _needs_species(mask):
                needs_species.append(mask['id'])
                continue
            mask['status'] = 'accepted'
            _stamp_reviewer(mask)
            count += 1
    seg['reviewed'] = True
    _stamp_provenance(seg, filename)
    out = {'ok': True, 'accepted': count}
    if needs_species:
        out['needs_species'] = needs_species
    return jsonify(out)


def _commit_candidate_to_seg(seg, cand):
    """Promote an ephemeral scan candidate into seg['masks'] as an accepted mask.

    Scan candidates (Task 12) live only on the client until the operator keeps
    one (Task 13 Enter). Keeping it commits a COPY here: a fresh id (max existing
    + 1) so it never collides with a live mask, status='accepted' so it exports
    like any reviewed mask. Rejected candidates are dropped client-side and never
    reach this helper, so nothing dropped is ever persisted.

    Expert-review lock (2026-09-06): the copy is wholesale, so a review tag on
    the posted candidate, and any expert round-trip identifier a client sends
    with it, would ride in. Both are stripped and the candidate is kept.
    """
    masks = seg.setdefault('masks', [])
    next_id = max((m['id'] for m in masks), default=-1) + 1
    # Strip rather than refuse: this helper takes one candidate at a time out of
    # a scan that can offer many, and refusing here would cost the operator the
    # whole tray for one tagged candidate.
    tag_dropped = _review_tag_refused(cand)
    mask = strip_review_tag(cand) if tag_dropped else dict(cand)
    ids_dropped = []
    if not REVIEW_TAGGING_ENABLED:
        ids_dropped = [k for k in CLIENT_REVIEW_PROVENANCE_KEYS if k in mask]
        for key in ids_dropped:
            mask.pop(key)
    if tag_dropped or ids_dropped:
        _record_candidate_stripped()
        dropped = (['its review tag'] if tag_dropped else []) + [
            f"a client-supplied {key}" for key in ids_dropped]
        log(f"[review-lock] scan candidate committed as mask #{next_id} without "
            f"{', '.join(dropped)}. {REVIEW_LOCKED_MSG}")
    mask['id'] = next_id
    mask['status'] = 'accepted'
    masks.append(mask)
    return mask


@app.route('/api/image/<path:filename>/commit_mask', methods=['POST'])
@_dirty_on_success
def commit_mask(filename):
    """Keep a scan candidate: append it to seg['masks'] as an accepted mask.

    Guard: same as update_masks / accept_all_masks. A candidate with no
    species assigned (e.g. seeded from an unlabeled source mask via
    exemplar_scan(mode='this')), or one left with none once the expert-review
    lock strips its review tag, is refused here too, so this third accept
    path cannot bypass the label guard. The candidate is not committed and
    the response reports needs_species instead of a mask, so the client can
    prompt for a label rather than silently accepting it unlabeled.

    Reviewer guard: this route only ever creates an ACCEPTED mask, so an
    unnamed session is refused outright with needs_reviewer.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    if not _current_reviewer():
        return _needs_reviewer_response()
    cand = (request.json or {}).get('mask')
    if not cand:
        return jsonify({'error': 'no mask'}), 400
    # Expert-review lock (2026-09-06): the label guard judges the candidate as
    # it will be STORED, with any review tag stripped. A candidate kept while
    # REVIEW was armed therefore reads as unlabelled and the operator is asked
    # for a species code, instead of the frame gaining an accepted mask with no
    # species. The strip itself happens in _commit_candidate_to_seg, which is
    # the backstop for every other caller of that helper.
    if _needs_species(_review_locked_candidate(cand)):
        return jsonify({'ok': False, 'needs_species': True})
    mask = _commit_candidate_to_seg(seg, cand)
    _stamp_reviewer(mask)
    seg['reviewed'] = True
    _stamp_provenance(seg, filename)
    return jsonify({'ok': True, 'mask': wire_mask(mask)})


@app.route('/api/image/<path:filename>/refine', methods=['POST'])
@_dirty_on_success
def refine_mask(filename):
    """Refine a mask with an additional positive/negative click."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json or {}
    mask_id = data['mask_id']
    # click_x/click_y are optional on a rerun (the stored clicks are re-prompted
    # as they are), so read them defensively.
    click_x = data.get('click_x')
    click_y = data.get('click_y')
    click_label = data.get('click_label', 1)  # 1=positive, 0=negative

    # Find the mask
    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    # A merged mask is deliberately multi-part and has no click history that
    # would reproduce it, so a SAM3 re-prompt would throw its other parts away.
    if target_mask.get('source_type') == 'merged':
        return jsonify({'error': 'Merged masks keep every part; edit them with Draw (D) or Shift+drag box redraw'}), 400

    # Ensure image is embedded
    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    # Build accumulated clicks: original source click + all refinement clicks + new
    clicks = [(target_mask['source_x'], target_mask['source_y'])]
    labels = [1]

    for rc in target_mask.get('refinement_clicks', []):
        clicks.append((rc['x'], rc['y']))
        labels.append(rc['label'])

    # rerun: re-prompt exactly what is already stored (the mask-size slider fires
    # this after switching variants). Nothing new is clicked, nothing is recorded.
    rerun = bool(data.get('rerun'))
    if not rerun:
        clicks.append((click_x, click_y))
        labels.append(click_label)

    # Run SAM3 refinement. A box-made mask keeps its box alongside the clicks so
    # the re-prompt reproduces the shape the operator drew.
    result = engine.refine_mask(clicks, labels, box=target_mask.get('source_box'))
    if result is None:
        return jsonify({'error': 'refinement produced no mask'}), 400

    # REFINE keeps the strict clip-to-neighbors rule on purpose — a positive
    # click on a mask shouldn't cause it to eat an accepted neighbor just
    # because SAM3 regrew into that region. User asked for this explicitly.
    # Rejected masks are always ignored (they're "deleted" in the UI), including
    # ones the client just deleted whose status PUT may not have landed yet.
    new_binary, blockers = _clip_existing_to_neighbours(
        seg, mask_id, result['mask'], data.get('rejected_ids'))
    if new_binary is None:
        names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'other masks'
        return jsonify({'error': f'refinement fully overlapped {names}', 'blocked_by': blockers}), 400

    # Update mask geometry
    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after refinement'}), 400
    _overlap_delta = {'updated': [], 'removed': []}

    # Record the refinement click (a rerun adds no click, so it records nothing)
    if not rerun:
        target_mask['refinement_clicks'].append({
            'x': click_x, 'y': click_y, 'label': click_label,
        })
    target_mask['score'] = round(result['score'], 4)

    return jsonify({
        'ok': True,
        'mask': wire_mask(target_mask),
        'affected': _overlap_delta,
    })


@app.route('/api/image/<path:filename>/rebox', methods=['POST'])
@_dirty_on_success
def rebox_mask(filename):
    """Redraw a mask from a bounding box prompt."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mask_id = data['mask_id']
    box = data['box']  # [x_min, y_min, x_max, y_max]

    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    result = engine.segment_box(*box)
    if result is None:
        return jsonify({'error': 'box prompt produced no mask'}), 400

    # Strict clip-to-neighbors (same rule as /refine and /add). Rejected
    # masks are ignored so you can redraw cleanly over a deleted one, including
    # ones the client just deleted whose status PUT may not have landed yet.
    new_binary, blockers = _clip_existing_to_neighbours(
        seg, mask_id, result['mask'], data.get('rejected_ids'))
    if new_binary is None:
        names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'other masks'
        return jsonify({'error': f'rebox fully overlapped {names}', 'blocked_by': blockers}), 400

    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after rebox'}), 400

    # Reset refinement history since we redrew from scratch
    target_mask['refinement_clicks'] = []
    target_mask['score'] = round(result['score'], 4)
    target_mask['source_type'] = 'manual_box'
    # The redraw replaces the whole shape, so the mask is no longer the union of
    # the ids it once absorbed. Drop the stale lineage with the merged status.
    target_mask.pop('merged_from', None)
    # Keep the box so a later refine or mask-size rerun re-prompts with it, and
    # move the source point to the box centre so it no longer points at wherever
    # the mask used to start (parity with /add_box).
    target_mask['source_box'] = [int(v) for v in box]
    target_mask['source_x'] = int((target_mask['source_box'][0] + target_mask['source_box'][2]) / 2)
    target_mask['source_y'] = int((target_mask['source_box'][1] + target_mask['source_box'][3]) / 2)

    return jsonify({'ok': True, 'mask': wire_mask(target_mask)})


@app.route('/api/image/<path:filename>/draw_edit', methods=['POST'])
@_dirty_on_success
def draw_edit(filename):
    """TagLab-style freehand edit on the selected mask.

    The client sends a path as [[x,y], ...] in image coordinates. We close
    the polygon (connecting end to start if the gesture was open), rasterize
    it, and decide add-vs-subtract by testing whether the polygon centroid
    falls inside the current mask:
      - centroid inside  -> subtract the polygon from the mask (carve chunk)
      - centroid outside -> union the polygon into the mask (add chunk)

    In both cases the result is then clipped against every live (non-rejected)
    neighbor via _clip_to_neighbors, so a draw edit can never overlap an
    already-made mask (existing masks win; same strict rule as /add and
    /refine). Neighbors are never modified by the edit.
    """
    import numpy as np
    try:
        import cv2
    except ImportError:
        return jsonify({'error': 'cv2 not available — draw edit needs opencv'}), 500
    from mask_ops import rle_decode, update_mask_geometry

    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    data = request.json or {}
    mask_id = data.get('mask_id')
    raw_path = data.get('path') or []
    if not isinstance(raw_path, list) or len(raw_path) < 3:
        return jsonify({'error': 'path needs at least 3 points'}), 400

    target_mask = None
    for m in seg.get('masks', []):
        if m.get('id') == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    h = int(seg.get('image_height') or 0)
    w = int(seg.get('image_width') or 0)
    if not h or not w:
        return jsonify({'error': 'missing image dimensions'}), 400

    # Current mask as a binary array (preserve existing RLE if available).
    rle = target_mask.get('rle')
    if rle:
        current = np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
    else:
        current = np.zeros((h, w), dtype=bool)

    # Rasterize the user-drawn polygon. If the gesture was open, OpenCV's
    # fillPoly auto-closes by connecting the last vertex to the first.
    poly_pts = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in raw_path],
                          dtype=np.int32)
    poly_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_img, [poly_pts], 255)
    poly_binary = poly_img > 0
    if not poly_binary.any():
        return jsonify({'error': 'drawn polygon has no interior'}), 400

    # Centroid of the drawn polygon, clipped to image bounds.
    cx = int(np.clip(np.mean(poly_pts[:, 0]), 0, w - 1))
    cy = int(np.clip(np.mean(poly_pts[:, 1]), 0, h - 1))
    is_subtract = bool(current[cy, cx])

    if is_subtract:
        new_binary = current & ~poly_binary
    else:
        new_binary = current | poly_binary

    if not new_binary.any():
        return jsonify({'error': 'edit left an empty mask — try a different stroke'}), 400

    # Strict clip-to-neighbors: the edited mask is trimmed to stay off every
    # other live (non-rejected) mask, so a draw edit can never overlap an
    # already-made mask. Same rule as /add and /refine.
    rejected_ids = data.get('rejected_ids') or []
    blocking = [m for m in _blocking_masks(seg.get('masks', []), rejected_ids)
                if m.get('id') != mask_id]
    new_binary = _clip_to_neighbors(new_binary, blocking)
    if not new_binary.any():
        return jsonify({'error': 'edit fully overlapped other masks — try a different stroke'}), 400

    # A merged mask is deliberately multi-part, so a draw edit must not collapse
    # it to its largest component (that is what /refine sends the operator here
    # to avoid). Every other mask keeps the single-component rule.
    # min_fragment=20 on the merged path rather than the merge default of 1: a
    # subtract stroke, or the neighbour clip just above, can shave a few-pixel
    # sliver off a part, and those are noise. Every real part of the merge is
    # far above 20 px, since each one passed a 500 px gate when it was made.
    _is_merged = target_mask.get('source_type') == 'merged'
    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON,
        keep_all_components=_is_merged,
        min_fragment=20 if _is_merged else None)
    if updated is None:
        return jsonify({'error': 'mask disappeared after draw edit'}), 400

    overlap_delta = {'updated': [], 'removed': []}  # neighbors are never modified now

    # Mark as manual and record for audit. Kept out of refinement_clicks so
    # the existing SAM3 undo path isn't confused by a non-point edit.
    target_mask['score'] = 1.0
    target_mask.setdefault('draw_edits', []).append({
        'op': 'subtract' if is_subtract else 'add',
        'n_points': len(raw_path),
    })

    return jsonify({
        'ok': True,
        'mask': wire_mask(target_mask),
        'op': 'subtract' if is_subtract else 'add',
        'affected': overlap_delta,
    })


@app.route('/api/image/<path:filename>/undo_refine', methods=['POST'])
@_dirty_on_success
def undo_refine(filename):
    """Undo the last refinement click on a mask."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mask_id = data['mask_id']
    rejected_ids = data.get('rejected_ids') or []

    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    clicks = target_mask.get('refinement_clicks', [])
    if not clicks:
        return jsonify({'error': 'nothing to undo'}), 400

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    # Pop last click
    clicks.pop()
    target_mask['refinement_clicks'] = clicks

    # Re-run SAM3 with original point + remaining clicks
    all_clicks = [(target_mask['source_x'], target_mask['source_y'])]
    all_labels = [1]
    for rc in clicks:
        all_clicks.append((rc['x'], rc['y']))
        all_labels.append(rc['label'])

    # Same box rule as /refine: a box-made mask re-prompts with its box so the
    # undo reproduces the shape the operator drew.
    result = engine.refine_mask(all_clicks, all_labels, box=target_mask.get('source_box'))
    if result is None:
        return jsonify({'error': 'undo produced no mask'}), 400

    # Same subtraction rule as /refine — the re-run can still re-grow into
    # neighboring masks.
    import numpy as np
    # Exclude the mask being edited AND any rejected masks: rejected masks are
    # "deleted" in the user's mental model, so they shouldn't block a refined
    # neighbor from spilling into that region. _blocking_masks honours the
    # client-side rejected_ids (a just-deleted mask whose 'rejected' PUT has not
    # arrived yet), closing the same race the /add fix closes. Unlike /refine,
    # an undo that clips to nothing is NOT refused: it falls through to the
    # "mask disappeared after undo" answer below.
    other_masks = [m for m in _blocking_masks(seg.get('masks', []), rejected_ids)
                   if m.get('id') != mask_id]
    new_binary = _clip_to_neighbors(np.asarray(result['mask'], dtype=bool), other_masks)

    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after undo'}), 400

    target_mask['score'] = round(result['score'], 4)
    return jsonify({'ok': True, 'mask': wire_mask(target_mask),
                    'remaining_undos': len(clicks)})


@app.route('/api/image/<path:filename>/reset_mask', methods=['POST'])
@_dirty_on_success
def reset_mask(filename):
    """Reset a mask to its original SAM3 segmentation (discard all refinements)."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mask_id = data['mask_id']

    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    # A merged mask has no single prompt to reset to, and re-prompting would
    # throw its other parts away. Same refusal and advice as /refine.
    if target_mask.get('source_type') == 'merged':
        return jsonify({'error': 'Merged masks keep every part; edit them with Draw (D) or Shift+drag box redraw'}), 400

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    # Re-run the original prompt: the drawn box when the mask has one, else the
    # source point. Resetting a box mask with a point would change its shape.
    source_box = target_mask.get('source_box')
    if source_box:
        result = engine.segment_box(*source_box)
    else:
        result = engine.segment_point(target_mask['source_x'], target_mask['source_y'], label=1)
    if result is None:
        return jsonify({'error': 'reset produced no mask'}), 400

    # Strict clip-to-neighbors, same rule as /refine, /rebox and /add. A mask
    # that was clipped when it was made would otherwise regrow straight back
    # into the neighbour that clipped it, because the reset re-runs the original
    # unclipped prompt.
    new_binary, blockers = _clip_existing_to_neighbours(
        seg, mask_id, result['mask'], data.get('rejected_ids'))
    if new_binary is None:
        names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'other masks'
        return jsonify({'error': f'reset fully overlapped {names}', 'blocked_by': blockers}), 400

    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after reset'}), 400

    target_mask['refinement_clicks'] = []
    target_mask['score'] = round(result['score'], 4)
    target_mask['source_type'] = 'manual_box' if source_box else 'auto'
    return jsonify({'ok': True, 'mask': wire_mask(target_mask)})


class MergeError(Exception):
    """A merge the operator asked for cannot be done.

    `status` is the HTTP code the route should answer with, so a genuine bug
    inside _merge_ids (a KeyError on a malformed seg, say) propagates as a 500
    instead of being mistaken for "masks not found".
    """

    def __init__(self, message, status=400, ids=None):
        super().__init__(message)
        self.status = status
        self.ids = list(ids or [])


# Expert-review fields an absorbed mask can hand to the survivor. Mirrors the
# policy mask_ops.merge_overlapping_same_id applies to automatic same-species
# grouping, so a hand-picked merge does not quietly discard a pending expert ID.
def _accepted_expert_code(mask):
    """The expert-ACCEPTED species code on a mask, or None.

    Same notion mask_ops._merge_review_state uses: only an expert_id dict whose
    mode is EXPERT counts. A pending or auto-suggested id is not an accepted ID
    and never conflicts with anything.
    """
    eid = (mask or {}).get('expert_id')
    if isinstance(eid, dict) and eid.get('mode') == 'EXPERT':
        return eid.get('code')
    return None


def _expert_id_conflict(by_id, ids):
    """First pair in `ids` (ascending) whose accepted expert IDs disagree.

    Returns (id_a, code_a, id_b, code_b) or None. Run BEFORE any geometry is
    touched so a refused merge leaves the frame exactly as it was.
    """
    first_id = None
    first_code = None
    for i in ids:
        code = _accepted_expert_code(by_id[i])
        if code is None:
            continue
        if first_code is None:
            first_id, first_code = i, code
        elif code != first_code:
            return first_id, first_code, i, code
    return None


def _merge_ids(seg, ids):
    """Union the masks with these ids into the lowest id. Returns (survivor, removed_ids).

    Every part survives (keep_all_components=True): a merge is deliberately
    multi-part, unlike a single SAM3 prompt which is cleaned to one component.
    Size is no longer a gate here: update_mask_geometry runs clean_mask with
    min_fragment=1 for a merge, so every part is kept including sub-500 px ones,
    because each input mask already passed its own 500 px gate when it was
    created. Merging a speck into a colony preserves the speck.

    The survivor inherits every absorbed mask's expert-review state via
    mask_ops._merge_review_state (the same helper the automatic same-species
    grouping uses, so a hand-picked merge and an automatic one treat review
    state identically) and its merge lineage, so re-merging a merged mask
    extends `merged_from` rather than overwriting it.

    Raises MergeError (carrying the HTTP status the route should use) for every
    condition the operator can fix: non-integer ids, fewer than two distinct
    ids, an id that is not on the frame (404), a rejected mask, two masks
    carrying different accepted expert IDs, two masks each awaiting their own
    expert ID, or a union that cleans away to nothing.
    """
    from mask_ops import rle_decode, _merge_review_state
    by_id = {m['id']: m for m in seg.get('masks', [])}
    try:
        ids = sorted({int(i) for i in ids})
    except (TypeError, ValueError):
        raise MergeError('mask_ids must be integers')
    if len(ids) < 2:
        raise MergeError('merge needs two different masks')
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise MergeError(f'masks not found: {missing}', status=404, ids=missing)
    rejected = [i for i in ids if by_id[i].get('status') == 'rejected']
    if rejected:
        raise MergeError(f'cannot merge a rejected mask (#{rejected[0]})')
    # Two different expert-accepted IDs cannot become one organism. Checked here,
    # before any geometry changes, so the refusal leaves the frame untouched.
    conflict = _expert_id_conflict(by_id, ids)
    if conflict:
        id_a, code_a, id_b, code_b = conflict
        raise MergeError(f'masks #{id_a} and #{id_b} carry different expert IDs '
                         f'({code_a} vs {code_b}); resolve that in Edit Masks before merging')
    # Two masks each waiting on their OWN expert ID cannot become one organism
    # either: a survivor carries a single review_uid, so the other round trip
    # would be dropped without a trace. Same placeholder notion
    # mask_ops.merge_overlapping_same_id uses when it passes REVIEW masks over,
    # and checked in the same pre-pass so a refusal changes no geometry.
    from mask_ops import PLACEHOLDER_SPECIES
    awaiting = [i for i in ids if (by_id[i].get('species') or '') in PLACEHOLDER_SPECIES
                and by_id[i].get('review_uid')]
    if len({by_id[i]['review_uid'] for i in awaiting}) > 1:
        a = awaiting[0]
        b = next(i for i in awaiting if by_id[i]['review_uid'] != by_id[a]['review_uid'])
        raise MergeError(f'masks #{a} and #{b} are each awaiting their own expert ID; '
                         f'resolve those before merging')

    h, w = seg['image_height'], seg['image_width']
    parts = [rle_decode(by_id[i]['rle'], (h, w)) for i in ids]
    survivor = by_id[ids[0]]
    updated = update_mask_geometry(survivor, union_many(parts), cfg.POLYGON_SIMPLIFY_EPSILON,
                                   keep_all_components=True, min_fragment=1)
    if updated is None:
        raise MergeError('merge produced empty mask')

    if any(by_id[i].get('status') == 'accepted' for i in ids):
        survivor['status'] = 'accepted'
        # The merge produced (or preserved) accepted work, so credit whoever
        # did the merge. The route refuses an unnamed session before we get
        # here, so this always has initials to write.
        _stamp_reviewer(survivor)
    for i in ids[1:]:
        # The pre-pass above already ruled out every conflict, so this cannot
        # refuse. Raise rather than silently drop review state if it ever does.
        if not _merge_review_state(survivor, by_id[i]):
            raise MergeError(f'masks #{ids[0]} and #{i} carry different expert IDs; '
                             f'resolve that in Edit Masks before merging')
    survivor['source_type'] = 'merged'
    # Lineage, not just this merge: a survivor that was already a merge keeps
    # every id it has ever absorbed.
    lineage = set()
    for i in ids:
        lineage.update(by_id[i].get('merged_from') or [])
        lineage.add(i)
    survivor['merged_from'] = sorted(lineage)
    # The click history no longer reproduces this shape, so drop it: /refine
    # refuses merged masks and points the operator at Draw instead.
    survivor['refinement_clicks'] = []
    # A merged shape has no single box that reproduces it, so drop the inherited
    # one: /rebox would re-prompt from it and collapse every other part.
    survivor.pop('source_box', None)
    removed = ids[1:]
    seg['masks'] = [m for m in seg['masks'] if m['id'] not in removed]
    return survivor, removed


@app.route('/api/image/<path:filename>/merge', methods=['POST'])
@_dirty_on_success
def merge_masks(filename):
    """Merge two masks into one.

    Reviewer guard: a merge whose survivor would be ACCEPTED is an accept, so
    an unnamed session is refused with needs_reviewer and no geometry changes.
    A merge of pending masks alone stays free.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    data = request.json
    id_a = data['mask_id_a']
    id_b = data['mask_id_b']

    if not _current_reviewer() and _merge_yields_accepted(seg, [id_a, id_b]):
        return _needs_reviewer_response()

    try:
        survivor, removed = _merge_ids(seg, [id_a, id_b])
    except MergeError as e:
        # Keep this route's long-standing not-found wording; every other
        # condition reports the shared message.
        msg = 'one or both masks not found' if e.status == 404 else str(e)
        return jsonify({'error': msg}), e.status

    return jsonify({'ok': True, 'mask': wire_mask(survivor), 'removed_id': removed[0]})


@app.route('/api/image/<path:filename>/merge_many', methods=['POST'])
@_dirty_on_success
def merge_many(filename):
    """Union 2+ masks into one (lowest id survives, every part kept)."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    # No pre-guard on the id list here: set() raises on an unhashable id (a list,
    # say) and answers 500. _merge_ids validates instead, and answers 400 for a
    # non-integer id and for fewer than two distinct ids.
    ids = (request.json or {}).get('mask_ids') or []
    if not _current_reviewer() and _merge_yields_accepted(seg, ids):
        return _needs_reviewer_response()
    try:
        survivor, removed = _merge_ids(seg, ids)
    except MergeError as e:
        return jsonify({'error': str(e)}), e.status
    return jsonify({'ok': True, 'mask': wire_mask(survivor), 'removed_ids': removed})


@app.route('/api/image/<path:filename>/merge_same_id', methods=['POST'])
@_dirty_on_success
def merge_same_id(filename):
    """Union OVERLAPPING masks that share the same species/label into one.

    On-demand, because segmentation-time overlap resolution uses
    OVERLAP_STRATEGY='larger_wins', which DROPS the smaller of two overlapping
    same-species masks rather than unioning them — so two SAM3 clicks on one
    colony can leave the colony split across masks (one trimmed away). This
    endpoint rebuilds the current frame's masks by unioning every overlapping
    same-species group via mask_ops.merge_overlapping_same_id (pure RLE math,
    no SAM3 engine needed) and reports how many masks were merged away.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    h = int(seg.get('image_height') or 0)
    w = int(seg.get('image_width') or 0)
    if not h or not w:
        return jsonify({'error': 'missing image dimensions'}), 400

    # merge_overlapping_same_id rebuilds survivors IN PLACE, so the reviewer
    # guard has to run before it, not on its result: any accepted mask on the
    # frame could be a survivor that absorbs a neighbour.
    if not _current_reviewer() and _frame_has_accepted(seg):
        return _needs_reviewer_response()

    masks = seg.get('masks', [])
    before_area = {m.get('id'): m.get('area') for m in masks}
    new_masks, merged, refused = merge_overlapping_same_id(
        masks, h, w, simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON)
    # Credit the merge on exactly the survivors it rebuilt: same id, new area.
    for m in new_masks:
        if m.get('status') == 'accepted' and m.get('area') != before_area.get(m.get('id')):
            _stamp_reviewer(m)
    seg['masks'] = new_masks
    return jsonify({
        'ok': True,
        'merged': merged,
        'refused': refused,
        'masks': wire_masks(new_masks),
    })


# Mask fields the operator owns locally — a disk reload must NOT clobber these
# (they hold unsaved accept/reject decisions and geometry edits).
_OPERATOR_OWNED_FIELDS = (
    'status', 'rle', 'polygon_px', 'polygon_norm', 'bbox', 'area',
    'refinement_clicks', 'draw_edits', 'source_type', 'source_x', 'source_y',
)
# Imported review fields a disk reload MAY merge in (expert pipeline output).
_IMPORTED_REVIEW_FIELDS = ('reviews', 'expert_id', 'review_uid', 'review')


def _merge_imported_review_fields(current_masks, disk_masks):
    """Merge expert-review fields from a freshly-read disk copy onto the
    operator's in-memory masks WITHOUT clobbering unsaved accept/reject or
    geometry edits.

    Matching key: review_uid first (stable across the expert round-trip), then
    id. For each matched mask we copy the imported review fields
    (reviews[], expert_id, review_uid, review) and — only when an expert
    actually accepted an ID (expert_id.mode == 'EXPERT') — the accepted species
    /name/category. Operator-owned fields (status, geometry, refinement) are
    left exactly as the operator has them in memory.

    Returns (merged_count, expert_count): masks that gained imported fields,
    and of those how many carry an expert-accepted ID.

    DELIBERATELY EXEMPT from the expert-review lock of 2026-09-06. Restoring a
    tag that is already in the file on disk is preservation, not addition: it is
    how the review tags written before the lock survive a reload, and how an
    expert identification coming back from the review repo reaches the operator
    without a restart. Do not add a refusal here.
    """
    def _key_index(masks):
        by_uid, by_id = {}, {}
        for m in masks:
            uid = m.get('review_uid')
            if uid:
                by_uid[uid] = m
            if m.get('id') is not None:
                by_id[m.get('id')] = m
        return by_uid, by_id

    disk_by_uid, disk_by_id = _key_index(disk_masks)

    merged_count = 0
    expert_count = 0
    for cm in current_masks:
        dm = None
        uid = cm.get('review_uid')
        if uid and uid in disk_by_uid:
            dm = disk_by_uid[uid]
        elif cm.get('id') in disk_by_id:
            dm = disk_by_id[cm.get('id')]
        if dm is None:
            continue

        touched = False
        for f in _IMPORTED_REVIEW_FIELDS:
            if f in dm and dm.get(f) != cm.get(f):
                cm[f] = dm[f]
                touched = True

        eid = dm.get('expert_id')
        if eid and isinstance(eid, dict) and eid.get('mode') == 'EXPERT':
            # An expert accepted an ID — promote the accepted species/name/cat
            # so step-5 colours it expert-green. This is review metadata, not an
            # operator geometry edit, so it's safe to merge.
            for f in ('species', 'name', 'category'):
                if f in dm and dm.get(f) != cm.get(f):
                    cm[f] = dm[f]
                    touched = True
            expert_count += 1
        if touched:
            merged_count += 1

    return merged_count, expert_count


@app.route('/api/reload_segmentations', methods=['POST'])
@_dirty_on_success
def reload_segmentations():
    """Re-read the current image's segmentations.json from disk and MERGE IN the
    imported review/expert fields (reviews[], expert_id, species-when-accepted)
    onto the in-memory masks by review_uid/id — WITHOUT clobbering the
    operator's unsaved accept/reject or geometry edits.

    Lets imported expert/pending colours appear without restarting the app:
    after the importer (or a teammate) writes accepted IDs into the on-disk
    segmentations.json, the operator hits "Reload IDs" and the freshly-confirmed
    masks light up green / pending-amber in place.
    """
    data = request.json or {}
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'filename required'}), 400

    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    export_dir = session.get('export_dir')
    if not export_dir:
        return jsonify({'error': 'no export dir configured'}), 400

    disk_segs = load_segmentations(export_dir)
    disk_seg = disk_segs.get(filename)
    if not disk_seg:
        return jsonify({'error': 'no on-disk segmentation for this frame'}), 404

    merged_count, expert_count = _merge_imported_review_fields(
        seg.get('masks', []), disk_seg.get('masks', []))

    return jsonify({
        'ok': True,
        'merged': merged_count,
        'expert': expert_count,
        'masks': wire_masks(seg.get('masks', [])),
    })


def _orchestrator_url():
    """The orchestrator base URL the 'Done' button can return to, if launched
    from the pipeline orchestrator (else '')."""
    return os.environ.get('TCRMP_ORCHESTRATOR_URL', '').rstrip('/')


def _build_exclusion_mask(seg, margin_px=20):
    """Build a binary mask of all accepted/pending regions + a margin buffer.
    This prevents exemplar candidates from being thin borders around existing masks."""
    h, w = seg['image_height'], seg['image_width']
    occupied = np.zeros((h, w), dtype=np.uint8)
    for m in seg.get('masks', []):
        if m.get('status') != 'rejected' and 'rle' in m:
            occupied |= rle_decode(m['rle'], (h, w)).astype(np.uint8)
    # Dilate to add margin — prevents thin halo masks
    if margin_px > 0 and occupied.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2, margin_px * 2))
        occupied = cv2.dilate(occupied, kernel)
    return occupied.astype(bool)


def _filter_exemplar_results(results, seg, species, category, threshold):
    """Filter exemplar scan results: no overlap, no halos, score >= threshold.
    Returns list of mask dicts, sorted by score descending."""
    from mask_ops import clean_mask
    h, w = seg['image_height'], seg['image_width']
    exclusion = _build_exclusion_mask(seg, margin_px=15)

    next_id = max((m['id'] for m in seg.get('masks', [])), default=-1) + 1
    candidates = []

    for r in results:
        if r['score'] < threshold:
            continue
        binary = r['mask']
        # Resize mask to image dimensions if needed (SAM3 may return at model resolution)
        if binary.shape != (h, w):
            binary = cv2.resize(binary.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        # Subtract exclusion zone (existing masks + margin)
        binary = binary & ~exclusion
        binary = clean_mask(binary, min_fragment=cfg.MIN_MASK_AREA_PX).astype(bool)
        if binary.sum() < cfg.MIN_MASK_AREA_PX:
            continue

        point_info = {
            'label': f"E{next_id}",
            'species': species,
            'name': species,
            'category': category,
            'x': float(np.mean(np.where(binary)[1])),
            'y': float(np.mean(np.where(binary)[0])),
        }

        mask_dict = build_mask_dict(
            mask_id=next_id,
            binary_mask=binary,
            score=r['score'],
            point_info=point_info,
            source_type='exemplar',
            simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON,
        )
        if mask_dict is not None:
            candidates.append((r['score'], mask_dict, binary))
            next_id += 1

    # Sort by score descending, remove inter-candidate overlaps (keep higher score)
    candidates.sort(key=lambda x: x[0], reverse=True)
    kept = []
    kept_union = np.zeros((h, w), dtype=bool)
    for score, md, binary in candidates:
        overlap = (binary & kept_union).sum()
        if overlap > binary.sum() * 0.3:
            continue  # too much overlap with a higher-scored candidate
        kept.append(md)
        kept_union |= binary

    return kept


class _StubExemplarEngine:
    """TEST-ONLY fake engine. Returns ``n`` synthetic candidate masks so the
    no-persist contract test can run ``_exemplar_candidates`` headlessly without
    loading SAM3. NOT used by the running app — the route always passes the real
    ``session['sam_engine']``. Kept here (not in tests/) so the pure helper has a
    drop-in stand-in with the exact ``exemplar_scan(bbox) -> [{mask, score, ...}]``
    contract.
    """

    def __init__(self, n=2):
        self.n = n

    def exemplar_scan(self, bbox):  # pragma: no cover - exercised only by tests
        # Emit n disjoint rectangles with high scores. Each is well over
        # MIN_MASK_AREA_PX and sits at x>=28 (clear of a seed mask in the
        # top-left + its exclusion margin), in distinct y-bands, so
        # _filter_exemplar_results keeps every one rather than subtracting or
        # de-duping them away. Assumes the test image is 48x64.
        out = []
        for i in range(self.n):
            y0 = 6 + i * 20
            binary = np.zeros((48, 64), dtype=bool)
            binary[y0:y0 + 16, 30:62] = True   # 16 x 32 = 512 px
            out.append({'mask': binary, 'score': 0.95 - i * 0.01, 'bbox': None})
        return out


def _exemplar_candidates(seg, mode, threshold, mask_id, engine):
    """Pure candidate finder for the exemplar scan — RETURNS a list of candidate
    mask dicts and NEVER mutates ``seg['masks']``.

    This holds the exact candidate-finding math the old route ran inline; the
    only change is WHERE results accumulate. To preserve the two behaviours the
    old code got from ``seg['masks'].extend(...)`` between species —
      (1) later species treat earlier candidates as occupied (exclusion), and
      (2) ids stay unique across species (``next_id = max(id)+1``) —
    we accumulate into a *working copy* of the mask list (``work_seg``) and pass
    that to ``_filter_exemplar_results``. The caller's ``seg`` is untouched, so
    candidates are ephemeral until the client decides to keep them (Task 13).

    Returns a flat list of candidate mask dicts (the route derives the
    ``species_scanned`` summary itself from ``seg``).
    """
    # Shallow working seg: same dims, a *copy* of the masks list (the dicts are
    # shared, but we only ever read them or append our own new dicts — never edit
    # the originals — so the real seg['masks'] list is never extended).
    work_seg = {
        'image_height': seg['image_height'],
        'image_width': seg['image_width'],
        'masks': list(seg.get('masks', [])),
    }

    all_new_masks = []

    if mode == 'this':
        source_mask = None
        for m in seg.get('masks', []):
            if m['id'] == mask_id:
                source_mask = m
                break
        if source_mask is None:
            raise LookupError('mask not found')
        bbox = source_mask.get('bbox')
        if not bbox:
            raise ValueError('mask has no bounding box')

        results = engine.exemplar_scan(bbox)
        all_new_masks = _filter_exemplar_results(
            results, work_seg,
            source_mask.get('species', 'UNK'),
            source_mask.get('category', ''),
            threshold,
        )

    elif mode == 'all':
        # Scan using every unique accepted/pending species as exemplar.
        for source in _exemplar_sources_all(seg):
            results = engine.exemplar_scan(source['bbox'])
            new_masks = _filter_exemplar_results(
                results, work_seg,
                source.get('species', 'UNK'),
                source.get('category', ''),
                threshold,
            )
            # Accumulate into the WORKING copy only — so the next species sees
            # these as occupied and gets fresh ids. seg['masks'] stays untouched.
            work_seg['masks'].extend(new_masks)
            all_new_masks.extend(new_masks)

    return all_new_masks


def _exemplar_sources_all(seg):
    """The unique accepted/pending masks (one per species, first seen) used as
    exemplars for a mode='all' scan. Shared by the helper and the route's
    species_scanned summary so they can never disagree."""
    seen_species = set()
    sources = []
    for m in seg.get('masks', []):
        if m.get('status') not in ('accepted', 'pending'):
            continue
        sp = m.get('species', '')
        if sp and sp not in seen_species and m.get('bbox'):
            seen_species.add(sp)
            sources.append(m)
    return sources


@app.route('/api/image/<path:filename>/exemplar', methods=['POST'])
def exemplar_scan(filename):
    """Scan for similar regions. mode='this' uses one mask, mode='all' uses all
    accepted masks. Candidates are RETURNED as new_masks and are NOT persisted to
    seg['masks'] — the client holds them ephemerally in a review panel (Task 12)."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mode = data.get('mode', 'this')  # 'this' or 'all'
    threshold = data.get('threshold', 0.4)
    mask_id = data.get('mask_id')

    if mode == 'this' and mask_id is None:
        return jsonify({'error': 'mask_id required for mode=this'}), 400

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    try:
        new_masks = _exemplar_candidates(seg, mode, threshold, mask_id, engine)
    except LookupError:
        return jsonify({'error': 'mask not found'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    species_scanned = ([s.get('species', '') for s in _exemplar_sources_all(seg)]
                       if mode == 'all' else [])

    return jsonify({
        'ok': True,
        'count': len(new_masks),
        'new_masks': new_masks,
        'species_scanned': sorted(species_scanned),
    })


@app.route('/api/image/<path:filename>/scrap', methods=['POST'])
@_dirty_on_success
def scrap_segmentation(filename):
    """Mark a frame as scrapped — excluded from review and export.
    Mirrors the step 4 scrap API so the two apps behave the same way.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    seg['scrapped'] = True
    seg['reviewed'] = True
    result, conflict_path = flush_all()
    _build_review_list()
    out = {'ok': result != 'conflict', 'remaining': len(session['review_files'])}
    out.update(_flush_result_fields(result, conflict_path))
    return jsonify(out)


@app.route('/api/image/<path:filename>/unscrap', methods=['POST'])
@_dirty_on_success
def unscrap_segmentation(filename):
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    seg['scrapped'] = False
    result, conflict_path = flush_all()
    _build_review_list()
    out = {'ok': result != 'conflict', 'remaining': len(session['review_files'])}
    out.update(_flush_result_fields(result, conflict_path))
    return jsonify(out)


def _blocking_masks(masks, rejected_ids=None):
    """Masks that a new or edited mask must clip against: the live (non-rejected)
    ones, minus any the client has just deleted (rejected_ids). The client sends
    its authoritative deleted set so a fast delete -> add/refine is NOT clipped by
    a mask whose 'rejected' status PUT has not landed yet. Flask runs threaded, so
    the reject PUT and the add/refine POST can race; without this, the just-deleted
    mask would intermittently block a new overlapping mask in its location.
    Backend status='rejected' is honored too.
    """
    rej = set(rejected_ids or [])
    return [m for m in (masks or [])
            if m.get('status') != 'rejected' and m.get('id') not in rej]


def _clip_to_neighbors(new_binary, blocking):
    """Subtract every blocking mask's footprint from new_binary so a new or
    edited mask stays on its own side of live neighbors (existing masks win).
    Same rule /add and /refine use. `blocking` is the output of _blocking_masks.
    """
    import numpy as np
    from mask_ops import rle_decode
    if not blocking:
        return new_binary
    h, w = new_binary.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for m in blocking:
        rle = m.get('rle')
        if not rle:
            continue
        try:
            forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
        except Exception:
            continue
    return new_binary & ~forbidden


def _overlapping_blockers(new_binary, blocking):
    """Which blocking masks actually intersect new_binary — for user-facing
    'blocked by mask #N' messages. `blocking` is the output of _blocking_masks.
    Returns [{'id','species','label'}, ...] ordered by descending overlap area.
    """
    import numpy as np
    from mask_ops import rle_decode
    h, w = new_binary.shape
    hits = []
    for m in blocking:
        rle = m.get('rle')
        if not rle:
            continue
        try:
            b = np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
        except Exception:
            continue
        inter = int((new_binary & b).sum())
        if inter > 0:
            hits.append((inter, {'id': m.get('id'),
                                 'species': m.get('species', ''),
                                 'label': m.get('label', '')}))
    hits.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in hits]


def _clip_existing_to_neighbours(seg, mask_id, binary, rejected_ids=None):
    """Clip an EDITED mask against every live neighbour but itself.

    The sibling of _new_mask_from_binary, for the routes that re-prompt a mask
    that already exists (/refine, /rebox, /reset_mask). Returns
    (clipped_binary, []) normally, or (None, blockers) when the clip leaves
    nothing, so the caller can answer 400 with its own wording plus blocked_by.

    Rejected masks never block: they are "deleted" in the operator's mental
    model. `rejected_ids` carries the client's just-deleted set so a fast
    delete then re-prompt is not blocked by a mask whose reject PUT has not
    arrived yet.
    """
    import numpy as np
    pre_clip = np.asarray(binary, dtype=bool)
    blocking = [m for m in _blocking_masks(seg.get('masks', []), rejected_ids or [])
                if m.get('id') != mask_id]
    if not blocking:
        return pre_clip, []
    clipped = _clip_to_neighbors(pre_clip, blocking)
    if not clipped.any():
        return None, _overlapping_blockers(pre_clip, blocking)
    return clipped, []


def _new_mask_from_binary(seg, binary, point_info, source_type, rejected_ids, score):
    """Clip a fresh SAM3 mask to live neighbours and build its dict, or return
    (None, blockers) when nothing survives. Shared by /add and /add_box.

    blockers is non-empty only when the clip emptied the mask; a None mask with
    an empty blockers list means build_mask_dict rejected it as too small.
    """
    import numpy as np
    from mask_ops import rle_decode
    existing_masks = seg.get('masks', [])
    blocking = _blocking_masks(existing_masks, rejected_ids or [])
    new_binary = np.asarray(binary, dtype=bool)
    pre_clip = new_binary.copy()
    if blocking:
        h, w = new_binary.shape
        forbidden = np.zeros((h, w), dtype=bool)
        for m in blocking:
            rle = m.get('rle')
            if not rle:
                continue
            try:
                forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
            except Exception:
                continue
        new_binary = new_binary & ~forbidden
        if not new_binary.any():
            return None, _overlapping_blockers(pre_clip, blocking)
    next_id = max((m['id'] for m in existing_masks), default=-1) + 1
    mask_dict = build_mask_dict(mask_id=next_id, binary_mask=new_binary, score=score,
                                point_info=point_info, source_type=source_type,
                                simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON)
    return mask_dict, []


@app.route('/api/image/<path:filename>/add', methods=['POST'])
@_dirty_on_success
def add_mask(filename):
    """Add a new mask at a click point (SAM3 point prompt).

    Edited-wins policy: the new mask claims whatever region SAM3 gives it.
    Any live (non-rejected) neighbor that overlaps yields that region —
    its RLE + polygon get re-computed without the new mask's footprint.
    Rejected masks are untouched (they're "deleted" in the user's mental
    model, so they can't block a new mask either).
    """
    import numpy as np

    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json or {}
    # Expert-review lock (2026-09-06): a body asking for a tag is refused here,
    # before any segmentation runs, so no mask is created and the frame is left
    # exactly as it was.
    if _review_tag_refused(data):
        return _review_locked_response()
    click_x = data['click_x']
    click_y = data['click_y']
    species = data.get('species', '')
    category = data.get('category', '')
    label = data.get('label', '?')
    is_review = bool(data.get('review'))

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    result = engine.segment_point(click_x, click_y, label=1)
    if result is None:
        return jsonify({'error': 'no mask generated at this point'}), 400

    new_binary = np.asarray(result['mask'], dtype=bool)
    if not new_binary.any():
        return jsonify({'error': 'no mask generated at this point'}), 400

    # Strict clip-to-neighbors: the new mask gets trimmed to stay on its side
    # of any LIVE (non-rejected) neighbor. Rejected masks are "deleted" in the
    # user's mental model and do NOT block a new overlapping mask.
    point_info = {
        'label': label, 'species': species, 'name': species,
        'category': category, 'x': click_x, 'y': click_y,
    }
    mask_dict, blockers = _new_mask_from_binary(
        seg, new_binary, point_info, 'manual_click',
        data.get('rejected_ids'), result['score'])
    if mask_dict is None:
        if blockers:
            names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'existing masks'
            return jsonify({
                'error': f'new mask fully overlaps {names} — delete it or click a clearer spot',
                'blocked_by': blockers,
            }), 400
        return jsonify({'error': 'mask too small or empty after clip'}), 400

    # REVIEW arming: flag the mask so YOLO export skips it and the batch export
    # ships it to the expert repo (parity with _process_single_image). The
    # source_type stays manual_click, a human placed it. Reachable only while
    # REVIEW_TAGGING_ENABLED is True: under the lock the request was refused at
    # the top of this route.
    if is_review:
        mask_dict['review'] = True
        mask_dict['species'] = 'REVIEW'
        mask_dict['category'] = 'Review'

    seg['masks'].append(mask_dict)
    return jsonify({'ok': True, 'mask': wire_mask(mask_dict)})


@app.route('/api/image/<path:filename>/add_box', methods=['POST'])
@_dirty_on_success
def add_mask_box(filename):
    """Add a new mask from a box prompt (Box mode, B). Same clip and provenance
    rules as /add; source_type manual_box; the box is kept for later re-prompts."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json or {}
    # Expert-review lock (2026-09-06): same refusal as /add, before the box is
    # even read, so no mask is created and the frame is left exactly as it was.
    if _review_tag_refused(data):
        return _review_locked_response()
    box = data.get('box') or []
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return jsonify({'error': 'box must be [x0, y0, x1, y1]'}), 400
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return jsonify({'error': 'box too small'}), 400

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    result = engine.segment_box(x0, y0, x1, y1)
    if result is None:
        return jsonify({'error': 'no mask generated in this box'}), 400
    import numpy as np
    if not np.asarray(result['mask'], dtype=bool).any():
        return jsonify({'error': 'no mask generated in this box'}), 400

    species = data.get('species', '')
    category = data.get('category', '')
    is_review = bool(data.get('review'))
    point_info = {'label': data.get('label', '?'), 'species': species, 'name': species,
                  'category': category, 'x': int((x0 + x1) / 2), 'y': int((y0 + y1) / 2)}
    mask_dict, blockers = _new_mask_from_binary(seg, result['mask'], point_info, 'manual_box',
                                                data.get('rejected_ids'), result['score'])
    if mask_dict is None:
        if blockers:
            names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'existing masks'
            return jsonify({'error': f'new mask fully overlaps {names}: delete it or draw a clearer box',
                            'blocked_by': blockers}), 400
        return jsonify({'error': 'mask too small or empty after clip'}), 400

    # Keep the drawn box so /refine and the mask-size rerun re-prompt with it.
    mask_dict['source_box'] = [int(x0), int(y0), int(x1), int(y1)]

    # REVIEW arming: flag the mask so YOLO export skips it and the batch export
    # ships it to the expert repo (parity with /add). The source_type stays
    # manual_box, a human drew it. Reachable only while REVIEW_TAGGING_ENABLED
    # is True: under the lock the request was refused at the top of this route.
    if is_review:
        mask_dict['review'] = True
        mask_dict['species'] = 'REVIEW'
        mask_dict['category'] = 'Review'

    seg['masks'].append(mask_dict)
    return jsonify({'ok': True, 'mask': wire_mask(mask_dict)})


def _derive_project_id(export_dir):
    parts = os.path.abspath(export_dir).split(os.sep)
    for p in reversed(parts):
        if p.startswith('run_'):
            return p
    return os.path.basename(os.path.dirname(os.path.abspath(export_dir))) or 'project'


def _stamp_provenance(seg, filename):
    """Compute per-(image,label) outcomes for a finished/exported frame and
    upsert the flat ledger. Stamps seg['label_outcomes'] in place. The CSV
    write is wrapped non-fatally so a provenance hiccup never blocks review or
    export.

    No-rescreening (2026-09-27): a target label another project already
    screened this same image for (session['prior_screening'][stem]['screened'])
    is passed to compute_label_outcomes as screened_elsewhere, so a frame
    left without a local mask for that label gets no fresh not_found row; a
    label this project DOES find a mask for is unaffected. Persistence of the
    block rides on the existing flush.
    """
    export_dir = session['export_dir']
    screened_elsewhere = frozenset(
        (_prior_screening_for(filename).get('screened') or {}).keys())
    outcomes = compute_label_outcomes(
        seg, _target_species(), source=cfg.PROVENANCE_SOURCE,
        reviewer=_current_reviewer(), screened_elsewhere=screened_elsewhere)
    seg['label_outcomes'] = outcomes
    try:
        write_provenance_csv(
            export_dir,
            os.path.splitext(filename)[0],
            outcomes,
            cfg.PROJECT_ID or _derive_project_id(export_dir))
    except Exception as e:
        log(f"[provenance] csv write failed for {filename} (non-fatal): {e}")


def _count_review_masks(to_export, batch_files):
    """How many masks in this batch carry an expert-review tag.

    Parameters:
        to_export: {filename: segmentation dict} for the batch.
        batch_files: the filenames in the batch, in export order.

    Returns:
        The number of tagged masks, counted with the same two spellings the
        export selector itself reads: the review boolean, or a species of
        REVIEW. A frame missing from to_export contributes nothing.
    """
    n_review = 0
    for fn in batch_files:
        seg = to_export.get(fn) or {}
        for m in seg.get('masks', []):
            if m.get('review') or m.get('species') == 'REVIEW':
                n_review += 1
    return n_review


def _export_review_bundle(to_export, batch_files):
    """Ship this batch's REVIEW-flagged masks to the expert-review repo +
    library, and auto-relabel any that a past expert already identified.
    Best-effort: any failure is logged and never breaks the YOLO export.

    SUSPENDED while REVIEW_TAGGING_ENABLED is False (2026-09-06). This is the
    one path that publishes outward: it mints a fresh review identifier onto
    every already-flagged mask and pushes the bundle to a public GitHub Pages
    repository. Publishing 130 masks outward while flagging is locked would
    contradict the lock, so the function returns early with the count of masks
    it did not publish, in the same shape it already uses when the exporter is
    unavailable. Reversing it is the one constant, never the environment
    variable TCRMP_REVIEW_GIT_PUSH, which any hand launch can flip back.
    """
    if not REVIEW_TAGGING_ENABLED:
        n_review = _count_review_masks(to_export, batch_files)
        log(f"[review-lock] expert-review export SUSPENDED: {n_review} flagged "
            f"mask(s) in this batch were not published and not given a review "
            f"identifier. {REVIEW_LOCKED_MSG}")
        return {'disabled': True, 'review_masks_skipped': n_review}
    if not _REVIEW_OK:
        # Surface the disabled state instead of dropping REVIEW masks silently:
        # if any frame in this batch carries a REVIEW-flagged mask, the operator
        # must know it did not ship to the expert repo.
        n_review = _count_review_masks(to_export, batch_files)
        if n_review:
            log(f"[review] WARNING: expert-review export DISABLED (_reefreview "
                f"import failed) but {n_review} REVIEW mask(s) in this batch were "
                f"NOT shipped. Fix the _reefreview import and re-export.")
        else:
            log("[review] expert-review export disabled (_reefreview import "
                "failed); no REVIEW masks in this batch.")
        return {'disabled': True, 'review_masks_skipped': n_review}
    try:
        export_dir = session['export_dir']
        contacts = session.get('contacts') or [
            c.strip() for c in (cfg.REVIEW_CONTACTS or '').split(',') if c.strip()]
        # Canonical identity from the orchestrator (project.json); fall back to
        # the path-derived id (and id-as-name) when launched standalone.
        project_id = cfg.PROJECT_ID or _derive_project_id(export_dir)
        project_name = cfg.PROJECT_NAME or project_id
        stats = _review_export.export_flagged_masks(
            to_export, batch_files,
            export_dir=export_dir,
            review_dir=session.get('review_dir') or cfg.REVIEW_DIR,
            repo_url=cfg.REVIEW_REPO_URL,
            library_dir=cfg.EXPERT_LIBRARY_DIR or None,
            master_codes=cfg.MASTER_CODES_CSV,
            contacts=contacts,
            featured_codes=_target_species(),
            project_id=project_id,
            project_name=project_name,
            pad_px=cfg.REVIEW_CROP_PAD_PX,
            max_edge=cfg.REVIEW_MAX_EDGE,
            full_max_edge=cfg.REVIEW_FULL_MAX_EDGE,
            overlap_thresh=cfg.REVIEW_OVERLAP_THRESH,
            git_push=cfg.REVIEW_GIT_PUSH,
            log_fn=lambda m: log(m),
        )
        if stats.get('new') or stats.get('auto_relabeled') or stats.get('skipped_missing_image'):
            log(f"[review] batch: +{stats.get('new', 0)} queued, "
                f"{stats.get('auto_relabeled', 0)} auto-relabeled, "
                f"{stats.get('skipped_missing_image', 0)} skipped (missing image), "
                f"pushed={stats.get('pushed')}, pending={stats.get('pending_total')}")
        return stats
    except Exception as e:
        log(f"[review] export bundle failed (non-fatal): {e}")
        return {'error': str(e)}


@app.route('/api/export_batch', methods=['POST'])
@_dirty_on_success
def do_export():
    """Export current batch to YOLO Segmentation format."""
    batch_size = session['review_batch_size']
    offset = session['review_offset']
    files = session['review_files']
    export_dir = session['export_dir']

    batch_files = files[offset:offset + batch_size]
    if not batch_files:
        return jsonify({'error': 'No images to export'}), 400

    # Collect segmentations for export
    to_export = {}
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if seg:
            to_export[fn] = seg

    stats = export_batch(
        to_export, export_dir, session['class_map'],
        symlink=cfg.SYMLINK_IMAGES,
        reviewer=_current_reviewer(),
    )

    # Ship REVIEW-flagged masks to the expert-review GitHub-Pages repo + the
    # permanent cross-project library (auto-relabels any already known to a past
    # expert). Runs before flush_all so in-place relabels persist.
    review_stats = _export_review_bundle(to_export, batch_files)

    # Mark exported and generate reviewed overlay images
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if seg:
            seg['exported'] = True
            _stamp_provenance(seg, fn)
            render_segmentation_overlays(seg, fn, export_dir, stage="reviewed")

    # Build per-frame manifest entries (one row per routed frame in this batch)
    batch_name = _next_batch_name(export_dir)
    manifest_entries = []
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if not seg:
            continue
        stem = os.path.splitext(fn)[0]
        accepted = [m for m in seg.get('masks', [])
                    if m.get('status') == 'accepted' and not m.get('review')
                    and m.get('species') != 'REVIEW' and not _needs_species(m)]
        has_review = any(m.get('review') or m.get('species') == 'REVIEW'
                         for m in seg.get('masks', []))
        image_path = seg.get('image_path_abs')
        if seg.get('scrapped'):
            outcome, lbl, img = 'scrapped', '', ''
        elif not image_path or not os.path.exists(image_path):
            outcome, lbl, img = 'image_missing', '', ''   # export_yolo skips it; no label is written
        elif accepted:
            outcome, lbl, img = 'exported_with_masks', stem + '.txt', fn
        elif has_review:
            outcome, lbl, img = 'review_only', '', ''
        else:
            outcome, lbl, img = 'exported_empty', stem + '.txt', fn
        preview = ''
        if outcome not in ('scrapped', 'image_missing'):
            preview = _publish_preview(export_dir, stem)
        from collections import Counter
        species_counts = Counter(m.get('species', 'UNK') for m in accepted)
        entry = {
            'basename': stem, 'n_masks': len(accepted),
            'label_file': lbl, 'image_file': img, 'outcome': outcome,
            'preview_file': preview, 'batch_name': batch_name,
            'image_path': image_path or '',
        }
        entry.update({sp: c for sp, c in species_counts.items()})
        manifest_entries.append(entry)
    missing_images = [e['basename'] for e in manifest_entries if e['outcome'] == 'image_missing']
    if missing_images:
        log(f"[export] WARNING: {len(missing_images)} frame(s) skipped — image file missing on disk "
            f"(Step-3 eligibility should prevent this): {', '.join(missing_images)}")
    write_export_manifest(export_dir, manifest_entries)
    reconcile_export(export_dir, manifest_entries)

    flush_all()

    # Rebuild review list under the removed-frames model: exported + scrapped
    # frames drop out, review_offset resets to 0. The rebuilt list IS the next
    # batch; "done" is review_files == [].
    _build_review_list()

    # GUARANTEE: when the queue is empty every routed (non-scrapped) frame must
    # be accounted for as exported. _build_review_list includes every
    # non-scrapped, non-exported frame, so an empty list means
    # exported + scrapped == routed. Assert it so a silently-dropped frame is
    # caught loudly instead of vanishing from the YOLO export.
    if not session['review_files']:
        routed = exported = scrapped = 0
        for _s in session['segmentations'].values():
            routed += 1
            if _s.get('scrapped', False):
                scrapped += 1
            elif _s.get('exported', False):
                exported += 1
        accounted = exported + scrapped
        if accounted != routed:
            log(f"[batch] ALL-DONE RECONCILE MISMATCH: routed={routed} "
                f"exported={exported} scrapped={scrapped} "
                f"accounted={accounted} (lost {routed - accounted})")
        else:
            log(f"[batch] all done: routed={routed} = exported={exported} "
                f"+ scrapped={scrapped}; every frame accounted for")
        assert accounted == routed, (
            f"review queue empty but {routed - accounted} routed frame(s) "
            f"unaccounted (exported={exported} scrapped={scrapped} routed={routed})")

    out = {'ok': True, **stats}
    if review_stats:
        out['review'] = review_stats
    return jsonify(out)


@app.route('/api/mask_size', methods=['POST'])
@_dirty_on_success
def set_mask_size():
    """Pick which SAM3 tracker variant later prompts return: tight, medium, generous."""
    engine = session.get('sam_engine')
    size = (request.json or {}).get('size', '')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500
    try:
        engine.set_mask_size(size)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    session['mask_size'] = size
    return jsonify({'ok': True, 'size': size})


@app.route('/api/status')
def status():
    segs = session['segmentations']
    agg = {
        'images': len(segs),
        'reviewed': sum(1 for s in segs.values() if s.get('reviewed', False)),
        'exported': sum(1 for s in segs.values() if s.get('exported', False)),
        'total_masks': sum(len(s.get('masks', [])) for s in segs.values()),
        'custom_imports': sum(1 for s in segs.values() if s.get('custom_import', False)),
    }
    # Batch progress under the removed-frames model: do_export() drops exported
    # frames from review_files and resets review_offset to 0, so the offset is
    # always 0 between batches. The number of batches still to do is simply how
    # many batch-sized chunks the remaining review_files split into. When
    # review_files is empty, nothing remains and the frontend shows "All Done".
    batch_size = session.get('review_batch_size') or cfg.REVIEW_BATCH_SIZE
    n_review = len(session['review_files'])
    batches_remaining = ((n_review + batch_size - 1) // batch_size) if (n_review and batch_size) else 0
    total_batches = batches_remaining
    batches_done = 0

    return jsonify({
        'configured': session['configured'],
        'phase': session['phase'],
        'processing_done': session['processing_idx'],
        'processing_total': len(session['processing_queue']),
        'review_total': n_review,
        'review_offset': session['review_offset'],
        'review_batch_size': batch_size,
        'total_classes': len(session['class_map']),
        'stats': agg,
        # Batch progress (orchestrator panel reads these).
        'batches_total': total_batches,
        'batches_done': batches_done,
        'batches_remaining': batches_remaining,
        'orchestrator_url': _orchestrator_url(),
        # Live SAM3 mask-size variant, the boot mode, and the labels this
        # session offers (standalone picks its own at start).
        'mask_size': getattr(session.get('sam_engine'), 'mask_size', cfg.SAM3_MASK_SIZE),
        'session_mode': getattr(cfg, 'SESSION_MODE', 'configure'),
        'target_labels': _target_labels(),
    })


def _host_open(path):
    """xdg-open a path on the host (host-only by design). Returns (ok, err)."""
    if not path or not os.path.exists(path):
        return False, f'path not found: {path}'
    try:
        subprocess.Popen(['xdg-open', path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, ''
    except FileNotFoundError:
        return False, 'xdg-open not available on this host'
    except Exception as e:
        return False, str(e)


@app.route('/api/open_export_folder', methods=['POST'])
def open_export_folder():
    export_dir = session.get('export_dir', '')
    ok, err = _host_open(export_dir)
    return (jsonify({'ok': True, 'path': export_dir}) if ok
            else (jsonify({'error': err}), 400))


@app.route('/api/open_export_manifest', methods=['POST'])
def open_export_manifest():
    export_dir = session.get('export_dir', '')
    manifest = os.path.join(export_dir, 'export_manifest.csv') if export_dir else ''
    ok, err = _host_open(manifest)
    return (jsonify({'ok': True, 'path': manifest}) if ok
            else (jsonify({'error': err}), 400))


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve raw image file.

    The frame bytes behind a given filename never change during a session, so
    the browser is told it may keep them for an hour. That makes stepping back
    to a frame, and the client's prefetch of the next frame, paint from cache
    instead of re-downloading a multi-megabyte JPEG. `private` keeps the
    imagery out of any shared proxy cache.
    """
    seg = _find_segmentation(filename)
    if seg:
        path = seg.get('image_path_abs', '')
        if path and os.path.exists(path):
            resp = send_file(path)
            resp.headers['Cache-Control'] = 'private, max-age=3600'
            return resp
    return jsonify({'error': 'not found'}), 404


# ── Main ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Annotate Images - SAM3 segmentation + review')
    parser.add_argument('--port', type=int, default=cfg.PORT)
    args = parser.parse_args()

    print(f"Loading SAM3 model...")
    from sam_engine import SAM3Engine
    session['sam_engine'] = SAM3Engine(cfg)
    # Load the exemplar model in the background so the first Scan does not stall.
    session['sam_engine'].warm_exemplar_async()

    # Save on quit (VEGA annotation safety net, 2026-09-27): a stop, a
    # restart, or an operator's Ctrl+C must not drop the last un-flushed
    # edit. SIGKILL cannot be caught, so the stop path that sends it (the
    # orchestrator's kill path) flushes first itself; this covers every
    # signal Python can intercept plus a normal interpreter exit.
    signal.signal(signal.SIGTERM, _handle_quit_signal)
    signal.signal(signal.SIGINT, _handle_quit_signal)
    atexit.register(flush_before_exit)

    log("SAM3 model loaded, server starting")
    print(f"Server ready on http://localhost:{args.port}")

    app.run(host='0.0.0.0', port=args.port, debug=False)
