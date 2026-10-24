"""Per-(image, label) provenance for the TCRMP segmentation pipeline.

Two artifacts:
  1. seg_dict['label_outcomes']  - an in-JSON block per image.
  2. label_provenance.csv        - a flat per-project ledger, one row per
                                    (basename, label), upserted in place.

Outcome rules (computed from a finished image's masks):
  found_expert   : an ACCEPTED non-review mask of that species carries an
                   accepted expert ID (expert_id.mode == 'EXPERT').
  found_manual   : an ACCEPTED non-review mask of that species has
                   source_type in MANUAL_SOURCE_TYPES (manual_box,
                   manual_click, merged; the one list in
                   _reefreview/manual_sources.py). A 'merged' mask is always
                   the product of an operator hand-merging reviewed masks
                   (POST .../merge_many, gated on a named reviewer), so it
                   counts as manual work even though no single
                   manual_box/manual_click prompt produced it.
  found_ai       : an ACCEPTED non-review mask of that species exists but is
                   only model-proposed (source_type in {auto, exemplar, model}).
  pending_expert : a non-rejected REVIEW mask is still awaiting an expert ID.
                   Labeled by each distinct tentative code in mask.reviews[]
                   (excluding the synthetic 'overlap' reviewer), or the
                   literal 'REVIEW' when there is none. Never overrides a
                   found_* outcome for the same code; beats not_found only.
  not_found      : the image is reviewed and a TARGET species has no accepted
                   mask. Only emitted for codes in target_species. Suppressed
                   for a code in screened_elsewhere (VEGA annotator tidy-up,
                   2026-09-27): another reef_point_seg project already
                   recorded an outcome for that label on this same image, so
                   this project emits no not_found row for it and leaves the
                   code out of the result entirely, rather than having the
                   operator re-screen an image someone else already covered.
                   A found_* outcome for the code is unaffected either way.

A mask with a blank species (missing, empty or whitespace) is not a label: it
yields no outcome and no ledger row. An unlabelled accepted mask is blocked at
export; it must not appear in the ledger as a label named "".

The ledger describes a frame as a whole. Writing a frame replaces every row
that frame has, from any source, when the writer is in the owner group
(step4test, step4loop, step5 and edit all describe one segmentations.json), so a label that left the frame (a species
changed, a mask deleted) leaves the ledger too. Locking and the atomic swap
live in _reefreview/ledger_io.py.
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone, timedelta

# scripts/ holds the shared _reefreview package; this file sits at
# scripts/<app>/src/provenance.py in every app that copies it.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _reefreview.ledger_io import (  # noqa: E402
    CSV_HEADER, LEDGER_FILE_MODE, frames_with_drift as frames_with_ledger_drift,
    ledger_lock, write_frames)
from _reefreview.manual_sources import MANUAL_SOURCE_TYPES  # noqa: E402

# Atlantic Standard Time, fixed UTC-4, no daylight saving.
AST = timezone(timedelta(hours=-4))


_MANUAL_SOURCES = MANUAL_SOURCE_TYPES
_AI_SOURCES = {'auto', 'exemplar', 'model'}


def ast_now():
    """AST ISO-8601 timestamp, second precision, e.g. 2026-06-25T08:13:42-04:00."""
    return datetime.now(AST).isoformat(timespec='seconds')


def _species_of(m):
    """The mask's species code, stripped; '' when missing, None or blank."""
    return str(m.get('species') or '').strip()


def _is_review(m):
    return bool(m.get('review')) or m.get('species') == 'REVIEW'


def _is_accepted(m):
    return m.get('status') == 'accepted'


def _is_expert(m):
    eid = m.get('expert_id')
    return isinstance(eid, dict) and eid.get('mode') == 'EXPERT'


def pending_labels(m):
    """Labels for a still-pending review mask: each distinct non-blank
    tentative code in mask.reviews[] excluding the synthetic 'overlap'
    reviewer, or ['REVIEW'] when there is none."""
    codes = []
    for r in (m.get('reviews') or []):
        if (r.get('reviewer') or '').strip() == 'overlap':
            continue
        code = (r.get('code') or '').strip()
        if code and code not in codes:
            codes.append(code)
    return codes or ['REVIEW']


def compute_label_outcomes(seg_dict, target_species, source, reviewer='',
                            screened_elsewhere=None):
    """Return the label_outcomes block for one finished image.

    Args:
        seg_dict: the per-image segmentation dict (has 'masks').
        target_species: iterable of target species codes for this project.
        source: 'step4test' | 'step4loop' | 'step5' | 'model' | 'edit'.
        reviewer: operator initials (may be '').
        screened_elsewhere: iterable of target codes another reef_point_seg
            project already recorded an outcome for on this same image (the
            no-rescreening rule; see the module docstring's not_found entry).
            A code in this set that finds no accepted mask HERE is left out
            of the result rather than getting not_found. A code that DOES
            find an accepted or pending mask here is unaffected: found_* and
            pending_expert are reported exactly as they would be without this
            argument. Optional and defaults to none excluded, so every
            existing caller's behaviour is unchanged.

    Returns:
        {code: {outcome, reviewed, source, reviewer, at}} where outcome is
        'found_expert' | 'found_manual' | 'found_ai' | 'pending_expert' |
        'not_found'.
    """
    at = ast_now()
    targets = set(target_species or [])
    screened = frozenset(screened_elsewhere or ())

    accepted = [m for m in seg_dict.get('masks', [])
                if _is_accepted(m) and not _is_review(m)]
    # Review masks stay OUT of found; they only ever yield pending_expert.
    pending = [m for m in seg_dict.get('masks', [])
               if _is_review(m) and m.get('status') != 'rejected']

    found = {}  # code -> {'manual': bool, 'expert': bool, 'ai': bool}
    for m in accepted:
        code = _species_of(m)
        if not code:
            continue  # an unlabelled mask is not a label: no outcome, no row
        rec = found.setdefault(code, {'manual': False, 'expert': False, 'ai': False})
        if m.get('source_type') in _MANUAL_SOURCES:
            rec['manual'] = True
        if m.get('source_type') in _AI_SOURCES:
            rec['ai'] = True
        if _is_expert(m):
            rec['expert'] = True

    pending_codes = set()
    for m in pending:
        pending_codes.update(pending_labels(m))

    outcomes = {}
    for code in (set(found) | targets | pending_codes):
        if code in found and found[code]['expert']:
            outcome = 'found_expert'
        elif code in found and found[code]['manual']:
            outcome = 'found_manual'
        elif code in found and found[code]['ai']:
            outcome = 'found_ai'
        elif code in pending_codes:
            outcome = 'pending_expert'
        elif code in targets:
            if code in screened:
                # Already screened for this label in another project, and
                # nothing accepted here: no fresh not_found, no re-screening.
                continue
            outcome = 'not_found'
        else:
            continue
        outcomes[code] = {
            'outcome': outcome,
            'reviewed': True,
            'source': source,
            'reviewer': reviewer or '',
            'at': at,
        }
    return outcomes


def write_provenance_frames(export_dir, frames, project_id, source=None,
                            protect_sources=frozenset()):
    """Replace several frames' ledger rows in one locked read-modify-write.

    See _reefreview.ledger_io.write_frames: `frames` is {basename: outcomes},
    the writer's `source` defaults to each frame's own outcome source, an
    owner-group writer (step4test, step4loop, step5, edit) replaces every row
    of the frame, and `protect_sources` rows are left alone.

    Returns:
        True when the ledger file changed.
    """
    return write_frames(export_dir, frames, project_id, source, protect_sources)


def write_provenance_csv(export_dir, basename, outcomes, project_id, source=None):
    """Replace one frame's rows in export_dir/label_provenance.csv.

    Args:
        export_dir: the project's export folder (holds the ledger).
        basename: the frame stem, e.g. 'TCRMP20140926_clip_BWR_T119'.
        outcomes: {label: outcome dict} as compute_label_outcomes returns.
        project_id: written to every row.
        source: the writer ('step4test', 'edit', 'expertids', ...). An
            owner-group writer (step4test, step4loop, step5, edit: they share
            one segmentations.json and recompute every found and pending
            label, expert ones included) replaces ALL of the frame's rows,
            whatever source wrote them, so a label that left the frame leaves
            the ledger. Any other writer (the importer's `expertids`)
            replaces only the labels it writes. Defaults to the source named
            by the first outcome.

    Rows of other frames are kept. A blank label is never written. The write
    waits at most LEDGER_LOCK_TIMEOUT_S for the ledger lock and raises
    LedgerLockTimeout after that.
    """
    write_frames(export_dir, {basename: outcomes}, project_id, source)


# prior_screening.json envelope version this reader understands.
PRIOR_SCREENING_VERSION = 1


def read_prior_screening(export_dir, log_fn):
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
        log_fn: one-argument callable that receives each log line.

    Returns:
        {stem: {'screened': {label_code: {...}}, 'masks': [{...}, ...]}},
        keyed by file stem WITHOUT an extension (session['segmentations'] and
        session['prior_screening'] use different keys for the same frame:
        see the app's _prior_screening_for). Never raises. A missing file, an unreadable
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
        log_fn(f"[prior-screening] no prior_screening.json at {path}; every "
            f"frame is treated as unscreened elsewhere")
        return {}

    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        log_fn(f"[prior-screening] could not read {path} ({e}); treating "
            f"every frame as unscreened elsewhere")
        return {}

    valid_envelope = (isinstance(payload, dict)
                       and payload.get('version') == PRIOR_SCREENING_VERSION
                       and isinstance(payload.get('frames'), dict))
    if not valid_envelope:
        log_fn(f"[prior-screening] {path} is not a version "
            f"{PRIOR_SCREENING_VERSION} prior-screening file "
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
