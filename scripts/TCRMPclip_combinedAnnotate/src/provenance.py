"""Per-(image, label) provenance for the TCRMP segmentation pipeline.

Two artifacts:
  1. seg_dict['label_outcomes']  - an in-JSON block per image.
  2. label_provenance.csv        - a flat per-project ledger, one row per
                                    (basename, label), upserted in place.

Outcome rules (computed from a finished image's masks):
  found_expert   : an ACCEPTED non-review mask of that species carries an
                   accepted expert ID (expert_id.mode == 'EXPERT').
  found_manual   : an ACCEPTED non-review mask of that species has
                   source_type in {manual_box, manual_click}.
  found_ai       : an ACCEPTED non-review mask of that species exists but is
                   only model-proposed (source_type in {auto, exemplar, model}).
  pending_expert : a non-rejected REVIEW mask is still awaiting an expert ID.
                   Labeled by each distinct tentative code in mask.reviews[]
                   (excluding the synthetic 'overlap' reviewer), or the
                   literal 'REVIEW' when there is none. Never overrides a
                   found_* outcome for the same code; beats not_found only.
  not_found      : the image is reviewed and a TARGET species has no accepted
                   mask. Only emitted for codes in target_species.
"""

import csv
import os
from datetime import datetime, timezone, timedelta

# Atlantic Standard Time, fixed UTC-4, no daylight saving.
AST = timezone(timedelta(hours=-4))

CSV_HEADER = ['basename', 'label', 'outcome', 'source', 'reviewer', 'at', 'project_id']

_MANUAL_SOURCES = {'manual_box', 'manual_click'}
_AI_SOURCES = {'auto', 'exemplar', 'model'}


def ast_now():
    """AST ISO-8601 timestamp, second precision, e.g. 2026-06-25T08:13:42-04:00."""
    return datetime.now(AST).isoformat(timespec='seconds')


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


def compute_label_outcomes(seg_dict, target_species, source, reviewer=''):
    """Return the label_outcomes block for one finished image.

    Args:
        seg_dict: the per-image segmentation dict (has 'masks').
        target_species: iterable of target species codes for this project.
        source: 'step4test' | 'step4loop' | 'step5' | 'model' | 'edit'.
        reviewer: operator initials (may be '').

    Returns:
        {code: {outcome, reviewed, source, reviewer, at}} where outcome is
        'found_expert' | 'found_manual' | 'found_ai' | 'pending_expert' |
        'not_found'.
    """
    at = ast_now()
    targets = set(target_species or [])

    accepted = [m for m in seg_dict.get('masks', [])
                if _is_accepted(m) and not _is_review(m)]
    # Review masks stay OUT of found; they only ever yield pending_expert.
    pending = [m for m in seg_dict.get('masks', [])
               if _is_review(m) and m.get('status') != 'rejected']

    found = {}  # code -> {'manual': bool, 'expert': bool, 'ai': bool}
    for m in accepted:
        code = m.get('species', 'UNK')
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


def write_provenance_csv(export_dir, basename, outcomes, project_id):
    """Upsert one row per (basename, label) into export_dir/label_provenance.csv.

    Existing rows for the same (basename, label) are replaced; all other rows
    are preserved. The whole file is rewritten atomically each call (it is
    small: one row per image-species).
    """
    path = os.path.join(export_dir, 'label_provenance.csv')

    rows = []
    if os.path.exists(path):
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                rows.append(r)

    incoming = {(basename, code) for code in outcomes}
    rows = [r for r in rows
            if (r.get('basename'), r.get('label')) not in incoming]

    for code, o in outcomes.items():
        rows.append({
            'basename': basename, 'label': code, 'outcome': o['outcome'],
            'source': o['source'], 'reviewer': o.get('reviewer', ''),
            'at': o['at'], 'project_id': project_id,
        })

    rows.sort(key=lambda r: (r.get('basename', ''), r.get('label', '')))

    os.makedirs(export_dir, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in CSV_HEADER})
    os.replace(tmp, path)
