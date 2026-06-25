"""Per-(image, label) provenance for the TCRMP segmentation pipeline.

Two artifacts:
  1. seg_dict['label_outcomes']  - an in-JSON block per image.
  2. label_provenance.csv        - a flat per-project ledger, one row per
                                    (basename, label), upserted in place.

Outcome rules (computed from a finished image's masks):
  found_manual : an ACCEPTED non-review mask of that species has
                 source_type in {manual_box, manual_click}.
  found_ai     : an ACCEPTED non-review mask of that species exists but is
                 only model-proposed (source_type in {auto, exemplar}).
  not_found    : the image is reviewed and a TARGET species has no accepted
                 mask. Only emitted for codes in target_species.
"""

import csv
import os
from datetime import datetime, timezone, timedelta

# Atlantic Standard Time, fixed UTC-4, no daylight saving.
AST = timezone(timedelta(hours=-4))

CSV_HEADER = ['basename', 'label', 'outcome', 'source', 'reviewer', 'at', 'project_id']

_MANUAL_SOURCES = {'manual_box', 'manual_click'}
_AI_SOURCES = {'auto', 'exemplar'}


def ast_now():
    """AST ISO-8601 timestamp, second precision, e.g. 2026-06-25T08:13:42-04:00."""
    return datetime.now(AST).isoformat(timespec='seconds')


def _is_review(m):
    return bool(m.get('review')) or m.get('species') == 'REVIEW'


def _is_accepted(m):
    return m.get('status') == 'accepted'


def compute_label_outcomes(seg_dict, target_species, source, reviewer=''):
    """Return the label_outcomes block for one finished image.

    Args:
        seg_dict: the per-image segmentation dict (has 'masks').
        target_species: iterable of target species codes for this project.
        source: 'step4test' | 'step5' | 'model'.
        reviewer: operator initials (may be '').

    Returns:
        {code: {outcome, reviewed, source, reviewer, at}} where outcome is
        'found_manual' | 'found_ai' | 'not_found'.
    """
    at = ast_now()
    targets = set(target_species or [])

    accepted = [m for m in seg_dict.get('masks', [])
                if _is_accepted(m) and not _is_review(m)]

    found = {}  # code -> {'manual': bool}
    for m in accepted:
        code = m.get('species', 'UNK')
        rec = found.setdefault(code, {'manual': False})
        if m.get('source_type') in _MANUAL_SOURCES:
            rec['manual'] = True

    outcomes = {}
    for code in (set(found) | targets):
        if code in found:
            outcome = 'found_manual' if found[code]['manual'] else 'found_ai'
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
