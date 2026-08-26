"""
Species/benthic code dictionary for the expert-review viewer.

Loads the canonical master_codes.csv (header: code,category,name; 130 rows) and
builds the structures the static viewer needs:
  - a flat list of {code, name, category, group}
  - a 2-level group hierarchy (authored here; master_codes.csv has no supergroup
    above `category`, and the category column has a few dirty one-offs)
  - confidence-level definitions and the special "I don't know" / "Something
    else" choices

Default source: github_repo/supporting_data/master_codes.csv (module-local
authoritative copy). pipeline.yaml also mirrors it at
_METADATA/library/definitions/tcrmp_species_codes.csv (byte-identical).
"""

import csv
import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(os.path.dirname(_PKG_DIR))   # scripts/.. = github_repo
DEFAULT_MASTER_CODES = os.path.join(_REPO_DIR, 'supporting_data', 'master_codes.csv')
DEFAULT_SITE_CODES = os.path.join(_REPO_DIR, 'supporting_data', 'site_codes.csv')

# Special, always-present choices on the expert form.
IDK_CODE = 'IDK'
IDK_LABEL = "I don't know"
SOMETHING_ELSE = 'OTHER_PICK'   # sentinel: user opened the nested picker

CONFIDENCE = {
    'high': {
        'value': 'high',
        'label': 'High confidence',
        'definition': 'Confident in this identification to the stated taxonomic '
                      'level — diagnostic features are clearly visible.',
    },
    'low': {
        'value': 'low',
        'label': 'Low confidence',
        'definition': 'Best guess only — the ID is uncertain (poor image quality, '
                      'ambiguous or partial view, or look-alike taxa).',
    },
}
DEFAULT_CONFIDENCE = 'high'   # per spec: default conf = high

# Authored supergroup -> list of master_codes `category` values it absorbs.
# Used to nest the "Something else" picker. Dirty one-offs (Spo, Dca) and the
# quoted "Other, Living" are folded into the nearest sensible group.
GROUP_MAP = [
    ('Coral',            ['Coral']),
    ('Coral condition',  ['Coral Condition']),
    ('Algae',            ['Macroalgae', 'Turf', 'Calcareous', 'Cyanobacteria']),
    ('Other living',     ['Sponge', 'Gorgonian', 'Zoanthid', 'Other, Living', 'Spo']),
    ('Non-living',       ['Non-living', 'Dca']),
    ('Other / markers',  ['Other']),
]


def _category_to_group(category):
    cat = (category or '').strip()
    for group, cats in GROUP_MAP:
        if cat in cats:
            return group
    return 'Other / markers'


def load_codes(master_csv_path=None):
    """Read master_codes.csv into the viewer's codes structure."""
    path = master_csv_path or DEFAULT_MASTER_CODES
    codes = []
    by_code = {}
    if os.path.exists(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get('code') or '').strip()
                if not code:
                    continue
                entry = {
                    'code': code,
                    'name': (row.get('name') or row.get('species_name') or '').strip(),
                    'category': (row.get('category') or '').strip(),
                }
                # Carry through an optional 'candidate' column when the master
                # codes file declares one, so ReviewRepo._candidate_codes can
                # harvest it (CONTRACTS §2). The canonical master_codes.csv has
                # no such column, so this key stays absent there and the harvest
                # stays []. Only forward a present value (never invent the key).
                if 'candidate' in row:
                    entry['candidate'] = row.get('candidate')
                entry['group'] = _category_to_group(entry['category'])
                codes.append(entry)
                by_code[code] = entry

    # Group hierarchy in display order: {group: [code, ...]}.
    groups = []
    for group, _cats in GROUP_MAP:
        members = sorted(
            [c['code'] for c in codes if c['group'] == group],
            key=lambda cc: (by_code[cc]['name'] or cc).lower(),
        )
        if members:
            groups.append({'group': group, 'codes': members})

    return {
        'codes': codes,
        'by_code': by_code,
        'groups': groups,
        'confidence': CONFIDENCE,
        'default_confidence': DEFAULT_CONFIDENCE,
        'idk': {'code': IDK_CODE, 'label': IDK_LABEL},
        'something_else': {'code': SOMETHING_ELSE, 'label': 'Something else'},
        'source': path,
    }


def load_sites(path=None):
    """Read site_codes.csv (header: site,site_code) into {site_code: site}.

    Maps the 3-letter site_code to its full name for viewer display (CONTRACTS
    §7). Tolerant: missing file or blank rows yield an empty/partial map, never
    raises. The site_code key is upper-cased to match parsed filename sites."""
    out = {}
    path = path or DEFAULT_SITE_CODES
    if not os.path.exists(path):
        return out
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                code = (row.get('site_code') or '').strip().upper()
                name = (row.get('site') or '').strip()
                if code:
                    out[code] = name
    except Exception:
        return out
    return out


def code_name(code, master_csv_path=None):
    """Look up a single code's full name (best-effort)."""
    if not code or code in (IDK_CODE, SOMETHING_ELSE, '', 'REVIEW'):
        return ''
    data = load_codes(master_csv_path)
    entry = data['by_code'].get(code)
    return entry['name'] if entry else ''
