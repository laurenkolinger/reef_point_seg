"""
Coverage planner for the global label tracker.

The premise: all_points.csv is a point-level census of EVERY CPCe/OCR point on
EVERY frame (~1.6M points over ~113k images), with each point's species_code --
including species that were never segmented for a given project. So we already
KNOW, without re-reviewing, which species co-occur in which image and how many
point-instances each has. That lets us:

  1. report coverage for any chosen set of codes (images + point-instances),
  2. cross-reference the global mask catalog (the expert-ID library) to see what
     is already SEGMENTED vs only point-census-known,
  3. pick a minimal set of images to reach a target instance count per class,
     exploiting co-occurrence so one review pass yields several classes,
  4. flag "gap" images: a frame that contains class X (per the census) but has
     no mask for X yet -- those go through route+SAM to add the missing mask
     into the global catalog.

Point-instances from the census are an UPPER BOUND on segmentable instances and
a planning proxy; the actual trainable instances are the masks in the catalog.

JOIN MODEL: census and catalog are joined on a normalized identity tuple
(date_compact, site, transect, frame) parsed from the FILENAME on BOTH sides --
NOT on raw image basename. The census key already IS that tuple; the catalog's
source_image clip filename is parsed with filename_parse to the same tuple. This
avoids basename collisions and removes the old "silent zero segmented" bug.

REMAP: the census stores RAW species_code. A project may remap codes (step 2
remap_log, e.g. MFAV -> OFAV). Pass remap (old->new) so coverage counts the
codes the model will ACTUALLY train on.

Pure-stdlib (csv/collections) + the catalog (library.Library). No heavy deps.
"""

import collections
import csv
import hashlib
import json
import os
import pickle
import re

from . import filename_parse


# ── identity key (join key on BOTH census and catalog sides) ──
def _norm_int(v):
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v or '').strip()


def _norm_date(date):
    """Digits-only YYYYMMDD so the census's '2020-10-22' and a filename's
    '20201022' join to the same key."""
    return re.sub(r'\D', '', str(date or ''))


def _norm_key(date, site, transect, frame):
    return (_norm_date(date),
            str(site or '').strip().upper(),
            _norm_int(transect),
            _norm_int(frame))


def image_key_from_row(row):
    """Census image identity: (date_compact, site, transect, frame)."""
    return _norm_key(row.get('date', ''), row.get('site', ''),
                     row.get('transect', ''), row.get('frame', ''))


def key_from_filename(filename):
    """Parse a clip filename into the same identity key, or None if it doesn't
    parse to a full identity (site+date+transect+frame)."""
    p = filename_parse.parse(filename)
    if not (p.get('site') and p.get('date')
            and p.get('transect') is not None and p.get('frame') is not None):
        return None
    return _norm_key(p['date'], p['site'], p['transect'], p['frame'])


def _coerce_year(row):
    """Year tolerating '2001.0', falling back to date[:4]."""
    try:
        return int(float(row.get('year')))
    except (ValueError, TypeError):
        pass
    try:
        return int(_norm_date(row.get('date', ''))[:4])
    except (ValueError, TypeError):
        return 0


def _is_identity_key(k):
    return isinstance(k, tuple) and len(k) == 4


# ── remap ──
def load_remap(remap_log_path):
    """Build {old_code: new_code} from a step-2 remap_log.json. {} if absent."""
    if not remap_log_path or not os.path.exists(remap_log_path):
        return {}
    try:
        with open(remap_log_path) as f:
            data = json.load(f)
    except Exception:
        return {}
    remap = {}
    for e in data.get('remaps', []):
        old, new = e.get('old_code', ''), e.get('new_code', '')
        if old and new and old != new:
            remap[old] = new
    return remap


# ── census construction (+ optional remap + on-disk cache) ──
def build_census(all_points_csv, year_min=None, year_max=None, sites=None,
                 remap=None):
    """Read all_points.csv into {image_key: Counter(code -> n_points)}.
    Returns (census, meta)."""
    census = collections.defaultdict(collections.Counter)
    sites = set(s.strip().upper() for s in sites) if sites else None
    remap = remap or {}
    n_rows = 0
    with open(all_points_csv, newline='') as f:
        for row in csv.DictReader(f):
            code = (row.get('species_code') or '').strip()
            if not code:
                continue
            if year_min is not None or year_max is not None:
                yr = _coerce_year(row)
                if year_min is not None and yr < year_min:
                    continue
                if year_max is not None and yr > year_max:
                    continue
            site = (row.get('site') or '').strip().upper()
            if sites is not None and site not in sites:
                continue
            code = remap.get(code, code)
            census[image_key_from_row(row)][code] += 1
            n_rows += 1
    meta = {'n_points': n_rows, 'n_images': len(census),
            'remapped': bool(remap),
            'filtered': bool(year_min or year_max or sites)}
    return census, meta


def build_census_cached(all_points_csv, cache_dir=None, **kwargs):
    """build_census with an on-disk cache keyed by (csv mtime, filters, remap).
    Avoids re-parsing 1.6M rows on every UI interaction."""
    try:
        mtime = os.path.getmtime(all_points_csv)
    except OSError:
        return build_census(all_points_csv, **kwargs)
    cache_dir = cache_dir or os.path.join(os.path.dirname(all_points_csv), '.census_cache')
    remap = kwargs.get('remap') or {}
    sig = (round(mtime, 3), kwargs.get('year_min'), kwargs.get('year_max'),
           tuple(sorted(s.strip().upper() for s in (kwargs.get('sites') or []))),
           tuple(sorted(remap.items())))
    h = hashlib.sha1(repr(sig).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f'census_{h}.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            return ({k: collections.Counter(v) for k, v in obj['census'].items()},
                    obj['meta'])
        except Exception:
            pass
    census, meta = build_census(all_points_csv, **kwargs)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'census': {k: dict(v) for k, v in census.items()},
                         'meta': meta}, f)
        os.replace(tmp, path)
        meta['cached_to'] = path
    except Exception:
        pass
    return {k: collections.Counter(v) for k, v in census.items()}, meta


# ── catalog (expert-ID library) index, keyed by the SAME identity tuple ──
def catalog_index(library):
    """Build a join index keyed by the normalized identity tuple parsed from
    each mask's source_image:
      masked_codes_by_key: {key: set(codes with a real mask)}  (excludes REVIEW/IDK/'')
      masks_by_key:        {key: [ {uid, code, mode} ... ]}
    """
    masked_codes_by_key = collections.defaultdict(set)
    masks_by_key = collections.defaultdict(list)
    unjoinable = 0
    for uid, rec in library.load().items():
        key = key_from_filename(rec.get('source_image', ''))
        if key is None:
            unjoinable += 1
            continue
        code = (rec.get('code') or '').strip()
        masks_by_key[key].append({'uid': uid, 'code': code, 'mode': rec.get('mode', '')})
        if code and code not in ('REVIEW', 'IDK', ''):
            masked_codes_by_key[key].add(code)
    return {'masked_codes_by_key': dict(masked_codes_by_key),
            'masks_by_key': dict(masks_by_key), 'unjoinable': unjoinable}


# ── coverage report ──
def coverage_report(census, codes, library=None, target_per_code=None):
    """Coverage for the chosen `codes`. `census` MUST be identity-tuple-keyed
    (from build_census) -- asserted, so a mis-keyed census fails loudly instead
    of silently reporting 0 segmented."""
    codes = list(dict.fromkeys(c for c in (codes or []) if c))
    if not codes:
        return {'codes': [], 'per_code': {}, 'cooccurrence': {}, 'empty': True}
    # Validate EVERY key, not a sample — a stray basename key anywhere would
    # otherwise be silently counted as 0-segmented (the old sampling bug).
    if census and any(not _is_identity_key(k) for k in census):
        raise ValueError(
            "coverage_report expects a census keyed by (date,site,transect,frame) "
            "tuples from build_census; got non-identity keys.")

    masked_by_key = catalog_index(library)['masked_codes_by_key'] if library else {}
    per_code = {c: {'census_images': 0, 'census_instances': 0,
                    'segmented_images': 0, 'gap_images': 0} for c in codes}
    cooc = {c: collections.Counter() for c in codes}

    for key, counter in census.items():
        present = [c for c in codes if c in counter]
        if not present:
            continue
        masked_here = masked_by_key.get(key, set())
        for c in present:
            per_code[c]['census_images'] += 1
            per_code[c]['census_instances'] += counter[c]
            if c in masked_here:
                per_code[c]['segmented_images'] += 1
            else:
                per_code[c]['gap_images'] += 1
        for a in present:
            for b in present:
                if a != b:
                    cooc[a][b] += 1

    report = {'codes': codes, 'per_code': per_code,
              'cooccurrence': {a: dict(cooc[a]) for a in codes},
              'has_catalog': library is not None}
    if target_per_code:
        report['shortfall'] = {
            c: max(0, int(target_per_code.get(c, 0)) - per_code[c]['census_instances'])
            for c in codes}
    return report


# ── greedy minimal image selection (with a budget cap) ──
def select_images(census, target_per_code, prefer_codes=None, max_images=None,
                  min_gain=1):
    """Greedy set-cover-ish selection minimizing total images reviewed while
    hitting per-code instance targets. Scores each image by ALL needed codes it
    contributes, so co-occurring multi-class frames win first.

    max_images: hard cap (the whole point -- avoid labeling thousands; the
        remaining shortfall is reported when capped).
    Returns {selected, reached, shortfall, images_reviewed, capped}."""
    need = {c: int(n) for c, n in (target_per_code or {}).items() if int(n) > 0}
    reached = collections.Counter()
    selected = []
    cands = {k: counter for k, counter in census.items()
             if any(c in counter for c in need)}
    prefer = set(prefer_codes or [])
    capped = False

    while need and any(reached[c] < need[c] for c in need) and cands:
        if max_images is not None and len(selected) >= max_images:
            capped = True
            break
        best_key, best_score = None, -1
        for k, counter in cands.items():
            score = 0
            for c in need:
                rem = need[c] - reached[c]
                if rem > 0 and c in counter:
                    gain = min(rem, counter[c])
                    score += gain * (2 if c in prefer else 1)
            if score > best_score:
                best_score, best_key = score, k
        if best_key is None or best_score < min_gain:
            break
        counter = cands.pop(best_key)
        selected.append(best_key)
        for c in need:
            if c in counter:
                reached[c] += counter[c]

    shortfall = {c: max(0, need[c] - reached[c]) for c in need}
    return {'selected': selected, 'reached': dict(reached), 'shortfall': shortfall,
            'images_reviewed': len(selected), 'capped': capped}


def gap_images(census, code, library):
    """Identity keys whose census contains `code` but the catalog has no mask
    for it. census must be identity-keyed (from build_census)."""
    masked = catalog_index(library)['masked_codes_by_key']
    return [key for key, counter in census.items()
            if code in counter and code not in masked.get(key, set())]
