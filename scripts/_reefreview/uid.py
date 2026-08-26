"""
Deterministic, human-readable, dedupe-stable mask UID.

mask['id'] is only a per-image index (reused across images, reshuffled on
edit), so it cannot identify a mask globally. We mint a UID from the source
image + the originating click coordinates, which are stable across re-exports
of the same flagged point:

    <SITE>-<YYYYMMDD>-T<transect><frame2>-x<sx>y<sy>
    e.g. SCP-20201022-T101-x1234y567

Re-exporting the same flagged mask yields the same UID (so the review repo and
library dedupe naturally). Falls back to the filename stem when the filename
does not parse.
"""

import hashlib
import os
import re

from . import filename_parse

# A UID is used to build filesystem paths (polygons/<uid>.json, items/<uid>/,
# images/<uid>.jpg). It MUST therefore be path-safe: no separators, no '..', no
# absolute markers. This is the single source of truth for that rule, enforced
# both when minting and when consuming an externally-supplied UID (e.g. a row
# from an expert's returned CSV — untrusted input).
_SAFE_UID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$')


def _slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '', str(s or ''))


def is_safe_uid(uid):
    """True if uid is safe to use as a single path component (no traversal)."""
    if not uid or not isinstance(uid, str):
        return False
    if uid in ('.', '..') or '/' in uid or '\\' in uid or '\x00' in uid:
        return False
    return bool(_SAFE_UID_RE.match(uid))


def sanitize_uid(uid):
    """Coerce an arbitrary string into a path-safe UID, or '' if nothing usable.
    Used defensively on UIDs that arrive from outside (CSV import) before they
    are ever joined into a filesystem path."""
    if not uid:
        return ''
    cleaned = re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(uid)).strip('_.')[:128]
    return cleaned if is_safe_uid(cleaned) else ''


def make_uid(filename, source_x, source_y, parsed=None, disc=None):
    """Mint a UID for a mask given its source filename and click point.

    disc: optional discriminator appended when two masks would otherwise collide
    on the same rounded source pixel (e.g. species code or a small index). Kept
    out of the default so re-exporting the SAME mask stays dedupe-stable.
    """
    p = parsed or filename_parse.parse(filename)
    try:
        sx = int(round(float(source_x)))
        sy = int(round(float(source_y)))
    except (ValueError, TypeError):
        sx, sy = 0, 0

    site = p.get('site') or ''
    date = p.get('date') or ''
    transect = p.get('transect')
    frame = p.get('frame')
    suffix = f"-{_slug(disc)}" if disc else ''

    if site and date and transect is not None and frame is not None:
        tblock = f"{transect}{frame:02d}"
        uid = f"{site}-{date}-T{tblock}-x{sx}y{sy}{suffix}"
    else:
        # Fallback: filename stem keeps it unique even when the name is off-pattern.
        stem = _slug(os.path.splitext(os.path.basename(str(filename or '')))[0])[:48]
        uid = f"{stem or 'UNK'}-x{sx}y{sy}{suffix}"
    return uid if is_safe_uid(uid) else (sanitize_uid(uid) or f"UNK-x{sx}y{sy}")


def content_disc(mask):
    """A short, STABLE discriminator derived from the mask's geometry, used to
    disambiguate two masks whose source click rounds to the same pixel. Unlike a
    batch index it depends only on the mask content, so the importer can
    reconstruct the SAME UID and the value is stable across re-exports.

    Hashes the RLE counts (the exact footprint); falls back to the bbox."""
    rle = mask.get('rle') if isinstance(mask, dict) else None
    basis = None
    if rle and isinstance(rle, dict) and rle.get('counts') is not None:
        basis = repr(rle.get('counts')) + repr(rle.get('size'))
    elif isinstance(mask, dict) and mask.get('bbox'):
        basis = repr(mask.get('bbox'))
    if not basis:
        return ''
    return hashlib.sha1(basis.encode()).hexdigest()[:8]


def mask_uid(filename, mask, parsed=None):
    """Canonical UID for a segmentation mask: source-click UID, plus a stable
    content discriminator so two masks at the same rounded pixel get distinct,
    reproducible UIDs. Used identically by the exporter and the importer so a
    UID minted at export time is re-derivable at import time.
    """
    return make_uid(filename, mask.get('source_x', 0), mask.get('source_y', 0),
                    parsed, disc=content_disc(mask))
