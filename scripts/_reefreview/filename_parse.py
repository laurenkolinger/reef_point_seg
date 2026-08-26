"""
TCRMP clip-filename parsing -> {year, date, site, transect, frame}.

Site and year survive downstream ONLY in the filename (build_sam_entry drops
the explicit site/t_id fields), so this is the single source of truth for
recovering context at review-export time.

Authoritative regex ported from TCRMPclip_cpcID/src/parser.py:62 plus the
three real filename dialects handled by TCRMPclip_ocrID_batch/src/scanner.py:17
(standard, extra-underscore "TCRMP_" 2025 variant, and date-first variant).
"""

import os
import re

# Canonical: TCRMP<YYYYMMDD>_clip_<SITE>_T<transect><frame2>[ _pts ].<ext>
#   e.g. TCRMP20201022_clip_SCP_T101.jpeg  -> 2020-10-22, SCP, transect 1, frame 01
#        TCRMP20140930_clip_BIT_T207.jpeg  -> 2014-09-30, BIT, transect 2, frame 07
_STD = re.compile(r'TCRMP(\d{8})_clip_([A-Za-z]+)(?:_T(\d+))?', re.I)

# 2025 extra-underscore: TCRMP_<YYYYMMDD>_clip_<SITE>...
_USCORE = re.compile(r'TCRMP_(\d{8})_clip_([A-Za-z]+)(?:_T(\d+))?', re.I)

# 2025 date-first: <YYYYMMDD>TCRMP_clip_<SITE>...
_DATEFIRST = re.compile(r'(\d{8})TCRMP_clip_([A-Za-z]+)(?:_T(\d+))?', re.I)

_PATTERNS = [_STD, _USCORE, _DATEFIRST]


def _strip(name):
    """Drop directory, the raw/ prefix, a trailing _pts/_cpc tag, and extension."""
    base = os.path.basename(str(name or ''))
    # Remove extension(s): handles .jpeg, .jpg, .png, and _cpc.json keys.
    base = re.sub(r'_cpc\.json$', '', base, flags=re.I)
    base = re.sub(r'\.(jpe?g|png|json)$', '', base, flags=re.I)
    base = re.sub(r'_pts$', '', base, flags=re.I)
    return base


def _split_transect_frame(tblock):
    """T-block digits -> (transect, frame). Last 2 digits are the frame, the
    rest is the transect. '101' -> (1, 1); '207' -> (2, 7); '315' -> (3, 15)."""
    if not tblock:
        return None, None
    tblock = re.sub(r'\D', '', tblock)
    if len(tblock) < 3:
        # Degenerate: treat the whole thing as a transect, frame unknown.
        try:
            return int(tblock), None
        except ValueError:
            return None, None
    transect = int(tblock[:-2])
    frame = int(tblock[-2:])
    return transect, frame


def parse(name):
    """Parse a clip filename. Returns a dict; unknown fields are None.

    Keys: year (int|None), date (YYYYMMDD str|''), site (UPPER str|''),
          transect (int|None), frame (int|None).
    Never raises — unparseable input yields all-empty fields.
    """
    base = _strip(name)
    out = {'year': None, 'date': '', 'site': '', 'transect': None, 'frame': None}
    for pat in _PATTERNS:
        m = pat.search(base)
        if not m:
            continue
        date = m.group(1)
        out['date'] = date
        try:
            out['year'] = int(date[:4])
        except (ValueError, TypeError):
            out['year'] = None
        out['site'] = (m.group(2) or '').upper()
        out['transect'], out['frame'] = _split_transect_frame(m.group(3))
        return out
    return out


def site_year(name):
    """Convenience: (site, year) tuple for display."""
    p = parse(name)
    return p['site'], p['year']
