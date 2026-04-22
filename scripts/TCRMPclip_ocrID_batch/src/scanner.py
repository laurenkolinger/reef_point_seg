"""
Recursive directory scanner for TCRMP_clip input directories.

Discovers all _pts annotated images across year directories (2020+),
handling both period-based subdirs (2020: Annual/PostBL/SCTLD,
2024-2025: Annual/PBL) and flat date_site directories (2021-2023).

READ-ONLY: This module never modifies the input directory.
"""

import os
import re
import glob


# Regex patterns for the various TCRMP filename formats
# Standard: TCRMP20201022_clip_SITE_T101_pts.jpg
# 2025 variant: 20251016TCRMP_clip_SCP (date-first)
# 2025 variant: TCRMP_20251010_clip_BWR (extra underscore)
_FILENAME_PATTERNS = [
    # Standard: TCRMP{YYYYMMDD}_clip_{SITE}_T{NUM}_pts
    re.compile(
        r'TCRMP(\d{8})_clip_([A-Za-z]+)_T(\d+?)_pts\.(jpe?g|png)', re.I
    ),
    # 2025 variant with extra underscore: TCRMP_{YYYYMMDD}_clip_{SITE}_T{NUM}_pts
    re.compile(
        r'TCRMP_(\d{8})_clip_([A-Za-z]+)_T(\d+?)_pts\.(jpe?g|png)', re.I
    ),
    # 2025 date-first variant: {YYYYMMDD}TCRMP_clip_{SITE}_T{NUM}_pts
    re.compile(
        r'(\d{8})TCRMP_clip_([A-Za-z]+)_T(\d+?)_pts\.(jpe?g|png)', re.I
    ),
]

# Patterns for directory names (to find raw image counterpart)
_DIR_NAME_PATTERNS = [
    re.compile(r'TCRMP(\d{8})_clip_([A-Za-z]+)'),
    re.compile(r'TCRMP_(\d{8})_clip_([A-Za-z]+)'),
    re.compile(r'(\d{8})TCRMP_clip_([A-Za-z]+)'),
]


def parse_pts_filename(filename):
    """Extract (date_str, site, t_id) from a _pts filename.

    Returns (date_str, site, t_id) or (None, None, None) if no match.
    date_str is YYYYMMDD format, site is uppercase, t_id is the full
    transect string (e.g. "101", "315").
    """
    for pat in _FILENAME_PATTERNS:
        m = pat.match(filename)
        if m:
            return m.group(1), m.group(2).upper(), m.group(3)
    return None, None, None


def pts_to_raw_name(pts_filename):
    """Convert a _pts filename to the corresponding raw filename.

    Handles both .jpg and .jpeg extensions.
    """
    # Remove _pts before the extension
    name = re.sub(r'_pts\.(jpe?g|png)', r'.jpeg', pts_filename, flags=re.I)
    return name


def extract_year(filename):
    """Extract the 4-digit year from a TCRMP filename.

    Returns int year or None if not parseable.
    """
    date_str, _, _ = parse_pts_filename(filename)
    if date_str and len(date_str) >= 4:
        try:
            return int(date_str[:4])
        except ValueError:
            pass
    return None


def _find_raw_for_pts(pts_path, pts_filename):
    """Find the raw image corresponding to a _pts file. READ-ONLY."""
    raw_name = pts_to_raw_name(pts_filename)
    raw_path = os.path.join(os.path.dirname(pts_path), raw_name)
    if os.path.exists(raw_path):
        return raw_path
    # Try .jpg instead of .jpeg
    alt = re.sub(r'\.jpeg$', '.jpg', raw_path, flags=re.I)
    if os.path.exists(alt):
        return alt
    # Try .jpg if was .jpeg
    alt2 = re.sub(r'\.jpg$', '.jpeg', raw_path, flags=re.I)
    if os.path.exists(alt2):
        return alt2
    return None


def scan_year_dir(year_dir):
    """Scan a single year directory for all _pts images.

    Handles both period subdirs (Annual/, PBL/, etc.) and flat
    date_site subdirs. Recursive glob traverses everything.

    Returns list of dicts: {pts_path, raw_path, filename, year,
                            date_str, site, t_id}
    """
    results = []
    patterns = [
        os.path.join(year_dir, '**', '*_pts.jpg'),
        os.path.join(year_dir, '**', '*_pts.jpeg'),
        os.path.join(year_dir, '**', '*_pts.png'),
    ]

    all_pts = set()
    for pat in patterns:
        all_pts.update(glob.glob(pat, recursive=True))

    for pts_path in sorted(all_pts):
        filename = os.path.basename(pts_path)
        date_str, site, t_id = parse_pts_filename(filename)
        if not date_str:
            continue

        year = int(date_str[:4])
        raw_path = _find_raw_for_pts(pts_path, filename)

        results.append({
            'pts_path': pts_path,
            'raw_path': raw_path,
            'filename': filename,
            'year': year,
            'date_str': date_str,
            'site': site,
            't_id': t_id,
        })

    return results


def scan_all_years(root_dir, selected_years=None):
    """Scan the TCRMP_clip root for all _pts images in selected years.

    Args:
        root_dir: Path to TCRMP_clip root (e.g. .../TCRMP/TCRMP_clip)
        selected_years: List of int years to include (e.g. [2020, 2021]).
                        If None, includes all years >= 2020.

    Returns:
        list of dicts sorted by (year, filename), each with:
        {pts_path, raw_path, filename, year, date_str, site, t_id}
    """
    if selected_years is None:
        selected_years = list(range(2020, 2030))

    all_results = []

    for year in sorted(selected_years):
        year_dir = os.path.join(root_dir, f'TCRMP{year}_clip')
        if not os.path.isdir(year_dir):
            continue
        year_results = scan_year_dir(year_dir)
        all_results.extend(year_results)

    # Sort by year, then filename for deterministic order
    all_results.sort(key=lambda r: (r['year'], r['filename']))
    return all_results


def get_available_years(root_dir, min_year=2020):
    """List available TCRMP year directories at the root.

    Returns list of int years (e.g. [2020, 2021, 2022, ...]).
    """
    years = []
    if not os.path.isdir(root_dir):
        return years
    for entry in os.listdir(root_dir):
        m = re.match(r'TCRMP(\d{4})_clip$', entry)
        if m:
            y = int(m.group(1))
            if y >= min_year:
                years.append(y)
    return sorted(years)


# ── Standalone test ──────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else (
        '/home/bizon/UVI Dropbox/SMITH LAB TEAM FOLDER/TCRMP/TCRMP_clip'
    )
    print(f"Scanning: {root}")
    years = get_available_years(root)
    print(f"Available years: {years}")

    for year in years:
        year_dir = os.path.join(root, f'TCRMP{year}_clip')
        results = scan_year_dir(year_dir)
        n_raw = sum(1 for r in results if r['raw_path'])
        print(f"  {year}: {len(results)} _pts files, {n_raw} with raw counterpart")
