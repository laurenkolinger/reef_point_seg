"""
CPC file parser - reads CPCe annotation files and extracts point data.

CPC format (from Coral Point Count with Excel extensions):
  Line 1:  header - code file path, image path, canvas_w, canvas_h, ...
  Lines 2-5: bounding box corners
  Line 6:  number of points (typically 20)
  Lines 7-N: X,Y coordinates in canvas space
  Lines N+1-M: label definitions ("A","species_code","Notes","note_value")
  Remaining: padding whitespace
"""

import os
import re
from pathlib import Path


def parse_cpc(cpc_path):
    """Parse a single .cpc file and return structured data.

    Returns dict with keys: image_filename, canvas_w, canvas_h,
                            num_points, coords, labels
    """
    with open(cpc_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f.readlines()]

    parts = _parse_csv_line(lines[0])

    image_path_win = parts[1] if len(parts) > 1 else ""
    canvas_w = float(parts[2]) if len(parts) > 2 else 57600
    canvas_h = float(parts[3]) if len(parts) > 3 else 32400

    num_points = int(lines[5])

    coords = []
    for i in range(num_points):
        xy = lines[6 + i].split(",")
        coords.append((float(xy[0]), float(xy[1])))

    labels = []
    label_start = 6 + num_points
    for i in range(num_points):
        if label_start + i >= len(lines):
            break
        fields = _parse_csv_line(lines[label_start + i])
        label = fields[0] if fields else chr(65 + i)
        species_code = fields[1] if len(fields) > 1 else ""
        labels.append({"label": label, "species_code_cpc": species_code})

    image_filename = os.path.basename(image_path_win)

    return {
        "image_filename": image_filename,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "num_points": num_points,
        "coords": coords,
        "labels": labels,
    }


def parse_filename(basename, fallback_date=None, fallback_site=None):
    """Extract date, site, transect, frame from a TCRMP CPC/image filename.

    Handles edge cases:
      - Standard: TCRMP20181101_clip_BIX_T101
      - 4-char sites: TCRMP20191021_clip_LBPD_T101
      - Truncated dates: TCRMP2017120_clip_MGN_T202 (uses fallback_date)
      - Extra digits: TCRMP201910230_clip_BPT_T205 (uses fallback_date)
    """
    m = re.match(r'TCRMP(\d{8})_clip_([A-Za-z]+)_T(\d)(\d{2})', basename)
    if m:
        date_str = m.group(1)
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return {
            "date": date_formatted,
            "year": int(date_str[:4]),
            "site": m.group(2).upper(),
            "transect": int(m.group(3)),
            "frame": int(m.group(4)),
        }

    # Fallback: extract transect/frame and use fallback date/site
    m_tf = re.search(r'_T(\d)(\d{2})$', basename)
    if m_tf and fallback_date and fallback_site:
        year = int(fallback_date[:4]) if fallback_date else 0
        return {
            "date": fallback_date,
            "year": year,
            "site": fallback_site.upper(),
            "transect": int(m_tf.group(1)),
            "frame": int(m_tf.group(2)),
        }

    return None


def parse_directory_name(dirname):
    """Extract date and site from a TCRMP directory name.

    Returns (date_str, site_code) or (None, None).
    """
    m = re.match(r'TCRMP(\d{8})_clip_([A-Za-z]+)', dirname)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}", m.group(2).upper()
    return None, None


def find_cpc_directories(root_path):
    """Recursively find all directories containing .cpc files.

    Handles special subdirectory patterns:
      - ColorCorr/ (2016): CPC files here, images in parent
      - JPEG/ (2009-2010): duplicate re-analyses, skip if parent has CPC files
      - _EDITED/ (2015): preferred version, parent has no CPC files
    """
    root = Path(root_path)
    results = []
    seen_dirs = set()
    # Track which parent dirs already have CPC entries, to skip JPEG duplicates
    parent_has_cpc = set()

    # First pass: collect all CPC directories
    all_cpc_dirs = []
    for cpc_file in sorted(root.rglob("*.cpc")):
        cpc_dir = cpc_file.parent
        if cpc_dir in seen_dirs:
            continue
        seen_dirs.add(cpc_dir)
        all_cpc_dirs.append(cpc_dir)

    # Sort so parent dirs come before their children
    all_cpc_dirs.sort(key=lambda d: len(d.parts))

    for cpc_dir in all_cpc_dirs:
        # Skip JPEG subdirectories when parent already has CPC files
        # These contain duplicate re-analyses (2009 CSE, 2010 GRP)
        if cpc_dir.name == "JPEG" and cpc_dir.parent in parent_has_cpc:
            continue

        # Only process TCRMP-named CPC files (skip CIG and other prefixes)
        cpc_files = sorted(f for f in cpc_dir.glob("*.cpc")
                           if f.stem.upper().startswith("TCRMP"))
        if not cpc_files:
            continue

        # For ColorCorr dirs, images might be in parent directory
        image_dir = cpc_dir
        if "_ColorCorr" in cpc_dir.name:
            jpgs_here = list(cpc_dir.glob("*.jpg")) + list(cpc_dir.glob("*.JPG"))
            jpgs_parent = list(cpc_dir.parent.glob("*.jpg")) + list(cpc_dir.parent.glob("*.JPG"))
            if not jpgs_here and jpgs_parent:
                image_dir = cpc_dir.parent

        # Walk up to find the TCRMP{date}_clip_{site} directory
        dir_date, dir_site = None, None
        for parent in [cpc_dir] + list(cpc_dir.parents):
            dir_date, dir_site = parse_directory_name(parent.name)
            if dir_date:
                break

        results.append({
            "cpc_dir": cpc_dir,
            "image_dir": image_dir,
            "cpc_files": cpc_files,
            "dir_date": dir_date,
            "dir_site": dir_site,
        })
        parent_has_cpc.add(cpc_dir)

    return results


def find_image_for_cpc(cpc_path, image_dir):
    """Find the matching image file for a CPC file."""
    basename = cpc_path.stem
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        candidate = image_dir / (basename + ext)
        if candidate.exists():
            return candidate

    # Mangled name fallback: match by transect suffix
    m = re.search(r'_(T\d{3})$', basename)
    if m:
        t_suffix = m.group(1)
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            candidates = list(image_dir.glob(f"*_{t_suffix}{ext}"))
            if len(candidates) == 1:
                return candidates[0]

    return None


def _parse_csv_line(line):
    """Parse a CSV-like line handling quoted fields."""
    parts = []
    in_quote = False
    current = ""
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        elif ch == ',' and not in_quote:
            parts.append(current.strip().strip('"'))
            current = ""
        else:
            current += ch
    parts.append(current.strip().strip('"'))
    return parts
