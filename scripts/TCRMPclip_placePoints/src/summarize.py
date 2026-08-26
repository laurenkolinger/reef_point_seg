"""
Generate dataset summary from OCR detection output.

Reads sam_click_prompts.json and produces dataset_summary.txt
with all periods folded into a single section per year.

Adapted from TCRMPclip_cpcID/src/summarize.py.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def summarize_year(year, sam_json_path):
    """Generate a dataset summary string for one year.

    Args:
        year: int year (e.g. 2020)
        sam_json_path: path to ids/sam_click_prompts.json

    Returns:
        Formatted summary string, or None if no data.
    """
    sam_path = Path(sam_json_path)
    if not sam_path.exists():
        return None

    with open(sam_path) as f:
        data = json.load(f)

    if not data:
        return None

    sites = defaultdict(lambda: {
        'transects': set(),
        'images': 0,
        'points': 0,
        'species_found': 0,
    })

    for img_name, entry in data.items():
        # Extract site and transect from filename
        m = re.match(r'TCRMP\d{8}_clip_([A-Za-z]+)_T(\d)\d{2}', img_name)
        if not m:
            # Try alternate patterns (2025 variants)
            m = re.match(r'TCRMP_\d{8}_clip_([A-Za-z]+)_T(\d)\d{2}', img_name)
        if not m:
            m = re.match(r'\d{8}TCRMP_clip_([A-Za-z]+)_T(\d)\d{2}', img_name)
        if not m:
            continue

        site = m.group(1).upper()
        transect = int(m.group(2))
        pts = entry.get('points', [])
        # REVIEW points carry species 'REVIEW' but are not a real ID — exclude
        # them from the species-found count so the percentage isn't inflated.
        labeled = sum(1 for p in pts
                      if p.get('species') and p.get('species') != 'REVIEW'
                      and not p.get('review'))

        sites[site]['transects'].add(transect)
        sites[site]['images'] += 1
        sites[site]['points'] += len(pts)
        sites[site]['species_found'] += labeled

    return _format_summary(year, sites, len(data))


def _format_summary(year, sites, total_images):
    """Format a flat site summary (all periods folded)."""
    lines = []

    total_t = sum(len(s['transects']) for s in sites.values())
    total_p = sum(s['points'] for s in sites.values())
    total_sp = sum(s['species_found'] for s in sites.values())
    pct = f"{100 * total_sp / total_p:.1f}%" if total_p else "N/A"

    lines.append(f"{year} TCRMP_clip OCR Dataset Summary")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        f"Dataset: {len(sites)} sites, {total_t} transects, "
        f"{total_images} images, {total_p} points ({pct} with species)"
    )
    lines.append(
        f"{'Site':>6} {'Trans':>6} {'Images':>7} {'Points':>7} {'Species':>8}"
    )
    lines.append(
        f"{'-' * 6} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 8}"
    )

    for site in sorted(sites):
        s = sites[site]
        t = sorted(s['transects'])
        sp_pct = (
            f"{100 * s['species_found'] / s['points']:.0f}%"
            if s['points'] else ""
        )
        lines.append(
            f"{site:>6} {len(t):>6} {s['images']:>7} {s['points']:>7} "
            f"{sp_pct:>8}   T{'.'.join(str(x) for x in t)}"
        )

    return "\n".join(lines)


def write_summary(year, export_dir):
    """Generate and write dataset_summary.txt for a year directory.

    Args:
        year: int year
        export_dir: root export dir (year subdir expected within)
    """
    year_dir = Path(export_dir) / str(year)
    sam_path = year_dir / 'ids' / 'sam_click_prompts.json'
    summary = summarize_year(year, sam_path)

    if summary:
        out_path = year_dir / 'dataset_summary.txt'
        with open(out_path, 'w') as f:
            f.write(summary + '\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python summarize.py <year> <export_dir>")
        sys.exit(1)
    year = int(sys.argv[1])
    export_dir = sys.argv[2]
    write_summary(year, export_dir)
    print(f"Summary written for {year}")
