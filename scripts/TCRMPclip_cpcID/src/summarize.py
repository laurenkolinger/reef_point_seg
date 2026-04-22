#!/usr/bin/env python3
"""
Generate dataset summary from CPC Reader output or source CPC directories.

Standalone usage:
    python summarize.py <output_dir>
    python summarize.py --source <input_dir>
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def summarize_output(output_dir):
    """Summarize from processed output (sam_click_prompts.json)."""
    output_dir = Path(output_dir)
    sam_path = output_dir / "ids" / "sam_click_prompts.json"
    if not sam_path.exists():
        return None

    with open(sam_path) as f:
        data = json.load(f)

    sites = defaultdict(lambda: {"transects": set(), "images": 0, "points": 0, "species_found": 0})

    for img_name, entry in data.items():
        m = re.match(r'TCRMP\d{8}_clip_([A-Za-z]+)_T(\d)\d{2}', img_name)
        if not m:
            continue
        site = m.group(1).upper()
        transect = int(m.group(2))
        pts = entry["points"]
        labeled = sum(1 for p in pts if p.get("species"))

        sites[site]["transects"].add(transect)
        sites[site]["images"] += 1
        sites[site]["points"] += len(pts)
        sites[site]["species_found"] += labeled

    return _format_summary(sites, len(data))


def summarize_source(input_dir):
    """Summarize from source CPC files, grouped by period subdirectory."""
    input_dir = Path(input_dir)
    # Group by first-level subdirectory (period) if they exist
    periods = defaultdict(lambda: defaultdict(lambda: {"transects": set(), "images": 0, "points": 0}))

    for cpc in sorted(input_dir.rglob("*.cpc")):
        # Skip JPEG duplicates
        if cpc.parent.name == "JPEG":
            continue

        m = re.match(r'TCRMP\d{8}_clip_([A-Za-z]+)_T(\d)\d{2}', cpc.stem)
        if not m:
            continue
        site = m.group(1).upper()
        transect = int(m.group(2))

        # Determine period from path
        rel = cpc.relative_to(input_dir)
        period = rel.parts[0] if len(rel.parts) > 2 else "All"

        try:
            with open(cpc, 'r', errors='replace') as f:
                lines = f.readlines()
            npts = int(lines[5].strip())
        except Exception:
            npts = 20

        periods[period][site]["transects"].add(transect)
        periods[period][site]["images"] += 1
        periods[period][site]["points"] += npts

    return _format_period_summary(periods)


def _format_summary(sites, total_images):
    """Format a flat site summary."""
    lines = []
    total_t = sum(len(s["transects"]) for s in sites.values())
    total_p = sum(s["points"] for s in sites.values())
    total_sp = sum(s["species_found"] for s in sites.values())
    pct = f"{100*total_sp/total_p:.1f}%" if total_p else "N/A"

    lines.append(f"Dataset: {len(sites)} sites, {total_t} transects, {total_images} images, {total_p} points ({pct} with species)")
    lines.append(f"{'Site':>6} {'Trans':>6} {'Images':>7} {'Points':>7} {'Species':>8}")
    lines.append(f"{'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
    for site in sorted(sites):
        s = sites[site]
        t = sorted(s["transects"])
        sp_pct = f"{100*s['species_found']/s['points']:.0f}%" if s["points"] else ""
        lines.append(f"{site:>6} {len(t):>6} {s['images']:>7} {s['points']:>7} {sp_pct:>8}   T{'.'.join(str(x) for x in t)}")
    return "\n".join(lines)


def _format_period_summary(periods):
    """Format a period-grouped summary."""
    lines = []
    grand_i = 0
    grand_p = 0
    for period in sorted(periods):
        sites = periods[period]
        total_t = sum(len(s["transects"]) for s in sites.values())
        total_i = sum(s["images"] for s in sites.values())
        total_p = sum(s["points"] for s in sites.values())
        grand_i += total_i
        grand_p += total_p

        lines.append(f"\n{period}: {len(sites)} sites, {total_t} transects, {total_i} images, {total_p} points")
        lines.append(f"  {'Site':>6} {'Trans':>6} {'Images':>7} {'Points':>7}")
        lines.append(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
        for site in sorted(sites):
            s = sites[site]
            t = sorted(s["transects"])
            lines.append(f"  {site:>6} {len(t):>6} {s['images']:>7} {s['points']:>7}   T{'.'.join(str(x) for x in t)}")

    lines.append(f"\nGrand total: {grand_i} images, {grand_p} points")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarize CPC dataset")
    ap.add_argument("path", help="Output dir (with ids/) or --source input dir")
    ap.add_argument("--source", action="store_true", help="Summarize from source CPC files")
    args = ap.parse_args()

    if args.source:
        result = summarize_source(args.path)
    else:
        result = summarize_output(args.path)

    if result:
        print(result)
    else:
        print("No data found", file=sys.stderr)
        sys.exit(1)
