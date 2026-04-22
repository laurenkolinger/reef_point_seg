#!/usr/bin/env python3
"""
TCRMPclip_cpcID - Parse CPCe .cpc annotation files and export point data.

Reads .cpc files, extracts point coordinates (A-T), scales from CPCe canvas
space to image pixels, joins species codes from all_points.csv, and exports
in SAM-click-prompt format compatible with downstream segmentation.

Two modes:
  Single:  python app.py <input_dir> <output_dir>
  Batch:   python app.py --batch <root_input> <root_output>
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from parser import find_cpc_directories
from species import SpeciesLookup
from process import process_directory
from export import export_sam_prompts, export_point_coords
from summarize import summarize_output, summarize_source


def log_to(logfile, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


def run_single(input_dir, output_dir, all_points_path, test_images):
    """Process a single directory of CPC + image files."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    os.makedirs(output_dir / "raw", exist_ok=True)
    os.makedirs(output_dir / "ids", exist_ok=True)
    os.makedirs(output_dir / "test_pts", exist_ok=True)

    logfile = open(output_dir / "log.txt", "w")
    log = lambda msg: log_to(logfile, msg)

    log("TCRMPclip_cpcID - single directory mode")
    log(f"Input:  {input_dir}")
    log(f"Output: {output_dir}")

    cpc_dirs = find_cpc_directories(input_dir)
    total_cpc = sum(len(d["cpc_files"]) for d in cpc_dirs)
    log(f"Found {len(cpc_dirs)} CPC director{'y' if len(cpc_dirs)==1 else 'ies'} with {total_cpc} files")

    species = _load_species(all_points_path, input_dir, log)

    all_sam = {}
    all_csv = []
    total_stats = _empty_stats()

    for cpc_info in cpc_dirs:
        log(f"Processing: {cpc_info['cpc_dir'].name} ({len(cpc_info['cpc_files'])} files)")
        sam, csv_rows, stats = process_directory(
            cpc_info, species, output_dir, log,
            test_image_limit=test_images,
            test_images_generated=total_stats["test_images"],
        )
        all_sam.update(sam)
        all_csv.extend(csv_rows)
        _merge_stats(total_stats, stats)

    export_sam_prompts(all_sam, output_dir / "ids" / "sam_click_prompts.json")
    export_point_coords(all_csv, output_dir / "ids" / "point_coords.csv")

    _log_summary(log, total_stats)

    # Dataset summary
    source_summary = summarize_source(input_dir)
    output_summary = summarize_output(output_dir)
    with open(output_dir / "dataset_summary.txt", "w") as f:
        if source_summary:
            log(f"\nSource (by period):\n{source_summary}")
            f.write(f"Source (by period):\n{source_summary}\n\n")
        if output_summary:
            log(f"\nProcessed output:\n{output_summary}")
            f.write(f"Processed output:\n{output_summary}\n")

    logfile.close()
    _print_summary(total_stats, output_dir)


def run_batch(root_input, root_output, all_points_path, test_images_per_dir):
    """Discover and process all CPC directories under root_input."""
    root_input = Path(root_input).resolve()
    root_output = Path(root_output).resolve()
    os.makedirs(root_output, exist_ok=True)

    master_log = open(root_output / "batch_log.txt", "w")
    log = lambda msg: log_to(master_log, msg)

    log("TCRMPclip_cpcID - BATCH mode")
    log(f"Root input:  {root_input}")
    log(f"Root output: {root_output}")

    log("Scanning for CPC directories...")
    all_cpc_dirs = find_cpc_directories(root_input)
    total_cpc = sum(len(d["cpc_files"]) for d in all_cpc_dirs)
    log(f"Found {len(all_cpc_dirs)} CPC directories with {total_cpc} total files")

    if not all_cpc_dirs:
        log("No CPC files found. Nothing to do.")
        master_log.close()
        return

    grouped = _group_by_site_date(all_cpc_dirs)
    log(f"Grouped into {len(grouped)} site-date combinations")

    species = _load_species(all_points_path, root_input, log)

    grand_stats = _empty_stats()
    grand_stats["directories"] = 0

    for group_key, group_dirs in sorted(grouped.items()):
        date_str, site = group_key
        dir_name = f"{site}_{date_str}" if date_str else site
        out_dir = root_output / dir_name

        os.makedirs(out_dir / "raw", exist_ok=True)
        os.makedirs(out_dir / "ids", exist_ok=True)
        os.makedirs(out_dir / "test_pts", exist_ok=True)

        dir_log = open(out_dir / "log.txt", "w")
        dlog = lambda msg, _dl=dir_log: log_to(_dl, msg)

        group_sam = {}
        group_csv = []
        group_stats = _empty_stats()

        for cpc_info in group_dirs:
            dlog(f"Processing: {cpc_info['cpc_dir']}")
            sam, csv_rows, stats = process_directory(
                cpc_info, species, out_dir, dlog,
                test_image_limit=test_images_per_dir,
                test_images_generated=group_stats["test_images"],
            )
            group_sam.update(sam)
            group_csv.extend(csv_rows)
            _merge_stats(group_stats, stats)

        export_sam_prompts(group_sam, out_dir / "ids" / "sam_click_prompts.json")
        export_point_coords(group_csv, out_dir / "ids" / "point_coords.csv")
        _log_summary(dlog, group_stats)
        dir_log.close()

        _merge_stats(grand_stats, group_stats)
        grand_stats["directories"] += 1

        status = "OK" if group_stats["species_missed"] == 0 else f"WARN({group_stats['species_missed']} missed)"
        fuzzy_note = f", {group_stats['fuzzy_dates']} fuzzy-date" if group_stats["fuzzy_dates"] else ""
        log(f"  {dir_name}: {group_stats['images']} imgs, {group_stats['points']} pts, "
            f"{group_stats['species_matched']} matched{fuzzy_note} [{status}]")

    log(f"\n{'='*60}")
    log("BATCH COMPLETE")
    _log_summary(log, grand_stats)
    log(f"Output directories: {grand_stats['directories']}")
    master_log.close()
    _print_summary(grand_stats, root_output, batch=True)


def _load_species(explicit_path, input_dir, log):
    if explicit_path:
        p = Path(explicit_path).resolve()
    else:
        script_dir = Path(__file__).resolve().parent.parent
        candidates = [
            script_dir / "output" / "all_points.csv",
            script_dir.parent / "output" / "all_points.csv",
        ]
        d = Path(input_dir)
        for _ in range(6):
            candidates.append(d / "output" / "all_points.csv")
            d = d.parent

        p = None
        for c in candidates:
            if c.exists():
                p = c
                break

    if p and p.exists():
        log(f"Loading species data from {p}")
        species = SpeciesLookup(str(p))
        log(f"Loaded {len(species)} species records")
        return species
    else:
        log("WARNING: all_points.csv not found - species codes will be empty")
        return None


def _group_by_site_date(cpc_dirs):
    """Group CPC directories by (date, site) for output organization."""
    groups = {}
    for d in cpc_dirs:
        key = (d["dir_date"] or "unknown", d["dir_site"] or "unknown")
        groups.setdefault(key, []).append(d)
    return groups


def _empty_stats():
    return {
        "images": 0, "points": 0, "species_matched": 0,
        "species_missed": 0, "fuzzy_dates": 0, "no_image": 0,
        "test_images": 0,
    }


def _merge_stats(target, source):
    for k in source:
        if k in target:
            target[k] += source[k]


def _log_summary(log, stats):
    log(f"Summary: {stats['images']} images, {stats['points']} points")
    log(f"  Species matched: {stats['species_matched']}, "
        f"missed: {stats['species_missed']}, "
        f"fuzzy-date: {stats['fuzzy_dates']}")
    if stats["no_image"]:
        log(f"  Missing images: {stats['no_image']}")
    if stats["test_images"]:
        log(f"  Test images: {stats['test_images']}")


def _print_summary(stats, output_dir, batch=False):
    total = stats["species_matched"] + stats["species_missed"]
    pct = f"{stats['species_matched']/total*100:.1f}%" if total else "N/A"
    print(f"\n{'='*50}")
    print(f"{'BATCH ' if batch else ''}COMPLETE")
    print(f"  Images:  {stats['images']}")
    print(f"  Points:  {stats['points']}")
    print(f"  Species: {stats['species_matched']}/{total} matched ({pct})")
    if stats["fuzzy_dates"]:
        print(f"  Fuzzy date matches: {stats['fuzzy_dates']}")
    if stats["no_image"]:
        print(f"  Missing images: {stats['no_image']}")
    print(f"  Output:  {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TCRMPclip_cpcID - Parse CPCe files and export point data"
    )
    ap.add_argument("input_dir", help="Input directory (single dir or root for --batch)")
    ap.add_argument("output_dir", help="Output directory")
    ap.add_argument("--batch", action="store_true",
                    help="Batch mode: recursively find and process all CPC directories")
    ap.add_argument("--all-points", default=None,
                    help="Path to all_points.csv (auto-detected if not specified)")
    ap.add_argument("--test-images", type=int, default=3,
                    help="Test overlay images per directory (default: 3, 0=skip, -1=all)")
    args = ap.parse_args()

    test_n = args.test_images if args.test_images >= 0 else 999999

    if args.batch:
        run_batch(args.input_dir, args.output_dir, args.all_points, test_n)
    else:
        run_single(args.input_dir, args.output_dir, args.all_points, test_n)
