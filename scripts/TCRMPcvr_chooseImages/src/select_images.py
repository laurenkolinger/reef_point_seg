#!/usr/bin/env python3
"""
TCRMPcvr_chooseImages — Balanced image selection for coral training data.

Selects frames evenly across years, sites, and transects to achieve
TARGET_INSTANCES_PER_SPECIES frame-instances for each target coral species.

Outputs:
  output/selected_frames.csv      — master list of selected frames + metadata
  output/route_cpc.csv            — frames with existing CPC point coords (pre-2020)
  output/route_ocr.csv            — frames needing OCR processing (2020+)
  output/route_missing.csv        — frames with no source image found
  output/selection_summary.txt    — human-readable summary
  output/selection_diagnostics.png — distribution plots

Usage:
  python src/select_images.py
  python src/select_images.py --all-points /path/to/all_points.csv
  python src/select_images.py --species OFRA PA OA
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

# Allow running from repo root or from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_recoded(recode_dir, fallback):
    """Use most recent recoded all_points if available, else fallback."""
    pattern = os.path.join(recode_dir, "all_points_*.csv")
    files = sorted(glob.glob(pattern))
    if files:
        return files[-1]
    return fallback


def build_image_filename(date_str, site, transect, frame):
    """Reconstruct the TCRMP image filename from frame metadata."""
    # date_str is YYYY-MM-DD, need YYYYMMDD
    ymd = date_str.replace("-", "")
    return f"TCRMP{ymd}_clip_{site}_T{int(transect)}{int(frame):02d}"


def find_source_image(basename, year_int, clip_dir):
    """Find the actual image file on disk. Returns (path, ext) or (None, None)."""
    year_dir_pattern = os.path.join(clip_dir, f"TCRMP{year_int}_clip")
    # Images can be in subdirectories (Annual/, PBL/, SCTLD/, or flat)
    for ext in ["jpg", "jpeg", "JPG", "JPEG"]:
        # Search recursively under the year directory
        pattern = os.path.join(year_dir_pattern, "**", f"{basename}.{ext}")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0], ext
    return None, None


def find_pts_image(basename, year_int, clip_dir):
    """Find the _pts annotated variant."""
    year_dir_pattern = os.path.join(clip_dir, f"TCRMP{year_int}_clip")
    for ext in ["jpg", "jpeg", "JPG", "JPEG"]:
        pattern = os.path.join(year_dir_pattern, "**", f"{basename}_pts.{ext}")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def load_cpc_basenames(cpc_dir, years):
    """Load all basenames that have CPC point coords, for given years.
    Returns a set of basenames (without extension)."""
    basenames = set()
    for yr in years:
        csv_path = os.path.join(cpc_dir, str(yr), "ids", "point_coords.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            names = df["raw_image"].str.replace(r"\.(jpg|jpeg)$", "", regex=True)
            basenames.update(names)
        except Exception:
            pass
    return basenames


def _load_species_remap():
    """Load old_code -> new_code mapping from recodeSpecies remap_log."""
    import json as _json
    remap = {}
    recode_dir = os.path.join(config._REPO_DIR, "TCRMPcvr_recodeSpecies", "output")
    logs = sorted(glob.glob(os.path.join(recode_dir, "remap_log_*.json")))
    if not logs:
        return remap
    with open(logs[-1]) as f:
        data = _json.load(f)
    for entry in data.get("remaps", []):
        old = entry.get("old_code", "")
        new = entry.get("new_code", "")
        if old and new and old != new:
            remap[old] = new
    return remap


def load_cpc_species(cpc_dir, years, remap):
    """Build a DataFrame of (frame_id, species_code, category, x, y) from cpc_all.

    Uses point_coords.csv from cpc_all (correct label-to-species mapping)
    with species remap applied. Returns rows compatible with all_points format.
    """
    import re
    rows = []
    for yr in years:
        csv_path = os.path.join(cpc_dir, str(yr), "ids", "point_coords.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            basename = os.path.splitext(r["raw_image"])[0]
            m = re.match(r"TCRMP(\d{4})(\d{2})(\d{2})_clip_([A-Za-z]+)_T(\d)(\d{2})", basename)
            if not m:
                continue
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            site = m.group(4).upper()
            transect = int(m.group(5))
            frame = int(m.group(6))
            sp = r.get("species_code", "")
            sp = remap.get(sp, sp)  # apply recode
            rows.append({
                "date": date_str,
                "year": int(m.group(1)),
                "site": site,
                "transect": transect,
                "frame": frame,
                "species_code": sp,
                "category": r.get("category", ""),
                "point_label": r.get("label", ""),
                "x": r.get("x", np.nan),
                "y": r.get("y", np.nan),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Selection algorithm
# ─────────────────────────────────────────────────────────────────────────────

def even_allocate(binary, species_list, years, target):
    """Allocate target instances evenly across years, capped by availability."""
    alloc = {}
    for sp in species_list:
        avail = {yr: int(binary[binary["year_int"] == yr][sp].sum()) for yr in years}
        sp_alloc = {yr: 0 for yr in years}
        remaining = target
        active = list(years)
        for _ in range(10):
            if remaining <= 0 or not active:
                break
            per_year = remaining / len(active)
            settled = []
            for yr in active:
                can_add = avail[yr] - sp_alloc[yr]
                give = min(per_year, can_add)
                sp_alloc[yr] += int(round(give))
                if sp_alloc[yr] >= avail[yr]:
                    settled.append(yr)
            remaining = target - sum(sp_alloc.values())
            for yr in settled:
                active.remove(yr)
        alloc[sp] = sp_alloc
    return alloc


def greedy_select(binary, species_list, alloc, years):
    """Year-stratified greedy frame selection.

    Two-pass: first selects frames with central target-species points
    (25-75% of image w/h) to fill 1/3 of each species allocation,
    then fills remaining slots with any eligible frame.

    Prefers single-species frames for diversity — frames with only one
    needed species score higher than frames with multiple, unless the
    extra species is also still needed and under-represented.
    """
    has_central = all(f"{sp}_central" in binary.columns for sp in species_list)

    selected = []
    achieved = {sp: 0 for sp in species_list}
    achieved_central = {sp: 0 for sp in species_list}
    used = set()

    for pass_name in (["central", "any"] if has_central else ["any"]):
        for yr in years:
            yr_data = binary[(binary["year_int"] == yr) & (~binary.index.isin(used))]
            yr_rem = {sp: max(0, alloc[sp].get(yr, 0) - sum(
                int(binary.loc[fid, sp]) for fid in selected
                if binary.loc[fid, "year_int"] == yr
            )) for sp in species_list}
            avail = yr_data.copy()

            while any(r > 0 for r in yr_rem.values()) and len(avail) > 0:
                scores = pd.Series(0.0, index=avail.index)
                n_needed = sum(1 for sp in species_list if yr_rem[sp] > 0)
                for sp in species_list:
                    if yr_rem[sp] > 0:
                        scores += avail[sp]

                # Penalise multi-species frames: prefer frames with just
                # one needed species so we get more variety across frames
                if n_needed > 1:
                    n_sp_in_frame = pd.Series(0, index=avail.index)
                    for sp in species_list:
                        if yr_rem[sp] > 0:
                            n_sp_in_frame += avail[sp]
                    # Single-species frames get +0.5 bonus
                    scores += (n_sp_in_frame == 1).astype(float) * 0.5

                if pass_name == "central":
                    central_mask = pd.Series(False, index=avail.index)
                    for sp in species_list:
                        if yr_rem[sp] > 0:
                            central_mask |= avail[f"{sp}_central"] > 0
                    scores = scores.where(central_mask, 0)
                if scores.max() == 0:
                    break
                best = scores.idxmax()
                selected.append(best)
                used.add(best)
                for sp in species_list:
                    added = int(avail.loc[best, sp])
                    yr_rem[sp] = max(0, yr_rem[sp] - added)
                    achieved[sp] += added
                    if has_central and added:
                        achieved_central[sp] += int(avail.loc[best, f"{sp}_central"])
                avail = avail.drop(best)

    if has_central:
        print("Central region coverage:")
        for sp in species_list:
            total = achieved[sp]
            central = achieved_central[sp]
            pct = central / total * 100 if total > 0 else 0
            print(f"  {sp}: {central}/{total} central ({pct:.0f}%)")

    return selected, achieved


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Balanced image selection")
    parser.add_argument("--all-points", default=None,
                        help="Path to all_points CSV (default: auto-detect recoded or raw)")
    parser.add_argument("--master-codes", default=None)
    parser.add_argument("--species", nargs="+", default=None,
                        help="Override target species (e.g., --species OFRA PA OA)")
    parser.add_argument("--target", type=int, default=None,
                        help="Frame-instances per species (default: from config)")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="Skip verifying images on disk (faster, no routing)")
    args = parser.parse_args()

    species_list = args.species or config.TARGET_SPECIES
    target = args.target or config.TARGET_INSTANCES_PER_SPECIES

    # Find input file
    if args.all_points:
        ap_path = args.all_points
    else:
        ap_path = find_latest_recoded(config._RECODE_DIR, config.DEFAULT_ALL_POINTS)
    mc_path = args.master_codes or config.DEFAULT_MASTER_CODES

    print(f"Input:   {ap_path}")
    print(f"Codes:   {mc_path}")
    print(f"Species: {species_list}")
    print(f"Target:  {target} frame-instances each")
    print()

    # Load species remap (e.g. MFRA->OFRA, MA->OA, MFAV->OFAV)
    remap = _load_species_remap()
    if remap:
        print(f"Species remap: {remap}")

    # Load data — use cpc_all for pre-2020, all_points for 2020+
    ap = pd.read_csv(ap_path)
    mc = pd.read_csv(mc_path)
    print(f"Loaded {len(ap):,} rows from all_points, {len(mc)} codes")

    # For pre-2020: replace species data with cpc_all (correct label mapping)
    cpc_years = list(range(config.MIN_YEAR, 2020))
    cpc_species = load_cpc_species(config.CPC_ALL_DIR, cpc_years, remap)
    if len(cpc_species) > 0:
        print(f"Loaded {len(cpc_species):,} CPC points from cpc_all (pre-2020, remapped)")
        # Remove pre-2020 rows from all_points, replace with cpc_all data
        ap = ap[ap["year"] >= 2020].copy()
        # Apply remap to post-2020 all_points too
        ap["species_code"] = ap["species_code"].replace(remap)
        ap = pd.concat([cpc_species, ap], ignore_index=True)
        print(f"Combined: {len(ap):,} rows (cpc_all pre-2020 + all_points 2020+)")
    else:
        # No cpc_all data, apply remap to all_points
        ap["species_code"] = ap["species_code"].replace(remap)

    # Filter
    ap = ap[(ap["year"] >= config.MIN_YEAR) & (ap["year"] <= config.MAX_YEAR)].copy()
    ap["frame_id"] = (
        ap["date"].astype(str) + "|" + ap["site"] + "|" +
        ap["transect"].astype(str) + "|" + ap["frame"].astype(str)
    )
    print(f"Post-{config.MIN_YEAR} rows: {len(ap):,}")

    coral = ap[ap["category"] == config.CATEGORY_FILTER].copy()

    # Build binary frame × species presence matrix
    target_rows = coral[coral["species_code"].isin(species_list)]
    frame_sp = target_rows.groupby(["frame_id", "species_code"]).size().unstack(fill_value=0)
    for sp in species_list:
        if sp not in frame_sp.columns:
            frame_sp[sp] = 0
    binary = (frame_sp[species_list] > 0).astype(int)

    # ── Central region flags ────────────────────────────────────────────────
    # For each frame+species, check if any target point is in the central
    # 25-75% region of the image. Image dims: 1920x1080 (pre-2017),
    # 3840x2160 (2017+). We use x,y from the data if available.
    if "x" in coral.columns and "y" in coral.columns:
        for sp in species_list:
            sp_pts = target_rows[target_rows["species_code"] == sp].copy()
            if sp_pts.empty or sp_pts["x"].isna().all():
                binary[f"{sp}_central"] = 0
                continue
            # Infer image dims per frame from year
            sp_pts = sp_pts.copy()
            sp_pts["img_w"] = np.where(sp_pts["year"] >= 2017, 3840, 1920)
            sp_pts["img_h"] = np.where(sp_pts["year"] >= 2017, 2160, 1080)
            sp_pts["in_central"] = (
                (sp_pts["x"] >= sp_pts["img_w"] * 0.25) &
                (sp_pts["x"] <= sp_pts["img_w"] * 0.75) &
                (sp_pts["y"] >= sp_pts["img_h"] * 0.25) &
                (sp_pts["y"] <= sp_pts["img_h"] * 0.75)
            )
            central_frames = sp_pts[sp_pts["in_central"]].groupby("frame_id").size()
            binary[f"{sp}_central"] = 0
            for fid in central_frames.index:
                if fid in binary.index:
                    binary.loc[fid, f"{sp}_central"] = 1
        n_central = {sp: int(binary[f"{sp}_central"].sum()) for sp in species_list}
        print(f"Frames with central-region target points: {n_central}")

    # Add metadata
    meta = ap.drop_duplicates("frame_id").set_index("frame_id")[
        ["date", "year", "site", "transect", "frame"]
    ]
    binary = binary.join(meta, how="left")
    binary["year_int"] = binary["year"].astype(int)

    years = sorted(binary["year_int"].unique())
    years = [y for y in years if config.MIN_YEAR <= y <= config.MAX_YEAR]

    print(f"Frames with any target species (before filtering): {len(binary):,}")

    # ── Pre-filter: exclude pre-2020 frames without CPC point coords ────────
    # These frames have species labels but no (x,y) pixel coordinates,
    # making them useless for downstream SAM3 segmentation.
    pre2020 = binary[binary["year_int"] < 2020]
    post2020 = binary[binary["year_int"] >= 2020]

    if len(pre2020) > 0:
        print("Loading CPC point_coords to filter pre-2020 frames...")
        cpc_years = [y for y in years if y < 2020]
        cpc_basenames = load_cpc_basenames(config.CPC_ALL_DIR, cpc_years)
        print(f"  CPC has coords for {len(cpc_basenames):,} frames")

        # Build basename for each pre-2020 frame to check against CPC
        pre2020_keep = []
        pre2020_drop = 0
        for frame_id in pre2020.index:
            r = pre2020.loc[frame_id]
            basename = build_image_filename(
                str(r["date"]), r["site"], int(r["transect"]), int(r["frame"])
            )
            if basename in cpc_basenames:
                pre2020_keep.append(frame_id)
            else:
                pre2020_drop += 1

        pre2020_filtered = pre2020.loc[pre2020_keep]
        binary = pd.concat([pre2020_filtered, post2020])
        print(f"  Kept {len(pre2020_filtered):,} pre-2020 frames with CPC coords")
        print(f"  Dropped {pre2020_drop:,} pre-2020 frames (no CPC coords)")

    # Recompute years after filtering
    years = sorted(binary["year_int"].unique())
    years = [y for y in years if config.MIN_YEAR <= y <= config.MAX_YEAR]

    print(f"Eligible frames after filtering: {len(binary):,}")
    print()

    # Availability
    print("Available frame-instances per species:")
    for sp in species_list:
        total = int(binary[sp].sum())
        status = "OK" if total >= target else f"WARNING: only {total} available"
        print(f"  {sp:6s}: {total:>6,} frames  [{status}]")
    print()

    # Allocate and select
    alloc = even_allocate(binary, species_list, years, target)
    selected, achieved = greedy_select(binary, species_list, alloc, years)

    sel = binary.loc[selected].copy()
    n = len(sel)
    print(f"Selected {n:,} frames")
    for sp in species_list:
        ok = "OK" if achieved[sp] >= target else f"SHORT by {target - achieved[sp]}"
        print(f"  {sp:6s}: {achieved[sp]:>5,} frame-instances  [{ok}]")
    print()

    # ── Build output with image paths and routing ────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    rows_out = []
    for frame_id in selected:
        r = sel.loc[frame_id]
        date_str = str(r["date"])
        site = r["site"]
        transect = int(r["transect"])
        frame = int(r["frame"])
        year_int = int(r["year_int"])
        basename = build_image_filename(date_str, site, transect, frame)

        # Which target species are in this frame?
        spp_present = [sp for sp in species_list if int(r[sp]) == 1]

        row = {
            "frame_id": frame_id,
            "basename": basename,
            "date": date_str,
            "year": year_int,
            "site": site,
            "transect": transect,
            "frame": frame,
            "species_present": ";".join(spp_present),
            "n_target_species": len(spp_present),
        }

        # Routing: simple split — pre-2020 already have CPC coords (we
        # filtered out those without), 2020+ all need OCR processing.
        row["route"] = "cpc" if year_int < 2020 else "ocr_needed"
        row["coords_source"] = os.path.join(
            config.CPC_ALL_DIR, str(year_int), "ids", "point_coords.csv"
        ) if year_int < 2020 else ""

        if not args.skip_image_check:
            img_path, ext = find_source_image(basename, year_int, config.CLIP_DIR)
            pts_path = find_pts_image(basename, year_int, config.CLIP_DIR)
            row["source_image"] = img_path or ""
            row["pts_image"] = pts_path or ""
            row["image_found"] = img_path is not None
        else:
            row["source_image"] = ""
            row["pts_image"] = ""
            row["image_found"] = None

        rows_out.append(row)

    df_out = pd.DataFrame(rows_out)

    # ── Save outputs ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Master list
    master_path = os.path.join(config.OUTPUT_DIR, "selected_frames.csv")
    df_out.to_csv(master_path, index=False)
    print(f"Saved: {master_path} ({len(df_out)} rows)")

    # Route splits
    for route_name, filename in [
        ("cpc", "route_cpc.csv"),
        ("ocr_needed", "route_ocr_needed.csv"),
    ]:
        subset = df_out[df_out["route"] == route_name]
        if len(subset) > 0:
            p = os.path.join(config.OUTPUT_DIR, filename)
            subset.to_csv(p, index=False)
            print(f"Saved: {p} ({len(subset)} rows)")

    if not args.skip_image_check:
        missing = df_out[df_out["image_found"] == False]  # noqa: E712
        if len(missing) > 0:
            p = os.path.join(config.OUTPUT_DIR, "route_missing.csv")
            missing.to_csv(p, index=False)
            print(f"Saved: {p} ({len(missing)} rows — images not found on disk)")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_lines = [
        f"TCRMPcvr_chooseImages — Selection Summary",
        f"Generated: {datetime.now().isoformat()}",
        f"",
        f"Input:   {os.path.basename(ap_path)}",
        f"Species: {', '.join(species_list)}",
        f"Target:  {target} frame-instances per species",
        f"",
        f"=== Results ===",
        f"Total frames selected: {n:,}",
        f"",
    ]
    for sp in species_list:
        summary_lines.append(f"  {sp:6s}: {achieved[sp]:>5,} frame-instances")
    summary_lines.append("")

    # Year distribution
    yr_dist = sel["year_int"].value_counts().sort_index()
    summary_lines.append("=== Frames per Year ===")
    for y, c in yr_dist.items():
        summary_lines.append(f"  {y}: {c:>5}")
    summary_lines.append("")

    # Site distribution
    site_dist = df_out["site"].value_counts().sort_values(ascending=False)
    summary_lines.append(f"=== Sites ({len(site_dist)}) ===")
    for s, c in site_dist.items():
        summary_lines.append(f"  {s:5s}: {c:>5}")
    summary_lines.append("")

    # Transect distribution
    trans_dist = df_out["transect"].value_counts().sort_index()
    summary_lines.append("=== Transects ===")
    for t, c in trans_dist.items():
        summary_lines.append(f"  T{int(t)}: {c:>5}")
    summary_lines.append("")

    # Species-per-frame distribution
    sp_count_dist = df_out["n_target_species"].value_counts().sort_index()
    summary_lines.append("=== Target species per frame ===")
    for nsp, c in sp_count_dist.items():
        label = f"{int(nsp)} species"
        summary_lines.append(f"  {label:12s}: {c:>5}")
    summary_lines.append("")

    # Central region coverage
    has_central = all(f"{sp}_central" in sel.columns for sp in species_list)
    if has_central:
        summary_lines.append("=== Central region (25-75% of image) ===")
        for sp in species_list:
            total = int(sel[sp].sum())
            central = int(sel[f"{sp}_central"].sum())
            pct = central / total * 100 if total > 0 else 0
            summary_lines.append(f"  {sp:6s}: {central:>3}/{total} frames with central point ({pct:.0f}%)")
        summary_lines.append("")

    # Routing summary
    route_dist = df_out["route"].value_counts()
    summary_lines.append("=== Routing ===")
    for r, c in route_dist.items():
        summary_lines.append(f"  {r:15s}: {c:>5}")
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    summary_path = os.path.join(config.OUTPUT_DIR, "selection_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved: {summary_path}")

    # Save config snapshot
    cfg_path = os.path.join(config.OUTPUT_DIR, "config_snapshot.txt")
    with open(cfg_path, "w") as f:
        f.write(f"timestamp: {ts}\n")
        f.write(f"species: {species_list}\n")
        f.write(f"target: {target}\n")
        f.write(f"min_year: {config.MIN_YEAR}\n")
        f.write(f"max_year: {config.MAX_YEAR}\n")
        f.write(f"all_points: {ap_path}\n")
        f.write(f"master_codes: {mc_path}\n")

    print(f"\nDone. Run 'python src/plot_diagnostics.py' for distribution plots.")


if __name__ == "__main__":
    main()
