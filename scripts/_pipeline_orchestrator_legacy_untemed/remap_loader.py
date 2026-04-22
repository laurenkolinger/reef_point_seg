"""
Remap loader — browse, parse, and auto-apply past remap_log JSON files.

Replicates the core recode logic from TCRMPcvr_recodeSpecies/src/app.py
so remaps can be applied without launching the interactive sub-app.
"""

import os
import json
import glob
from datetime import datetime

import pandas as pd


def browse_remap_logs(repo_dir, extra_dirs=None):
    """Find all remap_log_*.json files in known locations."""
    results = []

    # Standard location: recodeSpecies output
    pattern = os.path.join(repo_dir, "TCRMPcvr_recodeSpecies", "output", "remap_log_*.json")
    for path in sorted(glob.glob(pattern)):
        results.append(_parse_remap_meta(path))

    # Scan any extra directories (e.g., past project dirs)
    for d in (extra_dirs or []):
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.startswith("remap_log_") and f.endswith(".json"):
                    results.append(_parse_remap_meta(os.path.join(root, f)))
            # Don't recurse too deep
            if root.count(os.sep) - d.count(os.sep) > 2:
                dirs.clear()

    # Deduplicate by path
    seen = set()
    deduped = []
    for r in results:
        if r["path"] not in seen:
            seen.add(r["path"])
            deduped.append(r)

    return sorted(deduped, key=lambda x: x["timestamp"], reverse=True)


def _parse_remap_meta(path):
    """Extract metadata from a remap_log JSON."""
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            "path": path,
            "timestamp": data.get("timestamp", ""),
            "n_remaps": len(data.get("remaps", [])),
            "n_excludes": len(data.get("excludes", [])),
            "remaps_summary": [
                f"{r['old_code']}→{r['new_code']}" for r in data.get("remaps", [])[:10]
            ],
        }
    except (json.JSONDecodeError, OSError):
        return {
            "path": path,
            "timestamp": "?",
            "n_remaps": 0,
            "n_excludes": 0,
            "remaps_summary": [],
        }


def load_remap_log(path):
    """Load and return full remap_log JSON."""
    with open(path) as f:
        return json.load(f)


def apply_remaps(all_points_path, master_codes_path, remap_log_path, output_dir):
    """
    Apply remaps from a remap_log JSON to all_points and master_codes CSVs.
    Writes recoded files to output_dir. Returns summary dict.
    """
    # Load source data
    df_points = pd.read_csv(all_points_path)
    df_codes = pd.read_csv(master_codes_path)

    # Load remap log
    with open(remap_log_path) as f:
        remap_data = json.load(f)

    remaps = remap_data.get("remaps", [])
    excludes = set(remap_data.get("excludes", []))

    new_points = df_points.copy()
    new_codes = df_codes.copy()
    log_entries = []

    for rm in remaps:
        old_code = rm["old_code"]
        new_code = rm["new_code"]
        new_name = rm.get("new_name", "")
        new_cat = rm.get("new_category", "")

        mask = new_points["species_code"] == old_code
        n_affected = int(mask.sum())

        new_points.loc[mask, "species_code"] = new_code
        new_points.loc[mask, "species_name"] = new_name
        new_points.loc[mask, "category"] = new_cat

        if old_code != new_code:
            new_codes = new_codes[new_codes["code"] != old_code]
            if new_code not in new_codes["code"].values:
                new_codes = pd.concat([new_codes, pd.DataFrame([{
                    "code": new_code,
                    "category": new_cat,
                    "name": new_name,
                }])], ignore_index=True)
            else:
                idx = new_codes[new_codes["code"] == new_code].index[0]
                new_codes.loc[idx, "name"] = new_name
                new_codes.loc[idx, "category"] = new_cat
            action = "merge"
        else:
            idx = new_codes[new_codes["code"] == old_code].index
            if len(idx):
                new_codes.loc[idx[0], "name"] = new_name
                new_codes.loc[idx[0], "category"] = new_cat
            action = "rename"

        log_entries.append({
            "old_code": old_code,
            "new_code": new_code,
            "new_name": new_name,
            "new_category": new_cat,
            "action": action,
            "points_affected": n_affected,
        })

    new_codes.sort_values("code", inplace=True)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    ap_file = "all_points_recoded.csv"
    mc_file = "master_codes_recoded.csv"
    log_file = "remap_log.json"

    new_points.to_csv(os.path.join(output_dir, ap_file), index=False)
    new_codes.to_csv(os.path.join(output_dir, mc_file), index=False)

    log_out = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_all_points": os.path.basename(all_points_path),
        "source_master_codes": os.path.basename(master_codes_path),
        "source_remap_log": remap_log_path,
        "original_point_count": len(df_points),
        "original_code_count": len(df_codes),
        "new_code_count": len(new_codes),
        "excludes": sorted(excludes),
        "remaps": log_entries,
    }
    with open(os.path.join(output_dir, log_file), "w") as f:
        json.dump(log_out, f, indent=2)

    return {
        "success": True,
        "points_processed": len(df_points),
        "remaps_applied": len(log_entries),
        "excludes": len(excludes),
        "output_files": [ap_file, mc_file, log_file],
    }
