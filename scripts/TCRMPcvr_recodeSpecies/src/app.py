#!/usr/bin/env python3
"""
TCRMPcvr_recodeSpecies — Species code recoding tool.

Loads all_points.csv and master_codes.csv, presents a web UI for remapping
species codes, and outputs recoded CSVs with an audit log.
"""

import os
import sys
import json
import argparse
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)           # scripts/TCRMPcvr_recodeSpecies/
SCRIPTS_DIR = os.path.dirname(PROJECT_DIR)          # scripts/
REPO_DIR = os.path.dirname(SCRIPTS_DIR)             # seg_AI_img_full_april2026/

# Default inputs live in supporting_data/. The orchestrator overrides via CLI
# args (--all-points / --master-codes) so standalone use is the only path here.
DEFAULT_ALL_POINTS = os.path.join(REPO_DIR, "supporting_data", "all_points.csv")
DEFAULT_MASTER_CODES = os.path.join(REPO_DIR, "supporting_data", "master_codes.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
df_points: pd.DataFrame = None
df_codes: pd.DataFrame = None
code_counts: pd.Series = None

app = Flask(__name__, template_folder="templates")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    rows = []
    known_codes = set(df_codes["code"])

    for _, r in df_codes.iterrows():
        rows.append({
            "code": r["code"],
            "category": r["category"],
            "name": r["name"],
            "count": int(code_counts.get(r["code"], 0)),
        })

    # Orphan codes present in all_points but missing from master_codes
    for code, cnt in code_counts.items():
        if code not in known_codes:
            rows.append({
                "code": code,
                "category": "???",
                "name": "???",
                "count": int(cnt),
            })

    rows.sort(key=lambda r: r["count"], reverse=True)
    return render_template("index.html", rows=rows, total_points=len(df_points))


@app.route("/recode", methods=["POST"])
def recode():
    payload = request.get_json(force=True)
    remaps = payload.get("remaps", [])
    excludes = set(payload.get("excludes", []))

    if not remaps and not excludes:
        return jsonify({"success": False, "error": "Nothing to do."})

    # --- Apply remaps to all_points ------------------------------------------
    new_points = df_points.copy()
    new_codes = df_codes.copy()
    log_entries = []

    for rm in remaps:
        old_code = rm["old_code"]
        new_code = rm["new_code"]
        new_name = rm["new_name"]
        new_cat = rm["new_category"]

        mask = new_points["species_code"] == old_code
        n_affected = int(mask.sum())

        new_points.loc[mask, "species_code"] = new_code
        new_points.loc[mask, "species_name"] = new_name
        new_points.loc[mask, "category"] = new_cat

        # Update master_codes
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

    # --- Save outputs --------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ap_file = f"all_points_{ts}.csv"
    mc_file = f"master_codes_{ts}.csv"
    log_file = f"remap_log_{ts}.json"

    new_points.to_csv(os.path.join(OUTPUT_DIR, ap_file), index=False)
    new_codes.to_csv(os.path.join(OUTPUT_DIR, mc_file), index=False)

    log_data = {
        "timestamp": ts,
        "source_all_points": os.path.basename(args_global.all_points),
        "source_master_codes": os.path.basename(args_global.master_codes),
        "original_point_count": len(df_points),
        "original_code_count": len(df_codes),
        "new_code_count": len(new_codes),
        "excludes": sorted(excludes),
        "remaps": log_entries,
    }
    with open(os.path.join(OUTPUT_DIR, log_file), "w") as f:
        json.dump(log_data, f, indent=2)

    # --- Top 10 corals (excluding user-excluded codes) -----------------------
    coral_mask = new_points["category"] == "Coral"
    exclude_mask = ~new_points["species_code"].isin(excludes)
    coral_counts = (
        new_points[coral_mask & exclude_mask]["species_code"]
        .value_counts()
        .head(10)
    )

    code_to_name = dict(zip(new_codes["code"], new_codes["name"]))
    top10 = []
    for rank, (code, cnt) in enumerate(coral_counts.items(), 1):
        top10.append({
            "rank": rank,
            "code": code,
            "name": code_to_name.get(code, "?"),
            "count": int(cnt),
        })

    return jsonify({
        "success": True,
        "output_dir": "output/",
        "files": [ap_file, mc_file, log_file],
        "changes_applied": len(log_entries),
        "excludes_applied": len(excludes),
        "top10_corals": top10,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
args_global = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCRMP Species Recode Tool")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--all-points", default=DEFAULT_ALL_POINTS)
    parser.add_argument("--master-codes", default=DEFAULT_MASTER_CODES)
    args_global = parser.parse_args()

    print(f"Loading {args_global.all_points} ...")
    df_points = pd.read_csv(args_global.all_points)
    print(f"  {len(df_points):,} rows")

    print(f"Loading {args_global.master_codes} ...")
    df_codes = pd.read_csv(args_global.master_codes)
    print(f"  {len(df_codes)} codes")

    code_counts = df_points["species_code"].value_counts()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Server: http://localhost:{args_global.port}")

    app.run(host="0.0.0.0", port=args_global.port, debug=False)
