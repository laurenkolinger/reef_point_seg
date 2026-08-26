"""
Configuration for TCRMPclip_placePoints.

Edit paths here and rerun. All settings are logged to output/config_log.json
on each run so you know what was used.
"""

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)                # scripts/TCRMPclip_placePoints
_SCRIPTS_DIR = os.path.dirname(_PROJECT_DIR)               # scripts/
_REPO_DIR = os.path.dirname(_SCRIPTS_DIR)                  # seg_AI_img_full_april2026
_SUPPORTING = os.path.join(_REPO_DIR, 'supporting_data')

# ── Target species filter ────────────────────────────────────────────
# When 1, only process/review frames containing target species
# (reads species list from TCRMPcvr_chooseImages config)
TARGET_SPECIES_ONLY = 1

# ── Input paths ──────────────────────────────────────────────────────

# Selected frames CSV from TCRMPcvr_chooseImages. Standalone default is the
# sibling sub-tool's output/ dir; orchestrator overrides via TCRMP_SELECTED_FRAMES.
SELECTED_FRAMES = os.path.join(
    _SCRIPTS_DIR, 'TCRMPcvr_chooseImages', 'output', 'selected_frames.csv')

# TCRMP_clip root (READ-ONLY source images). Reads from the local
# supporting_data/TCRMP_clip tree (the full year/period set). The orchestrator
# may override via the TCRMP_CLIP_DIR env var.
CLIP_DIR = os.path.join(_SUPPORTING, 'TCRMP_clip')

# CPC point coords (pre-2020 frames) — static index in supporting_data.
CPC_DIR = os.path.join(_SUPPORTING, 'cpc_all')

# Recoded all_points.csv (auto-detects latest from recodeSpecies)
# Set to a specific path to override auto-detection
ALL_POINTS_OVERRIDE = None  # e.g., '/path/to/all_points.csv'

# ── Output ───────────────────────────────────────────────────────────

EXPORT_DIR = os.path.join(_PROJECT_DIR, 'output')

# ── Server ───────────────────────────────────────────────────────────

PORT = 5065

# Pipeline orchestrator overrides (no effect when env vars are unset/empty)
SELECTED_FRAMES = os.environ.get('TCRMP_SELECTED_FRAMES', '') or SELECTED_FRAMES
EXPORT_DIR = os.environ.get('TCRMP_EXPORT_DIR', '') or EXPORT_DIR
ALL_POINTS_OVERRIDE = os.environ.get('TCRMP_ALL_POINTS', '') or ALL_POINTS_OVERRIDE
CLIP_DIR = os.environ.get('TCRMP_CLIP_DIR', '') or CLIP_DIR
CPC_DIR = os.environ.get('TCRMP_CPC_DIR', '') or CPC_DIR
