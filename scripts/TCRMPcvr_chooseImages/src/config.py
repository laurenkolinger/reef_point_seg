"""
Configuration for TCRMPcvr_chooseImages.

Edit this file to change target species, thresholds, and paths.
Then rerun:  python src/select_images.py
             python src/plot_diagnostics.py
"""

# ── Target species (edit this list and rerun) ────────────────────────────────
TARGET_SPECIES = ["PA", "OFAV"]
# TARGET_SPECIES = ["OFRA", "PA", "OA", "OFAV", "AL", "MC", "AA"]

# ── Selection parameters ─────────────────────────────────────────────────────
TARGET_INSTANCES_PER_SPECIES = 100    # frame-instances (1 frame = 1 instance) default 1000
# MIN_YEAR / MAX_YEAR are fallback defaults only; the orchestrator passes the
# actual bounds via --min-year / --max-year.
MIN_YEAR = 2014
MAX_YEAR = 2025
# No category filter. Callers pass a `target_labels` list (via --species) and
# the script filters by label membership alone — any category is selectable.

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolved against the seg_AI_img_full_april2026 repo layout (scripts/, config/,
# supporting_data/, projects/). When the orchestrator launches this step it
# overrides OUTPUT_DIR via the TCRMP_OUTPUT_DIR env var.
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)                # scripts/TCRMPcvr_chooseImages
_SCRIPTS_DIR = os.path.dirname(_PROJECT_DIR)               # scripts/
_REPO_DIR = os.path.dirname(_SCRIPTS_DIR)                  # seg_AI_img_full_april2026
_SUPPORTING = os.path.join(_REPO_DIR, "supporting_data")

# Input: unified point-count dataset. Falls back to the recoded variant if
# TCRMPcvr_recodeSpecies/output/ contains a newer file (auto-detected elsewhere).
_RECODE_DIR = os.path.join(_SCRIPTS_DIR, "TCRMPcvr_recodeSpecies", "output")
DEFAULT_ALL_POINTS = os.path.join(_SUPPORTING, "all_points.csv")
DEFAULT_MASTER_CODES = os.path.join(_SUPPORTING, "master_codes.csv")

# Source images — the local full year/period clip set under supporting_data.
CLIP_DIR = os.path.join(_SUPPORTING, "TCRMP_clip")

# Pre-existing processed outputs from the legacy CPC/OCR routers.
CPC_ALL_DIR = os.path.join(_SUPPORTING, "cpc_all")
OCR_ALL_DIR = os.path.join(_SUPPORTING, "ocr_all")

# Output lives per-run inside this sub-tool's dir when run standalone; the
# orchestrator overrides it via TCRMP_OUTPUT_DIR.
OUTPUT_DIR = os.path.join(_PROJECT_DIR, "output")
OUTPUT_DIR = os.environ.get('TCRMP_OUTPUT_DIR', '') or OUTPUT_DIR
