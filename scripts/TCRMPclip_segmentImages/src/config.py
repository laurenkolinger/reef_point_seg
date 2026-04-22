"""
Configuration for TCRMPclip_segmentImages.

Edit paths and thresholds here. Settings are logged to output/config_log.json
on each run so you know what was used.
"""

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)                # scripts/TCRMPclip_segmentImages
_SCRIPTS_DIR = os.path.dirname(_PROJECT_DIR)               # scripts/
_REPO_DIR = os.path.dirname(_SCRIPTS_DIR)                  # seg_AI_img_full_april2026

# ── Target species filter ────────────────────────────────────────────
# When 1, only segment/review points for target species
# (reads species list from TCRMPcvr_chooseImages config)
TARGET_SPECIES_ONLY = 1

# ── Input (from routeChosenImages output) ───────────────────────────
# Standalone default is the sibling sub-tool's output/ dir. The orchestrator
# overrides this via TCRMP_INPUT_DIR to point at the active project's
# step4_routeChosenImages/ folder.
INPUT_DIR = os.path.join(
    _SCRIPTS_DIR, 'TCRMPclip_routeChosenImages', 'output')

# ── SAM3 Model (HuggingFace Transformers: facebook/sam3) ────────────

SAM3_DEVICE_TRACKER = "cuda:1"         # Sam3TrackerModel (point/box prompts)
SAM3_DEVICE_EXEMPLAR = "cuda:0"        # Sam3Model (exemplar/visual scan)
SAM3_MASK_SIZE = "generous"            # tight | medium | generous — halves VRAM usage

# ── Segmentation Thresholds ─────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.5             # medium sensitivity for point segmentation
EXEMPLAR_THRESHOLD = 0.5              # default threshold for exemplar scan
MIN_MASK_AREA_PX = 500                 # discard masks smaller than this (pixels)
MERGE_DISTANCE_PX = 30                 # merge same-species masks within this distance
OVERLAP_STRATEGY = "larger_wins"       # larger_wins | higher_score | first_wins
THIN_MASK_RATIO = 0.10                 # discard sliver masks (area/bbox_area < this)
POLYGON_SIMPLIFY_EPSILON = 0.001       # approxPolyDP epsilon as fraction of perimeter

# ── Default Categories ──────────────────────────────────────────────
# These are pre-selected in the UI. User can change per run.

DEFAULT_CATEGORIES = [
    "Target species only",
]

# ── Processing ──────────────────────────────────────────────────────

FLUSH_INTERVAL = 20                    # save state every N images
REVIEW_BATCH_SIZE = 10                 # images per review batch

# ── Export ──────────────────────────────────────────────────────────

EXPORT_DIR = os.path.join(_PROJECT_DIR, 'output')
SYMLINK_IMAGES = True                  # symlink raw images (saves disk space)

# ── Server ──────────────────────────────────────────────────────────

PORT = 5070

# Pipeline orchestrator overrides (no effect when env vars are unset/empty)
INPUT_DIR = os.environ.get('TCRMP_INPUT_DIR', '') or INPUT_DIR
EXPORT_DIR = os.environ.get('TCRMP_EXPORT_DIR', '') or EXPORT_DIR
SAM3_DEVICE_TRACKER = os.environ.get('TCRMP_SAM3_DEVICE_TRACKER', '') or SAM3_DEVICE_TRACKER
SAM3_DEVICE_EXEMPLAR = os.environ.get('TCRMP_SAM3_DEVICE_EXEMPLAR', '') or SAM3_DEVICE_EXEMPLAR
_ct = os.environ.get('TCRMP_CONFIDENCE_THRESHOLD', '')
if _ct: CONFIDENCE_THRESHOLD = float(_ct)
_ma = os.environ.get('TCRMP_MIN_MASK_AREA_PX', '')
if _ma: MIN_MASK_AREA_PX = int(_ma)
_md = os.environ.get('TCRMP_MERGE_DISTANCE_PX', '')
if _md: MERGE_DISTANCE_PX = int(_md)
OVERLAP_STRATEGY = os.environ.get('TCRMP_OVERLAP_STRATEGY', '') or OVERLAP_STRATEGY
