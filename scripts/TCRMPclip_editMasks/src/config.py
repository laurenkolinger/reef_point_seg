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

# ── Input (from Place Points output) ───────────────────────────
# Standalone default is the sibling sub-tool's output/ dir. The orchestrator
# overrides this via TCRMP_INPUT_DIR to point at the active project's
# step4_routeChosenImages/ folder.
INPUT_DIR = os.path.join(
    _SCRIPTS_DIR, 'TCRMPclip_placePoints', 'output')

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

PORT = 5085

# ── Combined-annotator mode (step 4.test) ───────────────────────────
# When 1, skip the auto-segment-all pass. Prompt points are loaded as
# read-only reference; the operator drives masks by hand. Default 1
# because this fork is the combined annotator.
MANUAL_ANNOTATE = (os.environ.get('TCRMP_MANUAL_ANNOTATE', '1') == '1')

# Per-(image,label) provenance source tag written into label_provenance.csv
# and each seg_dict's label_outcomes block. This fork defaults to step4test.
PROVENANCE_SOURCE = os.environ.get('TCRMP_PROVENANCE_SOURCE', 'edit')

# Operator initials for provenance attribution (best-effort; may be empty).
REVIEWER = os.environ.get('TCRMP_REVIEWER', '')

# Session boot mode for orchestrated launches. 'configure' (default) drives
# the normal Step-4-routed POST /api/configure boot. 'resume' is the active-
# learning loop mode: the app boots straight into the review queue via
# POST /api/resume, skipping configure/routing entirely (the export_dir
# already carries a seeded segmentations.json with pending model masks).
SESSION_MODE = os.environ.get('TCRMP_SESSION_MODE', 'edit').strip().lower()

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

# ── Combined Step-4+5 settings overrides (set by the orchestrator's step4test
# panel; honored by configure() when the orchestrated /api/configure is called
# with an empty body, since defaults must then come from env). Each falls back
# to the inherited Step-5 saved value already baked into the constants above. ──

# Target-labels-only filter. "1" (default) keeps only target-species masks/
# points; "0" keeps all categories. Read as an explicit flag by configure()
# instead of inferring it from the Categories string list.
_tso = os.environ.get('TCRMP_TARGET_SPECIES_ONLY', '')
if _tso != '':
    TARGET_SPECIES_ONLY = 1 if _tso == '1' else 0

# Review batch size (frames per export batch).
_rbs = os.environ.get('TCRMP_REVIEW_BATCH_SIZE', '')
if _rbs:
    if str(_rbs).strip().lower() == 'all':
        # "All (no batching)" — one batch holding every routed frame.
        REVIEW_BATCH_SIZE = 10 ** 9
    else:
        try:
            REVIEW_BATCH_SIZE = int(_rbs)
        except ValueError:
            pass

# Symlink-vs-copy of exported images (was a hardcoded constant; now env-exposed
# so the 4.test panel's "Symlink exported images" control actually takes effect).
_sym = os.environ.get('TCRMP_SYMLINK_IMAGES', '')
if _sym != '':
    SYMLINK_IMAGES = _sym == '1'

# SAM3 advanced overrides not previously env-exposed.
SAM3_MASK_SIZE = os.environ.get('TCRMP_SAM3_MASK_SIZE', '') or SAM3_MASK_SIZE
_tmr = os.environ.get('TCRMP_THIN_MASK_RATIO', '')
if _tmr:
    try:
        THIN_MASK_RATIO = float(_tmr)
    except ValueError:
        pass
_pse = os.environ.get('TCRMP_POLYGON_SIMPLIFY_EPSILON', '')
if _pse:
    try:
        POLYGON_SIMPLIFY_EPSILON = float(_pse)
    except ValueError:
        pass
_et = os.environ.get('TCRMP_EXEMPLAR_THRESHOLD', '')
if _et:
    try:
        EXEMPLAR_THRESHOLD = float(_et)
    except ValueError:
        pass

# Page initial state for the read-only reference markers (the "Show existing
# points read-only" toggle / G key). "1" (default) shows them on load; they are
# guidance only and never become masks or labels. The template reads this to
# seed the JS `showReference` default before localStorage prefs apply.
REFERENCE_DEFAULT = 0 if os.environ.get('TCRMP_REFERENCE_DEFAULT', '') == '0' else 1

# ── Expert-review export (REVIEW-flagged masks) ──────────────────────
# REVIEW masks are segmented like targets but never trained; at each batch
# export they are given a UID, rendered, and pushed to the expert-review
# GitHub-Pages repo. A permanent cross-project library (gitignored, in the
# module) records every reviewed UID + its eventual expert ID.

# Canonical project identity (set by the orchestrator from project.json).
# Empty when step 5 is launched standalone -> the app falls back to deriving
# the id from the export dir and uses the id as the display name.
PROJECT_ID = os.environ.get('TCRMP_PROJECT_ID', '')
PROJECT_NAME = os.environ.get('TCRMP_PROJECT_NAME', '')

# Pages repo working tree (a git clone of laurenkolinger/reefpointseg-review).
REVIEW_DIR = os.environ.get('TCRMP_REVIEW_DIR', '') or '/mnt/tear/REVIEW_reefpointseg'
REVIEW_REPO_URL = (os.environ.get('TCRMP_REVIEW_REPO_URL', '')
                   or 'https://github.com/laurenkolinger/reefpointseg-review.git')
# Permanent library dir; '' -> _reefreview.library.default_dir()
# (<module>/inprocess/_expert_id_library).
EXPERT_LIBRARY_DIR = os.environ.get('TCRMP_EXPERT_LIBRARY_DIR', '')
# Auto git-push the review repo after each export / import.
REVIEW_GIT_PUSH = (os.environ.get('TCRMP_REVIEW_GIT_PUSH', '1') == '1')
# Closeup crop padding + downscale caps (keep the Pages repo small).
REVIEW_CROP_PAD_PX = int(os.environ.get('TCRMP_REVIEW_CROP_PAD_PX', '') or 40)
REVIEW_MAX_EDGE = int(os.environ.get('TCRMP_REVIEW_MAX_EDGE', '') or 720)
REVIEW_FULL_MAX_EDGE = int(os.environ.get('TCRMP_REVIEW_FULL_MAX_EDGE', '') or 1400)
# >50% auto-relabel threshold (intersection / area-of-new-mask).
REVIEW_OVERLAP_THRESH = float(os.environ.get('TCRMP_REVIEW_OVERLAP_THRESH', '') or 0.5)
# Default reviewer recipients (operator adds their own + their student's).
REVIEW_CONTACTS = (os.environ.get('TCRMP_REVIEW_CONTACTS', '')
                   or 'lauren.olinger@uvi.edu')
# Canonical species-code dictionary for the viewer's codes.json.
#
# Resolution order (first existing file wins, except env which always wins):
#   1. env TCRMP_MASTER_CODES — the orchestrator already passes the correct
#      (recoded) dictionary here, so it must win unconditionally.
#   2. The step-2 RECODED dictionary for THIS project. A standalone step-5
#      launch into an existing project (INPUT_DIR/EXPORT_DIR point inside
#      inprocess/<project>/step{4,5}_...) should prefer the post-recode codes
#      over the canonical defaults. step2_recodeSpecies/ is a sibling of the
#      step-4/step-5 folders, so the project root is the step dir's parent.
#   3. A repo-level step2_recodeSpecies/master_codes_recoded.csv, if present.
#   4. The canonical supporting_data/master_codes.csv (last resort).

def _recoded_candidates():
    """Yield possible recoded-dictionary paths, most-specific first."""
    seen = set()
    # Project roots derived from the (already env-resolved) step dirs.
    for step_dir in (INPUT_DIR, EXPORT_DIR):
        if not step_dir:
            continue
        # EXPORT_DIR may be <project>/step5_.../output, so walk up to the
        # project root by looking for a step2_recodeSpecies sibling at the
        # step-dir parent and at its grandparent.
        for root in (os.path.dirname(step_dir),
                     os.path.dirname(os.path.dirname(step_dir))):
            cand = os.path.join(root, 'step2_recodeSpecies',
                                'master_codes_recoded.csv')
            if cand not in seen:
                seen.add(cand)
                yield cand
    # Repo-level step2 recoded dictionary (mirrors the contract's example path).
    repo_cand = os.path.join(_REPO_DIR, 'step2_recodeSpecies',
                             'master_codes_recoded.csv')
    if repo_cand not in seen:
        yield repo_cand


def _resolve_master_codes_csv():
    env_override = os.environ.get('TCRMP_MASTER_CODES', '')
    if env_override:
        return env_override
    for cand in _recoded_candidates():
        if cand and os.path.isfile(cand):
            return cand
    return os.path.join(_REPO_DIR, 'supporting_data', 'master_codes.csv')


MASTER_CODES_CSV = _resolve_master_codes_csv()
