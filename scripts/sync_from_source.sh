#!/usr/bin/env bash
# Pull updates from the source repo (/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI)
# into the new seg_AI_img_full_april2026 layout — never the other way around.
#
# Use this when agents/collaborators continue working in the source repo and
# their edits (new scripts, updated step code, new supporting-data artifacts)
# need to land in this pared-down copy.
#
# Usage:
#   ./sync_from_source.sh                    # dry-run (default — prints what would change)
#   ./sync_from_source.sh --apply            # sync everything (scripts + docs + supporting data)
#   ./sync_from_source.sh --apply step7      # sync just step-7 (evaluate_model) code
#   ./sync_from_source.sh --apply step5      # sync just step-5 (SAM3 segmentImages) code
#   ./sync_from_source.sh --apply scripts    # sync all scripts/ (all sub-tools + orchestrator)
#   ./sync_from_source.sh --apply data       # sync supporting_data/ deltas
#   ./sync_from_source.sh --apply projects   # sync projects/ deltas
#
# Never moves files in the source repo. Only reads + copies to this location.
set -euo pipefail
cd "$(dirname "$0")/.."
DST="$(pwd)"
# Source tree default is this box's hopper copy; override with TCRMP_SYNC_SRC.
SRC="${TCRMP_SYNC_SRC:-/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI}"

DRY="--dry-run"
TARGET="all"
for arg in "$@"; do
  case "$arg" in
    --apply) DRY="" ;;
    step1|step2|step3|step4|step5|step6|step7|step8|scripts|data|projects|all) TARGET="$arg" ;;
    *)       echo "Unknown argument: $arg"; exit 2 ;;
  esac
done

if [ -n "$DRY" ]; then
  echo "[DRY-RUN] (use --apply to commit). Target: $TARGET"
else
  echo "[APPLY]   Target: $TARGET"
fi

RSYNC_BASE="rsync -a $DRY --info=progress2"
# Skip ephemeral / rebuilt / misplaced things. Root-level yolo*.pt weights belong
# in supporting_data/model_weights/, not inside TCRMPtrain_oceankindCV/.
COMMON_EX="--exclude=env/ --exclude=__pycache__/ --exclude=*.pyc --exclude=output/ --exclude=output_*/ --exclude=.DS_Store --exclude=.git/ --exclude=/yolo*.pt"

sync_scripts() {
  # pipeline_orchestrator (scripts/patches live here; prefer not to overwrite patches blindly)
  # Only sync non-patched files: we DO NOT pull app.py, orchestrator_config.py, or run_pipeline.sh
  # because we've customised them. Everything else (project_manager.py, stage_runner.py, remap_loader.py,
  # static/, templates/) is safe.
  echo "  scripts/pipeline_orchestrator/  (skipping app.py, orchestrator_config.py, run_pipeline.sh)"
  $RSYNC_BASE $COMMON_EX \
    --exclude=presets/ --exclude=eval_presets/ \
    --exclude=app.py --exclude=orchestrator_config.py --exclude=run_pipeline.sh \
    "$SRC/pipeline_orchestrator/" "$DST/scripts/pipeline_orchestrator/"

  # Sub-tools (safe wholesale; we did not patch them)
  for d in TCRMPcvr_makeAllPoints TCRMPcvr_recodeSpecies TCRMPcvr_chooseImages \
           TCRMPclip_placePoints TCRMPclip_segmentImages \
           TCRMPclip_cpcID TCRMPclip_ocrID_batch \
           TCRMPtrain_oceankindCV; do
    if [ -d "$SRC/$d" ]; then
      echo "  scripts/$d/"
      $RSYNC_BASE $COMMON_EX "$SRC/$d/" "$DST/scripts/$d/"
    fi
  done

  # config/presets + config/eval_presets (live under pipeline_orchestrator/ in the source)
  echo "  config/presets/ and config/eval_presets/"
  $RSYNC_BASE "$SRC/pipeline_orchestrator/presets/"      "$DST/config/presets/"
  $RSYNC_BASE "$SRC/pipeline_orchestrator/eval_presets/" "$DST/config/eval_presets/"

  # top-level docs
  for f in PIPELINE_OVERVIEW.md NOTES_pipeline_improvements.md; do
    [ -f "$SRC/$f" ] && $RSYNC_BASE "$SRC/$f" "$DST/docs/$f"
  done
}

sync_one_step() {
  local step="$1"
  # Step -> source sub-dir mapping
  declare -A MAP=(
    [step1]="TCRMPcvr_makeAllPoints"
    [step2]="TCRMPcvr_recodeSpecies"
    [step3]="TCRMPcvr_chooseImages"
    [step4]="TCRMPclip_placePoints"
    [step5]="TCRMPclip_segmentImages"
    [step6]="TCRMPtrain_oceankindCV"
    [step7]="TCRMPtrain_oceankindCV"
    [step8]="TCRMPtrain_oceankindCV"
  )
  local sub="${MAP[$step]:-}"
  [ -z "$sub" ] && { echo "Unknown step $step"; exit 2; }
  echo "  scripts/$sub/  (for $step)"
  $RSYNC_BASE $COMMON_EX "$SRC/$sub/" "$DST/scripts/$sub/"
}

sync_data() {
  # Small static supporting-data assets (catches re-run step 1 outputs and added weights)
  echo "  supporting_data/  (diff-sync only)"
  [ -f "$SRC/output/all_points.csv" ]   && $RSYNC_BASE "$SRC/output/all_points.csv"   "$DST/supporting_data/"
  [ -f "$SRC/output/master_codes.csv" ] && $RSYNC_BASE "$SRC/output/master_codes.csv" "$DST/supporting_data/"
  # cpc_all delta (rare — only if cpcID was re-run)
  [ -d "$SRC/output/cpc_all" ]          && $RSYNC_BASE "$SRC/output/cpc_all/"         "$DST/supporting_data/cpc_all/"
  # TCRMP_CVR deltas (rare — only if new year added)
  [ -d "$SRC/input/TCRMP_CVR" ]         && $RSYNC_BASE "$SRC/input/TCRMP_CVR/"        "$DST/supporting_data/TCRMP_CVR/"
  # TCRMP_clip local subset
  [ -d "$SRC/input/TCRMP_clip" ]        && $RSYNC_BASE "$SRC/input/TCRMP_clip/"       "$DST/supporting_data/TCRMP_clip/"
}

sync_projects() {
  echo "  projects/  (diff-sync only — preserves local path rewrites via rsync deltas)"
  rsync -aHAX $DRY --info=progress2 "$SRC/projects/" "$DST/projects/"
  if [ -z "$DRY" ]; then
    echo "  rerunning project-path rewriter on newly-synced content..."
    "$DST/env/bin/python" "$DST/scripts/rewrite_project_paths.py" \
      --old "$SRC" --new "$DST" --apply || true
  fi
}

case "$TARGET" in
  all)       sync_scripts; sync_data; sync_projects ;;
  scripts)   sync_scripts ;;
  data)      sync_data ;;
  projects)  sync_projects ;;
  step*)     sync_one_step "$TARGET" ;;
esac

echo ""
if [ -n "$DRY" ]; then
  echo "dry-run complete. Re-run with --apply to commit."
else
  echo "sync complete."
fi
