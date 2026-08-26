#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_all_tests.sh — CONTRACTS §9 top-level test aggregator.
#
# Runs EVERY tool's no-pytest test suite (Python via __main__ + Node .js)
# with the unified env python and node, prints a per-suite PASS/FAIL line
# plus a final tally, and exits NONZERO if ANY suite fails.
#
# Works from ANY CWD: the repo root is resolved from this script's own
# location (scripts/run_all_tests.sh -> repo root is its parent's parent).
#
#   <repo>/env/bin/python scripts/_reefreview/tests/test_reefreview.py   (and friends)
#   node                  scripts/_reefreview/tests/test_viewer_core.js  (and friends)
#
# Usage:  bash scripts/run_all_tests.sh   (or ./scripts/run_all_tests.sh)
# ─────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO="$(cd "$HERE/.." && pwd)"                          # github_repo root
SCRIPTS="$REPO/scripts"

PY="$REPO/env/bin/python"
if [ ! -x "$PY" ]; then
  echo "FATAL: unified env python not found at $PY" >&2
  exit 2
fi

# Tests must see a deterministic lock state, independent of the LIVE platform
# module-lock registry (with reef_point_seg locked for real, every orchestrator
# suite would 423 and the whole board goes red for reasons unrelated to the
# code under test). lock_status.py reads this env at call time, and
# test_lock_gate.py pops it before exercising the gate with its own registry,
# so the lock gate itself is still fully tested.
export VICARIUS_LOCK_BYPASS=1

# Fixture trees built by the suites must not leave persisted clip-index files
# under the real supporting_data/clip_index/ (Task 1.4): point the index store
# at a throwaway dir for the whole run and remove it on exit.
TMP_CLIP_INDEX="$(mktemp -d /tmp/clip_index_test.XXXXXX)"
export TCRMP_CLIP_INDEX_DIR="$TMP_CLIP_INDEX"
trap 'rm -rf "$TMP_CLIP_INDEX"' EXIT
if ! command -v node >/dev/null 2>&1; then
  echo "FATAL: node not on PATH (required for the .js suites)" >&2
  exit 2
fi

# Every suite: "<runner> <path-relative-to-scripts>". py -> env python, js -> node.
# Mirrors CONTRACTS §9: each tool's tests/ harness, run with the unified env.
SUITES=(
  "py _reefreview/tests/test_reefreview.py"
  "py _reefreview/tests/test_review_schema.py"
  "py _reefreview/tests/test_project_identity.py"
  "py _reefreview/tests/test_coverage.py"
  "js _reefreview/tests/test_viewer_core.js"
  "js _reefreview/tests/test_viewer_dom.js"
  "py TCRMPcvr_chooseImages/tests/test_selection_topup.py"
  # Season-agnostic image resolver + informative outputs (Step 3).
  "py TCRMPcvr_chooseImages/tests/test_image_resolver.py"
  "py TCRMPcvr_chooseImages/tests/test_outputs_informative.py"
  # Step-3 eligibility, random selection, reserve, and AT_test integration.
  "py TCRMPcvr_chooseImages/tests/test_eligibility_gate.py"
  "py TCRMPcvr_chooseImages/tests/test_random_selection.py"
  "py TCRMPcvr_chooseImages/tests/test_reserve_list.py"
  "py TCRMPcvr_chooseImages/tests/test_at_test_integration.py"
  # Site-balanced selection (2026-07-09): every site included, proportional, no site > 25%.
  "py TCRMPcvr_chooseImages/tests/test_site_balance.py"
  "js TCRMPclip_placePoints/tests/test_pp_core.js"
  "py TCRMPclip_placePoints/tests/verify_apps.py"
  # Step-4 routing trusts Step 3 paths + recursive clip-index fallback.
  "py TCRMPclip_placePoints/tests/test_routing_resolve.py"
  # EasyOCR must construct offline-first (lab DNS drops out intermittently).
  "py TCRMPclip_placePoints/tests/test_ocr_offline_load.py"
  "py TCRMPclip_segmentImages/tests/test_segment_app.py"
  # Combined annotator (4.test) — now the promoted Step 4, so its suites run here.
  "py TCRMPclip_combinedAnnotate/tests/test_segment_app.py"
  # SAM3 must boot from the local HF cache when lab DNS is down (2026-08-13).
  "py TCRMPclip_combinedAnnotate/tests/test_sam_offline_load.py"
  # Cached SAM3 tracker backfill (2026-08-26): second boot skips the CPU
  # video-model load; TCRMP_SAM3_REAL_PARITY=1 adds the real-checkpoint check.
  "py TCRMPclip_combinedAnnotate/tests/test_sam_tracker_cache.py"
  "py TCRMPclip_combinedAnnotate/tests/test_draw_clip.py"
  "py TCRMPclip_combinedAnnotate/tests/test_batch_flow.py"
  "py TCRMPclip_combinedAnnotate/tests/test_provenance.py"
  "py TCRMPclip_combinedAnnotate/tests/test_batch_all.py"
  # Export, manifest, exemplar, commit-mask, delete-remask, export-completeness.
  "py TCRMPclip_combinedAnnotate/tests/test_export_negatives.py"
  "py TCRMPclip_combinedAnnotate/tests/test_export_manifest.py"
  "py TCRMPclip_combinedAnnotate/tests/test_export_previews.py"
  "py TCRMPclip_combinedAnnotate/tests/test_exemplar_no_persist.py"
  "py TCRMPclip_combinedAnnotate/tests/test_commit_mask.py"
  "py TCRMPclip_combinedAnnotate/tests/test_delete_remask.py"
  "py TCRMPclip_combinedAnnotate/tests/test_add_after_delete.py"
  "py TCRMPclip_combinedAnnotate/tests/test_clip_visibility.py"
  "py TCRMPclip_combinedAnnotate/tests/test_clean_mask_kernel.py"
  "py TCRMPclip_combinedAnnotate/tests/test_at_test_export_complete.py"
  # Host-open buttons (export folder / manifest via xdg-open).
  "py TCRMPclip_combinedAnnotate/tests/test_open_routes.py"
  # Custom image imports (2026-08-26): bring-your-own images into the Step-4
  # annotator, lores rule + project.json recording (T20260815-203910 part 2).
  "py TCRMPclip_combinedAnnotate/tests/test_custom_imports.py"
  # Step-4 reserve refill.
  "py TCRMPclip_placePoints/tests/test_reserve_refill.py"
  "py TCRMPclip_placePoints/tests/test_lores_scaling.py"
  "py pipeline_orchestrator/tests/test_orchestrator_ui.py"
  "py pipeline_orchestrator/tests/test_step4test_clipdir.py"
  "py pipeline_orchestrator/tests/test_step6_source.py"
  # Species filter passed to routing thread (C2 fix: no hardcoded []).
  "py pipeline_orchestrator/tests/test_route_species_filter.py"
  # Staleness-based routed_input/ invalidation (C3 fix: stale cache cleared when Step 3 selection newer).
  "py pipeline_orchestrator/tests/test_route_clears_stale.py"
  # A routing pass in flight must not be reported to the UI as a ready annotator
  # on a stale port, and must not launch into a project it never routed.
  "py pipeline_orchestrator/tests/test_step4test_route_inflight.py"
  # Cached ui_ready/review_ready must never outlive the process (2026-08-13).
  "py pipeline_orchestrator/tests/test_status_liveness.py"
  # Closing the launch window cancels the routing pass: no GPU held by work nobody
  # is watching, and no annotator killed+relaunched by an abandoned pass.
  "py pipeline_orchestrator/tests/test_step4test_route_cancel.py"
  "py pipeline_orchestrator/tests/test_project_naming.py"
  # routed_input/<year> flattened to flat routed_input/{ids,raw} after routing.
  "py pipeline_orchestrator/tests/test_flatten_routed_input.py"
  # Module-lock gate (2026-08-16): lock page on /, 423 on APIs, fail-open registry.
  "py pipeline_orchestrator/tests/test_lock_gate.py"
  "py _expertids/tests/test_expertids.py"
  "py _expertids/tests/test_step4test_relabel.py"
  # Dual-pattern (flat + legacy year-nested) segmentation readers.
  "py _expertids/tests/test_flat_segmentations.py"
  # End-to-end expert-review pathway (export -> viewer CSV -> consensus -> accept -> matrix -> remove_project).
  "py _expertids/tests/test_e2e_expert_review.py"
  # Coverage matrix (found_expert/pending_expert outcome vocabulary, CONTRACTS section 13).
  "py _matrix/tests/test_matrix.py"
  # Lores generator + manifest (A1: 1920px mirror of oversized 4K clip frames).
  "py tests/test_make_lores.py"
  # Persisted clip index (2026-08-26): hit/stale paths + consumer listing parity.
  "py tests/test_clip_index.py"
  # Model-assist loop (2026-07-09): inference predictions.json, prediction seeder, transect frozen split.
  "py TCRMPtrain_oceankindCV/tests/test_predictions_output.py"
  "py TCRMPclip_combinedAnnotate/tests/test_seed_from_predictions.py"
  "py TCRMPtrain_oceankindCV/tests/test_pinned_split.py"
  # Model-assist loop wave 2: annotator resume mode, train --freeze, evaluate rounds ledger.
  "py TCRMPclip_combinedAnnotate/tests/test_loop_resume.py"
  "py TCRMPtrain_oceankindCV/tests/test_train_freeze.py"
  "py TCRMPtrain_oceankindCV/tests/test_rounds_ledger.py"
  # Model-assist loop wave 3: orchestrator step4loop routes, step6 fine-tune, step7 rounds/promote.
  "py pipeline_orchestrator/tests/test_step4loop_routes.py"
  "py pipeline_orchestrator/tests/test_step6_finetune.py"
  "py pipeline_orchestrator/tests/test_rounds_routes.py"
  # Canonical mask registry (2026-07-10): per-mask store, export upsert hook, matrix derivation.
  "py _reefreview/tests/test_mask_registry.py"
  "py TCRMPclip_combinedAnnotate/tests/test_export_registry_upsert.py"
  "py _matrix/tests/test_matrix_registry.py"
  # Label guard (2026-07-10): block empty-species accept/export + unlabeled filter.
  "py TCRMPclip_combinedAnnotate/tests/test_label_guard.py"
  "py TCRMPclip_combinedAnnotate/tests/test_unlabeled_filter.py"
  # Edit Masks app (2026-07-10): standalone editor - boot, expert-lock, registry-on-edit; orchestrator tile.
  "py TCRMPclip_editMasks/tests/test_edit_boot.py"
  "py TCRMPclip_editMasks/tests/test_expert_lock.py"
  "py TCRMPclip_editMasks/tests/test_edit_registry.py"
  "py pipeline_orchestrator/tests/test_editmasks_routes.py"
  "py _matrix/tests/test_registry_matrix_e2e.py"
)

PASS=0
FAIL=0
FAILED_SUITES=()

echo "============================================================"
echo " run_all_tests.sh  —  repo: $REPO"
echo "============================================================"

for entry in "${SUITES[@]}"; do
  kind="${entry%% *}"
  rel="${entry#* }"
  path="$SCRIPTS/$rel"

  if [ ! -f "$path" ]; then
    echo "FAIL  $rel  (missing test file)"
    FAIL=$((FAIL + 1))
    FAILED_SUITES+=("$rel (missing)")
    continue
  fi

  case "$kind" in
    py) "$PY" "$path" >/tmp/run_all_tests.$$.log 2>&1 ;;
    js) node    "$path" >/tmp/run_all_tests.$$.log 2>&1 ;;
    *)  echo "FAIL  $rel  (unknown runner '$kind')"; FAIL=$((FAIL + 1));
        FAILED_SUITES+=("$rel (bad runner)"); continue ;;
  esac
  rc=$?

  if [ "$rc" -eq 0 ]; then
    echo "PASS  $rel"
    PASS=$((PASS + 1))
  else
    echo "FAIL  $rel  (exit $rc)"
    FAIL=$((FAIL + 1))
    FAILED_SUITES+=("$rel")
    # Surface the tail of the failing suite's output so failures are diagnosable.
    echo "----- last 25 lines of $rel -----"
    tail -n 25 /tmp/run_all_tests.$$.log | sed 's/^/    /'
    echo "---------------------------------"
  fi
  rm -f /tmp/run_all_tests.$$.log
done

TOTAL=$((PASS + FAIL))
echo "============================================================"
echo " SUITE TALLY:  $PASS/$TOTAL passed, $FAIL failed"
if [ "$FAIL" -ne 0 ]; then
  echo " FAILED SUITES:"
  for s in "${FAILED_SUITES[@]}"; do echo "   - $s"; done
  echo "============================================================"
  exit 1
fi
echo " ALL SUITES GREEN"
echo "============================================================"
exit 0
