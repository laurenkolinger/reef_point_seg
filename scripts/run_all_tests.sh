#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_all_tests.sh: CONTRACTS §9 top-level test aggregator.
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
# Never pop host file-manager windows from a test fixture (host_open.py honors this).
export VICARIUS_HOST_OPEN=0

# Run-start marker (Task 13a): anchors the exit-time fixture sweep below so
# only /tmp dirs created during THIS run are ever removed. A file, not a
# dir, so it can never collide with the -type d sweep it feeds.
RUN_START_STAMP="$(mktemp /tmp/run_all_tests_start.XXXXXX)"

# Fixture trees built by the suites must not leave persisted clip-index files
# under the real supporting_data/clip_index/ (Task 1.4): point the index store
# at a throwaway dir for the whole run and remove it on exit.
TMP_CLIP_INDEX="$(mktemp -d /tmp/clip_index_test.XXXXXX)"
export TCRMP_CLIP_INDEX_DIR="$TMP_CLIP_INDEX"
# Same rule for the cross-project mask registry (D7, 2026-08-26): annotator
# and export suites upsert rows keyed by fixture project ids, which must never
# reach the real inprocess/_mask_registry/manifest.csv. One throwaway registry
# root for the whole run, removed with the clip index on exit.
TMP_MASK_REGISTRY="$(mktemp -d /tmp/mask_registry_test.XXXXXX)"
export TCRMP_MASK_REGISTRY_DIR="$TMP_MASK_REGISTRY"

# Guard (Task 13a): several suites only redirect away from the live registry
# when this var is set (see task-13a-report.md for the current offender
# list). An empty value here would mean every one of them silently writes
# into the real inprocess/_mask_registry/manifest.csv, so refuse to run
# rather than risk it.
if [ -z "$TCRMP_MASK_REGISTRY_DIR" ]; then
  echo "FATAL: TCRMP_MASK_REGISTRY_DIR did not get set, refusing to run" >&2
  exit 2
fi

# Fixture-dir prefixes suites can leave behind in /tmp when a
# tempfile.mkdtemp() has no matching shutil.rmtree(), or the suite ran
# outside this script (Task 13a). Extend this list if a new suite mints a
# new prefix. Only dirs newer than RUN_START_STAMP are ever swept, so a
# pre-existing leak from an earlier run is never touched by this trap (see
# the one-off cleanup recorded in task-13a-report.md).
LEAKED_FIXTURE_PREFIXES=(
  combined_smoke_ edit_boot_test_ exp_neg_ exp_complete_ flatten_
  host_open_guard_
  label_guard_ lock_gate_ loop_resume_test_ lores_ man_ open_test_
  pp_refill_ proj_naming_ recon_ routed_ stale_ unlabeled_
  pullback_test_guard_ server_api_test_guard_ tcrmp_registry_test_guard_
)

_sweep_leaked_fixture_dirs() {
  local name_args=()
  for p in "${LEAKED_FIXTURE_PREFIXES[@]}"; do
    [ ${#name_args[@]} -gt 0 ] && name_args+=(-o)
    name_args+=(-name "${p}*")
  done
  find /tmp -maxdepth 1 -newer "$RUN_START_STAMP" -type d \( "${name_args[@]}" \) -exec rm -rf {} + 2>/dev/null
}

trap 'rm -rf "$TMP_CLIP_INDEX" "$TMP_MASK_REGISTRY"; _sweep_leaked_fixture_dirs; rm -f "$RUN_START_STAMP"' EXIT
if ! command -v node >/dev/null 2>&1; then
  echo "FATAL: node not on PATH (required for the .js suites)" >&2
  exit 2
fi

# Every suite: "<runner> <path-relative-to-scripts>". py -> env python, js -> node.
# Mirrors CONTRACTS §9: each tool's tests/ harness, run with the unified env.
SUITES=(
  # Suites that were on disk but absent from this array until 2026-09-16 (LO).
  # Every one passes; leaving them off the board hid seven working suites.
  "py TCRMPclip_placePoints/tests/test_export_all.py"
  "py TCRMPclip_placePoints/tests/test_routing_report.py"
  "py _labels/tests/test_labels_io.py"
  "py _labels/tests/test_labels_states.py"
  "py _labels/tests/test_labels_design.py"
  "py pipeline_orchestrator/tests/test_remap_loader_recode.py"
  "py tests/test_clip_index.py"
  "py tests/test_host_open_guard.py"
  "py tests/test_make_lores.py"
  "py TCRMPclip_combinedAnnotate/tests/test_annotator_copy.py"
  "py pipeline_orchestrator/tests/test_champion_scope.py"
  "py pipeline_orchestrator/tests/test_project_atomic_write.py"
  "py pipeline_orchestrator/tests/test_rail_family.py"
  "py pipeline_orchestrator/tests/test_reset_guard.py"
  "py pipeline_orchestrator/tests/test_run_attribution.py"
  "py pipeline_orchestrator/tests/test_seed_holdout_guard.py"
  "py pipeline_orchestrator/tests/test_source_family_guard.py"
  "py pipeline_orchestrator/tests/test_workflow_design_conformance.py"
  "py _ablation/tests/test_fish_ablation.py"
  "py TCRMPclip_combinedAnnotate/tests/test_annotator_design.py"
  "py TCRMPclip_combinedAnnotate/tests/test_annotator_layout_stability.py"
  # VEGA annotator tidy-up and no re-screening, 2026-09-27 (LO, task W1): the
  # gear settings popover and the cross-project prior-mask echo.
  "py TCRMPclip_combinedAnnotate/tests/test_annotator_settings_menu.py"
  "py TCRMPclip_combinedAnnotate/tests/test_prior_mask_render.py"
  "py _reefreview/tests/test_reefreview.py"
  "py _reefreview/tests/test_review_schema.py"
  "py _reefreview/tests/test_project_identity.py"
  "py _reefreview/tests/test_coverage.py"
  # EML metadata writer (2026-09-02, quality-sweep Wave 2): well-formed XML,
  # required fields, AST timestamp, no em/en dashes, rows summarized as counts-only.
  "py _reefreview/tests/test_eml_writer.py"
  # Shared training-project minting helper (2026-09-02, quality-sweep Tasks
  # H/N2): composite and bbox source contracts, run-dir naming, symlink-not-
  # copy rule, collision suffix, bad-initials/unknown-source guards.
  "py _reefreview/tests/test_training_projects.py"
  # No-re-screening ledger (2026-09-27 VEGA annotator tidy-up plan):
  # screened_by_stem's full/partial/none cases, project exclusion by id and
  # by directory, archive/standalone/composite-training all ignored,
  # write_prior_screening's contract (screened labels, other projects'
  # accepted masks, polygon normalisation, master_codes name resolution),
  # and a read-only reconciliation against the real inprocess tree.
  "py _reefreview/tests/test_screening.py"
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
  # Site-balance regressions (2026-09-01, quality-sweep Task 2): small-site-count
  # convergence, per-species targets (no cross-species crosstalk), honest shortfall reasons.
  "py TCRMPcvr_chooseImages/tests/test_site_balance_regressions.py"
  # No-re-screening at Step 3 (2026-09-27 VEGA annotator tidy-up plan):
  # --prior-screening drops a frame fully screened elsewhere from selection
  # and reserve alike, never touches a partly-screened one, and behaves
  # byte-identically to before this flag existed when it is omitted.
  "py TCRMPcvr_chooseImages/tests/test_prior_screening_exclusion.py"
  "js TCRMPclip_placePoints/tests/test_pp_core.js"
  "py TCRMPclip_placePoints/tests/verify_apps.py"
  "py TCRMPclip_placePoints/tests/test_review_lock_points.py"
  # Step-4 routing trusts Step 3 paths + recursive clip-index fallback.
  "py TCRMPclip_placePoints/tests/test_routing_resolve.py"
  # EasyOCR must construct offline-first (lab DNS drops out intermittently).
  "py TCRMPclip_placePoints/tests/test_ocr_offline_load.py"
  "py TCRMPclip_segmentImages/tests/test_segment_app.py"
  "py TCRMPclip_segmentImages/tests/test_review_lock_routes.py"
  # Combined annotator (4.test), now the promoted Step 4, so its suites run here.
  "py TCRMPclip_combinedAnnotate/tests/test_segment_app.py"
  # SAM3 must boot from the local HF cache when lab DNS is down (2026-08-13).
  "py TCRMPclip_combinedAnnotate/tests/test_sam_offline_load.py"
  # Cached SAM3 tracker backfill (2026-08-26): second boot skips the CPU
  # video-model load; TCRMP_SAM3_REAL_PARITY=1 adds the real-checkpoint check.
  "py TCRMPclip_combinedAnnotate/tests/test_sam_tracker_cache.py"
  # Per-click SAM3 embedding cache (2026-08-26): vision encoder runs once per
  # frame, not once per prompt; exemplar vision-embeds cache mirrors it.
  "py TCRMPclip_combinedAnnotate/tests/test_sam_embedding_cache.py"
  # Live mask-size slider (2026-08-26): tight/medium/generous, re-prompts the
  # selected mask.
  "py TCRMPclip_combinedAnnotate/tests/test_mask_size.py"
  "py TCRMPclip_combinedAnnotate/tests/test_draw_clip.py"
  "py TCRMPclip_combinedAnnotate/tests/test_batch_flow.py"
  "py TCRMPclip_combinedAnnotate/tests/test_provenance.py"
  # No-rescreening cross-project preview (2026-09-27): prior_screening.json
  # load, scaling, provenance exclusion, and the export/registry/undo/dirty
  # boundaries it must never cross.
  "py TCRMPclip_combinedAnnotate/tests/test_prior_screening_server.py"
  "py TCRMPclip_combinedAnnotate/tests/test_batch_all.py"
  # Export, manifest, exemplar, commit-mask, delete-remask, export-completeness.
  "py TCRMPclip_combinedAnnotate/tests/test_export_negatives.py"
  "py TCRMPclip_combinedAnnotate/tests/test_export_manifest.py"
  "py TCRMPclip_combinedAnnotate/tests/test_export_previews.py"
  "py TCRMPclip_combinedAnnotate/tests/test_exemplar_no_persist.py"
  "py TCRMPclip_combinedAnnotate/tests/test_commit_mask.py"
  "py TCRMPclip_combinedAnnotate/tests/test_review_lock_core.py"
  "py TCRMPclip_combinedAnnotate/tests/test_review_lock_routes.py"
  "py TCRMPclip_combinedAnnotate/tests/test_review_lock_ui.py"
  "py TCRMPclip_combinedAnnotate/tests/test_target_only_shows_pending.py"
  "py TCRMPclip_combinedAnnotate/tests/test_delete_remask.py"
  "py TCRMPclip_combinedAnnotate/tests/test_add_after_delete.py"
  # Add box (B) (2026-08-26): drag-to-box creates a new mask via /add_box.
  "py TCRMPclip_combinedAnnotate/tests/test_add_box.py"
  "py TCRMPclip_combinedAnnotate/tests/test_clip_visibility.py"
  "py TCRMPclip_combinedAnnotate/tests/test_clean_mask_kernel.py"
  # Merge keeps every part (2026-08-26): clean_mask(keep_largest=False) no
  # longer drops disconnected components; /merge_many unions N checked masks
  # into one survivor in a single request.
  "py TCRMPclip_combinedAnnotate/tests/test_merge_components.py"
  "py TCRMPclip_combinedAnnotate/tests/test_merge_many_route.py"
  "py TCRMPclip_combinedAnnotate/tests/test_at_test_export_complete.py"
  # Host-open buttons (export folder / manifest via xdg-open).
  "py TCRMPclip_combinedAnnotate/tests/test_open_routes.py"
  # Custom image imports (2026-08-26): bring-your-own images into the Step-4
  # annotator, lores rule + project.json recording (T20260815-203910 part 2).
  "py TCRMPclip_combinedAnnotate/tests/test_custom_imports.py"
  # Standalone "Annotate Images" (2026-08-26): sessions outside any project,
  # inprocess/_standalone/<session>/, reuse the same custom-imports ingest.
  "py TCRMPclip_combinedAnnotate/tests/test_standalone_session.py"
  # Step-4 reserve refill.
  "py TCRMPclip_placePoints/tests/test_reserve_refill.py"
  # Site-aware reserve refill (2026-09-01, quality-sweep Task 3): deficit-ordered
  # pulls from selection_summary.json, fallback preserved, routing_report columns.
  "py TCRMPclip_placePoints/tests/test_reserve_site_balance.py"
  "py TCRMPclip_placePoints/tests/test_lores_scaling.py"
  # 2018 frame fix (2026-09-27, VEGA annotation safety net): a pre-2020
  # frame's CPC points can sit in a season folder other than its own
  # calendar year (cpc_all/2017 holds frames dated into 2018).
  "py TCRMPclip_placePoints/tests/test_cpc_cross_season.py"
  "py pipeline_orchestrator/tests/test_orchestrator_ui.py"
  # Orchestrator SPA sweep (2026-09-01, quality-sweep Task 1): linkage,
  # relics, internal-number leaks, tooltips, cache-bust.
  "py pipeline_orchestrator/tests/test_ui_sweep_20260901.py"
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
  # Auto project names (2026-08-26): {INITIALS}_{SPECIES}_{N}img_{YY-MM-DD}
  # rebuilt on every save for name_auto projects + /api/project/name-preview.
  "py pipeline_orchestrator/tests/test_auto_project_name.py"
  # routed_input/<year> flattened to flat routed_input/{ids,raw} after routing.
  "py pipeline_orchestrator/tests/test_flatten_routed_input.py"
  # Module-lock gate (2026-08-16): lock page on /, 423 on APIs, fail-open registry.
  "py pipeline_orchestrator/tests/test_lock_gate.py"
  # GPU right-of-way gate (2026-09-17): StageRunner asks before every Popen
  # for a runner key listed in module.yaml's gpu.stages, claims with the
  # child pid, releases on process exit; a key absent from that map is
  # never gated.
  "py pipeline_orchestrator/tests/test_gpu_gate.py"
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
  # Host-open guard (2026-09-02 hotfix): VICARIUS_HOST_OPEN=0 and temp-dir
  # paths are refused before subprocess.Popen; a PATH-shim xdg-open drives
  # the real spawn path end to end without touching the desktop.
  "py tests/test_host_open_guard.py"
  # Nightly run-data safety net (2026-09-27): rsync --link-dest snapshots of
  # inprocess/ with count + sha256 verification, atomic .partial rename,
  # single-instance locking, and 14-snapshot retention that never deletes
  # anything unverified, outside dest, or reached through a symlink.
  "py tests/test_snapshot_runs.py"
  # Model-assist loop (2026-07-09): inference predictions.json, prediction seeder, transect frozen split.
  "py TCRMPtrain_oceankindCV/tests/test_predictions_output.py"
  "py TCRMPclip_combinedAnnotate/tests/test_seed_from_predictions.py"
  "py TCRMPtrain_oceankindCV/tests/test_pinned_split.py"
  # Model-assist loop wave 2: annotator resume mode, train --freeze, evaluate rounds ledger.
  "py TCRMPclip_combinedAnnotate/tests/test_loop_resume.py"
  "py TCRMPtrain_oceankindCV/tests/test_train_freeze.py"
  "py TCRMPtrain_oceankindCV/tests/test_rounds_ledger.py"
  # Detection-dataset builder (2026-09-02, quality-sweep Task N1): bbox
  # pairing, seeded 80/10/10 split with replayed pins, polygon-to-box
  # conversion, segment-vs-detect dataset_label_kind, CLI prepare.
  "py TCRMPtrain_oceankindCV/tests/test_detect_dataset.py"
  # train_wrapper.py task support, segment | detect (2026-09-02,
  # quality-sweep Task N1): pure arg/kwarg paths only, no GPU touched.
  "py TCRMPtrain_oceankindCV/tests/test_train_task.py"
  # run_step6.sh task plumbing, segment | detect (2026-09-02, quality-sweep
  # Task N1): driven for real against a stub python, no model loaded.
  "py TCRMPtrain_oceankindCV/tests/test_run_step6_task.py"
  # Model-assist loop wave 3: orchestrator step4loop routes, step6 fine-tune, step7 rounds/promote.
  "py pipeline_orchestrator/tests/test_step4loop_routes.py"
  "py pipeline_orchestrator/tests/test_step6_finetune.py"
  "py pipeline_orchestrator/tests/test_rounds_routes.py"
  # Canonical mask registry (2026-07-10): per-mask store, export upsert hook, matrix derivation.
  "py _reefreview/tests/test_mask_registry.py"
  "py TCRMPclip_combinedAnnotate/tests/test_export_registry_upsert.py"
  # Reviewer identity on masks + /api/reviewer (re-runs itself for the editMasks twin).
  "py TCRMPclip_combinedAnnotate/tests/test_reviewer_identity.py"
  # Accepting a mask requires a reviewer identity (2026-09-02, quality-sweep
  # Task J): every ACCEPT path 200s with needs_reviewer instead of mutating,
  # re-runs itself for the editMasks twin (no separate editMasks copy).
  "py TCRMPclip_combinedAnnotate/tests/test_reviewer_required.py"
  "py _matrix/tests/test_matrix_registry.py"
  # Partial-registry merge (2026-09-01, quality-sweep Task 4): registry rows merge
  # into a project's ledger/legacy contribution instead of replacing it wholesale.
  "py _matrix/tests/test_partial_registry_merge.py"
  # Label guard (2026-07-10): block empty-species accept/export + unlabeled filter.
  "py TCRMPclip_combinedAnnotate/tests/test_label_guard.py"
  "py TCRMPclip_combinedAnnotate/tests/test_unlabeled_filter.py"
  # Shared-file parity (2026-08-26): mask_ops.py, sam_engine.py, provenance.py,
  # export_yolo.py stay byte-identical between combinedAnnotate and editMasks.
  "py TCRMPclip_combinedAnnotate/tests/test_shared_file_parity.py"
  # Annotator polish and speed (2026-09-01, quality-sweep Task 7): scan-refusal
  # keep, flush-to-disk, wire_mask payload trim, warm-up prefetch, relic removal.
  # Drives both combinedAnnotate and editMasks from this one file.
  "py TCRMPclip_combinedAnnotate/tests/test_polish_20260901.py"
  # VEGA annotation safety net (2026-09-27): SIGTERM/SIGINT/atexit flush the
  # session before the process exits, a closed tab's edit reaches disk via a
  # pagehide sendBeacon, and a merged mask records found_manual, not
  # not_found. Drives both combinedAnnotate and editMasks from these files.
  "py TCRMPclip_combinedAnnotate/tests/test_save_on_quit.py"
  "py TCRMPclip_combinedAnnotate/tests/test_merged_outcome_found.py"
  # Review defect fix (2026-09-27): flush_all overwrote segmentations.json
  # from memory unconditionally; the expert-ID importer can rewrite the same
  # file while an annotator sits idle. A dirty flag now gates the quiet quit
  # and pre-kill save paths, and save_segmentations diverts to a conflict
  # sibling file rather than clobbering an externally-changed original.
  "py TCRMPclip_combinedAnnotate/tests/test_flush_conflict_guard.py"
  # Edit Masks app (2026-07-10): standalone editor - boot, expert-lock, registry-on-edit; orchestrator tile.
  "py TCRMPclip_editMasks/tests/test_edit_boot.py"
  "py TCRMPclip_editMasks/tests/test_expert_lock.py"
  "py TCRMPclip_editMasks/tests/test_edit_registry.py"
  # Merge-many + add-box parity in Edit Masks (2026-08-26).
  "py TCRMPclip_editMasks/tests/test_merge_many_route.py"
  "py TCRMPclip_editMasks/tests/test_add_box.py"
  "py TCRMPclip_editMasks/tests/test_review_lock_routes.py"
  "py pipeline_orchestrator/tests/test_editmasks_routes.py"
  # Open-folder routes: launch bindings forgotten on project switch, error bodies carry the path (2026-08-26).
  "py pipeline_orchestrator/tests/test_open_folder_routes.py"
  # Reviewer initials threaded into every annotator launch (2026-08-26).
  "py pipeline_orchestrator/tests/test_reviewer_env.py"
  # Standalone "Annotate Images" launch route (2026-08-26): /annotate, run/status/stop.
  "py pipeline_orchestrator/tests/test_annotate_routes.py"
  # Step 3 site-balance controls + editMasks port move off 5085 (2026-09-02,
  # quality-sweep Task 11): config round-trip, _run_step3 command building,
  # site_summary attached on step completion.
  "py pipeline_orchestrator/tests/test_step3_site_controls.py"
  # Composite training set opened as a pipeline project (2026-09-02,
  # quality-sweep Task H): opens on Train, refuses steps 1-4 and the
  # annotator routes, Step 6 reads the symlinked export.
  "py pipeline_orchestrator/tests/test_composite_training_project.py"
  # Bbox training project opened in the orchestrator (2026-09-02,
  # quality-sweep Task N2): state/is_training_project, annotation-route
  # guards, Step 6 builds its own detect split from the staged dataset.
  "py pipeline_orchestrator/tests/test_bbox_training_project.py"
  # Step 6 "Training task" selector, segment vs detect (2026-09-02,
  # quality-sweep Task N2): UI placement, config round-trip, env forwarding,
  # model default per task.
  "py pipeline_orchestrator/tests/test_step6_task.py"
  # Which preset a fresh project opens with (2026-09-14/15): the fish
  # recipe carries run3 verbatim, every applier resolves to a live helper
  # and form field, and a source-tagged preset never opens on a project of
  # another source kind.
  "py pipeline_orchestrator/tests/test_step6_preset_default.py"
  # VEGA training fold-in (2026-09-04/05): the four capabilities of the
  # retired Train + Infer page moved into the main window's Train, Evaluate
  # and Inference panels, with the engine fixes the audit called for.
  # Safety: a second Start Training press is refused before any write.
  "py pipeline_orchestrator/tests/test_step6_launch_guard.py"
  # Engine: per-class metrics indexed through ap_class_index, the headless
  # Chrome PDF print, and the inference manifest's skipped-detection counts.
  "py TCRMPtrain_oceankindCV/tests/test_evaluate_metrics.py"
  "py TCRMPtrain_oceankindCV/tests/test_evaluate_pdf.py"
  "py TCRMPtrain_oceankindCV/tests/test_inference_manifest.py"
  # Routes: the cross-project runs list, the run-keyed evaluation report and
  # its image route, and the inference run list behind the gallery picker.
  "py pipeline_orchestrator/tests/test_step6_all_runs.py"
  "py pipeline_orchestrator/tests/test_all_runs_trained_on.py"
  "py pipeline_orchestrator/tests/test_path_containment.py"
  "py pipeline_orchestrator/tests/test_standalone_training_source.py"
  "py pipeline_orchestrator/tests/test_rounds_ledger_layout.py"
  "py pipeline_orchestrator/tests/test_vega_tour_content.py"
  "js pipeline_orchestrator/tests/test_vega_tour.js"
  "py pipeline_orchestrator/tests/test_status_note_hold.py"
  "py pipeline_orchestrator/tests/test_step7_report_by_run.py"
  # Evaluate follows the run that was picked (2026-09-14/15): a lab run is
  # scored on the lab dataset and written to inprocess/_models/_eval, the
  # open project keeps its own Step 7 pointers and its Evaluate step, and
  # the path refusals hold.
  "py pipeline_orchestrator/tests/test_step7_foreign_run.py"
  "py pipeline_orchestrator/tests/test_step8_runs.py"
  # The three new cards: Jinja fragment and prose checks (py), then the pure
  # render functions driven in Node with no browser (js).
  "py pipeline_orchestrator/tests/test_step6_runs_card.py"
  "py pipeline_orchestrator/tests/test_step7_report_card.py"
  "py pipeline_orchestrator/tests/test_step8_gallery_card.py"
  "js pipeline_orchestrator/tests/test_vega_runs.js"
  "js pipeline_orchestrator/tests/test_vega_report.js"
  "js pipeline_orchestrator/tests/test_vega_gallery.js"
  # Panel wiring and control fixes: the three includes, the removed inline
  # report and gallery code, Stop Training, the per-panel warnings container,
  # box-dataset mode, the preset notes, and the tooltip rules.
  "py pipeline_orchestrator/tests/test_ui_sweep_20260905.py"
  # Starting model and training data on the Train panel (2026-09-21, LO):
  # the read-only source summary route (pictures and split sentences), the
  # three-section template with the starting model select and the Continue
  # from a VEGA model disclosure, and the script's continue-from routing.
  "py pipeline_orchestrator/tests/test_step6_source_summary.py"
  "py pipeline_orchestrator/tests/test_step6_starting_model.py"
  "py pipeline_orchestrator/tests/test_step6_continue_from_js.py"
  # 2026-09-27 annotation safety-net plan, Task T2: every stop path flushes
  # the annotator/Edit Masks process (POST /api/flush) before it kills it,
  # including the step4loop/run reorder (flush + stop precede the seeder).
  "py pipeline_orchestrator/tests/test_flush_before_kill.py"
  # No-re-screening at the orchestrator layer (2026-09-27 VEGA annotator
  # tidy-up plan): _run_step3 writes prior_screening_input.json and passes
  # --prior-screening; every annotator launch (configure via step4test,
  # resume via step4loop, and editmasks) refreshes prior_screening.json in
  # the project's step4test export dir; a screening or write failure is
  # logged and never blocks Step 3 or a launch.
  "py pipeline_orchestrator/tests/test_prior_screening_launch.py"
  "py _matrix/tests/test_registry_matrix_e2e.py"
  # Matrix cell drill-down, preferences, composite export, stats (2026-08-26).
  "py _matrix/tests/test_matrix_cell.py"
  "py _matrix/tests/test_preferences.py"
  "py _matrix/tests/test_composite.py"
  # Composite route layer (2026-09-01, quality-sweep Task 6): source-type filter,
  # rebuild-in-place validation and 409/400 handling on POST /matrix/api/composite.
  "py _matrix/tests/test_composite_routes.py"
  # Composite-to-project pipeline (2026-09-02, quality-sweep Task H):
  # project_ids filtering, create_training_project run-dir + project.json
  # shape, matrix scan/stats self-exclusion for a composite training run.
  "py _matrix/tests/test_composite_projects.py"
  # Matrix minter delegates to the shared _reefreview helper (2026-09-02,
  # quality-sweep Task N2): identical project.json, one RunDirExhausted
  # definition, matrix scans skip both training-project kinds.
  "py _matrix/tests/test_training_project_delegation.py"
  # Matrix cell-info cache (2026-09-02, quality-sweep micro-task).
  "py _matrix/tests/test_cellinfo_cache.py"
  "py _matrix/tests/test_stats.py"
  "py _matrix/tests/test_matrix_routes.py"
  "py _matrix/tests/test_matrix_ui.py"
  "py _matrix/tests/test_workflow_assets.py"
  # Label + site code -> name resolution (2026-09-02, coverage-matrix
  # stage-rail Task 2): names.py CSV readers, label_names/site_names on
  # /api/data and /api/stats.
  "py _matrix/tests/test_names.py"
  # Cross-run library, kept as library code after the Train + Infer page was
  # retired (2026-09-05): source discovery (composite exports and staged bbox
  # datasets), name and path validation, lab and project run scanning with the
  # champion flag, infer and run guards. The Train panel's runs card reads it.
  "py _modellab/tests/test_labruns.py"
  # Bounding Boxes tool (2026-09-02, quality-sweep Wave 2): Coral Intel ingest
  # (staged layout, verified class map, degenerate handling) and the
  # box-to-mask SAM3 bridge, plus every blueprint route (subprocess.run /
  # Popen stubbed at module level, no GPU or Dropbox access).
  "py _boxes/tests/test_ingest_coral_intel.py"
  # Generic YOLO box-dataset stager (2026-09-14/15): basename
  # normalisation, the staged layout with names.csv, split_pins.json and
  # rename_map.csv, the manifest counts, every refusal, and the
  # case-unique staged names the Synology Drive sync forces.
  "py _boxes/tests/test_ingest_yolo_boxes.py"
  "py _boxes/tests/test_bbox_to_seg.py"
  "py _boxes/tests/test_boxes_routes.py"
  # One-click staged box dataset to pipeline project (2026-09-02,
  # quality-sweep Task N2): mints project_type bbox_training at Step 6, no
  # imagery or symlink enters the run dir, validation guards on 400.
  "py _boxes/tests/test_boxes_create_project.py"
  # Runs trained outside VEGA imported as a model lab (2026-09-14/15):
  # the lab layout, the runs card listing them, the dataset rebuilt on the
  # staged split pins, no invented ledger row or champion, every refusal,
  # and the rollback that leaves no half-written lab.
  "py _boxes/tests/test_import_external_runs.py"
  # Class-count subset of a staged box dataset (2026-09-15): the renumbered
  # class ids, the images an emptied label file takes with it, class_map.csv
  # and dropped_classes.csv, the licence carried over, the manifest counts and
  # every refusal, with nothing written until the last check has passed.
  "py _boxes/tests/test_subset_by_class_count.py"
  # Label Manager theme contract (2026-09-02, quality-sweep theme-unify
  # pass): pins the VICARIUS palette and the Inter, Syne and Space Mono font
  # roles across Label Manager, the Annotate boot page and Add Expert IDs.
  # The Train + Infer stylesheet left this list when the page was retired.
  "py _labels/tests/test_theme_unify.py"
  # The VEGA tutorial box dataset (2026-09-16): the fixture on disk at
  # supporting_data/bbox_datasets/tutorial_boxes, checked pair by pair, class
  # by class and split by split (every class in all three splits, or the
  # tutorial's evaluation lesson has a missing row), plus the builder itself
  # over temp fixtures: the reproducible seed, the picture carrying a dropped
  # class that is passed over, and every refusal it promises.
  "py _tutorial/tests/test_tutorial_set.py"
  "py _tutorial/tests/test_tutorial_projects.py"
)

PASS=0
FAIL=0
FAILED_SUITES=()

echo "============================================================"
echo " run_all_tests.sh  repo: $REPO"
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

# ---------------------------------------------------------------------------
# Drift check (2026-09-16, LO): a test suite that exists on disk but is absent
# from SUITES above is a silent hole in the board, which is how seven working
# suites went unrun. Vendored upstream tests (ultralytics_src) and virtualenvs
# are not ours and are excluded by name.
# ---------------------------------------------------------------------------
DRIFT=""
while IFS= read -r found; do
  case "$found" in
    */env/*|*site-packages*|*ultralytics_src*) continue ;;
  esac
  hit=""
  for entry in "${SUITES[@]}"; do
    [ "${entry#* }" = "$found" ] && { hit=1; break; }
  done
  [ -z "$hit" ] && DRIFT="$DRIFT $found"
done < <(cd "$HERE" 2>/dev/null && find . -path '*/tests/test_*.py' | sed 's|^\./||' | sort)

if [ -n "$DRIFT" ]; then
  echo "============================================================"
  echo " BOARD DRIFT: these suites exist on disk but are not on the board:"
  for d in $DRIFT; do echo "   - $d"; done
  echo " Add each to the SUITES array, or exclude it by name with a reason."
  echo "============================================================"
  exit 1
fi

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
