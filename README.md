# Reef Point Seg (`reef_point_seg`)

Reef Point Seg turns decades of TCRMP CVR coral point-count data into a trained YOLO11 instance-segmentation model, for the TCRMP benthic-monitoring team.

- Tags: AI, TCRMP | Version: 1.3.0 | Status: active | Owner: Lauren K Olinger
- Repo: `vicarius/modules/reef_point_seg/github_repo` | Related studies: TCRMP (Territorial Coral Reef Monitoring Program), study default `S1_historical`

## What it does and why it exists

Reef Point Seg is a launcher-style VICARIUS pipeline that converts the point-count records the TCRMP program has collected for decades into pixel-level coral masks and, from those masks, a trained segmentation model. TCRMP surveys record benthos as CoralNet-style point annotations: a survey photo, a grid of labelled points, and a species code at each point. That format records what sits under each point, and it leaves the colony location and outline undefined. Reef Point Seg supplies the colony outline. It reads the point labels, verifies each point against the survey image, segments the object under each point with SAM3, and assembles the accepted masks into a YOLO11-segmentation training set. Steps 6 through 8 then train, evaluate, and run the model.

The module solves a labelling-cost problem. Drawing coral outlines by hand across every TCRMP site, transect, and year would cost more labor than the project can spend. The existing point labels already carry an established species identity at a known pixel, so the pipeline uses each point as a SAM3 click prompt and lets the segmenter propose the outline. A human reviewer then accepts, edits, or discards each proposed mask before it enters training data. The reviewer flags uncertain corals REVIEW, and the module ships them to an outside taxonomic expert through a GitHub-Pages viewer. The expert returns identifications, the operator folds them back in, and later projects reuse them.

The operator runs Reef Point Seg as an orchestrated, resumable project. Each project is a directory on disk that records the exact step, configuration, and outputs, so a run can stop after any step and resume later with its state restored. One person drives the whole flow through a browser: the orchestrator serves a single sidebar UI and folds each sub-tool behind it, so the operator never sees a sub-app startup form.

Version 1.1.0 added the REVIEW flag, the expert-review round-trip, and the Add Expert IDs tile (relabelled Expert Review I/O in the current UI). Version 1.1.0 also folded the separate point-placement and segmentation steps into one combined annotator, so the operator now places a point and SAM3 segments it in a single pass. Version 1.2.0 added the model-assist Refine loop: inference persists machine-readable predictions, a seeder turns them into editable pre-labels in the combined annotator, a frozen transect holdout keeps every training round comparable, and a rounds ledger with promotion gates tracks whether each retrain is actually better. See "Refine loop (model-assisted labeling)" below. Version 1.3.0 added a standalone Edit Masks tool for going back into an already-exported project to relabel, delete, redraw, or add masks, a permanent per-mask canonical registry that every export upserts into, and a guard that blocks accepting or exporting an unlabeled mask. See "Edit Masks and canonical mask registry" below.

## Where it sits in the pipeline

Reef Point Seg is the head of its own chain, separate from the 3D photogrammetry survey chain. The inputs come from the TCRMP program and from a legacy point-index tool, and no other VICARIUS module feeds them:

- The TCRMP CVR Excel workbooks (`tcrmp_cvr`) are the raw point-count exports the monitoring program produces.
- The TCRMP clip frames (`tcrmp_clip`) are the survey still frames sampled from transect video.
- The `cpc_all` index is a static pre-2020 CPCe point store built once by the legacy `TCRMPclip_cpcID` tool.

Downstream, the module produces a trained YOLO11-segmentation checkpoint (`best.pt`) plus evaluation and inference artifacts. Within this repo, no VICARIUS module automatically consumes those outputs; the trained model and its per-image detections are the handoff artifacts for later coral-cover and demography analysis. Any automatic downstream wiring beyond this repo is unverified here.

## Inputs

| Input | Type | Formats | Required | What it is |
|-------|------|---------|----------|------------|
| `tcrmp_cvr` | directory | `.xls`, `.xlsx` | Yes for a from-scratch run | Raw TCRMP CVR Excel workbooks. Step 1 parses every workbook into one long point table. The operator often skips Step 1 by linking an existing `all_points.csv` plus a matching master-codes table. Location: `supporting_data/TCRMP_CVR`. |
| `tcrmp_clip` | directory | `.jpg`, `.jpeg` | Yes | TCRMP video-derived clip frames for the full year and period set. The pipeline reads them from the local supporting-data tree and resolves each frame by a recursive whole-tree scan keyed on the globally-unique basename. Location: `supporting_data/TCRMP_clip`. |
| `cpc_all` | directory | CPCe point index | Yes for pre-2020 frames | Static pre-2020 CPCe point index, about 67 GB, produced once by the legacy `TCRMPclip_cpcID` tool. Step 4 reads it to recover point coordinates for pre-2020 frames. Location: `supporting_data/cpc_all`. |

The master-codes table (the species-code lookup) resolves to the VICARIUS metadata library at `vicarius/_METADATA/library/definitions/tcrmp_species_codes.csv`. Pretrained YOLO weights live under `supporting_data/model_weights`; Ultralytics base weights auto-download on first use. All three bulk input trees plus `model_weights` live under `supporting_data/`, which is gitignored (about 68 GB); `scripts/sync_from_source.sh` syncs it from the hopper source tree.

## Parameters

These are the operator-facing parameters surfaced in `module.yaml`. Every parameter also has a per-step default in `config/pipeline.yaml`, and a per-project value saved into `project.json`; the project value takes precedence at run time.

| Parameter | Type | Default | What it controls |
|-----------|------|---------|------------------|
| `target_species` | string | `OFRA, PA, OA, OFAV, AL, MC, AA` | Priority species codes for balanced selection in Step 3 and review filtering in Step 4. The live Step 3 picker opens with nothing preselected (the `config/pipeline.yaml` default is empty), so the operator chooses target labels for each project. |
| `target_instances_per_species` | integer | 1000 | Minimum count of point-labels per target species that Step 3 collects before selection stops. |
| `epochs` | integer | 500 | Maximum training epochs for Step 6. |
| `imgsz` | integer | 512 | Training, evaluation, and inference image size, in pixels on a side. |
| `model` | string | `yolo11m-seg.pt` | Base pretrained YOLO11-seg weights. Scale letter is one of n, s, m, l, x. |

## Outputs and data-dictionary entries

Every declared output binds to a descriptor under `vicarius/_METADATA/dictionary/datasets/`. Each entry carries `eml: true` in `module.yaml`, so the runner writes an EML 2.2.0 record for the output at run finalization. Paths use `{project_id}` for the `run_*` project directory under `inprocess/`.

| Output dataset | Path template | Descriptor | What it is |
|----------------|---------------|------------|------------|
| all points | `{project_id}/step1_makeAllPoints/all_points.csv` | [reef_point_seg_all_points](../../../_METADATA/dictionary/datasets/reef_point_seg_all_points.yaml) | Point-level unified CVR table, 9 columns, one row per CPC point across every site, transect, and year. About 1.6M rows on recent runs. |
| master codes | `{project_id}/step2_recodeSpecies/master_codes.csv` | [reef_point_seg_master_codes](../../../_METADATA/dictionary/datasets/reef_point_seg_master_codes.yaml) | Species-code lookup, 3 columns (code, category, name), one row per code. It joins codes to names and supplies the species menu for point routing. |
| recoded all points | `{project_id}/step2_recodeSpecies/all_points_recoded.csv` | [reef_point_seg_recoded_all_points](../../../_METADATA/dictionary/datasets/reef_point_seg_recoded_all_points.yaml) | Copy of `all_points.csv` after Step 2 applies the rename, merge, and exclude rules the operator defines. Same 9-column schema. |
| remap log | `{project_id}/step2_recodeSpecies/remap_log.json` | [reef_point_seg_remap_log](../../../_METADATA/dictionary/datasets/reef_point_seg_remap_log.yaml) | JSON audit of every rename, merge, and exclusion, with per-remap point counts. It lets the operator replay a recode deterministically. |
| selected frames | `{project_id}/step3_chooseImages/selected_frames.csv` | [reef_point_seg_selected_frames](../../../_METADATA/dictionary/datasets/reef_point_seg_selected_frames.yaml) | Master list of the frames Step 3 chooses by year-balanced allocation, 18 columns, one row per selected frame. Sibling splits `route_cpc.csv`, `route_ocr_needed.csv`, and `route_missing.csv` share the schema. |
| reserve frames | `{project_id}/step3_chooseImages/reserve_frames.csv` | [reef_point_seg_reserve_frames](../../../_METADATA/dictionary/datasets/reef_point_seg_reserve_frames.yaml) | Eligible-but-unselected frames, 12 columns, randomly ranked. Step 4 refills from this pool when OCR detects no points on a 2020-and-later frame. |
| SAM click prompts | `{project_id}/step4_routeChosenImages/{year}/ids/sam_click_prompts.json` | [reef_point_seg_sam_click_prompts](../../../_METADATA/dictionary/datasets/reef_point_seg_sam_click_prompts.yaml) | Per-year JSON of click prompts, keyed by raw-image filename. Each point carries the pixel x and y, the resolved species, and `point_type` 1 (positive click). |
| export manifest | `{project_id}/step4test_combinedAnnotate/export_manifest.csv` | [reef_point_seg_export_manifest](../../../_METADATA/dictionary/datasets/reef_point_seg_export_manifest.yaml) | Per-basename reconciliation of the YOLO export, 8 base columns plus one dynamic count column per species. `outcome` records exported_with_masks, exported_empty, review_only, image_missing, or scrapped. The manifest accumulates rows across export batches. |
| YOLO training set | `{project_id}/step5_segmentImages/` | [reef_point_seg_yolo_training_set](../../../_METADATA/dictionary/datasets/reef_point_seg_yolo_training_set.yaml) | Self-contained YOLO11-seg bundle: `data.yaml`, `class_map.json`, `all_images/`, `all_labels/`. The combined annotator excludes REVIEW-flagged masks from the bundle. See the drift note below on the live output location. |
| trained model | `{project_id}/step6_trainModel/runs/{run_name}/weights/best.pt` | [reef_point_seg_trained_model](../../../_METADATA/dictionary/datasets/reef_point_seg_trained_model.yaml) | Best-epoch Ultralytics segmentation checkpoint. A sibling `last.pt` and the full run artifacts (`args.yaml`, `results.csv`, curve and confusion-matrix PNGs, sample mosaics) sit beside it. |
| training rounds ledger | `{project_id}/step6_trainModel/rounds.csv` | [reef_point_seg_training_rounds](../../../_METADATA/dictionary/datasets/reef_point_seg_training_rounds.yaml) | Cumulative, idempotent ledger of every evaluated training run: identity, dataset composition, headline metrics, `split_pinned`, and the three promotion-gate verdicts (`gate_map`, `gate_class`, `gate_recall`). See the Refine-loop section below. |
| evaluation report | `{project_id}/step7_evaluateModel/eval_{run_name}/report.pdf` | [reef_point_seg_evaluation_report](../../../_METADATA/dictionary/datasets/reef_point_seg_evaluation_report.yaml) | Multi-page plain-language PDF with a stable page layout, so a reader can compare two reports page for page. A `report.md` twin sits beside it. |
| evaluation metrics | `{project_id}/step7_evaluateModel/eval_{run_name}/metrics.json` | [reef_point_seg_evaluation_metrics](../../../_METADATA/dictionary/datasets/reef_point_seg_evaluation_metrics.yaml) | Machine-readable companion to the PDF: one Ultralytics `val()` pass with overall and per-class box and mask metrics, provenance, and preview references. |
| inference manifest | `{project_id}/step8_inference/{run_name}/manifest.json` | [reef_point_seg_inference_manifest](../../../_METADATA/dictionary/datasets/reef_point_seg_inference_manifest.yaml) | Inference-run gallery and summary: a per-image `items` array with detection rows, run-wide totals, per-class confidence aggregates, and a plain-language `blind_spots` string. |
| inference predictions | `{project_id}/step8_inference/{run_name}/predictions.json` | [reef_point_seg_inference_predictions](../../../_METADATA/dictionary/datasets/reef_point_seg_inference_predictions.yaml) | Machine-readable, per-detection companion to the manifest, written when the run passes `--save_predictions`: per-image detections with class, confidence, and a normalized polygon. The sole input to the Refine-loop seeder; see below. |
| expert-review bundle | `/mnt/tear/REVIEW_reefpointseg/review_manifest.json` | [reef_point_seg_expert_review_bundle](../../../_METADATA/dictionary/datasets/reef_point_seg_expert_review_bundle.yaml) | Living queue of REVIEW-flagged masks awaiting an outside expert. The module writes it into the review repo working tree and auto-pushes it to GitHub Pages. Items leave the queue once the operator imports an expert ID. |
| expert-ID library | `inprocess/_expert_id_library/manifest.csv` | [reef_point_seg_expert_id_library](../../../_METADATA/dictionary/datasets/reef_point_seg_expert_id_library.yaml) | Permanent, cross-project master record tying every reviewed mask UID to the assigned code and MODE (USER or EXPERT). Gitignored, shared across all projects on the box. |
| mask registry | `inprocess/_mask_registry/manifest.csv` | [reef_point_seg_mask_registry](../../../_METADATA/dictionary/datasets/reef_point_seg_mask_registry.yaml) | Permanent, cross-project canonical census of every mask, keyed by content-stable `mask_uid`, upserted by every export (combined annotator, segment-images, Edit Masks). Manifest-only, no geometry sidecars. The label coverage matrix derives from this registry. `eml: false`, a meta-level store sibling to `_expert_id_library/`, not nested under `{project_id}`. |

Two further Refine-loop artifacts have data-dictionary descriptors but no `module.yaml` outputs entry, since they are per-project working state rather than run deliverables: `dataset/split_manifest.json` under Step 6 ([reef_point_seg_split_manifest](../../../_METADATA/dictionary/datasets/reef_point_seg_split_manifest.yaml)) and `step4test_combinedAnnotate/loop_rounds/round_*.json` written by each Refine seeding pass ([reef_point_seg_loop_round_manifest](../../../_METADATA/dictionary/datasets/reef_point_seg_loop_round_manifest.yaml)).

Drift and known-issue notes, carried from the descriptors and verified against the code:

- The YOLO training set descriptor path template points at `step5_segmentImages/`, the legacy separate-segment output. In the current combined-annotator flow the live bundle and `export_manifest.csv` land under `step4test_combinedAnnotate/`, and Step 6 sources the training directory from there (`_run_step6` reads `step4test_combinedAnnotate`, verified against `pipeline_orchestrator/app.py` and `tests/test_step6_source.py`).
- The writer writes blank species codes verbatim as a separate class in `class_map.json` and `data.yaml`; it does not filter empty codes. Whether an upstream step should filter blank codes is open, flagged in the descriptor.
- The Step 8 `crops/` directory appears only when the operator passes `--save_crops`, and the inspected `run_inference.py` does not populate it. Treat `crops/` as effectively unimplemented despite the CLI flag.
- Several output basenames carry a run-date stamp on disk (for example `all_points_{YYYYMMDD}.csv`) that differs from the plain names in the `module.yaml` path templates; the header order and columns match the real outputs (verified).

## How to run it

### From the UI

The VICARIUS platform UI runs on port 5090. For bringing the VICARIUS UI itself up, see `/mnt/rip/vicarius_drive/vicarius/_DOCS/START_HERE.md`; these steps assume the platform is already running. If port 5090 is down and you only need to run this module, skip the platform entirely: the port-5050 orchestrator path in "From the CLI" below brings the module up on its own with `scripts/restart_reef_point_seg.sh` and requires neither port 5090 nor START_HERE.md.

1. Open the VICARIUS UI at `http://localhost:5090` and go to the Reef Point Seg module page. Because `module.yaml` sets `ui.style: launcher`, the page renders the launcher view rather than a generic input or output form.
2. Choose "create a new project" or "open an existing one". There is no standalone "Launch Orchestrator" button: creating or opening a project is what boots the orchestrator, with that project loaded.
3. Clicking a project (or a dashboard button) boots the Reef Point Seg orchestrator on port 5050 (idempotent; it reuses a running one) and opens it in a new window. Two cross-project dashboards are available from the launcher: Label Coverage Matrix (`/matrix`) and Label Manager (`/labels`).
4. Work the sidebar steps in order. The orchestrator folds each sub-tool behind the single window, so a sub-tool opens in its own window with no startup form. The current sidebar presents: 1 Fetch Points, 2 Fetch Labels, 3 Choose Images, 4 Place + Segment (the combined annotator), 5 Train Model, 6 Evaluate Model, 7 Model Inference, plus an Expert Review I/O tile and an Edit Masks tile that both run at any time. Stop after any step and resume later by reopening the same project.

### From the CLI

Reef Point Seg has no headless per-step CLI and no `cli.flag_map`; `module.yaml` declares `cli.interactive: true`. Every run is driven through the orchestrator in the browser. The terminal action available to you is bringing the orchestrator up.

Do not call `scripts/run_pipeline.sh` directly, and do not rely on `vicarius restart reef_point_seg`: that command routes through `run_pipeline.sh`, whose `lsof` stale-sweep hangs on this box (stale NFS mounts under `/mnt`). Use the dedicated restart script instead, which avoids `lsof`:

```
cd /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo
bash scripts/restart_reef_point_seg.sh        # default port 5050
```

The script hard-kills the orchestrator and its sub-app ports, starts the Flask app via `run_pipeline.py`, blocks until the current page is actually served, and self-tests live markers before it reports success. After it prints the verified line, open `http://localhost:5050/` and drive the pipeline in the browser. When launched through the VICARIUS UI on 5090, the launcher performs the boot for you.

## How it works inside

The entry script `run_pipeline.py` inserts `scripts/pipeline_orchestrator/` on the path and calls `create_app()` from `app.py`, which builds a Flask application and serves it on port 5050 (`app.run(host=0.0.0.0, port=5050)`). The orchestrator is the single UI; it launches each sub-tool as a child process on its own port and proxies or mounts it behind port 5050.

`config/pipeline.yaml` is the source of truth for paths, ports, interpreters, and per-step defaults. It defines the eight internal step entry points, all run with the unified environment Python at `env/bin/python`:

1. Step 1, `TCRMPcvr_makeAllPoints/run.py`: parses every CVR workbook into `all_points.csv` and the master-codes table.
2. Step 2, `TCRMPcvr_recodeSpecies/src/app.py` (sub-app port 5055, or a headless `remap_loader`): applies rename, merge, and exclude rules and writes the recoded points plus `remap_log.json`.
3. Step 3, `TCRMPcvr_chooseImages/src/select_images.py`: runs the greedy, year-balanced, fresh-random frame selection gated on image and point-source eligibility, and writes `selected_frames.csv`, the route splits, and `reserve_frames.csv`.
4. Step 4 routing, `TCRMPclip_placePoints/src/app.py` (sub-app port 5065): resolves the points of each selected frame from CPC coordinates (pre-2020) or OCR (2020 and later) and writes the per-year `sam_click_prompts.json` into `routed_input/`.
5. Place + Segment, `TCRMPclip_combinedAnnotate/src/app.py` (sub-app port 5080): the combined annotator that folds legacy point placement and segmentation into one pass. The operator arms a species, clicks, and SAM3 segments the object under the click; the operator accepts, edits, or scraps each mask. It reads `routed_input/` as a read-only reference layer, writes the YOLO bundle and `export_manifest.csv` under `step4test_combinedAnnotate/`, and pushes REVIEW-flagged masks to the expert-review repo. The review topbar's Import Images control (`POST /api/import_images`, `src/custom_imports.py`) also accepts operator-supplied images from outside the pipeline: each file is copied verbatim into `step4test_combinedAnnotate/custom_imports/originals/`, delivered into `custom_imports/raw/` (downscaled to the 1920px lores standard when oversized, the same rule as routed frames, with the scale recorded in `custom_imports/imports_manifest.json`), added to the review queue with a Custom badge, logged in the run's `project.json` (`access_log` + `custom_imports`), and exported through the normal batch-export path like any routed frame.
6. Step 6, `TCRMPtrain_oceankindCV/run_step6.sh`: splits the combined-annotator export into train, valid, and test and trains YOLO11-seg (Ultralytics), writing `runs/{run_name}/weights/best.pt`.
7. Step 7, `TCRMPtrain_oceankindCV/src/evaluate_run.py`: runs one Ultralytics `val()` pass and writes `report.pdf`, `report.md`, and `metrics.json`.
8. Step 8, `TCRMPtrain_oceankindCV/src/run_inference.py`: runs `best.pt` over a chosen image source and writes `manifest.json` (and `overlays/` when the operator sets `--save_overlays`).

External software and hardware:

- SAM3 (Meta AI Segment Anything with Concepts) provides the click-to-mask segmentation. In the orchestrated flow the combined annotator runs SAM3 from the unified env at `env/bin/python`: `pipeline_orchestrator/app.py` launches it with the Step-5 interpreter (`PYTHON_PATHS.get(5)`), and `config/pipeline.yaml` sets `python_paths.5` to the unified env, which carries SAM3 through `transformers>=4.51` and the torch cu128 wheels. The named conda env `sam3reef` at `/home/bizon/anaconda3/envs/sam3reef` is a legacy fallback that the standalone `run.sh` launchers use only when `env/bin/python` is absent. `config/pipeline.yaml` step 5 places the SAM3 tracker on `cuda:1` and the exemplar model on `cuda:0`, so segmentation expects two CUDA GPUs.
- YOLO11 training and evaluation use Ultralytics from the unified environment. The `TCRMP_STEP6_DEVICE` environment variable controls the Step 6 training device (per the prior README); the training run sets the exact multi-GPU default, and `config/pipeline.yaml` does not fix it (unverified default).
- Additional sub-app ports the module owns: 5055 (recode), 5065 (place points routing), 5070 (legacy segment), 5075 (Expert Review I/O / Add Expert IDs), 5080 (Place + Segment combined annotator), 5085 (Edit Masks). The restart script frees 5050, 5055, 5065, 5066, 5070, 5075, and 5080 and never touches the platform UI.

When `vicarius.enabled: true` (the shipped default), the orchestrator emits `process_start`, `process_end`, and `user_note` events to the central VICARIUS event stream through `pipeline_orchestrator/vicarius_bridge.py`. View activity with `vicarius story --days 1`.

## Refine loop (model-assisted labeling)

Steps 1 through 8 take a project from raw point counts to a trained, evaluated, inference-ready model once. The Refine loop is what closes that into a repeatable cycle: run inference, let the model propose masks on new frames, have a human correct only what the model got wrong, and retrain on the corrected set. Version 1.2.0 adds the machinery for this loop; the plan is `docs/superpowers/plans/2026-07-09-reefpointseg-model-assist-loop.md`.

**The loop path.** Step 3 (Choose Images) selects a fresh batch of frames. Step 4 (Place + Segment) is where a human labels them from scratch the first time. Train (Step 5/6) fits a model on the labeled set. Evaluate (Step 6/7) scores it and appends a row to the rounds ledger (`rounds.csv`). Infer (Step 7/8) runs the trained model over a new batch of frames and, with `--save_predictions`, writes `predictions.json`, a machine-readable list of per-detection class, confidence, and polygon. Refine reopens Step 4 in loop mode: `seed_from_predictions.py` reads `predictions.json` and seeds each above-threshold detection into the combined annotator as a pending, `source_type: model` mask, so the operator reviews and corrects model output instead of drawing every mask by hand. The corrected labels flow back into Train (Step 5/6), and the cycle repeats:

```
3 (Choose Images) -> 4 (Place + Segment) -> Train 5 -> Evaluate 6 -> Infer 7
                                                 ^                      |
                                                 |                      v
                                        Train 5 <- Refine (Step 4, loop mode)
```

**Two retrain strategies.** Once a Refine round has corrected labels, the operator chooses how the next training run starts:

- **Full retrain from the COCO base weights** is the default and the robust choice. It trains from the same pretrained base (`yolo11m-seg.pt` or whichever scale is configured) over the full merged labelset (original labels plus every accepted Refine correction). Slower, because it retrains from scratch, but it never inherits a bias from an earlier round and gives the cleanest apples-to-apples comparison in the rounds ledger.
- **Fine-tune from the champion** is fast. It resumes from the current champion's `best.pt` instead of the COCO base, with a lower learning rate (`lr0 0.001`) and the first 10 layers frozen (`freeze 10`), so the run adapts to the new corrections without unlearning what the champion already does well. Use it for quick iteration between labeling sessions; fall back to a full retrain periodically, or whenever the labelset has grown enough that a from-scratch fit is worth the wall time, or if fine-tuning starts trending down on the gates below.

**Frozen transect holdout.** Every round needs to be judged against the same yardstick, so the validation and test splits are frozen by transect rather than re-randomized on every training run: transect 5 is always validation, transect 6 is always test, and transects 1 through 4 always go to train (`dataset/split_manifest.json`, `holdout_mode: "transect"`). This keeps video-adjacent frames from splitting across train and holdout, which would otherwise inflate the reported mAP, and it means a round-over-round metric comparison is actually comparable. If a project has no labeled frames yet on the val or test transect, the split does not hard-fail: it gracefully degrades to a seeded random holdout for that project (`holdout_mode: "transect-degraded"`, with a `degrade_warning` naming the empty transect) and self-heals back to the clean transect split the moment both holdout transects have labeled frames. **Never seed a holdout transect.** The Refine seeder is fail-closed on this: it requires `split_manifest.json` to exist and excludes any frame pinned to valid/test, or (in clean transect mode) any new frame that parses to a holdout transect, from seeding. A Refine round can never leak model-proposed labels onto the frozen gate it is being measured against.

**Gate meanings.** Each row in the rounds ledger (`rounds.csv`) carries three promotion-gate verdicts comparing that round against the current champion (or, if none is promoted yet, the best prior round):

- `gate_map`: pass if mask mAP50-95 has not dropped more than 0.005 versus the baseline, else fail.
- `gate_class`: pass if no individual class's mask AP50-95 (among classes actually present in train) dropped more than 0.05, else `flag:<codes>` naming the regressed classes.
- `gate_recall`: pass if mask recall has not dropped more than 0.05 versus the baseline, else flag.

All three gates read **unpinned** whenever the dataset's split is not a clean frozen transect split, i.e. `holdout_mode` is `adopted-random`, `pinned-random`, or `transect-degraded` (or no `split_manifest.json` exists at all). Round-over-round comparison is only trustworthy on a frozen, leak-free holdout; an unpinned split is honestly flagged as unmeasurable rather than shown as a false pass. A project that inherited a legacy random split (kept for continuity with its prior training history) always reads `unpinned` here, which is expected.

**Annotator single-instance rule.** The combined annotator is a single long-lived process per orchestrator instance. Launching Refine closes any Step-4 annotator that is already open before starting the loop-mode session, so exports always land in the project the operator intended and two annotator instances never race on the same `segmentations.json`.

## Edit Masks and canonical mask registry

Version 1.3.0 added a direct edit surface for masks a project has already exported, plus a permanent, cross-project canonical registry of every mask that every export upserts into.

**Edit Masks (sidebar tile, port 5085).** The Edit Masks tile opens the standalone `TCRMPclip_editMasks` app against the current project's Step 4 export. It never creates a fresh export the way Step 4 (Place + Segment) or Refine can; it requires an existing `segmentations.json` to edit. Inside it, the operator can relabel a mask's species, delete a mask, redraw a mask's outline, or add a new mask to an already-processed frame, then export the changes. Masks carrying an accepted EXPERT-mode ID are locked against relabel and against any new REVIEW round-trip, so an outside expert's identification cannot be silently overwritten from the editor; a locked mutation is refused and the mask id is reported back rather than applied. An "unlabeled only" filter and a "next unlabeled" jump (`GET /api/unlabeled`) let the operator move directly through the backlog of masks that were exported without a species assigned, instead of paging through every frame by hand. Every edit action writes a `source: edit` row to `label_provenance.csv` (the shared `provenance.py` module, one copy per sub-tool, kept byte-identical) and upserts the mask's canonical registry row.

**Unlabeled-mask guard.** An unlabeled mask can no longer be accepted or exported anywhere in the pipeline. The combined annotator refuses to accept a mask with no species assigned, and the export path carries a matching backstop (`blocked_unlabeled` count in the export summary) so a mask that reached export some other way still cannot ship into `all_labels/` without a label. This closes the gap that produced the pre-1.3.0 backlog of masks exported with an empty species code.

**Canonical mask registry.** `scripts/_reefreview/mask_registry.py` defines `MaskRegistry`, writing `inprocess/_mask_registry/manifest.csv`: one row per content-stable `mask_uid`, across every project on the box. Every export upserts into it, whichever tool produced the export (the combined annotator, the standalone segment-images path, or Edit Masks), so the registry always reflects each mask's current species, category, review state, and source/project context, not just its state at first export. Concurrent writers are safe: upserts are serialized with a POSIX advisory lock so two pipeline steps writing at the same time cannot lose each other's rows. The label coverage matrix (`/matrix`) now derives its image x label grid from this registry rather than re-scanning raw per-project export state, with a fallback path preserved for masks recorded before the registry existed. See the [reef_point_seg_mask_registry](../../../_METADATA/dictionary/datasets/reef_point_seg_mask_registry.yaml) descriptor for the full field list, and [reef_point_seg_expert_id_library](../../../_METADATA/dictionary/datasets/reef_point_seg_expert_id_library.yaml) for the sibling permanent store that additionally keeps geometry sidecars for expert-reviewed masks.

## Gotchas and troubleshooting

- `vicarius restart reef_point_seg` hangs on this box because it routes through `run_pipeline.sh`, which calls `lsof`, and `lsof` stalls on the stale NFS mounts under `/mnt`. Use `scripts/restart_reef_point_seg.sh` instead; it uses `pkill`, `fuser`, and `ss`, never `lsof`.
- After any orchestrator or sub-app edit, restart with the script so Flask reloads its cached templates and static assets, then hard-refresh the browser tab with Ctrl+Shift+R. The restart script self-tests live page markers and exits non-zero if it served stale assets.
- Large parts of the live orchestrator are untracked in git. Verified counts: `scripts/TCRMPclip_combinedAnnotate` has only `src/provenance.py` and its test tracked (the annotator `src/app.py` is untracked), and `scripts/_expertids`, `scripts/_matrix`, `scripts/_labels`, and `scripts/_reefreview` have no tracked files at all. A fresh `git clone` does not reconstruct the working orchestrator; operate from the on-box working tree.
- `supporting_data/` (about 68 GB: `cpc_all`, `TCRMP_clip`, `TCRMP_CVR`, `model_weights`) and `env/` are gitignored. A clone has no data and no interpreter until the operator syncs the data and builds the interpreter.
- Step 3 selection is fresh-random with no fixed seed, so two runs with the same targets select different frames. Row counts vary with the per-species instance target and differ from run to run.
- The combined annotator is a single long-lived process, and it freezes the export directory at the first configure call. The orchestrator kills any running annotator before launching a fresh one so exports land in the active project; if exports appear in the wrong project, a stale annotator process was still bound.
- The combined annotator excludes REVIEW-flagged masks from the YOLO training set even when a reviewer accepted them. They route to the expert-review bundle instead and re-enter training only after the operator imports an expert ID.
- Blank species codes surface as real training classes (empty name in `class_map.json`). The inspected `run_inference.py` does not populate the Step 8 `crops/` directory despite the `--save_crops` flag. Both are known, flagged for review.

## Resume from cold

These steps assume the VICARIUS platform is already running. To bring the VICARIUS UI itself up, follow `/mnt/rip/vicarius_drive/vicarius/_DOCS/START_HERE.md`. Do not reconstruct the module from a fresh git clone: the working orchestrator and its data live only on this box (see the untracked-git note above).

Environment:

- Module root: `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/`. The code lives under `github_repo/`.
- Interpreter: the unified environment at `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python`. If it is missing, rebuild it by running `bash bootstrap.sh` from the `github_repo` root. `bootstrap.sh` calls `scripts/setup_unified_env.sh` to create the conda env at `env/` with Python 3.11, the torch and torchvision cu128 wheels, an editable ultralytics from the vendored `scripts/TCRMPtrain_oceankindCV/ultralytics_src`, `transformers>=4.51` for SAM3, `easyocr==1.7.2`, flask, pandas, numpy, and the rest; it then rewrites absolute paths inside any `projects/` and runs an import smoke test. Run `bash scripts/setup_unified_env.sh` on its own to rebuild only the env, without the path rewrite and smoke test. Both scripts are idempotent.
- SAM3: the orchestrated Step 4 Place + Segment runs SAM3 from the unified env at `env/bin/python`, so `bash bootstrap.sh` is the only build needed for segmentation. Two CUDA GPUs are expected: `config/pipeline.yaml` step 5 sets `sam3_device_tracker: cuda:1` and `sam3_device_exemplar: cuda:0`, so confirm `nvidia-smi` lists at least two GPUs before segmenting. The named conda env `sam3reef` at `/home/bizon/anaconda3/envs/sam3reef` is a legacy fallback that the standalone `scripts/TCRMPclip_combinedAnnotate/run.sh` and `scripts/TCRMPclip_segmentImages/run.sh` launchers use only when `env/bin/python` is absent. To rebuild a SAM3 conda env from clean, run `bash scripts/TCRMPclip_combinedAnnotate/setup_env.sh`: it creates a conda env with Python 3.12, installs the torch, torchvision, and torchaudio cu130 wheels, `transformers` and `huggingface_hub` for SAM3, flask, flask-cors, opencv, numpy, Pillow, scipy, and pycocotools, then verifies by importing `Sam3TrackerModel` and `Sam3Processor`. That script builds a prefix env at `env/` inside the sub-tool directory, a separate location from the named `sam3reef` env; the repo ships no script that rebuilds `sam3reef` by name, so treat the named env as the historical hand-built fallback (unverified as script-reproducible).
- Node.js must be on PATH for the JavaScript test suites.
- Bulk data: `config/pipeline.yaml` reads `cpc_all`, `TCRMP_CVR`, `model_weights`, and `all_points.csv` directly from the hopper tree at `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/` (keys `cpc_all_dir`, `tcrmp_cvr_dir`, `model_weights_dir`, `all_points_csv`), so those three need no repo-local copy. Only `tcrmp_clip_local_dir` resolves to the repo-local `github_repo/supporting_data/TCRMP_clip`, so Steps 3 and 4 need the clip frames present under `supporting_data/`. `supporting_data/` is gitignored (about 68 GB), so a fresh checkout starts empty. Populate the local clip tree with `rsync -a /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/TCRMP_clip/ supporting_data/TCRMP_clip/`. For ongoing deltas from the source repo at `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI`, run `bash scripts/sync_from_source.sh` for a dry run, then `bash scripts/sync_from_source.sh --apply data`; the script rsyncs the supporting-data deltas (all_points.csv, master_codes.csv, cpc_all, TCRMP_CVR, TCRMP_clip) and only reads from the source, never writing back.

Where inputs and outputs live:

- Inputs: `supporting_data/TCRMP_CVR`, `supporting_data/TCRMP_clip`, `supporting_data/cpc_all`, `supporting_data/model_weights`. The master-codes lookup is `vicarius/_METADATA/library/definitions/tcrmp_species_codes.csv`.
- Projects (runs): `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess/`. Each project is a directory named `run_*` holding `project.json` plus `step1_makeAllPoints/` through `step8_inference/` and `step4test_combinedAnnotate/`. `project.json` records `steps.N.status`, so reopening a project restores the exact step and configuration.
- Cross-project stores: `inprocess/_expert_id_library/` (the permanent expert-ID library, gitignored). The expert-review queue lives outside the module at `/mnt/tear/REVIEW_reefpointseg/`.

Build from cold (first time on a fresh checkout, or after wiping `env/` or `supporting_data/`):

```
cd /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo

# 1. Build the unified interpreter at env/. bootstrap.sh calls
#    scripts/setup_unified_env.sh, rewrites project paths, and smoke-tests
#    torch CUDA and transformers (SAM3). Idempotent. This one env also serves
#    SAM3 for Step 4 Place + Segment, so no separate SAM3 build is needed.
bash bootstrap.sh

# 2. Populate the repo-local clip frames. config/pipeline.yaml reads cpc_all,
#    TCRMP_CVR, and model_weights straight from the hopper, so only TCRMP_clip
#    needs a local copy under supporting_data/.
rsync -a /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/TCRMP_clip/ \
         supporting_data/TCRMP_clip/
#    Ongoing deltas from the source repo instead of a first-time copy:
#    bash scripts/sync_from_source.sh --apply data
```

Run it:

```
cd /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo
bash scripts/restart_reef_point_seg.sh        # brings the orchestrator up on port 5050
```

Then open `http://localhost:5050/`, create or open a `run_*` project, and work the sidebar steps. From the VICARIUS UI on port 5090, opening the module page and choosing a project performs the same boot.

Verify success:

- The restart script blocks until the orchestrator serves the current page and prints `Reef Point Seg orchestrator restarted and verified on http://localhost:5050/`. If it exits non-zero, it served stale assets or the process died on startup; read the tail it prints and the log at `/tmp/reef_point_seg_orch_5050.log`.
- Independent check that the orchestrator is live and current: `curl -s http://127.0.0.1:5050/ | grep -q s3-label-chips` returns success, and `ss -ltn 'sport = :5050'` shows a LISTEN socket.
- House testing uses no pytest. Run every sub-tool's suite with the unified env Python and node:

```
cd /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo
bash scripts/run_all_tests.sh
```

A clean run ends with `ALL SUITES GREEN` and exits 0; any failing suite prints its last lines and the aggregator exits 1.

- Per-project success: the `project.json` in a project directory shows the intended step at status complete, and the step output exists on disk at the path in the Outputs table (for example `runs/{run_name}/weights/best.pt` after Step 6).

## Provenance and links

- Repo: `vicarius/modules/reef_point_seg/github_repo`. Interface contract: `module.yaml`. Configuration source of truth: `config/pipeline.yaml`.
- Related studies: TCRMP (Territorial Coral Reef Monitoring Program, University of the Virgin Islands, Smith Lab). Study default `S1_historical`.
- Related tools and models: Oceankind_CV (github.com/laurenkolinger/oceankind_CV, Lauren K Olinger); SAM3 (Meta AI, Segment Anything with Concepts, github.com/facebookresearch/sam3); YOLO11 (Ultralytics).
- Restart script: `scripts/restart_reef_point_seg.sh`. Test aggregator: `scripts/run_all_tests.sh`. Platform bring-up: `vicarius/_DOCS/START_HERE.md`.
- Data-dictionary descriptors (19), under `vicarius/_METADATA/dictionary/datasets/`: `reef_point_seg_all_points`, `reef_point_seg_master_codes`, `reef_point_seg_recoded_all_points`, `reef_point_seg_remap_log`, `reef_point_seg_selected_frames`, `reef_point_seg_reserve_frames`, `reef_point_seg_sam_click_prompts`, `reef_point_seg_export_manifest`, `reef_point_seg_yolo_training_set`, `reef_point_seg_trained_model`, `reef_point_seg_training_rounds`, `reef_point_seg_evaluation_report`, `reef_point_seg_evaluation_metrics`, `reef_point_seg_inference_manifest`, `reef_point_seg_inference_predictions`, `reef_point_seg_split_manifest`, `reef_point_seg_loop_round_manifest`, `reef_point_seg_expert_review_bundle`, `reef_point_seg_expert_id_library`.
- Pipeline orchestration and VICARIUS integration: Lauren K Olinger.
</content>
</invoke>
