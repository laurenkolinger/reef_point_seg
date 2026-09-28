# Reef Point Seg: pipeline overview

One page on what this system is, what runs where, and which document to read next.

Last updated: 2026-09-05

Repo root: `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo`

## What it is

Reef Point Seg turns two decades of TCRMP coral point-count records into a trained YOLO11 instance-segmentation model. TCRMP surveys record benthos as points: a survey frame, twenty labelled points on it, and a species code at each point. A point says which pixel is coral. It does not say where the colony starts and stops. This module supplies the outline: it reads each point label, verifies the point against the frame, prompts SAM3 with that point to propose a colony outline, has a person accept or fix the outline, and assembles the accepted masks into a YOLO training set.

Everything is local. The photographs stay on this machine's NVMe, the models train on its two RTX 5090s, and the annotations live in per-project directories on disk.

The full module manual, including inputs, outputs, data-dictionary bindings, environment build, and troubleshooting, is `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/README.md`. The operator-facing user guide is `/mnt/rip/vicarius_drive/vicarius/_DOCS/modules/reef_point_seg.md`.

## Two data eras

How the twenty points per frame were recorded changed in 2020, and harmonizing the two shapes is a real job the pipeline does.

| Era | How points were stored | How the pipeline recovers them |
| --- | --- | --- |
| Pre-2020 (CPCe) | A `.cpc` text file per image with the pixel coordinates of all twenty points and the analyst's species code for each | Read the coordinates directly from the static `cpc_all` index. No computer vision needed. |
| 2020 and later (OCR) | Points baked into the image as red letters A to T with a crosshair; species codes only in an Excel workbook | Run EasyOCR on the GPU over the annotated frame, detect the letters and crosshairs, and pair them up. |

## The seven display steps

The orchestrator sidebar shows seven numbered steps. Internal step names and output directory names never change, so the display number and the on-disk name differ for the later steps. Both are given here.

| Display | Sidebar name | What it does | On disk |
| --- | --- | --- | --- |
| 1 | Fetch Points | Link or build `all_points.csv` (one row per point annotation) and `master_codes.csv` (the label lookup) | `step1_makeAllPoints/` |
| 2 | Fetch Labels | Rename, merge and exclude species codes, with an auditable remap log | `step2_recodeSpecies/` |
| 3 | Choose Images | Pick frames balanced across years and across sites, subject to a per-site cap | `step3_chooseImages/` |
| 4 | Place + Segment | The combined annotator: arm a species, click, SAM3 segments, accept or fix, export YOLO labels | `step4_routeChosenImages/` for routing, `step4test_combinedAnnotate/` for the export |
| 5 | Train Model | Split into train, valid and test (recording the seed, the fractions and a split fingerprint), then train YOLO11 as a segmenter or a detector; a runs card lists every run the module can see, with Evaluate, Promote and Fine-tune per row | `step6_trainModel/` |
| 6 | Evaluate Model | One Ultralytics validation pass; `metrics.json`, `report.md`, `report.html` and a PDF printed from it by headless Chrome; a rounds-ledger row with gates judged in the run's own metric family; a report card in the window with the per-species table, curves, confusion matrices and samples | `step7_evaluateModel/` |
| 7 | Model Inference | Run the trained weights over a chosen image source on a chosen device; overlays with a labelled chip per detection, `predictions.json`, `manifest.json`, and a gallery in the window with Yes, No and Skip verdicts | `step8_inference/` |

Three tiles sit outside the numbered chain:

- **Refine (model loop)** seeds a chosen inference batch's predictions back into the annotator as pending model masks, so a reviewer corrects the model instead of drawing from scratch. The validation and test splits are frozen by transect (5 is validation, 6 is test) and the seeder is fail-closed against seeding a holdout transect, so round-over-round comparison stays honest.
- **Edit Masks** reopens a project's existing export to relabel, delete, redraw or add a mask. It never creates a fresh export.
- **Expert Review I/O** is disabled as of 2026-07-09; the tile documents its own re-enable path.

## The four cross-project pages

These live beside the pipeline rather than in it. Each is a blueprint mounted inside the orchestrator on port 5050, reachable from the orchestrator sidebar's Tools cluster and from the VICARIUS launcher's DASHBOARDS row.

| Page | Route | What it is for |
| --- | --- | --- |
| Coverage Matrix | `/matrix` | Every image against every label across every project: what has been reviewed, what was found, and by whom. Three tabs: MATRIX, STATS, TRAINING SETS. |
| Label Manager | `/labels` | The canonical species vocabulary: add labels, edit names and categories, compose a remap. |
| Annotate Images | `/annotate` | Annotate any folder of photos with SAM3 outside any project. Sessions live in `inprocess/_standalone/`. |
| Bounding Boxes | `/boxes` | Ingest external box datasets, convert boxes to outlines with SAM3, and hand a dataset to the main window as a training project (the Train in VEGA button). Never enters the coverage matrix. |

Training, evaluation and inference have no page of their own. Since 2026-09-05 they happen only in the main window's Steps 5 to 7, inside a project; the project-free training page that shipped on 2026-09-02 was retired that day, and `scripts/_modellab/labruns.py` stays as the run library the orchestrator imports. The decision is recorded in `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/docs/adr/0002-one-training-surface-the-vega-main-window.md`; the rule that two runs are compared only on the same split and the same metric family is `docs/adr/0003-runs-compare-only-on-the-same-split-and-metric-family.md` in the same folder.

## Ports

| Port | What listens |
| --- | --- |
| 5090 | The VICARIUS platform UI, which launches this module. Not owned by this module. |
| 5050 | The Reef Point Seg orchestrator, and every cross-project page mounted inside it. |
| 5055 | Step 2 recode sub-app. |
| 5065 | Step 4 point-routing sub-app. |
| 5070 | Legacy segment sub-app. |
| 5075 | Expert Review I/O. |
| 5080 | Place + Segment combined annotator (preferred port). |
| 5083 | Edit Masks (preferred port). It moved off 5085 because vicariusLLM autostarts there. |
| 5096 | Annotate Images standalone. Deliberately outside the 5080 to 5089 sweep so a standalone session survives an orchestrator restart. |

Do not bind 5085. It belongs to vicariusLLM. Preferred ports are not guarantees: when the preferred port is taken the launcher falls back to an arbitrary free port, so read the live port from the orchestrator rather than assuming one.

## The AI stack

- **SAM3** (Meta AI, Segment Anything with Concepts) turns each verified point into a colony outline. Click prompts, box prompts and exemplar scans are all used. It runs from the unified environment at `env/bin/python`; `config/pipeline.yaml` puts the tracker on `cuda:1` and the exemplar model on `cuda:0`, so segmentation expects two CUDA GPUs.
- **YOLO11-seg** (Ultralytics, vendored editable at `scripts/TCRMPtrain_oceankindCV/ultralytics_src/`) is the production model trained on the SAM3-labelled set. SAM3 is the one-time labelling pass; YOLO11 is what runs over tens of thousands of frames afterwards.
- **EasyOCR** reads the red letter stamps on 2020-and-later frames, GPU-accelerated, preloaded once at sub-app startup.

Base weights live under `supporting_data/model_weights/`. Segmentation weights cover n through x; detection weights `yolo11n.pt`, `yolo11s.pt` and `yolo11m.pt` are staged for the Detector training task.
- **Headless Chrome** (`/usr/bin/google-chrome`, or the binary named by `TCRMP_CHROME_BINARY`) prints the Step 6 evaluation PDF from the evaluation folder's `report.html`; the evaluator checks the output's existence, size and first bytes because Chrome exits 0 on failure.

## The machine

One Linux workstation, no cluster and no cloud: two NVIDIA RTX 5090 (32 GB each), two AMD EPYC 9124, 256 GB RAM, Ubuntu 24.04, primary NVMe at `/mnt/rip`.

## Configuration

Three layers, each with its own lifetime:

1. **Pipeline defaults.** `config/pipeline.yaml` is the source of truth for paths, ports, interpreters and per-step defaults; `scripts/pipeline_orchestrator/orchestrator_config.py` reads it. Changing a default here changes it for every new project.
2. **Per-project state.** Each project is a directory `inprocess/run_{YY-MM-DD}_{INITIALS}` holding `project.json` plus its `step*` outputs. Defaults are deep-copied in at creation; after that `project.json` is the single source of truth for that project, written atomically. This is what makes a project reproducible, resumable and auditable.
3. **Sub-app defaults.** Each sub-tool keeps its own `src/config.py` for standalone use; the orchestrator overrides those with environment variables when it launches one.

Training presets live at `config/presets/`.

## Running it

```
cd /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo
bash scripts/restart_reef_point_seg.sh
```

The script frees the orchestrator and sub-app ports, starts the Flask app, blocks until the current page is served, and self-tests live markers before reporting success. Then open `http://localhost:5050/`.

Never call `scripts/run_pipeline.sh` directly, and do not use `vicarius restart reef_point_seg`: it routes through that script, whose `lsof` sweep hangs on this box.

The normal path is the VICARIUS launcher on port 5090: open the Reef Point Seg module page, create or open a project, and the orchestrator boots with that project loaded.

## Testing

House harness, no pytest. Every suite is a plain Python or Node script run by the aggregator:

```
cd /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo
bash scripts/run_all_tests.sh
```

The `SUITES` array listed 129 suites at 2026-09-05 15:54 AST (the training fold-in of that day registers its new suites there; its baseline before any change was 126 of 126 green). A clean run ends with `ALL SUITES GREEN` and exits 0. The aggregator redirects the mask registry and clip index to throwaway directories so no suite can write live data, and an EXIT trap sweeps the fixture directories a run creates under `/tmp`. Any orchestrator test that touches a launch route must stub `subprocess.run`, because those routes sweep the annotator ports.

## Where to read next

| Document | What it covers |
| --- | --- |
| `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/README.md` | The module manual: inputs, parameters, every output and its data-dictionary descriptor, internals, environment build, troubleshooting. |
| `/mnt/rip/vicarius_drive/vicarius/_DOCS/modules/reef_point_seg.md` | The user guide: quick start, the coverage matrix, the multi-class training walkthrough, and a page-to-place linkage map. |
| `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/changelog.md` | What changed and why, newest first. |
| `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/docs/adr/README.md` | The module's decision records: the non-TCRMP scope rule, one training surface, and when two runs may be compared. |
| `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/CONTEXT.md` | The module glossary: the words the orchestrator, its pages and the training steps share. |
| `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/docs/VICARIUS_INTEGRATION.md` | How this module binds to the VICARIUS platform. |
| `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/docs/GLOBAL_LABEL_TRACKER.md` | The label vocabulary tracker. |
| `/mnt/rip/vicarius_drive/vicarius/_DOCS/START_HERE.md` | Bringing the VICARIUS platform itself up. |

## Sources

- Meta SAM 3 announcement, 2025-11-19: https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/
- SAM 3 paper, "Segment Anything with Concepts": https://arxiv.org/abs/2511.16719
- SAM 3 reference implementation: https://github.com/facebookresearch/sam3
- Ultralytics SAM 3 documentation: https://docs.ultralytics.com/models/sam-3/
- oceankind_CV, the YOLO training pipeline this module wraps: https://github.com/laurenkolinger/oceankind_CV
