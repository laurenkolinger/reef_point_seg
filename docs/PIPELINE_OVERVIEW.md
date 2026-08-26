# TCRMP CVR-CLIP Pipeline Orchestrator — Comprehensive Overview

*A non-specialist walkthrough of what this system is, why it exists, how it works, and where it's going.*

Last updated: 2026-05-30

---

## 1. The short version

This repository is an **eight-step, browser-driven pipeline** that turns two decades of Caribbean coral reef monitoring photographs into a **trained, production-ready AI model** that can recognize and outline individual corals (and other reef organisms) in new images — automatically.

The pipeline is coordinated by a single **Pipeline Orchestrator** — a small Flask web app that runs on our GPU workstation, presents a friendly sidebar in the browser, and launches the right sub-tool for each step, passing outputs forward so the user never has to hand-wire file paths between tools.

Everything is local. No cloud services, no external uploads. Source photographs stay on our NVMe storage; AI models train on our GPUs; annotations live in per-project folders we control.

Below: what each step does, what the source data means, the AI technology we use (SAM3 from Meta; YOLO11 from Ultralytics), the workstation the system runs on, how we maintain the configs and parameters, where we are in training, and how this plugs into the broader **VICARIUS** research platform ([/mnt/rip/vicarius_drive/vicarius](../../vicarius)).

---

## 2. The source data — in plain English

### 2.1 TCRMP: Twenty years of divers with cameras

The **Territorial Coral Reef Monitoring Program (TCRMP)**, run out of the University of the Virgin Islands, has surveyed the same reef sites in the US Virgin Islands since 2001. Divers run a measuring tape across the reef bottom (a **transect**), then either photograph along it or film a video clip for each transect. From each transect you get a sequence of **frames** — still images that sample a strip of reef floor.

For decades the standard analysis method was simple: take a handful of frames, drop **20 random points** on each frame, and have a trained analyst identify what each point landed on — a coral, an algae, a sponge, bare rock, sand, etc. Multiply by dozens of transects × multiple sites × many years and you get hundreds of thousands of point-by-point **percent-cover observations** of Caribbean reefs. That's the **Coral Visual Record (CVR)**, and it's what lives in the 839 Excel spreadsheets this pipeline starts from.

### 2.2 CPCe and OCR — two eras, two data shapes

There is a significant wrinkle: **how** those 20 points per frame were recorded changed in 2020.

- **Pre-2020 (2009–2019) — CPCe era.** Analysts used a Windows program called **Coral Point Count with Excel extensions (CPCe)**. CPCe saves a small text file (`.cpc`) per image that contains the **(x, y) pixel coordinates** of all 20 random points plus the species code the analyst entered for each one. We have the actual click coordinates on disk — no computer-vision work needed to recover them.
- **2020–present — OCR era.** CPCe was retired. Analysts moved to a workflow where the points are baked into the image as red letter annotations (A, B, C, …, T) with a small red crosshair under each letter. The species codes live only in an Excel workbook; the point coordinates live only as pixels in an image. To get structured coordinates back, we have to *read* the image — detect the red letters, detect the crosshairs, match each letter to its crosshair, and OCR the letter (A–T). That's the **OCR route**.

So a fundamental job of this pipeline is **harmonizing these two data shapes** — CPCe files for old images, OCR-on-image for newer ones — into a single, uniform format that can feed the same AI segmentation model.

### 2.3 What "segmentation" actually means

A percent-cover point is just a pixel location labeled with a species. An **instance segmentation mask** is a full outline — every pixel belonging to one specific coral colony, traced along its boundary. Point data tells us *that this pixel is coral*; segmentation tells us *here is the full shape of this individual colony*. Shape lets us measure colony size, track colonies across years, detect bleaching and disease, and do everything else that a pixel-count can't.

The pipeline uses the existing 20 point-and-species annotations as **free training prompts** for a segmentation model — the point says "coral of species X is here", the AI expands that point into the full colony outline, a human reviews the outline, and we end up with a rich segmentation-labeled training set without paying for new annotations.

---

## 3. The eight pipeline steps

Each step runs as its own sub-tool with its own Python environment (conda / venv). The orchestrator launches them, watches them, and wires their inputs and outputs together.

| # | Step | What it does | Sub-tool dir |
|---|------|--------------|--------------|
| 1 | **Make All Points** | Parse 839 CVR Excel files → one unified `all_points.csv` (~1.6–2.3M rows) and `master_codes.csv` | [TCRMPcvr_makeAllPoints/](TCRMPcvr_makeAllPoints/) |
| 2 | **Recode Species** | Web UI to remap species codes (e.g. `MFAV → OFAV` after taxonomy revisions) and produce a consistent recoded dataset | [TCRMPcvr_recodeSpecies/](TCRMPcvr_recodeSpecies/) |
| 3 | **Choose Images** | Balanced selection of ~4,500 frames across years, sites, transects, and target species (≥1,000 instances per target) | [TCRMPcvr_chooseImages/](TCRMPcvr_chooseImages/) |
| 4 | **Route & Verify Points** | For each selected frame, load CPCe coords (pre-2020) OR run GPU OCR (2020+), then review/edit point coords in a browser UI | [TCRMPclip_placePoints/](TCRMPclip_placePoints/) |
| 5 | **SAM3 Segmentation** | Turn every point-click into a per-organism mask using Meta's SAM3 model, review/refine each mask in a browser UI, export to YOLO format | [TCRMPclip_segmentImages/](TCRMPclip_segmentImages/) |
| 6 | **Train Model** | Stratified train/val/test split + Ultralytics YOLO11 segmentation training on local GPUs | [TCRMPtrain_oceankindCV/](TCRMPtrain_oceankindCV/) |
| 7 | **Evaluate Model** | Run the trained model on the held-out test split, dump metrics + a PDF report with example predictions | same dir |
| 8 | **Inference** | Apply the trained model to new/unused frames; save masks, overlays, optional crops | same dir |

The user never calls these directly. They click through the orchestrator sidebar and the tools launch themselves.

### 3.1 Step 1 — Make All Points
Parses three generations of Excel formats spanning 2001–2025:
- **Legacy (2001–2016)**: per-transect sheets (`01TRAN`–`06TRAN`) with interspersed metadata rows and a hand-rolled species lookup column.
- **Intermediate (2017–2019)**: standard `RAW_DATA` / `TO_MASTER` sheets.
- **Modern (2020–2025)**: formal `ReadMe` / `Codes` / `DataValidation` / `RawData` / `ToMaster` / `Summary` workbook.

Output is one giant tidy CSV (`all_points.csv`) with columns `date, year, site, transect, frame, point_label (A–T), species_code, species_name, category` plus a lookup table `master_codes.csv`. Any weirdness in the source data — corrupted `.xls` files, typoed site codes, date offsets, numeric-instead-of-alphabetical species codes in 2009 — is handled or logged here.

### 3.2 Step 2 — Recode Species
Coral taxonomy has moved: *Montastraea faveolata* → *Orbicella faveolata*, plus many smaller reshufflings, plus manual analyst-to-analyst code drift over 20 years. A small Flask web app lets an expert sort the master codes by frequency, merge/rename them, and exclude noisy ones. The recode is **non-destructive** and timestamped: every change lands in `remap_log_{YYYYMMDD}.json` so the recode history is auditable and reversible.

### 3.3 Step 3 — Choose Images
We want to train on a **balanced** dataset — not whatever happens to dominate the archive. Given a set of target labels (species or categories), the step allocates roughly equal counts of frames per year (2014–present), greedily picks frames that contribute the most needed species, and emits a manifest of ~4,500 selected frames along with diagnostic plots of the distribution across years, sites, and species.

### 3.4 Step 4 — Route & Verify Points ("Verify Points")
Each selected frame follows one of two routes:
- **CPC route (pre-2020)**: read the matching `.cpc` file, scale the (x,y) from canvas space to image pixels, re-enrich species codes with the recoded taxonomy.
- **OCR route (2020+)**: run **EasyOCR on GPU** over the `_pts.jpg` image to detect the red letters A–T and the crosshairs under them, greedily assign labels by confidence.

All frames — regardless of route — then show up in a single browser review UI where an expert flips through images, verifies/drags point positions, adds aux observations, and scraps bad frames. Key affordances added in April 2026:
- **Hold-number + click quick-add** for each target label (`1` = PA, `2` = OFRA, …). The held button lights up magenta and clicks drop a labeled point instantly.
- **Hover magnifier (loupe)** with crosshair and label overlay.
- **Click-and-drag pan**, scroll-zoom, arrow-nudge, scrap-frame, reference mode, aux observations.
- **Auto-trigger SAM3** on each batch export so step 5 runs in the background.

### 3.5 Step 5 — SAM3 Segmentation
This is the AI heart of the pipeline. Each verified (x, y) point becomes a **click prompt** for **SAM3** (Meta's Segment Anything Model 3, released November 19, 2025). SAM3 produces a pixel-accurate outline of whatever organism the click landed on. The server-side pipeline:
1. Loads SAM3 on two GPUs (tracker on `cuda:1`, exemplar on `cuda:0`).
2. For each frame, runs all its point-clicks through SAM3, producing a set of candidate masks.
3. Resolves overlapping masks (configurable strategy — default: larger wins).
4. Merges nearby same-species masks.
5. Simplifies polygons (configurable epsilon).
6. Saves per-frame state incrementally, so crashes never lose more than one frame.

Humans then review in a browser UI that mirrors step 4's ergonomics — quick-add, loupe, scrap, refine. New capabilities added in April 2026:
- **Click-to-refine** (click a mask to auto-enter refine mode; next click is a positive SAM3 click).
- **Draw mode (D)** — freehand edit: release the cursor inside the mask to carve a chunk out, outside to add a chunk on. Boolean ops via OpenCV's `fillPoly` + numpy.
- **Clean-boundary rule** — every refine / add op subtracts the union of existing masks on that frame so neighbors don't get "eaten" by re-segmentations.
- **Accept / Reject / Space** auto-advance to the next mask or the next frame.

Output is a **YOLO segmentation training set**: `all_images/` symlinks, `all_labels/*.txt` (class_id + polygon vertices), `data.yaml`, and a persistent `class_map.json`.

#### Expert-review round-trip (REVIEW flag + Add Expert IDs)

Masks the reviewer cannot confidently identify are flagged **REVIEW**. At each step-5 export, the `scripts/_reefreview/` helpers handle them:

1. Each REVIEW mask gets a stable UID.
2. If a past EXPERT mask in the cross-project library overlaps it on the same image, that expert ID is inherited in place and the mask is not re-queued.
3. Otherwise a closeup plus full frame is rendered, added to a GitHub-Pages review repo (`review_dir` `/mnt/tear/REVIEW_reefpointseg`, pushed to `reefpointseg-review.git`), and recorded in the permanent `inprocess/_expert_id_library/` as awaiting an expert ID.

An outside taxonomic expert reviews on the GitHub-Pages site and returns a CSV (`uid,code,confidence,reviewer,project_id`; as of 2026-07-09 the confidence column is always blank from the viewer, and the importer accepts legacy values). The **Expert Review I/O** panel (the `scripts/_expertids/` blueprint embedded in the orchestrator; its sidebar tile is temporarily disabled as of 2026-07-09) folds those IDs back in as rolling per-reviewer tentative reviews; the operator's ACCEPT in the consensus table is what stamps `expert_id`, relabels the mask, and removes the item from the queue. It also auto-relabels any remaining TO-REVIEW mask whose footprint is >50% covered by a now-EXPERT mask on the same image, then pushes the shrunken queue back to the repo. It is idempotent and never blocks steps 6-8.

### 3.6 Step 6 — Train Model
A wrapper around [oceankind_CV](https://github.com/laurenkolinger/oceankind_CV) (Lauren's existing CVAT-to-YOLO training pipeline, already adapted for the Oceankind collaborator network — UVI, QUT, Berkeley, Point Blue).

The orchestrator:
1. Runs a **stratified train/val/test split** (`bal_train_test_split.py`) honoring `min_samples` per class so every rare coral is present in all three splits.
2. Launches **Ultralytics YOLO11 segmentation** training (model family: `yolo11{n,s,m,l,x}-seg.pt`) with the full parameter surface — epochs, image size, batch, LR schedule, loss weights, augmentation pipeline — exposed in the orchestrator UI.
3. Streams `stdout` (Ultralytics' per-epoch metrics) live into the step 6 log panel so progress is visible without TensorBoard.

Runs land in `{project}/step6_trainModel/runs/<run_name>/` with `best.pt`, `last.pt`, and the usual Ultralytics artifacts.

### 3.7 Step 7 — Evaluate Model
Runs `model.val()` against the held-out split (default: `test`), renders a PDF report with metrics (mAP@0.5, mAP@0.5–0.95, precision, recall per class) plus a configurable number of preview images showing predicted vs ground-truth masks side by side.

### 3.8 Step 8 — Inference
Apply `best.pt` to:
- a chosen directory, OR
- a random/systematic sample of TCRMP frames **not** in step 3's selection (gets you a quick read on out-of-training performance), OR
- the full TCRMP clip set.

Outputs: overlay renders, optional crops, JSON detection records.

---

## 4. The AI stack, explained

### 4.1 SAM3 — "Segment Anything with Concepts" (Meta AI, Nov 2025)

**The one-paragraph explanation.** SAM3 is a foundation model that takes an image and a **prompt** — a click, a box, or a short phrase like "yellow school bus" — and returns a pixel-accurate outline of whatever you pointed at. It's the third generation of Meta's Segment Anything family. It was released November 19, 2025 ([facebookresearch/sam3](https://github.com/facebookresearch/sam3), [arXiv:2511.16719](https://arxiv.org/abs/2511.16719)) and represents roughly a **2× improvement over SAM 2** on Meta's new promptable concept segmentation benchmark (SA-Co).

**Why it matters for us.** Our point-count legacy is a gold mine of **pre-labeled clicks**. Each of the ~1.6M CVR points is effectively "a paid coral taxonomist already told us there is a *Porites astreoides* at exactly this pixel". SAM3 converts every one of those clicks into a full-organism outline with no additional human labeling cost. That's how we bootstrap a high-quality segmentation training set from an existing percent-cover dataset.

**What SAM3 brings on top of SAM2:**
- **Concept prompts** — natural-language phrases that match *all* instances of a concept in the image (useful for later: "segment all sponges", "segment all tape", etc.).
- **Exemplar prompts** — draw a box around one coral, get masks on all similar corals in the frame. Already hooked up in our step 5 review UI ("Exemplar Scan" button) to help fill in missed organisms.
- **Image + video unified** — tracks instances across video frames with consistent IDs. Relevant for Study S3 (individual coral tracking).
- **Multi-GPU** — we split tracker and exemplar encoders across our two RTX 5090s.

Further reading: [Meta's announcement](https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/), [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3/), [Roboflow explainer](https://blog.roboflow.com/what-is-sam3/).

### 4.2 YOLO11 via Ultralytics

Once we have a SAM3-labeled training set, we train a **smaller, faster** production model — **YOLO11-seg** — that can run in near real-time on new TCRMP frames without needing to round-trip through a foundation model per prediction.

**Why a two-stage approach** (SAM3 → YOLO)?
- SAM3 is huge, slow per image, and needs a good prompt per instance. Perfect for a **one-time labeling pass** over ~4,500 frames with human review.
- YOLO11-seg is small, extremely fast (milliseconds per image on our GPUs), and runs end-to-end with no prompt needed. Perfect for **production inference** on tens of thousands of frames and future video.

**Ultralytics** is the company/framework that maintains the YOLO lineage today. They publish a clean Python API (`from ultralytics import YOLO`), full training/validation/export tooling, and a well-understood hyperparameter surface. We vendor their source at `TCRMPtrain_oceankindCV/ultralytics_src/` (editable install) so we can tweak if needed while staying on released versions.

**Model sizes available** in the orchestrator: `yolo11{n,s,m,l,x}-seg.pt` (nano → extra-large). Default: `yolo11m-seg.pt` — a middle-of-the-road choice that trains in hours on our workstation and has delivered good coral segmentation accuracy in the Oceankind collaborator experiments.

**Hyperparameter coverage in the UI:** run name, base model, epochs (default 500), image size (default 512), batch (auto), patience, optimizer, seed, cosine LR schedule toggle, mosaic close epoch, full LR schedule (lr0, lrf, momentum, weight decay, warmup), loss weights (box, cls, dfl, label smoothing), full augmentation pipeline (HSV, geometric, mosaic/mixup/cutmix/copy-paste/erasing, auto-augment policy). Defaults come from `oceankind_CV/training/train_segment.py`; a tuned preset for underwater coral imagery ships at [pipeline_orchestrator/presets/tuned_v1_regularized.yaml](pipeline_orchestrator/presets/tuned_v1_regularized.yaml).

### 4.3 EasyOCR

Used only in step 4 (the OCR route) to read the red A–T letter stamps on 2020+ `_pts` images. GPU-accelerated; preloaded once at sub-app startup. Benchmarked at ~0.6 s per image (4.8× faster than CPU Tesseract, 50% higher letter recall at 97.5% confidence).

---

## 5. The server workstation

Everything runs on one Linux workstation (hostname: `vicar-mainframe`). No cluster, no cloud.

| Component | Spec |
|---|---|
| GPU | **2× NVIDIA RTX 5090** (32 GB VRAM each, SM 12.0 / Blackwell, driver 580.65.06, CUDA 12.8/13.0) |
| CPU | **2× AMD EPYC 9124** (16-core each, 64 threads total) |
| RAM | **256 GB** |
| OS | Ubuntu 24.04, Linux 6.8.0 |
| Primary storage | NVMe at `/mnt/rip` — 14 TB, currently 7.7 TB used |
| Data backup | Synology NAS (per VICARIUS backup policy) |

Why the two GPUs matter: SAM3's concept-segmentation head and its tracking head can be split across devices (`cuda:0` exemplar, `cuda:1` tracker), which is the default in [orchestrator_config.py](pipeline_orchestrator/orchestrator_config.py#L84-L99) and keeps step 5 responsive while YOLO training runs on the same box.

---

## 6. The lightweight UI

The orchestrator is a **Flask app** (~2000 LOC backend + a single-page HTML/CSS/JS frontend at [pipeline_orchestrator/static/](pipeline_orchestrator/static/) and [pipeline_orchestrator/templates/index.html](pipeline_orchestrator/templates/index.html)). Design goals:

- **One URL, one window.** Launching the module from VICARIUS (`vicarius restart reef_point_seg` or the UI launcher) starts the orchestrator on `http://localhost:5050` and opens a browser. The user never sees a terminal. Do not call `run_pipeline.sh` directly.
- **Sidebar of steps.** Each step has its own panel with a config form, Start / Open UI / Reset buttons, and a live log.
- **Sub-apps are "folded in".** Steps 4 and 5 launch their own Flask apps on separate ports (5065, 5070), but the user sees the orchestrator-styled loading veil, not the raw sub-app startup form. Sub-apps detect they are orchestrated (via env vars we pass) and auto-advance past their own forms.
- **Live progress.** SAM3 status polled from the background driver thread; YOLO training logs streamed line-by-line.
- **State is a file.** Each live run is a directory named `run_*` under `inprocess/` (at the module root, beside `github_repo/`) with a `project.json` describing status, config, and output paths per step. Back-compat for older projects is built into [project_manager.py `load_project`](pipeline_orchestrator/project_manager.py#L85-L143) — pre-step-6 projects grow step 6/7/8 entries on open.
- **Stale-process cleanup at launch.** `run_pipeline.sh` sweeps zombie orchestrators and sub-apps on startup to prevent session leaks between demo-day runs.

Everything is kept intentionally **simple** — plain Flask, plain JavaScript, no React, no build step, no SPA framework. The surface area you'd need to teach a new researcher is one HTML file, one JS file, one CSS file.

---

## 7. Maintenance of configs and parameters

Three layers of configuration, each with its own lifetime and ownership:

### 7.1 Pipeline-level defaults — [pipeline_orchestrator/orchestrator_config.py](pipeline_orchestrator/orchestrator_config.py)
One Python file that names, for each of the 8 steps:
- the Python interpreter path (one conda/venv per step, explicit)
- the entry point script
- the working directory
- the default port
- the default config values (all hyperparameters, all toggles)

Changing a default **here** changes it for all new projects.

### 7.2 Per-project config — `inprocess/run_<name>_<YYYYMMDD>_<uuid>/project.json`
On project creation, the defaults are **deep-copied** into `project.json`. After that, the file is the single source of truth for *this* project's configuration. Every UI form field binds to a field in `project.json`. Every step config change does an atomic `tmp → rename` write. This means:
- Projects are **reproducible.** Zip the project dir, send it to a collaborator, they can re-open it and see exactly what parameters you used.
- Projects are **resumable.** Close the orchestrator in the middle of step 5, come back tomorrow, click "Resume" — state is recovered from disk.
- Projects are **auditable.** `project.json` + each step's on-disk outputs (CSV, JSON, logs) together form a complete audit trail.

### 7.3 Step-level sub-app configs
Each sub-tool also has its own `src/config.py` with CLI-friendly defaults for running it standalone. When the orchestrator launches a sub-app, it passes env vars that **override** those defaults. This is how we keep sub-apps usable independently (for one-off runs) without forcing duplicate maintenance.

### 7.4 Rolling change log
[NOTES_pipeline_improvements.md](NOTES_pipeline_improvements.md) is a working journal of known issues, triaged priorities, and a running `[DONE]` log of what landed and when. As of 2026-04-21, the `[DONE]` entries for April 16–18 capture the step 4 / step 5 UX overhaul, SAM3 auto-start, step 6 wiring, and the current freehand draw-edit mode.

### 7.5 Presets
[pipeline_orchestrator/presets/](pipeline_orchestrator/presets/) holds YAML presets of hyperparameter bundles that we've settled on. `tuned_v1_regularized.yaml` is the current regularized-training baseline for underwater coral imagery.

### 7.6 Conda/venv environments
Each sub-tool owns a `./env/` created by its `setup_env.sh`. This isolates dependency conflicts (step 5's SAM3 + PyTorch 2.11 nightly vs step 6's PyTorch 2.6 + Ultralytics 8.4.38) and makes each step reproducible by re-running its setup script. SAM3's env is the shared `sam3reef` conda env at `/home/bizon/anaconda3/envs/sam3reef` rather than per-step because SAM3's install is both large and picky about CUDA versions.

---

## 8. Model training — what's done so far

**Training target:** a YOLO11-seg model that segments the current top coral species — **OFRA, PA, OA, OFAV, AL, MC, AA** — plus category-level labels for non-coral organisms where we have enough data (algae, sponges).

**Training data:**
- **Source:** ~4,500 frames selected in step 3, balanced across 2014–2025 years and TCRMP sites.
- **Points per frame:** 20 (A–T), each with a verified (x, y) and a recoded species code.
- **Masks per frame:** one per point (SAM3-derived), human-reviewed.
- **Split:** stratified 70% train / 20% valid / 10% test, `min_samples=10` per class.

**Training infrastructure:**
- Ultralytics YOLO11-seg, editable install at [TCRMPtrain_oceankindCV/ultralytics_src/](TCRMPtrain_oceankindCV/ultralytics_src/).
- Project-local env at [TCRMPtrain_oceankindCV/env/](TCRMPtrain_oceankindCV/env/) with `torch 2.6.0+cu124`, `ultralytics 8.4.38`, and friends.
- Runs land in `{project}/step6_trainModel/runs/<run_name>/`.

**Status (2026-04-21):**
- Pipeline end-to-end works — projects have been taken from raw Excel (step 1) to trained YOLO weights (step 6) on the existing workstation.
- Step 7 (evaluation) and step 8 (inference on unused frames) are wired in. Early inference results against held-out TCRMP frames are the current focus.
- The model is trained on **locally generated data** only — no public coral segmentation datasets mixed in. This is deliberate: TCRMP's imagery has its own illumination, water clarity, substrate, and taxonomic composition, and a model trained on TCRMP data generalizes to TCRMP data best. Public data can be blended in later as an augmentation, once we know baseline performance on our own distribution.

**What's pending:**
- Broader class coverage beyond the initial 7 target corals (sponges, octocorals, bleaching vs disease states).
- Hyperparameter sweep using the tuned preset + Ultralytics' built-in genetic tuner.
- Multi-year longitudinal evaluation — same sites re-surveyed annually — to validate that the model's outputs can drive percent-cover and colony-count trend analyses without human review.

---

## 9. Why it matters — benefits to our efforts

1. **Twenty years of institutional labor becomes structured training data.** Every CPCe click and every OCR annotation that already exists gets converted, for free, into segmentation masks. No new labeling budget required.
2. **Human review where it's worth it.** Experts never again type 20 species codes per image — they only review pre-populated SAM3 predictions, reject the bad ones, and nudge boundaries on the edge cases. That is the highest-leverage use of their time.
3. **One pipeline, many projects.** Swapping the target species list, the target organism category (corals → sponges → macroalgae), or the year range is a config change — not a new codebase.
4. **Full provenance.** Every project is one self-contained directory with `project.json`, step outputs, and logs. Every recode is timestamped. Every SAM3 mask edit is persisted. Reviewer B can audit what Reviewer A did.
5. **Local, air-gapped, reproducible.** No cloud drift, no vendor lock-in, no "that model got deprecated". Swap the SAM3 checkpoint and re-run? It's one config line.
6. **The same workstation trains and serves.** Inference on new frames (step 8) runs on the same GPUs that trained the model. No "we need to port this to a different environment" step.
7. **Extensible to the full VICARIUS mission.** The same pipeline, pointed at video frames from 3D surveys or AUV footage (see §10), trivially generalizes to the rest of the benthic community — not just corals.

---

## 10. Integration with VICARIUS — the longer plan

**VICARIUS** ([/mnt/rip/vicarius_drive/vicarius](../../vicarius)) is the broader data platform this pipeline plugs into. Its [README](../../vicarius/README.md) is worth reading end-to-end, but the short version:

> *Virgin Islands Center for Autonomous Research Integrated Undersea Survey* — a platform that integrates historical TCRMP monitoring data, 3D photogrammetric surveys, AUV imagery, and AI-powered classification under one enforced set of conventions (naming, metadata, provenance, logging).

### 10.1 What plugging in looks like

The current orchestrator repo will live as a **VICARIUS module** under `vicarius/modules/`. Specifically, each of the 8 sub-tools maps naturally to the template layout VICARIUS already uses:

```
vicarius/modules/reef_point_seg/
├── github_repo/       # this repo, versioned
├── inprocess/         # one folder per training/inference run
│   └── run_YYYYMMDD_<purpose>/
│       ├── analysis_params.yaml     # auto-populated (git SHA, user, timestamp)
│       ├── outputs/
│       └── logs/
└── misc/
```

Concretely, the integration work that will happen over the next weeks:

1. **Run initialization via VICARIUS.** `init_run.py` prompts for purpose/study/data context and stamps `analysis_params.yaml` with the current git commit of this repo, the operator, and the start date. The orchestrator launches under that run's umbrella so `project.json` and `analysis_params.yaml` sit side by side.
2. **Source images from VICARIUS `/raw/`.** Step 4's `CLIP_DIR` currently points at the Dropbox-synced clip share. Under VICARIUS it will point at `$VICARIUS_ROOT/raw/tcrmp_video/` (or a processed frame subdir). The immutability rule (Commandment I) means we read, never write, this directory.
3. **Models to VICARIUS `/models/`.** After step 6, `best.pt` gets versioned and copied to `$VICARIUS_ROOT/models/tcrmp_coral_seg/v{major}.{minor}.{patch}/` with a model card describing training set, metrics, and hyperparameters.
4. **Predictions to VICARIUS `/annotations/ai_predictions/`.** Step 8's inference outputs land in the shared annotation tree so downstream studies (S1 historical dynamics, S3 colony tracking, S4 non-coral, S5 community interactions) can consume them.
5. **Event logging.** Every start/stop/export hits the VICARIUS event stream via `vicarius note` and module-level `process_start` / `process_end` events — so there's one chronological story of the entire platform's activity (Commandment VIII).
6. **Shared controlled vocabularies.** The `master_codes_recoded.csv` from step 2 becomes a first-class VICARIUS vocabulary in `$VICARIUS_ROOT/_METADATA/library/`. One source of truth for species codes across all modules.

### 10.2 Why this integration matters

VICARIUS directly addresses the failure mode our field keeps hitting — lost institutional knowledge, unreproducible scripts, scattered data, unrecorded analytical decisions. Hooking this pipeline into VICARIUS means:
- A new researcher can walk up to the machine and, from `vicarius log show`, see every pipeline run this year and the purpose statement for each.
- The provenance of any model version is one SQLite query away: "which raw data, which recode log, which step 3 selection, which SAM3 checkpoint, what hyperparameters?"
- Adding sponges, octocorals, substrate classification, or brand-new AUV imagery is a module or a run — not a rewrite.

---

## 11. Long-term maintenance & documentation plan

The goal is that this system outlives its builders. The discipline is:

1. **One README per sub-tool**, kept in sync with the code. Already in place for all 8 tools; treat README changes as a required part of any code PR.
2. **This overview document** (`PIPELINE_OVERVIEW.md`) is refreshed whenever a step's responsibility changes materially — new step added, step removed, major sub-tool rewrite, major model swap.
3. **`NOTES_pipeline_improvements.md`** stays the living triage log. Every session's `[DONE]` entries form a de facto changelog; every unchecked item is visible backlog.
4. **Environments are scripted, not hand-built.** Every `setup_env.sh` must reproduce the env from a clean machine. If it breaks, fix the script — don't patch in place.
5. **Model cards for every trained model.** Each run under `step6_trainModel/runs/` ships with the full `project.json` + a one-page summary (training set size, class map, metrics, known limitations). These become the authoritative record when the model is copied into VICARIUS's `/models/` tree.
6. **Recodes are timestamped and audit-logged.** Never edit `all_points.csv` in place; always produce a new timestamped copy with a matching `remap_log_{YYYYMMDD}.json`.
7. **Source data is sacred.** Per VICARIUS Commandment I — never write into `/raw/`. All writes go into per-project `step*` directories.
8. **Documentation lives next to the code it describes.** No wiki, no external Notion page. If it's not in this repo (or VICARIUS's `_DOCS/`), it will eventually be lost.
9. **Test projects are cheap.** Before a major run, always create a small test project with 10–20 frames to validate the full chain end-to-end. This is how the step 4 / step 5 overhauls got shaken out in April 2026.
10. **On hiring a new operator.** Have them (a) read this document top-to-bottom, (b) read the VICARIUS README + Ten Commandments, (c) run one new project end-to-end on a subset of data, (d) trigger a retraining. Two days and they are productive.

---

## 12. Quick reference

| Thing | Where |
|---|---|
| Start the orchestrator | `vicarius restart reef_point_seg` or the UI launcher (opens `http://localhost:5050`) |
| Pipeline defaults | [pipeline_orchestrator/orchestrator_config.py](pipeline_orchestrator/orchestrator_config.py) |
| Per-project state | `inprocess/run_<name>_<date>_<uuid>/project.json` |
| Step directory names | [pipeline_orchestrator/project_manager.py:15-24](pipeline_orchestrator/project_manager.py#L15-L24) |
| Orchestrator backend | [pipeline_orchestrator/app.py](pipeline_orchestrator/app.py) |
| Orchestrator frontend | [pipeline_orchestrator/templates/index.html](pipeline_orchestrator/templates/index.html), [static/orchestrator.js](pipeline_orchestrator/static/orchestrator.js) |
| Running improvement log | [NOTES_pipeline_improvements.md](NOTES_pipeline_improvements.md) |
| YOLO training preset | [pipeline_orchestrator/presets/tuned_v1_regularized.yaml](pipeline_orchestrator/presets/tuned_v1_regularized.yaml) |
| VICARIUS platform | [/mnt/rip/vicarius_drive/vicarius/README.md](../../vicarius/README.md) |

---

## Sources

- [Meta SAM 3 announcement (Nov 19, 2025)](https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/)
- [SAM 3 paper — "Segment Anything with Concepts" (arXiv:2511.16719)](https://arxiv.org/abs/2511.16719)
- [SAM 3 reference implementation — facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- [Ultralytics SAM 3 documentation](https://docs.ultralytics.com/models/sam-3/)
- [Meta AI research blog — SAM 3.1 / real-time tracking](https://ai.meta.com/blog/segment-anything-model-3/)
- [Roboflow — What is SAM3?](https://blog.roboflow.com/what-is-sam3/)
- [Ultralytics — Exploring SAM 3](https://www.ultralytics.com/blog/exploring-sam-3-meta-ais-new-segment-anything-model)
- [Edge AI Vision — SAM3 for open-vocabulary segmentation](https://www.edge-ai-vision.com/2025/11/sam3-a-new-era-for-open%E2%80%91vocabulary-segmentation-and-edge-ai/)
- [oceankind_CV — YOLO training pipeline](https://github.com/laurenkolinger/oceankind_CV)
