# Pipeline Orchestrator — Improvement Backlog

Running notes from the 2026-04-16/17 review sessions. Nothing here is implemented unless marked **[DONE]**. Pick items off as we go.

## What's the user's next priority?

Quick triage of items NOT yet done, roughly ordered by how much they block real work:

| Rank | Item | Section | Why it matters |
|---|---|---|---|
| 1 | **"Aux observation" UX polish** — better species picker than `prompt()`; maybe a quick-add button bar for common aux codes; cursor marker in aux mode | D-follow | Feature is functional but the double-`prompt()` is crude |
| 2 | **Session isolation per sub-app launch** (rotate Flask secret per launch; delete legacy `output_11april26` fallbacks) | A | Still the stated root cause of the demo-day annotation leak. Not yet seen again, but unfixed. |
| 3 | **Category filter autoderive from target species** (step 3) | B | Still a footgun for non-coral targets |
| 4 | **Click-to-select target species from master_codes** (step 3) | B | Current free-text input is error-prone |
| 5 | **Rename "target species" → "target label"** (config + UI) | D | Wide-reaching rename; best done in one PR |
| 6 | **Step 3 "+5% backup frames"** | B | Pairs nicely with scrap-frame; not urgent |
| 7 | **In-window diagnostics after step 3 selection** | B | Nice-to-have; plots already exist on disk |
| 8 | **Move cpc_all to a permanent canonical location** (step 4 infra) | C | Still brittle; easy to forget to regenerate after a recode |
| 9 | **Long-term frame-manifest CSV** (cross-project tracking) | F-follow | Infrastructure for the multi-year project lifespan |
| 10 | **Lock down legacy `output_11april26` fallbacks** | A | Already works because orchestrator supplies env vars; removing fallback is cleanup |
| 11 | **Step 4 tile-mode** — view 2–4 frames side-by-side and annotate as one continuous bunch | Z | Ergonomic win for long review sessions; moderate scope (see § Z below) |

Ask which you want to tackle when you're back and I'll pick it up.

---

## Z. Step 4 tile-mode (proposed, not implemented 2026-04-18)

User asked: *"is there a way to just tile 2-4 images in the batch that I can zoom in and out of to annotate as one continuous bunch? like one on top one on bottom… that could feel like less work."*

### Design sketch
- **Layout toggle** in the review top-bar: `Single | 2-stack | 2×2`.
- Each tile owns its own canvas, its own `scale` / `offsetX` / `offsetY`, its own list of points (server data for that frame).
- The held quick-add label and loupe are shared (one mouse).
- `Click+drag`, `Scroll`, and all the number-tap placement keys route to the **hovered** tile.
- `Space` / `Shift+Space` still advance the batch, but by `tileCount` frames at a time.
- The per-frame timer strip shows one line per visible tile + the rolling average across all of them.
- Save policy: save all visible tiles before advancing; partial-save if the user hits `Export Batch`.

### Why it's not a 10-minute change
- `render()`, `fitImage()`, `screenToImage()`, `findPointAt()`, `updateLoupe()`, and the entire `mousedown/move/up` chain assume a single canvas + scale. Every one of them needs a "which tile?" parameter.
- `currentData` would become a list of size `tileCount`. `selectedPoint` would need a tile index too. Same for `sortedIndices`, `pendingAddIdx`, etc.
- Draw-edit / refine / merge interactions in step 5's equivalent code are similarly single-canvas-flavored.

### Scope
~1 day of focused work (≈ 400 LOC change in the step 4 template + careful retesting). Step 5 would need a parallel version (~another 0.5 day). Low priority compared to the items in the main triage table but a clear ergonomic win on long sessions — keep on the list.

### Middle-ground quick win (if we want something NOW)
Split the existing canvas horizontally into two halves and load the **current + next** frame side-by-side, with both sharing a single set of interaction handlers but separate pan/zoom state. User moves on with one `Space` press (advances by 2). This is still a decent chunk of work (~200 LOC) but avoids the harder parts of arbitrary grid layout. Ping me if you want this as an intermediate step.

---

## A. Fresh-start / state hygiene (the demo bug)

**Context:** On 2026-04-16 demo, annotations from prior runs leaked into a fresh project (step 4/5 showed stale state even though project dirs were empty). Root causes identified:
1. Stale orchestrator + zombie sub-apps from prior days still running, holding Flask sessions.
2. Sub-app default paths (`output_11april26`) act as fallback when env-var overrides are empty.
3. Flask session cookies persist across sub-app restarts on the same port.

### Done
- **[DONE]** Landing page always shows on page load; Resume banner appears only if server still has a project. ([index.html](pipeline_orchestrator/templates/index.html), [orchestrator.js](pipeline_orchestrator/static/orchestrator.js))
- **[DONE]** `run_pipeline.sh` sweeps stale orchestrator + sub-app processes (by entry-point path) and frees default ports before starting. ([run_pipeline.sh](run_pipeline.sh))

### Still to do
- **Session isolation per sub-app launch.** Rotate Flask `app.secret_key` on each launch (or set `SESSION_COOKIE_NAME` per project) so old browser cookies can't re-attach. Sub-apps: [routeChosenImages/src/app.py](TCRMPclip_routeChosenImages/src/app.py), [segmentImages/src/app.py](TCRMPclip_segmentImages/src/app.py).
- **Make env-var overrides required, not fallback.** When orchestrator launches a sub-app, set `TCRMP_ORCHESTRATED=1`; sub-apps should *require* paths to be explicit and refuse to fall back to `output_11april26` when that flag is set.
- **Delete the legacy `output_11april26` fallbacks** from [TCRMPclip_routeChosenImages/src/config.py](TCRMPclip_routeChosenImages/src/config.py) and [TCRMPclip_segmentImages/src/config.py](TCRMPclip_segmentImages/src/config.py). They only cause leaks now.
- **Assert empty step dirs on project create.** After `create_project`, assert step3–5 dirs are empty; if not, bail with a clear error.
- **Browser-side test script** — automate the create/quit/resume flow so we don't repeat the demo bug.

---

## B. Step 3 UX (Choose Images) — user-flagged 2026-04-16

### `category_filter` field is confusing / footgun
- Silently legacy from coral-only era.
- If target species include non-Coral codes (e.g. sponges) and field stays `Coral`, selection returns zero.
- **Fix direction:** derive categories automatically from `master_codes.csv` given the selected target species. User should not have to touch this field. (Or multi-select that auto-populates.)
- Code site: [TCRMPcvr_chooseImages/src/config.py:17](TCRMPcvr_chooseImages/src/config.py#L17), [select_images.py:324](TCRMPcvr_chooseImages/src/select_images.py#L324).

### Target species: click-to-select, not free-text
- Replace the text input on Step 3 with a **checkable list pulled from `master_codes.csv`** (post-recode).
- For each row show: code, species name, category, point count.
- Support **sort / filter by category** (Coral, Sponge, Octocoral, etc.).
- Show **definition / common name / notes** inline when hovering or expanding a row.
- Populate from the project's step 2 output, not a hardcoded default.

### "Skip image verification" checkbox
- User doesn't know what it does. Either:
  - Explain it in the description (what it skips, when to use it), or
  - Remove it if defunct.
- Audit what this flag actually toggles in [select_images.py](TCRMPcvr_chooseImages/src/select_images.py).

### "+5% backup frames" — user-flagged 2026-04-16
When selecting frames in Step 3, also emit a secondary pool of ~5% *extra* candidate frames that aren't in the primary set but would have been next-best picks. Purpose: if the user scrap-frames some of the primary picks in Step 4 (see section C), they have pre-vetted replacements ready, keeping the dataset balanced.

- Emit alongside `selected_frames.csv` as `selected_frames_backup.csv` with same schema + a `rank` column.
- Backups should honor the same per-species/per-year/per-site balance constraints so swapping one in maintains the distribution.
- Step 4 UI: once a user scraps a primary frame, offer "Load a backup replacement?" — pulls the highest-ranked backup for the same species/year/site and inserts it into the review queue.
- Make the 5% configurable in the orchestrator Step 3 panel (default 5%).
- Code site: [TCRMPcvr_chooseImages/src/select_images.py](TCRMPcvr_chooseImages/src/select_images.py), config [TCRMPcvr_chooseImages/src/config.py](TCRMPcvr_chooseImages/src/config.py).

### In-window diagnostics after selection
- After Step 3 finishes, show in the orchestrator panel (not just in stdout / output files):
  - Count per target species (bar chart).
  - Distribution across years / sites / transects.
  - **Thumbnail previews** of a handful of selected frames per species.
- Diagnostics currently get written to `output/diagnostics/` by [plot_diagnostics.py](TCRMPcvr_chooseImages/src/plot_diagnostics.py) — wire those PNGs into the Step 3 panel, and optionally add a thumbnail grid.

---

## C. Step 4 UX (currently "Route Images (OCR)") — user-flagged 2026-04-16

### Done 2026-04-16 evening (sub-app UI only — no orchestrator changes)
- **[DONE]** Removed the "Show only" category filter bar.
- **[DONE]** Added **Hold-number-and-click placement**: one button per target label (code + species name + number hint). Hold `1`–`N` and click on the image → places a new point with that label instantly. Held button highlights magenta; cursor switches to `cell`.
- **[DONE]** `A + click` preserved as the fallback for non-target labels (opens the species/category popup).
- **[DONE]** Points placed via quick-add hydrate `pt.species_name` and `pt.category` from the master_codes lookup (previously `species_name` was just the code and category was hardcoded to `Coral`).
- **[DONE]** Added **hover magnifier (loupe)**: 180 px circular magnifier, 3× zoom over view scale, yellow crosshair. When a number is held, the label code (e.g. `PA`) is overlaid in large magenta text on the loupe. Toggle with `H`.
- **[DONE]** Backend: `_load_target_label_info()` reads `master_codes_recoded.csv` (falls back to scanning `all_points_recoded.csv`) and exposes `{code, name, category}` for each target label to the template.
- **[DONE]** **Click-and-drag to pan** — no Alt needed. Uses a 4 px threshold to distinguish click-to-deselect from drag-to-pan. Alt+drag and middle-click still work as power-user shortcuts.
- **[DONE]** **Persisted prefs** (`localStorage` key `tcrmp_route_prefs_v1`): `autoCenter` and `loupeEnabled` survive frame changes, sub-app relaunches, and exports.
- **[DONE]** **Review Batch Size: "All"** — new radio option sends batch_size = 999999 so the backend returns the whole review queue in one go (no batching).
- **[DONE]** **Scrap Frame** — magenta button in the review panel marks a frame `scrapped: true` on disk; it's excluded from the review list *and* from export. Backend: new `POST /api/image/<fn>/scrap` and `/unscrap` routes; `_build_review_list()` skips scrapped detections.
- **[DONE]** Autocenter default is `false`. Help text updated.

### Done 2026-04-17 (Step 4)
- **[DONE]** **Reference mode** (startup checkbox): existing OCR / CPC points render as read-only reference — **larger font (56 px letter, 22 px info)**, muted cyan, no circle, no letter-to-cross line. They don't appear in the Labels panel and `findPointAt` skips them, so drag/select never grabs them. User places their own fresh points via hold-number+click. Session flag `reference_mode` stored server-side; echoed in `/api/image/<fn>` so the client re-syncs on each frame load. ([app.py:545,694,849](TCRMPclip_routeChosenImages/src/app.py), [index.html render / panel / findPointAt](TCRMPclip_routeChosenImages/src/templates/index.html))
- **[DONE]** Export filters: in reference mode, `do_export` keeps only user-added points (`pt.added is True`) — existing reference points stay on disk but don't flow into the YOLO / SAM3 training set. ([app.py do_export](TCRMPclip_routeChosenImages/src/app.py))
- **[DONE]** **Review queue no longer wraps.** `_build_review_list()` used to append exported frames at the end (`unexported + exported`); now it emits only unexported, non-scrapped frames. When the user finishes, "All Done!" is sticky. Re-opening via Resume also respects this (done is done).

### Quick mental model (for anyone reading this cold)
Step 4 verifies **point coordinates** for each selected frame, from two sources:
- **Pre-2020 frames (CPC route):** point (x,y) + species code already exist in `.cpc` files. The app loads them from `TCRMP_clip/TCRMPyyyy_clip/**.cpc`, applies the recoded species codes, and displays them for review. **No OCR runs.**
- **2020+ frames (OCR route):** only `_pts.jpg` annotated images exist. The app runs GPU OCR to detect letters (A–T) + crosshairs, then the user verifies/edits.

So the step has two input flavors with very different data paths. The current UI calls the whole thing "OCR" which is misleading — OCR only applies to ~30% of frames (post-2020).

### Rename throughout: "Verify Points"
- Sidebar label: "Route Images (OCR)" → **"Verify Points"**
- Panel title, run button ("Launch OCR Service" → "Launch Point Verification" or just "Open Verify Points UI"), mark-done button, success messages.
- Code sites: [index.html step 4 panel](pipeline_orchestrator/templates/index.html), [project_manager.py STEP_NAMES](pipeline_orchestrator/project_manager.py#L23-L29), [orchestrator.js](pipeline_orchestrator/static/orchestrator.js), sub-app [routeChosenImages index.html](TCRMPclip_routeChosenImages/src/templates/index.html) copy.
- Keep the directory name `step4_routeChosenImages` on disk to avoid breaking existing projects — it's an internal key, user never sees it.

### Drop the "Target species only" checkbox
- Always on by default. Remove the checkbox from the Step 4 panel in [index.html](pipeline_orchestrator/templates/index.html).
- Keep the `TARGET_SPECIES_ONLY` env var / config flag alive for CLI use, but don't expose it in the orchestrator UI.

### Clarify what `TCRMP_clip` (CLIP_DIR) actually is
**User asked "is this only for OCR images?" — no.** `TCRMP_clip` is the source dir for *both* routes:
- CPC route reads `.cpc` files AND raw frames from `TCRMP_clip/TCRMPyyyy_clip/**`
- OCR route reads `_pts.jpg` files from the same tree

So the field should be labeled something clearer, e.g.:
> **Source image directory** (read-only) — `.cpc` files for pre-2020, `_pts.jpg` files for 2020+

### Lock cpc_all (recoded CPC data) to a permanent reference
**User requirement: always reference `output/cpc_all` for pre-2020 data, and move it somewhere durable.**

Current state:
- Orchestrator hardcodes `TCRMP_CPC_DIR = {REPO_DIR}/output/cpc_all` at [app.py:233](pipeline_orchestrator/app.py#L233).
- Sub-app default is the same path via [config.py:29](TCRMPclip_routeChosenImages/src/config.py#L29).
- Contents today: `output/cpc_all/{2013..2019}/{ids, raw, test_pts, dataset_summary.txt, log.txt}`. **~7 years of extracted CPC coords with remapped species.**

**Suggested changes:**
1. **Define a single canonical path** (e.g. `/mnt/rip/vicarius_drive/TCRMP_shared/cpc_all_recoded/` outside the working repo).
2. Make it a **required** setting in [orchestrator_config.py](pipeline_orchestrator/orchestrator_config.py) — no per-project override; no silent fallback. If missing, fail fast with a clear error.
3. Add a small README/manifest inside the canonical dir explaining: "These are recoded CPC points (species codes remapped per `remap_log_YYYYMMDD.json`). Regenerate via ... . Do not edit by hand."
4. Once moved: delete stale copies in the working repo + update `run_pipeline.sh` / config pointers.
5. (Separate task) Document how to regenerate cpc_all when the remap log changes — this path must stay in sync with whatever the latest remap is.

---

## D. Terminology — "target species" → "target label"

User flagged: sometimes the target is a coarse category (e.g. "Sponge"), not a species. The code, UI copy, and docs should use **"target label"** — neutral term covering species codes *and* category-level labels.

Scope of rename (do as one coordinated pass):
- UI copy: all orchestrator step panels, sub-app templates.
- Config vars: `TARGET_SPECIES` → `TARGET_LABELS` in [TCRMPcvr_chooseImages/src/config.py](TCRMPcvr_chooseImages/src/config.py), [TCRMPclip_routeChosenImages/src/config.py](TCRMPclip_routeChosenImages/src/config.py), [TCRMPclip_segmentImages/src/config.py](TCRMPclip_segmentImages/src/config.py).
- Env vars: `TCRMP_TARGET_SPECIES` → `TCRMP_TARGET_LABELS` (keep old name as alias for one release to avoid breaking existing scripts).
- Function/arg names: `target_species`, `species_list`, `species_filter` → `target_labels`, `label_list`, `label_filter`.
- Project JSON key: `steps.3.config.target_species` → `target_labels` (add a migration path in `load_project` so old projects keep working).

This is cross-cutting — keep a single PR so the rename is atomic.

---

## E. Species-agnostic / non-coral targets

The pipeline is coral-first throughout. Generalize so sponges (or anything in `master_codes.csv`) work end-to-end.

- Step 3 `CATEGORY_FILTER` default (see B above).
- Step 4 target_species env var handling — assume target species are always coral in the UI copy? Audit.
- Step 5 SAM3 class definitions, export class map — check [TCRMPclip_segmentImages/src/export_yolo.py](TCRMPclip_segmentImages/src/export_yolo.py) and `class_map.json`.
- UI copy across all panels talks about "coral" specifically — generalize to "target species" / "instances".

---

## F. SAM3 refinement tool (Step 5) — bigger workstream, revisit

### Done 2026-04-17 (orchestrator + step 5 sub-app)
- **[DONE]** **Orchestrator drives SAM3 segmentation.** Clicking "Start SAM3 Segmentation" now launches the sub-app AND (in a background thread) calls `/api/configure` + `/api/process` on it, then polls `/api/process_status` continuously. Progress surfaces via `GET /api/step/5/sam3_status` (phase, processed/total, message, error). ([app.py _sam3_drive / _run_step5](pipeline_orchestrator/app.py))
- **[DONE]** **Review UI gated on SAM3 progress.** The "Open Review UI" button stays disabled until at least one frame is segmented (or the queue was already complete). Button label reflects state: *waiting for first segmentation* → *N frames ready — SAM3 still running* → *segmentation complete*. ([orchestrator.js updateSam3Panel](pipeline_orchestrator/static/orchestrator.js), [index.html #s5-sam3-panel](pipeline_orchestrator/templates/index.html))
- **[DONE]** **Sub-app auto-skips startup** when orchestrator has already configured it. On `DOMContentLoaded`, the segment-images sub-app calls `/api/status`; if `configured === true`, it jumps straight to the review or processing screen — no "Start Review" click needed. ([TCRMPclip_segmentImages/src/templates/index.html](TCRMPclip_segmentImages/src/templates/index.html))

### Done 2026-04-17 (auto-start SAM3 on each step 4 export)
- **[DONE]** Checkbox **"Auto-start SAM3 segmentation on each batch export"** added to the step 4 panel (default ON). Persisted as `steps.4.config.auto_start_sam3`. ([index.html](pipeline_orchestrator/templates/index.html), [orchestrator.js populate/collect](pipeline_orchestrator/static/orchestrator.js))
- **[DONE]** `_run_step4` passes `TCRMP_ORCHESTRATOR_URL` and `TCRMP_AUTO_START_SAM3` env vars to the step 4 sub-app. Orchestrator URL is computed from `request.host` at launch time.
- **[DONE]** After a successful `do_export` in the step 4 sub-app, `_nudge_sam3_if_configured()` fires a background POST to `{ORCH_URL}/api/step/5/kick` (only if export produced ≥1 frame). Failures are logged but never block the export response. ([TCRMPclip_routeChosenImages/src/app.py _nudge_sam3_if_configured](TCRMPclip_routeChosenImages/src/app.py))
- **[DONE]** Orchestrator `POST /api/step/5/kick` is idempotent:
  - Sub-app not running → full launch (same path as Start button) + `_sam3_drive` thread.
  - Sub-app running, driver thread dead → spawn a fresh driver that re-configures and processes new frames.
  - Sub-app running, driver alive → no-op (queue is rebuilt on the driver's next cycle).
- **[DONE]** `DEFAULT_STEP_CONFIGS["4"]["auto_start_sam3"] = True` so new projects opt in by default. ([orchestrator_config.py](pipeline_orchestrator/orchestrator_config.py))

### Known limitations / follow-ups
- The kick endpoint rebuilds the SAM3 queue from scratch on each re-configure call — already-segmented frames are skipped by the sub-app's existing "already_processed" check, so this is safe, but it *does* re-parse the whole `sam_click_prompts.json`. Fine for now; revisit if export becomes frequent enough for the overhead to matter.
- There is no UI anywhere saying "SAM3 is running in the background because you enabled auto-start." To peek at progress while still on step 4, switch to the step 5 panel — the SAM3 status panel updates regardless of which step tab you're viewing once `sam3PollTimer` has been started by entering the step 5 panel. **Follow-up:** start the SAM3 poll timer whenever a project is loaded (not just when the user runs step 5), so a small status badge can surface on the step 4 panel too. Low priority; do when time permits.

### Done 2026-04-18 (step 6 — YOLO segmentation training via oceankind_CV)
- **[DONE]** Cloned [laurenkolinger/oceankind_CV](https://github.com/laurenkolinger/oceankind_CV.git) into a project-owned wrapper at [TCRMPtrain_oceankindCV/](TCRMPtrain_oceankindCV/). The whole repo lives under `TCRMPtrain_oceankindCV/oceankind_CV/`; our glue is in `TCRMPtrain_oceankindCV/{setup_env.sh, run_step6.sh, src/train_wrapper.py}`.
- **[DONE]** **Project-local env** at `TCRMPtrain_oceankindCV/env/` (conda prefix, not the shared `OK_CV` env). Dependencies: `torch 2.6.0 + cu124`, `torchvision`, `ultralytics 8.4.38` (editable from git clone at `TCRMPtrain_oceankindCV/ultralytics_src/`), `pyyaml`, `tqdm`, `scikit-learn`, `pandas`, `opencv-python`, `wandb`. Verified: CUDA available, 2 devices detected. Re-run [setup_env.sh](TCRMPtrain_oceankindCV/setup_env.sh) to rebuild from scratch.
- **[DONE]** **`run_step6.sh` driver** with sub-commands `split | train | all`. Orchestrator calls `all`, which chains `bal_train_test_split.py` (stratified train/valid/test split) → `train_wrapper.py` (YOLO11 segmentation training with the exact hyperparameters from `oceankind_CV/training/train_segment.py`, but with `project` dir configurable so runs go into `{step6_dir}/runs/<run_name>/`).
- **[DONE]** **Orchestrator step 6 wiring** — new panel in [index.html](pipeline_orchestrator/templates/index.html), `_run_step6` in [app.py](pipeline_orchestrator/app.py), STEP_KEYS extended to `["1"…"6"]` in [project_manager.py](pipeline_orchestrator/project_manager.py). Back-compat: older projects without a `steps.6` entry auto-grow one on `load_project`. `auto_link_outputs` now chains step 5 → step 6 (step 5's export dir becomes step 6's input).
- **[DONE]** **Step 6 UI**: sidebar nav item, panel with run name, base model dropdown (yolo11{n,s,m,l,x}-seg), epochs, image size, valid/test fractions, min-samples. "Start Training" streams stdout via the existing `run_cli_stage` + log polling infrastructure — training progress (Ultralytics' per-epoch log lines) shows up in the Step 6 log panel.

### Known gaps / next up for step 6
- No TensorBoard or wandb auto-launch button in the panel — training writes to `{step6_dir}/runs/<name>/` and the user can point TensorBoard at it manually.
- No "test.py" wrapper (oceankind_CV has `training/test.py`). Easy to add as a third sub-command on `run_step6.sh` when we want post-training evaluation inside the orchestrator.
- `run_step6.sh` currently hard-codes valid/test/min_samples defaults in `all` mode even though the UI collects them. Forwarding them is a 10-line fix — flag for next session.
- Training is long; no graceful "pause/resume" handling. A ctrl-stop from the orchestrator will send SIGTERM to the run, cutting off mid-epoch. Ultralytics writes best/last weights periodically, but you'd need to re-launch with `--resume` manually. Good enough for v1.

### Done 2026-04-17 evening (robustness + folding + aux)
- **[DONE]** **SAM3 per-image flush.** Background processor now saves `{export_dir}/segmentations/{year}/segmentations.json` after **every** image instead of every 20. Orchestrator restarts (or crashes, or abrupt close) can no longer rewind progress more than 1 image. ([app.py _background_process_all](TCRMPclip_segmentImages/src/app.py))
- **[DONE]** **Refine / rebox / undo_refine all subtract neighbors.** Same clean-boundary rule as `/add`: when SAM3 regenerates a mask (on positive click, box redraw, or undo-pop), we mask off the union of other masks on this frame before polygonizing. This is what the user kept hitting: a positive refine click on one coral half would regrow SAM3's idea of the whole colony and overlap its neighbor on the far side of the tape. Fixed. ([app.py refine_mask / rebox_mask / undo_refine](TCRMPclip_segmentImages/src/app.py))
- **[DONE]** **Enter / Reject progression auto-advances frames.** Space still cycles *all* masks (manual review). Enter / X / assign-species-popup all call `advanceToNextOrFrame`, which picks the next pending mask or — if none left — moves to the next frame and fit-views it. ([templates/index.html selectNextMask + advanceToNextOrFrame](TCRMPclip_segmentImages/src/templates/index.html))
- **[DONE]** **Draw mode (D) — TagLab-style freehand boolean edit.** Select a mask, press `D` (or click Draw button), click-drag over the image. Release closes the path. Backend `POST /api/image/<fn>/draw_edit` rasterizes the polygon and decides **add vs subtract by centroid-in-mask test**: centroid inside mask → subtract (carve chunk); centroid outside → union (add chunk). Add ops also respect the "don't eat neighbors" rule. Cyan border + badge in draw mode. ([app.py draw_edit](TCRMPclip_segmentImages/src/app.py) + [template render / gesture](TCRMPclip_segmentImages/src/templates/index.html))
- **[DONE]** **Orchestrator folds sub-app launch UIs.** When either sub-app is launched by the orchestrator (detected via the env vars we already set: `TCRMP_ORCHESTRATOR_URL` for step 4, `TCRMP_INPUT_DIR` for step 5), the browser lands on an **orchestrator-styled loading veil** instead of the standalone setup form. Step 4 auto-calls `doStart` on `DOMContentLoaded`; step 5 polls `/api/status` every 2 s (up to 10 min for model loading) until `configured=true`, then advances to processing/review. The standalone form still works when a dev runs the sub-app directly.
- **[DONE]** **Aux observations in step 4** (`added_points.csv`). Press `L` → aux mode (green border + badge). Click → prompt for species code + optional notes → appended to `{step4_export_dir}/added_points.csv`. Reloaded on `/api/configure` and `/api/resume`. Rendered as green diamonds with the code; they never flow into SAM3 or YOLO exports. `POST /add_aux` + `POST /delete_aux` endpoints. ([TCRMPclip_routeChosenImages/src/app.py](TCRMPclip_routeChosenImages/src/app.py), [templates/index.html](TCRMPclip_routeChosenImages/src/templates/index.html))

### Done 2026-04-17 (step 5 overhaul — mirrors step 4 feel)
- **[DONE]** **Add-mode clean boundary fix.** `/api/image/<fn>/add` now subtracts the union of all existing masks (decoded from RLE) from the freshly-segmented binary before `build_mask_dict`. `clean_mask` still enforces single-connected-component and polygon simplification, so the new mask is smooth and can't "lump into" a neighbor. ([app.py add_mask](TCRMPclip_segmentImages/src/app.py))
- **[DONE]** **Pink add-mode chrome.** When the user enters Add mode, the canvas wrapper gets an inset pink border + a floating "ADD MODE — hold 1–N + click to place" badge. Impossible to miss. Leaving add mode also clears any held quick-label.
- **[DONE]** **Target-label button bar** (same component as step 4): one button per target species with code + name + number hint. Held number + click in Add mode calls `addMaskAt` with species + category pre-assigned, so the new mask is born already labeled.
- **[DONE]** **Full view on frame load.** Dropped the auto-zoom-to-first-pending call in `selectImage`. User asked: "each new frame starts in full view mode." Space cycles through masks (see next).
- **[DONE]** **Spacebar cycles ALL masks** (was: pending only). Also auto-enters refine mode on the selected mask, so the next click edits instead of deselects.
- **[DONE]** **Click-select auto-enters refine mode.** Clicking any mask in SELECT mode now flips to refine; any follow-up click is an immediate positive refinement on the selected mask.
- **[DONE]** **Accept/Reject/Merge progressions** now advance via `selectNextMask` (not `selectNextPending`) to match the Space cycle.
- **[DONE]** **Click + drag = pan** (Alt+drag and middle-click preserved for muscle memory). Uses a deferred-click pattern: mousedown stores the intent, mousemove ≥ 4 px promotes to pan, mouseup runs the mode-specific click action if no drag happened.
- **[DONE]** **Hover loupe, hidden by default, `H` toggles.** Shows the held label code overlay in large pink letters when a number is held. Per user: "could be distracting. hide by default."
- **[DONE]** **Persisted prefs** (`localStorage` key `tcrmp_segment_prefs_v1`): loupe on/off, autocenter, mask opacity. Kept the legacy `seg_opacity` key for back-compat with old sessions.
- **[DONE]** **Review-batch "All (no batching)"** + **Scrap Frame** button + backend `/scrap` + `/unscrap` routes (same API shape as step 4). `_build_review_list` skips scrapped and exported frames. No wrap-around.
- **[DONE]** **Shuffle checkbox on startup** (both step 4 and step 5). Passed to `/configure` + `/resume` as `shuffle: bool`, stored in `session['shuffle']`, applied in `_build_review_list` with `random.shuffle`. Per-frame processed state is still persisted on disk so shuffled order doesn't lose progress tracking.

### Still to do — user-flagged 2026-04-17 evening
- **Draw mode (`D`) in refine mode — TagLab-style freehand boolean edit.** User saw this in TagLab and wants it in step 5. Design sketch:
  - **Trigger:** `D` while a mask is selected and we're in refine mode. Canvas border turns a distinct color (blue?) + badge "DRAW MODE".
  - **Interaction:** mousedown + drag = capture a freehand path (array of image-space points). Release to close.
  - **Shape detection:**
    - If release point is within ~10 px of start point → **closed polygon**. Compute its interior; union (add) or difference (subtract) with current mask depending on whether the polygon's centroid lies inside or outside the current mask.
    - If open path → **extension/clip** along the path. Simplest impl: fatten the path into a thin region (Minkowski sum with a small disk, ~3 px radius in image space), then union OR diff:
      - If most of the path lies **outside** the current mask → union (adds "outward" curves as new material).
      - If most of the path lies **inside** the current mask → diff (carves "inward" curves away).
  - **Backend:** new `POST /api/image/<fn>/draw_edit` with `mask_id, path:[[x,y],...], closed:bool`. Server rasterizes the path (scipy or OpenCV `fillPoly`/`polylines`), applies the boolean op via numpy, then reuses `update_mask_geometry` for cleanup + polygon simplification.
  - **Undo:** push prior mask onto `mask.refinement_clicks`-style undo stack so existing `undoRefine` works.
  - **Gotcha:** `clean_mask` keeps only the largest connected component. If a draw edit bisects a mask into two pieces, the smaller piece is dropped. Consider a `keep_all_components` flag for this endpoint, or an explicit "split mask" action.
  - **Rough LOC:** ~80 lines backend, ~120 lines frontend (freehand capture, path rendering, submit). Half a day.
- **Sparse organisms aux-ID** (better suited for step 4, per user). When reviewing images the user sometimes spots other interesting critters; they want to log these separately. User's hint: "exported completely differently and maybe just as a table even (showing that I found them)." Design sketch:
  - New side panel or overlay in step 4 labeled "Aux sightings".
  - Hold `L` (or dedicated key) + click → add aux point, prompt for species/name, category defaults to "Aux".
  - Stored separately from the main detections: `{export_dir}/aux_sightings.csv` with columns: `frame_filename, year, date, site, transect, x, y, label, species, name, category, notes, timestamp`.
  - Skipped by SAM3 segmentation and YOLO export — purely an observational log.
  - Nice-to-have: a "summary" view in the orchestrator showing the aux table.
- **Long-term frame processing log (both step 4 and step 5).** User wants persistence across the project's years-long life:
  - Status per-frame per-step: `makeAllPoints_done / chooseImages_selected / routeImages_reviewed / routeImages_exported / segmentImages_reviewed / segmentImages_exported` — plus per-organism-type counts.
  - Most of this already exists ad-hoc (the detections.json / segmentations.json per year). The missing piece is a **consolidated manifest** for cross-project reporting.
  - Candidate: write `{project_dir}/frame_manifest.csv` on every orchestrator state save, with one row per frame: filename, year, site, transect, route, step-status flags, counts per target label.
  - Also useful: an orchestrator "Stats" panel showing progress across all open projects.

### Parity with the Step 4 UX work done 2026-04-16
Once Step 4 is locked in, bring the same affordances to the SAM3 review UI ([TCRMPclip_segmentImages/src/app.py](TCRMPclip_segmentImages/src/app.py) + its template). Review + decide per-item:

- **Hold-number-and-click placement of target-labeled points.** SAM3 already uses point prompts — this becomes "hold `1` + click to prompt SAM3 with label PA", etc. Button row sourced from `master_codes_recoded.csv`.
- **Hover magnifier (loupe).** Same 180 px circular magnifier, 3× zoom, crosshair, label overlay when number is held. Toggle with `H`.
- **Click-and-drag pan** (replace any Alt+drag requirement); 4 px click/drag threshold.
- **Persisted prefs** via `localStorage` under a separate key (e.g. `tcrmp_sam3_prefs_v1`): `autoCenter`, `loupeEnabled`, plus anything SAM3-specific (mask opacity, outline thickness, confidence threshold display).
- **Autocenter default off.**
- **"All" review-batch option** — if segmentImages batches frames in review, add an "All" mode that sends the whole queue.
- **Scrap Frame button** — same semantics: marks the frame's SAM3 output record `scrapped: true`, skipped by review and YOLO export. Needs backend support: `POST /api/frame/<id>/scrap` + update export_yolo.py to skip scrapped.
- **Rename "target species" → "target label"** in UI + config (part of section D).
- **Remove any "Show only <category>" filter** if present (same legacy footgun as Step 4).

Implementation note: much of the JS can be lifted from Step 4's template verbatim. Consider extracting the shared blocks (loupe, pan, quick-add, prefs) to a common `static/review_common.js` served from both sub-apps, rather than copy-paste. Low priority but avoids future divergence.

### Bigger open questions for SAM3 itself (original placeholder)

User wants refinement of the SAM3 review UI with tests alongside. Not scoped yet. To do when we get there:
- Inventory what refinement operations exist ([TCRMPclip_segmentImages/src/app.py](TCRMPclip_segmentImages/src/app.py) is ~1200 LOC).
- Identify the pain points (merge? split? boundary edit? re-prompt?).
- Define test fixtures (small mask set we can regression against).
- Add test harness before changes.

---

## G. Field definitions across steps

Audit every form field in the orchestrator and make sure each has:
- A clear label
- A one-line tooltip / help text explaining what it does and when to change it
- A sensible default

Fields flagged so far: Step 3 `category_filter`, Step 3 `skip_image_check`. More to come — capture them here as user surfaces them.
