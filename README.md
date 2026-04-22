# seg_AI_img_full_april2026

> TCRMP coral reef segmentation pipeline — from decades of Caribbean reef
> point-count data to a production YOLO11 segmentation model, via SAM3
> auto-labelling and a browser-based review UI.

This is a pared-down, reproducible snapshot of the CVR-CLIP coral
segmentation pipeline, organized for git version-control (code + configs)
with data artifacts excluded. It is the staging ground for eventual
integration as a VICARIUS module (`modules/cvr_clip_segmentation/`).

---

## Quick start

```bash
# 1. Clone
git clone <url> seg_AI_img_full_april2026 && cd seg_AI_img_full_april2026

# 2. Repopulate supporting_data/ and projects/ (git-ignored — see below)
#    On the lab workstation this is already present; on a fresh box, use:
#      rsync -aHAX bizon@vicar-mainframe:/mnt/rip/vicarius_drive/hopper/\
#        CVR_CLIP_forAI/seg_AI_img_full_april2026/{supporting_data,projects} ./

# 3. Build the unified environment + rewrite project paths + smoke-test
./bootstrap.sh

# 4. Launch the orchestrator
./scripts/run_pipeline.sh

# 5. Open http://localhost:5050 in a browser
```

---

## Repository layout

```
seg_AI_img_full_april2026/
├── README.md                    -- this file (GitHub landing page)
├── .gitignore                   -- excludes supporting_data/, projects/, env/
├── bootstrap.sh                 -- one-shot env build + project path migration
├── env/                         -- unified venv (rebuilt, never committed)
├── scripts/                     -- ALL executable code (git-tracked)
│   ├── pipeline_orchestrator/   -- Flask orchestrator + sub-app launcher
│   ├── TCRMPcvr_*               -- steps 1-3 (CVR point parsing, recode, select)
│   ├── TCRMPclip_*              -- steps 4-5 (CPC+OCR review, SAM3 segmentation)
│   ├── TCRMPtrain_oceankindCV/  -- steps 6-8 (YOLO training, eval, inference)
│   ├── setup_unified_env.sh     -- builds env/ with all 8 steps' dependencies
│   ├── rewrite_project_paths.py -- migrates absolute paths in projects/*/project.json
│   └── run_pipeline.sh          -- orchestrator launcher (opens browser)
├── config/                      -- git-tracked configs
│   ├── pipeline.yaml            -- THE canonical config (every path, every default)
│   ├── module.yaml              -- VICARIUS module-interface stub
│   ├── presets/                 -- step-6 training presets (tuned_v1_regularized.yaml)
│   └── eval_presets/            -- step-7 evaluation presets (standard.yaml)
├── supporting_data/             -- GIT-IGNORED, large static data
│   ├── all_points.csv (82MB)    -- unified CVR point dataset (step 1 output)
│   ├── master_codes.csv         -- species code lookup
│   ├── cpc_all/ (67GB)          -- pre-2020 CPCe point index consumed by step 4
│   ├── TCRMP_CVR/ (572MB)       -- raw Excel sources (step 1 input)
│   ├── TCRMP_clip/ (462MB)      -- offline test subset (2018 + 2024 years)
│   └── model_weights/           -- pretrained YOLO weights (n/m/l sizes + yolov8n)
├── projects/                    -- GIT-IGNORED, per-project outputs + state
│   └── <name>_<YYYYMMDD>_<uuid>/ -- one dir per project; project.json + step1..step8 subdirs
└── docs/                        -- git-tracked documentation
    ├── PIPELINE_OVERVIEW.md     -- comprehensive non-specialist walkthrough
    ├── NOTES_pipeline_improvements.md  -- rolling change log / backlog
    └── VICARIUS_INTEGRATION.md  -- event schema, palette, module-drop-in recipe
```

---

## The eight pipeline steps

| # | Step | Sub-tool | UI port | Input | Output |
|---|------|----------|--------:|-------|--------|
| 1 | Make All Points | TCRMPcvr_makeAllPoints | — | `supporting_data/TCRMP_CVR/` (Excel) | `all_points.csv`, `master_codes.csv` |
| 2 | Recode Species | TCRMPcvr_recodeSpecies | 5055 | `all_points.csv` | `all_points_recoded.csv`, `remap_log_*.json` |
| 3 | Choose Images | TCRMPcvr_chooseImages | — | recoded `all_points.csv` | `selected_frames.csv` + routing CSVs |
| 4 | Route & Verify Points | TCRMPclip_routeChosenImages | 5065 | `selected_frames.csv` + Dropbox clips + `cpc_all/` | per-year `sam_click_prompts.json` |
| 5 | SAM3 Segmentation | TCRMPclip_segmentImages | 5070 | step-4 export | `all_images/`, `all_labels/`, `data.yaml` (YOLO format) |
| 6 | Train Model | TCRMPtrain_oceankindCV | — | step-5 export | `runs/<run_name>/weights/best.pt` |
| 7 | Evaluate Model | TCRMPtrain_oceankindCV | — | step-6 run | `report.pdf` + `metrics.json` |
| 8 | Inference | TCRMPtrain_oceankindCV | — | step-6 `best.pt` | `manifest.json`, `overlays/`, `crops/` |

Sub-tools 2, 4, and 5 are browser apps; the rest are CLI. The orchestrator
folds them all behind a single http://localhost:5050 sidebar so the user
never sees the sub-app startup forms.

See `docs/PIPELINE_OVERVIEW.md` for the full story.

---

## Configuration

All paths, ports, interpreter choices, per-step defaults, and VICARIUS
integration toggles live in `config/pipeline.yaml`. Every field is commented
inline. The orchestrator loads it at startup; its values become defaults
baked into each new `projects/<id>/project.json` at project creation.

Per-project overrides then live in `project.json` (written from the UI) and
take precedence over the YAML defaults. This means a project is fully
reproducible: zip the project dir + the pipeline.yaml at the SHA it was
created against, and you can replay the run.

---

## Supporting data (not in git)

`supporting_data/` holds the multi-gigabyte assets the pipeline reads. It
is git-ignored — these files are too large to commit and are backed up to
the Synology NAS per VICARIUS policy.

To repopulate on a fresh workstation:
- **From a lab box**: `rsync -aHAX bizon@vicar-mainframe:.../supporting_data/ ./`
- **From raw sources** (if rebuilding from scratch):
  - `TCRMP_CVR/` — from UVI Smith Lab share
  - `cpc_all/` — re-run the legacy `scripts/TCRMPclip_cpcID/run.sh` pipeline
  - `all_points.csv` + `master_codes.csv` — re-run step 1
  - `TCRMP_clip/` — optional local subset; Dropbox is the production source
  - `model_weights/` — Ultralytics YOLO11 weights auto-download on first use

See `supporting_data/README.md` for details on each asset.

---

## Projects (not in git)

`projects/` holds per-project outputs + `project.json` state. Each project
is self-contained and portable — copying a project dir preserves its full
state. Step subdirs (`step1_makeAllPoints/`, …, `step8_inference/`) are
created by `project_manager.create_project()`.

**Start / stop / resume**: close the orchestrator (or the whole browser)
at any time. Relaunch `run_pipeline.sh`, click "Open Project", pick the
same dir — the UI restores the exact step and config you were on.

---

## VICARIUS integration

When `vicarius.enabled: true` in `config/pipeline.yaml`, the orchestrator
emits `process_start` / `process_end` / `user_note` events to the central
VICARIUS event stream (`/mnt/rip/vicarius_drive/vicarius/_logging/db/vicarius_events.db`).
Query with `vicarius story --days 1` or the SQLite views.

`config/module.yaml` is the drop-in contract for the day this directory is
copied under `vicarius/modules/cvr_clip_segmentation/github_repo/`.

Ten Commandments compliance:
- **I** Raw data sacred: `supporting_data/TCRMP_CVR/` + Dropbox clips are never written to.
- **II** Naming conventions: TCRMP frame/transect pattern honored throughout.
- **III** Every file has metadata: every step writes a `config_log.json`.
- **IV** Modules atomic: each sub-tool has one responsibility + standalone runner.
- **V** Document as you build: `docs/NOTES_pipeline_improvements.md` is the rolling log.
- **VI** Prompt for purpose: project creation asks for name + purpose at UI-time.
- **VII** Test before production: `projects/demo_*` and `projects/test_*` exist for smoke tests.
- **VIII** Log everything: VICARIUS event stream + per-step log files.
- **IX** Version code, track data: git tracks `scripts/` + `config/` + `docs/` only.
- **X** Keep it simple, then grow: plain Flask + plain JS + single YAML. No React, no build step.

See `docs/VICARIUS_INTEGRATION.md` for event schema, UI palette reference,
and the module-drop-in recipe.

---

## Development

- **Code style**: conform to the VICARIUS aesthetic — plain text, readable
  fonts (Inter body, Space Mono code), no emoji in code or UI strings,
  border-radius 2px, high contrast, crosshair cursor site-wide.
- **Testing**: small test projects live under `projects/test_*`; exercise
  them as the first smoke test after any change to orchestrator internals.
- **Logs**: for any recent activity, run `vicarius story --days 1`.
- **Dependencies**: all Python deps live in the unified `env/` built by
  `scripts/setup_unified_env.sh`; do not add sub-tool-level envs.

---

## Attribution

- **TCRMP** — Territorial Coral Reef Monitoring Program, University of the Virgin Islands, Smith Lab.
- **Oceankind_CV** — [github.com/laurenkolinger/oceankind_CV](https://github.com/laurenkolinger/oceankind_CV) (Lauren K Olinger).
- **SAM3** — Meta AI, [Segment Anything with Concepts (Nov 2025)](https://github.com/facebookresearch/sam3).
- **YOLO11** — Ultralytics.
- **Pipeline orchestration + integration** — Lauren K Olinger.
