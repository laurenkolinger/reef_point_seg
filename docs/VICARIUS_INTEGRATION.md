# VICARIUS Integration Notes

This document captures how the coral-segmentation pipeline plugs into the
VICARIUS platform (`/mnt/rip/vicarius_drive/vicarius`). It covers the event
stream, the UI aesthetic, the module contract, and the planned migration
into `vicarius/modules/cvr_clip_segmentation/`.

Read `/mnt/rip/vicarius_drive/vicarius/_DOCS/TEN_COMMANDMENTS.md` alongside
this document — the commandments are the governing discipline.

---

## 1. Event stream instrumentation

The orchestrator emits events to the central VICARIUS event DB
(`/mnt/rip/vicarius_drive/vicarius/_logging/db/vicarius_events.db`) via a
thin bridge (`scripts/pipeline_orchestrator/vicarius_bridge.py`) that calls
into the platform's Python API (`vicarius_log.VICARIUSLog`).

### 1.1 Toggle

```yaml
# config/pipeline.yaml
vicarius:
  enabled: true
  study_default: S1_historical          # default study tag on every event
  module_name: cvr_clip_segmentation    # drives the event module prefix
  event_prefix: orch                    # recorded in metadata_json
```

When `enabled: false`, the bridge degrades to no-ops and the orchestrator
still runs. The bridge also silently no-ops if the VICARIUS logging lib is
absent (useful for collaborators without the platform installed).

### 1.2 Events emitted

| Orchestrator action | VICARIUS event | Module tag | Notes |
|--|--|--|--|
| `POST /api/project/create` | `user_note` | `cvr_clip_segmentation` | `"[orch] project <name> created at <dir>"` |
| `POST /api/project/open` | `user_note` | `cvr_clip_segmentation` | `"[orch] project <name> reopened"` |
| `POST /api/step/<n>/run` | `process_start` | `cvr_clip_segmentation.step<n>_<name>` | `purpose` drawn from step config + project purpose; returns `event_id` for later linkage |
| step completion | `process_end` (status=success) | same | `duration_sec` + `outputs=[step_dir]`; `parent_event_id` links to the `process_start` |
| step error | `process_end` (status=failed) | same | `notes` contains the last ~20 log lines |
| `POST /api/project/quit` | `user_note` | `cvr_clip_segmentation` | `"[orch] project closed"` |
| SAM3 driver kick | `user_note` | `cvr_clip_segmentation.step5_segmentImages` | `"[orch] SAM3 driver kicked for batch <n>"` |

### 1.3 Query recipes

```bash
# Narrative view of today
vicarius story --days 1

# All orchestrator events in the past week
vicarius log show -t process_start -d 7 | grep cvr_clip_segmentation

# All events for a specific study
sqlite3 /mnt/rip/vicarius_drive/vicarius/_logging/db/vicarius_events.db \
  "SELECT timestamp, event_type, process_module, context_purpose
     FROM events
    WHERE context_study = 'S1_historical'
      AND process_module LIKE 'cvr_clip_segmentation%'
    ORDER BY timestamp DESC LIMIT 50;"

# Duration rollup per step across all projects
sqlite3 ... \
  "SELECT process_module, COUNT(*) AS runs,
          SUM(process_duration_sec) AS total_sec
     FROM events
    WHERE event_type = 'process_end'
      AND process_module LIKE 'cvr_clip_segmentation.%'
    GROUP BY process_module ORDER BY total_sec DESC;"
```

---

## 2. UI aesthetic — mapping to VICARIUS style

The orchestrator UI was re-themed to match
`/mnt/rip/vicarius_drive/vicarius_ui/static/theme.css`. The contract:

### 2.1 Colors

| Role | Value |
|------|-------|
| Background | `#0a0a0f` / `#050508` (near-black layers) |
| Body text | `#F5EDF0` (soft off-white) |
| Primary accent | `#FFB3D9` (bubblegum pink) |
| Strong accent | `#E20074` (T-Mobile magenta) |
| Success | `#00FFB3` |
| Warning | `#FFB800` |
| Error | `#FF4D6D` |

### 2.2 Typography

- **Body**: Inter 15px, line-height 1.6, letter-spacing 0.01em — never fall below 14px.
- **Code / logs**: Space Mono, same size as body.
- **Headings**: Syne / Space Grotesk, 600-700 weight, no tracking games.
- **Wordmark accent only**: Silkscreen (the pixel font) for decorative "VICARIUS"-style stamps. Never use it for content.

### 2.3 Chrome

- Border-radius 2px everywhere (square-ish).
- `cursor: crosshair` on the body.
- 20px pixel grid overlay, opacity 0.04.
- Horizontal scanline overlay, opacity 0.22.
- No emoji — replace with plain text (`[done]`, `[running]`, `[ERR]`).

### 2.4 File-level changes (relative to upstream orchestrator)

- `scripts/pipeline_orchestrator/static/style.css` — full palette + type rewrite.
- `scripts/pipeline_orchestrator/static/orchestrator.js` — any emoji literals replaced with plain text.
- `scripts/pipeline_orchestrator/templates/index.html` — Google Fonts imports + `.grid-overlay` + `.scanlines` divs.

CSS class names and HTML structure are identical to upstream so no
JavaScript logic changes.

---

## 3. Module contract — `config/module.yaml`

See `config/module.yaml` in this repo for the full spec. Highlights:

- `type: pipeline` — multi-step; launches its own orchestrator UI.
- Three external inputs declared: `tcrmp_cvr`, `tcrmp_clip`, `cpc_all`.
- Six first-class outputs declared: everything a downstream study would
  consume (trained model, eval report, inference manifest, etc.).
- Five user-facing parameters (target species, target instances, epochs,
  imgsz, base model) — the full hyperparameter surface still exists behind
  the UI but these five are the common customization points.

---

## 4. Migration into `vicarius/modules/cvr_clip_segmentation/`

When ready to formally plug in:

```bash
cd /mnt/rip/vicarius_drive/vicarius/modules
mkdir -p cvr_clip_segmentation/{github_repo,inprocess,misc}

# Copy the whole repo (code + config only; NOT supporting_data or projects)
rsync -a --exclude='supporting_data' --exclude='projects' --exclude='env' \
      /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/ \
      cvr_clip_segmentation/github_repo/

# Symlink the data + runs so they stay outside the module's git repo
ln -s /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data \
      cvr_clip_segmentation/github_repo/supporting_data
ln -s /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/projects \
      cvr_clip_segmentation/github_repo/projects

# VICARIUS init_run.py / shelve_run.py are dropped into the module root
# (not the github_repo subdir) per platform convention; copy from template:
cp /mnt/rip/vicarius_drive/vicarius/modules/_template/github_repo/src/init_run.py \
   cvr_clip_segmentation/github_repo/src/
cp /mnt/rip/vicarius_drive/vicarius/modules/_template/github_repo/src/shelve_run.py \
   cvr_clip_segmentation/github_repo/src/
```

Then re-run `bootstrap.sh` from the new location.

Post-migration, `vicarius list` should show `cvr_clip_segmentation` in its
module table (via `module.yaml`).

---

## 5. Naming convention checklist

VICARIUS Commandment II requires strict naming. Our outputs already
inherit the TCRMP pattern since inputs are TCRMP-named
(`TCRMP{YYYYMMDD}_clip_{SITE}_T{N}`). When the pipeline writes derived
products:

- Step 4 raw-image copies: keep source basename (`TCRMP20180502_clip_BIX_T103.jpeg`).
- Step 5 YOLO labels: `{source_id}.txt` sharing the source basename.
- Step 6 runs: `{project_id}_{YYYYMMDD_HHMMSS}` if user leaves `run_name` blank.
- Step 8 inference outputs: `overlays/{source_id}.jpg`,
  `crops/{source_id}_det{NN}.jpg`.

When the model is promoted to VICARIUS `/models/`, use semantic versioning
per the naming convention:

```
coral_seg_v1.0.0_20260421.pt      # initial release
coral_seg_v1.1.0_20260505.pt      # minor update (more training data)
coral_seg_v2.0.0_20261001.pt      # major update (class set changed)
```

Model cards go alongside with the same basename + `.md`.

---

## 6. Ten Commandments compliance checklist

- [x] **I. Raw data sacred** — `supporting_data/TCRMP_CVR/` and the
      Dropbox clip dir are read-only. No step writes into them.
- [x] **II. Naming conventions** — TCRMP pattern honored through every
      stage; all intermediate outputs derive from source basenames.
- [x] **III. Every file has metadata** — each step writes a
      `config_log.json`; `project.json` captures the full run state.
- [x] **IV. Modules atomic** — each sub-tool does one thing and has a
      standalone `run.sh` / `setup_env.sh`.
- [x] **V. Document as you build** — `docs/NOTES_pipeline_improvements.md`
      is the rolling changelog; this file updates alongside code changes.
- [x] **VI. Prompt for purpose** — project creation UI asks for name +
      purpose; `process_start` events carry that purpose string.
- [x] **VII. Test before production** — `projects/demo_*` and
      `projects/test_*` are retained for smoke tests.
- [x] **VIII. Log everything** — VICARIUS event stream captures start,
      end, user notes; each step also writes a per-step log file.
- [x] **IX. Version code, track data** — `.gitignore` excludes
      `supporting_data/` + `projects/` + `env/`; NAS handles data backup.
- [x] **X. Keep it simple, then grow** — plain Flask, plain JS, single
      YAML config, one env. No React, no build step, no SPA framework.
