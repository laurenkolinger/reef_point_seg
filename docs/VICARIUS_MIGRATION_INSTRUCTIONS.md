# Migration Instructions for the VICARIUS Agent

**Target audience**: the VICARIUS migration agent (or a human operator with file-system tools). This document prescribes, phase by phase, how to move `seg_AI_img_full_april2026/` into the VICARIUS platform at `/mnt/rip/vicarius_drive/vicarius/` as a first-class module.

**Non-negotiable**: nothing is moved. Every operation is a copy. The source repo at `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/` must remain intact and operable throughout the migration.

---

## 0. Preflight — things to confirm before writing a single byte

Run these checks first. If any fails, stop and ask the operator.

```bash
# 0.1 VICARIUS platform exists + is writable
ls -ld /mnt/rip/vicarius_drive/vicarius
test -w /mnt/rip/vicarius_drive/vicarius || echo "NOT WRITABLE — stop"

# 0.2 Ten Commandments present
test -f /mnt/rip/vicarius_drive/vicarius/_DOCS/TEN_COMMANDMENTS.md

# 0.3 Module template present (we'll copy init_run.py / shelve_run.py from it)
test -d /mnt/rip/vicarius_drive/vicarius/modules/_template

# 0.4 Logging lib importable (the bridge expects vicarius_log.VICARIUSLog)
python3 -c "import sys; sys.path.insert(0, '/mnt/rip/vicarius_drive/vicarius/_logging/src'); from vicarius_log import VICARIUSLog; print('ok')"

# 0.5 Source repo intact
test -f /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/pipeline.yaml
test -f /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/scripts/pipeline_orchestrator/app.py

# 0.6 Nothing is currently writing to projects/ on the source side
pgrep -fc 'seg_AI_img_full_april2026.*run_pipeline' || echo "  clean"

# 0.7 Free space: need at least 100 GB headroom on /mnt/rip
df -BG /mnt/rip/vicarius_drive | tail -1
```

Read `/mnt/rip/vicarius_drive/vicarius/README.md` and `/mnt/rip/vicarius_drive/vicarius/_DOCS/TEN_COMMANDMENTS.md` end to end before proceeding. Every action below is constrained by those documents.

---

## 1. What goes where (the contract — with FULL absolute paths)

The source is partitioned four ways already. The destination splits them across FIVE VICARIUS locations. The user has been explicit: **outputs do NOT go into `github_repo/`**; they live under VICARIUS data roots or inside the module's `inprocess/` tree.

### 1.0 Path constants used throughout this document

Copy-paste these into shell at the top of any session running the migration, or just substitute them inline if the agent prefers verbatim paths:

```
SOURCE REPO:
    /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026

VICARIUS PLATFORM ROOT:
    /mnt/rip/vicarius_drive/vicarius

VICARIUS MODULE DIR:
    /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation

MODULE github_repo:
    /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo

MODULE inprocess:
    /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess

MODULE misc:
    /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/misc
```

### 1.1 Full source-to-destination manifest

| Source (full path) | Destination (full path) | Why |
|---|---|---|
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/scripts/` (excluding `env/`, `__pycache__/`) | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts/` | Code — git-tracked, version-controlled. Commandment IX. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/pipeline.yaml` | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/pipeline.yaml` | Canonical config — git-tracked. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/module.yaml` | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/module.yaml` | **Note**: moves UP one level to the repo root (VICARIUS `module_registry.py` discovers `module.yaml` in `github_repo/`, not inside a `config/` subdir). |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/presets/` | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/presets/` | Step-6 training preset YAMLs — git-tracked. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/eval_presets/` | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/eval_presets/` | Step-7 evaluation preset YAMLs — git-tracked. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/docs/*.md` | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/docs/` | Documentation — git-tracked. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/README.md` | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/README.md` | Meta README / GitHub landing page. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/bootstrap.sh` | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/bootstrap.sh` | One-shot env build + project path rewrite. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/.gitignore` | REWRITTEN — do NOT copy verbatim. See §6.3. | inprocess/ and data-root excludes no longer live inside github_repo/. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/all_points.csv` (82 MB) | `/mnt/rip/vicarius_drive/vicarius/processed/derived_tabular/tcrmp_cvr_all_points.csv` | Derived tabular product (Commandment II naming). |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/master_codes.csv` (4 KB) | `/mnt/rip/vicarius_drive/vicarius/_METADATA/library/definitions/tcrmp_species_codes.csv` | Controlled vocabulary — first-class VICARIUS metadata. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/cpc_all/` (67 GB) | `/mnt/rip/vicarius_drive/vicarius/processed/cpc_index/` *(see §4.2 — may need operator confirmation)* | Pre-computed per-year point index, derived from CPCe files. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/TCRMP_CVR/` (572 MB, 839 Excel files) | `/mnt/rip/vicarius_drive/vicarius/raw/tcrmp_tabular/cvr/` | Raw CVR survey workbooks. **IMMUTABLE** per Commandment I — `chmod -R a-w` after landing. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/TCRMP_clip/` (462 MB, 2018+2024 offline test subset) | *DO NOT COPY* — path-reference only | `/home/bizon/UVI Dropbox/SMITH LAB TEAM FOLDER/TCRMP/TCRMP_clip` is authoritative. Module config already points there via `paths.dropbox_clip_dir`. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/model_weights/*.pt` | `/mnt/rip/vicarius_drive/vicarius/models/yolo_pretrained/` | Pretrained YOLO bases (n/m/l + yolov8n + yolo26n) — versioned per Commandment IX. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/projects/<id>/project.json` + small state files | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess/run_<id>/project.json` + `analysis_params.yaml` | One VICARIUS "run" per project. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/projects/<id>/step*_*/` (step-dir outputs — step-6 runs/, step-7 reports, step-8 inference overlays) | `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess/run_<id>/outputs/step*_*/` | Outputs stay scoped to the run that produced them until promoted to `/processed/` or `/annotations/ai_predictions/` via `shelve_run.py`. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/env/` | **skipped** | Rebuilt via `bootstrap.sh` on target host. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/scripts/**/env/` | **skipped** | Rebuilt. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/scripts/**/__pycache__/`, `*.pyc` | **skipped** | Rebuilt. |
| `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/scripts/TCRMPtrain_oceankindCV/oceankind_CV/.git/`, `ultralytics_src/.git/` | **skipped** | Re-cloneable from upstream. |

---

## 2. Create the module skeleton

```bash
mkdir -p /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo
mkdir -p /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess
mkdir -p /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/misc
mkdir -p /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts
mkdir -p /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/presets
mkdir -p /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/eval_presets
mkdir -p /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/docs
```

Populate the standard VICARIUS run-template files:

```bash
cp /mnt/rip/vicarius_drive/vicarius/modules/_template/github_repo/src/init_run.py \
   /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts/init_run.py

cp /mnt/rip/vicarius_drive/vicarius/modules/_template/github_repo/src/shelve_run.py \
   /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts/shelve_run.py

# The _template dir may or may not have this file — skip gracefully if absent.
cp /mnt/rip/vicarius_drive/vicarius/modules/_template/inprocess/_analysis_params_template.yaml \
   /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess/_analysis_params_template.yaml \
   2>/dev/null || echo "  (no template file found — will generate one in §6)"
```

---

## 3. Copy scripts, configs, docs (code only, no data)

### 3.1 scripts/ — all sub-tools + orchestrator + helpers (~14 MB)

```bash
rsync -a \
  --exclude=env/ \
  --exclude=__pycache__/ \
  --exclude=*.pyc \
  --exclude=output/ \
  --exclude=output_*/ \
  --exclude=.git/ \
  /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/scripts/ \
  /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts/
```

### 3.2 config/pipeline.yaml + presets + eval_presets

```bash
cp /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/pipeline.yaml \
   /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/pipeline.yaml

rsync -a \
  /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/presets/ \
  /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/presets/

rsync -a \
  /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/eval_presets/ \
  /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/eval_presets/
```

### 3.3 module.yaml goes at the repo root (not inside config/)

```bash
cp /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/config/module.yaml \
   /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/module.yaml
```

### 3.4 docs + top-level README + bootstrap.sh

```bash
rsync -a \
  /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/docs/ \
  /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/docs/

cp /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/README.md \
   /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/README.md

cp /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/bootstrap.sh \
   /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/bootstrap.sh

chmod +x /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/bootstrap.sh
```

### 3.5 .gitignore — DO NOT copy verbatim, write the VICARIUS variant

See §6.3 for the content; the write target is
`/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/.gitignore`.

### 3.6 Verify

```bash
# No env/ or __pycache__/ in the copy
find /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo \
     -type d \( -name env -o -name __pycache__ \) -print 2>/dev/null | head
# Expected: empty

du -sh /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo
# Expected: ~14 MB (scripts) + a few hundred KB (configs + docs) = ~15 MB total
```

---

## 4. Place supporting data under VICARIUS data roots

### 4.1 Controlled vocabularies and tabular products

```bash
# Species code lookup — first-class vocabulary
mkdir -p /mnt/rip/vicarius_drive/vicarius/_METADATA/library/definitions

cp /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/master_codes.csv \
   /mnt/rip/vicarius_drive/vicarius/_METADATA/library/definitions/tcrmp_species_codes.csv

# Unified point-count dataset — derived tabular product
mkdir -p /mnt/rip/vicarius_drive/vicarius/processed/derived_tabular

cp /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/all_points.csv \
   /mnt/rip/vicarius_drive/vicarius/processed/derived_tabular/tcrmp_cvr_all_points.csv

# Sidecar metadata (Commandment III)
cat > /mnt/rip/vicarius_drive/vicarius/processed/derived_tabular/tcrmp_cvr_all_points.csv.meta.yaml <<'YAML'
produced_by: cvr_clip_segmentation.step1_makeAllPoints
source_rows: 839 Excel files under /mnt/rip/vicarius_drive/vicarius/raw/tcrmp_tabular/cvr/
columns: [date, year, site, transect, frame, point_label, species_code, species_name, category]
row_count_at_ingest: ~2300000
provenance: |
  Parser: /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts/TCRMPcvr_makeAllPoints/run.py
  See the sub-tool's README.md for format-version handling.
YAML
```

### 4.2 The 67 GB cpc_index — **operator confirmation required**

`cpc_all/` is a per-year, per-frame pre-computed index of CPCe `.cpc` annotation files. It's an intermediate derived product — neither raw nor a finished analytical output. Recommended placement: `processed/cpc_index/`. **Before copying, ask the operator**:

> The 67 GB pre-2020 CPCe point index (`supporting_data/cpc_all/`) is a derived
> static asset consumed by step 4 of the pipeline. Should it land at
> `processed/cpc_index/` (my recommendation), or does VICARIUS have a different
> canonical location for bulk derived indexes? If the operator prefers, it can
> stay as a pointer — the pipeline config (`config/pipeline.yaml` →
> `paths.cpc_all_dir`) can reference `$ORIGINAL_REPO/supporting_data/cpc_all/`
> instead, which avoids duplicating 67 GB.

If green-lit for copy:

```bash
mkdir -p /mnt/rip/vicarius_drive/vicarius/processed/cpc_index

rsync -aHAX --info=progress2 \
  /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/cpc_all/ \
  /mnt/rip/vicarius_drive/vicarius/processed/cpc_index/

# Verify byte-identical
src_sz=$(du -sb /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/cpc_all | awk '{print $1}')
dst_sz=$(du -sb /mnt/rip/vicarius_drive/vicarius/processed/cpc_index | awk '{print $1}')
[ "$src_sz" = "$dst_sz" ] && echo "cpc_index sizes match ($src_sz bytes)" || echo "SIZE MISMATCH — stop"
```

### 4.3 Raw CVR workbooks — **immutable per Commandment I**

```bash
mkdir -p /mnt/rip/vicarius_drive/vicarius/raw/tcrmp_tabular/cvr

rsync -a --info=progress2 \
  /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/TCRMP_CVR/ \
  /mnt/rip/vicarius_drive/vicarius/raw/tcrmp_tabular/cvr/

# After landing, mark read-only (Commandment I)
chmod -R a-w /mnt/rip/vicarius_drive/vicarius/raw/tcrmp_tabular/cvr
```

### 4.4 Pretrained model weights

```bash
mkdir -p /mnt/rip/vicarius_drive/vicarius/models/yolo_pretrained

cp /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/supporting_data/model_weights/*.pt \
   /mnt/rip/vicarius_drive/vicarius/models/yolo_pretrained/

# Ship a model card (Commandment III)
cat > /mnt/rip/vicarius_drive/vicarius/models/yolo_pretrained/README.md <<'MD'
# Pretrained YOLO bases

Ultralytics-provided starting weights used as step-6 training initializations.
Not fine-tuned on TCRMP data. Auto-downloaded on first use if missing; kept
here for offline reproducibility.

| File | Architecture | Size | Params |
|------|--------------|------|--------|
| yolo11n-seg.pt | YOLO11 nano seg | 6 MB  | 2.9M  |
| yolo11m-seg.pt | YOLO11 medium seg | 44 MB | 22.4M |
| yolo11l-seg.pt | YOLO11 large seg | 54 MB | 27.6M |
| yolov8n.pt     | YOLOv8 nano det | 6 MB  | 3.2M  |
| yolo26n.pt     | experimental    | 5 MB  | —     |
MD
```

### 4.5 TCRMP_clip local subset — **do not copy**

The 462 MB local subset is for offline-testing only and is a subset of Dropbox.
Dropbox remains the authoritative source. In `config/pipeline.yaml` the module
already knows how to point `${paths.dropbox_clip_dir}` at the mounted path.
Record the location in `module.yaml` as a referenced input, not a stored input.

---

## 5. Migrate in-flight runs

Each `projects/<id>/` in the source maps to one `inprocess/run_<id>/` on the VICARIUS side. Small state files travel verbatim; large step outputs land inside the run's `outputs/` tree.

```bash
for pdir in /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/projects/*/; do
    pname=$(basename "$pdir")
    rundir=/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess/run_${pname}
    mkdir -p "$rundir/outputs" "$rundir/logs"

    # 5.1 project.json verbatim
    cp "$pdir/project.json" "$rundir/project.json"

    # 5.2 Step outputs -> outputs/step<N>/  (preserve per-step separation)
    for stepdir in "$pdir"step*_*; do
        [ -d "$stepdir" ] || continue
        step_name=$(basename "$stepdir")
        rsync -aHAX "$stepdir/" "$rundir/outputs/$step_name/"
    done
    echo "migrated $pname -> $rundir"

    # 5.3 Stamp analysis_params.yaml (see §6 for the template body)
done
```

**Concrete source-side project inventory** (as of 2026-04-22 snapshot):

```
/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/projects/
  18aprilpm_PA_20260418_431d1b/                (3.0 GB — production run; includes step 6 training + step 7 eval + step 8 inference)
  test_april16pm_20260416_4eaf2f/              (241 MB — smoke test, retain)
  demo2_20260415_279c6c/                       (85 MB — demo)
  demo3_20260415_9f5664/                       (85 MB — demo)
  demo4_20260415_c1247e/                       (123 MB — demo)
  demo_labmeeting_16april_20260416_b0803c/     (85 MB — lab meeting demo)
  demo_try1/                                   (85 MB — first-run demo)
```

Each becomes a `run_<id>/` under:
```
/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess/
  run_18aprilpm_PA_20260418_431d1b/
  run_test_april16pm_20260416_4eaf2f/
  run_demo2_20260415_279c6c/
  ... etc
```

For each resulting `run_<id>/analysis_params.yaml`, the agent must populate:

```yaml
run:
  name: run_<id>
  module_name: cvr_clip_segmentation
  created_at: <ISO date — copy from project.json.created_at>
  created_by: <operator username from project.json if available>
  status: active | completed | archived    # infer from project.json step statuses

version:
  git_commit_hash: <to be filled by init_run.py equivalent>
  git_commit_short: ""
  git_branch: main
  git_remote_url: ""
  git_tag: ""
  is_clean: true

temporal:
  start_date: <project.json.created_at truncated to date>
  end_date: <project.json.updated_at if status == completed>
  duration_days: <derive>

processing:
  input:
    source_pattern: ""
    source_type: "cvr_clip"
  parameters:
    # Full copy of project.json["steps"]["N"]["config"] for every step
    step1: {...}
    ...
    step8: {...}
  output:
    output_dir: "outputs/"
    naming_prefix: ""

context:
  study: S1_historical          # default; operator may override
  purpose: |
    <extract from project.json if any; else "Migrated from CVR_CLIP_forAI pre-VICARIUS pipeline">
  notes: ""
  data_description: ""

shelving:
  shelved_at: ""
  shelved_by: ""
  disposition: ""
  final_notes: ""
  archive_location: ""
```

---

## 6. Rewrite paths and config

Every hardcoded absolute path that referenced the old layout must now reference the VICARIUS layout.

### 6.1 Update `config/pipeline.yaml` in the destination

The new `repo_root` will auto-resolve as `parent(parent(pipeline.yaml))` = `github_repo/`. That means `${scripts_dir}`, `${config_dir}`, `${unified_env_python}` all continue to resolve correctly against `github_repo/`. The **data-pointing** paths, however, must be repointed to the VICARIUS data roots:

```bash
python3 <<'PY'
import yaml
from pathlib import Path

p = Path("/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/pipeline.yaml")
doc = yaml.safe_load(p.read_text())

doc["paths"]["all_points_csv"]    = "/mnt/rip/vicarius_drive/vicarius/processed/derived_tabular/tcrmp_cvr_all_points.csv"
doc["paths"]["master_codes_csv"]  = "/mnt/rip/vicarius_drive/vicarius/_METADATA/library/definitions/tcrmp_species_codes.csv"
doc["paths"]["cpc_all_dir"]       = "/mnt/rip/vicarius_drive/vicarius/processed/cpc_index"
doc["paths"]["tcrmp_cvr_dir"]     = "/mnt/rip/vicarius_drive/vicarius/raw/tcrmp_tabular/cvr"
doc["paths"]["model_weights_dir"] = "/mnt/rip/vicarius_drive/vicarius/models/yolo_pretrained"

# projects_dir now lives under the module's inprocess/ tree
doc["paths"]["projects_dir"]      = "/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess"

# supporting_data_dir is no longer meaningful in VICARIUS (data lives elsewhere).
# Repoint to the inprocess/ tree so anything referencing ${supporting_data_dir}
# degrades to a valid path rather than pointing at a ghost directory.
doc["paths"]["supporting_data_dir"] = "/mnt/rip/vicarius_drive/vicarius/processed"

# vicarius_root and vicarius_logging_lib remain correct
# unified_env_python auto-resolves against repo_root (github_repo/env)
# step_defaults stay untouched — they reference the path aliases via ${...}

p.write_text(yaml.safe_dump(doc, sort_keys=False))
print("updated", p)
PY
```

After editing, run the loader to verify every `${...}` resolves:

```bash
cd /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo
python3 -c "
import sys
sys.path.insert(0, 'scripts/pipeline_orchestrator')
import orchestrator_config as C
import json
assert '\${' not in json.dumps({k:v for k,v in vars(C).items() if k.isupper()}, default=str)
print('ok -- every var resolved')
print('step1 input_dir:', C.DEFAULT_STEP_CONFIGS['1']['input_dir'])
print('step4 clip_dir :', C.DEFAULT_STEP_CONFIGS['4']['clip_dir'])
print('cpc_all_dir    :', C.PATHS['cpc_all_dir'])
print('all_points_csv :', C.PATHS['all_points_csv'])
print('projects_dir   :', C.PATHS['projects_dir'])
"
```

### 6.2 Rewrite absolute paths inside every migrated `run_<id>/project.json`

The migrated `run_*/` dirs still contain absolute paths pointing at the **source** repo (per-step `dir:`, `repo_dir`, absolute symlinks under `step5_segmentImages/all_images/`, and `step5_segmentImages/data.yaml` `path:` line). Run the rewriter against the VICARIUS run locations:

```bash
python3 /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts/rewrite_project_paths.py \
  --old /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026 \
  --new /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo \
  --projects-dir /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess \
  --apply
```

**Also** rewrite references to the OLD `CVR_CLIP_forAI/` repo root (pre-reorganization paths that might have survived in older `project.json` entries):

```bash
python3 /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/scripts/rewrite_project_paths.py \
  --old /mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI \
  --new /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo \
  --projects-dir /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess \
  --apply
```

After both passes, grep to confirm no stale references remain:

```bash
grep -rn "CVR_CLIP_forAI\|seg_AI_img_full_april2026" \
  /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess \
  2>/dev/null || echo "clean"
```

### 6.2b Re-target per-step output paths inside `project.json`

`project.json` still stores per-step paths like:
```
steps.N.dir = ".../seg_AI_img_full_april2026/projects/<id>/stepN_xxx"
```
The rewriter above changed the prefix to `.../github_repo/projects/<id>/stepN_xxx`, but the project outputs now actually live under `.../inprocess/run_<id>/outputs/stepN_xxx/`. Run this one-off patch:

```bash
python3 <<'PY'
import json
from pathlib import Path

inproc = Path("/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess")
github_repo = Path("/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo")

for pj in inproc.glob("run_*/project.json"):
    rundir = pj.parent
    state = json.loads(pj.read_text())
    # repo_dir: the github_repo location
    state["repo_dir"] = str(github_repo)
    # project_dir: the run_<id>/ itself
    state["project_dir"] = str(rundir)
    # Per-step dirs: moved under outputs/
    for k, step in state.get("steps", {}).items():
        old = Path(step.get("dir", ""))
        step["dir"] = str(rundir / "outputs" / old.name)
    tmp = pj.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(pj)
    print(f"repointed {pj.relative_to(inproc)}")
PY
```

### 6.3 Write `.gitignore` for the VICARIUS location

The `inprocess/` tree and the VICARIUS data roots live OUTSIDE the module's `github_repo/`. So the module's own `.gitignore` only needs to cover `github_repo/`-local exclusions. Write it to:

```
/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/.gitignore
```

…with this content:

```gitignore
# env/ and caches live inside github_repo/ and must not be committed
env/
__pycache__/
*.pyc
**/*.egg-info/

# Vendored tree .git dirs (re-cloneable)
scripts/TCRMPtrain_oceankindCV/oceankind_CV/.git/
scripts/TCRMPtrain_oceankindCV/ultralytics_src/.git/

# OS / editor cruft
.DS_Store
.idea/
.vscode/
```

`inprocess/`, `supporting_data/`, `projects/` entries from the source `.gitignore` are obsolete — those directories no longer live inside `github_repo/`.

---

## 7. Register the module with VICARIUS

```bash
# 7.1 Trigger module re-discovery
python3 -c "
import sys
sys.path.insert(0, '/mnt/rip/vicarius_drive/vicarius_ui')
from module_registry import discover_modules
mods = discover_modules()
print('discovered:', [m['name'] for m in mods])
assert any(m['name'] == 'cvr_clip_segmentation' for m in mods), 'module not found'
"

# 7.2 Verify the CLI sees it
vicarius list | grep cvr_clip_segmentation || \
    echo "MISSING -- check /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/module.yaml"
```

If the CLI can't see the module, the most common cause is `module.yaml` at the wrong path. It must live at:

```
/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/module.yaml
```

…NOT inside a `config/` subdirectory. Re-check step 3.3.

---

## 8. Hook up VICARIUS logging

The orchestrator already imports `vicarius_bridge` (no-ops if the log lib isn't reachable). In the VICARIUS location, the lib path is valid so the bridge will initialize for real.

Verify live logging:

```bash
# Launch the orchestrator from the VICARIUS module
cd /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo
./scripts/run_pipeline.sh 5050 &
sleep 5

# From another shell
vicarius story --days 0 | grep cvr_clip_segmentation

# Or query the event DB directly
sqlite3 /mnt/rip/vicarius_drive/vicarius/_logging/db/vicarius_events.db \
  "SELECT timestamp, event_type, process_module, context_purpose
     FROM events
    WHERE process_module LIKE 'cvr_clip_segmentation%'
    ORDER BY timestamp DESC LIMIT 20;"
```

If no events appear, check:
- `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo/config/pipeline.yaml` → `vicarius.enabled: true` is still set.
- `paths.vicarius_logging_lib` points at `/mnt/rip/vicarius_drive/vicarius/_logging/src` and that directory is readable.
- `/mnt/rip/vicarius_drive/vicarius/_logging/db/vicarius_events.db` is writable by the operator running the orchestrator.

---

## 9. UI integration — match VICARIUS aesthetic

The module already ships a re-themed orchestrator (VICARIUS palette: dark bg, bubblegum pink + T-Mobile magenta, Inter body, Space Mono code, Syne headings, Silkscreen wordmark accent, crosshair cursor, subtle pixel grid + scanlines). Rules for landing it in VICARIUS:

1. **Do not diverge from `vicarius_ui/static/theme.css`**. The module's own `style.css` is a deliberate mirror of the platform theme. If the platform palette changes, regenerate the module's `style.css` to match — do not drift.

2. **Font choices are content, not decoration**. Keep Inter ≥ 14 px, Space Mono for any monospace surface (logs, code, path inputs), Silkscreen only for the top-bar wordmark and setup-screen logo. Never use Silkscreen for body text.

3. **Emoji is banned platform-wide**. The source has been purged (0 decorative unicode remaining in live code). If new sub-tool UIs are added, run this check before accepting:
   ```bash
   python3 -c "
   import re, pathlib
   rx = re.compile('[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U00002B00-\U00002BFF\u2190-\u21FF]')
   for p in pathlib.Path('github_repo/scripts').rglob('*'):
       if p.suffix in ('.py','.js','.html','.css','.yaml') and p.is_file():
           m = rx.findall(p.read_text())
           if m: print(p, sorted(set(m)))
   "
   ```
   The only permitted exception is the `★` semantic-cost marker in the orchestrator's step-6 training panel (visual-weight indicator for hyperparameter tuning cost, not decoration).

4. **Port allocation**: use the documented ports (orchestrator 5050, recode 5055, route 5065, segment 5070). When VICARIUS itself is running on 5077, there is no collision — but confirm with `lsof -ti:5050` before first launch.

5. **Navigation integration**: the module's orchestrator opens in its own browser tab at `http://localhost:5050`. It does not embed inside the VICARIUS `vicarius_ui` shell at this time. `vicarius list` points operators at it; `vicarius run cvr_clip_segmentation` launches it. Do **not** try to iframe-embed the orchestrator into the main `vicarius_ui` — the sub-apps it launches (step 2, 4, 5) run on different ports and Flask sessions that do not survive iframing.

6. **Readability over personality**. If a styling choice makes a form field or log line harder to scan, bias toward the readable choice. Every token in the current theme passed this bar; hold that line.

---

## 10. Testing protocol (in order — stop on first failure)

### 10.1 Config loader
```bash
cd /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo
python3 -c "
import sys, os, json
sys.path.insert(0, 'scripts/pipeline_orchestrator')
import orchestrator_config as C
assert '\${' not in json.dumps({k:v for k,v in vars(C).items() if k.isupper()}, default=str)
for p in [C.DEFAULT_STEP_CONFIGS['1']['input_dir'],
          C.DEFAULT_STEP_CONFIGS['4']['clip_dir'],
          C.PRESET_DIRS['train'],
          C.PATHS['all_points_csv'],
          C.PATHS['master_codes_csv'],
          C.PATHS['cpc_all_dir']]:
    print(f'{p}  exists={os.path.exists(p)}')"
# Dropbox clip_dir may show exists=False on a fresh host; that's fine as long
# as the Dropbox mount is set up when step 4 is actually exercised.
```

### 10.2 Bootstrap the unified env
```bash
cd /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo
./bootstrap.sh
# Expect: env/ created (conda, Python 3.11, torch cu128, ultralytics, transformers, easyocr, flask, pandas)
#         project paths rewritten inside inprocess/*
#         smoke test reports [ok] for every import
```

### 10.3 Orchestrator launches
```bash
cd /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/github_repo
./scripts/run_pipeline.sh 5050
# In another shell:
curl -sf http://localhost:5050/ > /dev/null && echo "index ok"
curl -sf http://localhost:5050/api/defaults > /dev/null && echo "defaults ok"
curl -sf http://localhost:5050/api/step/6/presets > /dev/null && echo "presets ok"
curl -sf http://localhost:5050/api/remap/browse > /dev/null && echo "remaps ok"
```

### 10.4 Open a migrated run
Open http://localhost:5050, click "Open Project", paste this path:

```
/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess/run_18aprilpm_PA_20260418_431d1b
```

Expected:
- All completed steps read `completed` on the sidebar.
- Step 7 "Open Report" serves the existing `report.pdf` out of `.../outputs/step7_evaluateModel/`.
- No broken absolute paths in `project.json` (confirmed by the path rewriter).

### 10.5 Small end-to-end smoke test
Create a tiny new project in the UI:
- Project name: `vicarius_smoke_test`
- Base dir: `/mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation/inprocess`
- Step 1: run against `/mnt/rip/vicarius_drive/vicarius/raw/tcrmp_tabular/cvr/` (read-only — will not write)
- Step 2: launch the recode UI on 5055, confirm it loads `/mnt/rip/vicarius_drive/vicarius/processed/derived_tabular/tcrmp_cvr_all_points.csv` and closes cleanly
- Step 3: run with `target_species="PA"`, `target_instances=10`, `min_year=2018`, `max_year=2018` — tiny selection for speed
- Step 4: launch on 5065, override `clip_dir` to `/home/bizon/UVI Dropbox/SMITH LAB TEAM FOLDER/TCRMP/TCRMP_clip` (or whatever mount is available on the host)
- Skip steps 5-8 (GPU-heavy; require SAM3 + full training setup)

After each step completes, this must show a `process_start` + `process_end` pair linked to that step:

```bash
sqlite3 /mnt/rip/vicarius_drive/vicarius/_logging/db/vicarius_events.db \
  "SELECT timestamp, event_type, process_module, process_status
     FROM events
    WHERE process_module LIKE 'cvr_clip_segmentation%'
    ORDER BY timestamp DESC LIMIT 10;"
```

### 10.6 Rollback check
The source repo must still work unchanged. Launch this from the original location:

```bash
/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/scripts/run_pipeline.sh 5052
```

Both the source (port 5052 in this test) and the VICARIUS-hosted copy (port 5050) should coexist without any shared state leaking between them.

---

## 11. Ten Commandments compliance checklist

Fill this in as a sanity check before declaring the migration complete.

- [ ] **I. Raw data sacred** — `raw/tcrmp_tabular/cvr/` was `chmod -R a-w` after landing. No step writes there.
- [ ] **II. Naming conventions** — `tcrmp_cvr_all_points.csv`, `tcrmp_species_codes.csv`, `coral_seg_v1.0.0_<YYYYMMDD>.pt` pattern documented for future model promotions.
- [ ] **III. Every file has metadata** — every migrated run has `analysis_params.yaml`; every derived product has a sidecar `.meta.yaml`.
- [ ] **IV. Modules atomic** — `cvr_clip_segmentation` is one module with one `module.yaml`, one inputs contract, one outputs contract.
- [ ] **V. Document as you build** — `docs/PIPELINE_OVERVIEW.md`, `docs/NOTES_pipeline_improvements.md`, `docs/VICARIUS_INTEGRATION.md`, this file: present.
- [ ] **VI. Prompt for purpose** — project-creation UI asks for name + purpose; `process_start` events carry that purpose string.
- [ ] **VII. Test before production** — §10 must pass end to end before the module is marked production-ready.
- [ ] **VIII. Log everything** — VICARIUS event stream captures orchestrator start, every step start/end, every error, every run open/close.
- [ ] **IX. Version code, track data** — `.gitignore` excludes `env/` + editable-install artifacts; data lives on `/mnt/rip` + NAS, not in git.
- [ ] **X. Keep it simple, then grow** — no new frameworks introduced; the module remains plain Flask + vanilla JS + a single YAML.

---

## 12. Rollback plan

If something breaks after migration:

1. **Stop the VICARIUS-hosted orchestrator** (`kill` any pid bound to 5050/5055/5065/5070 that is running from `modules/cvr_clip_segmentation/`).
2. **The source repo is unchanged**. Operators can continue working from `/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/` exactly as before.
3. **Revert by deleting the module directory** (after confirming nothing under `inprocess/` has been touched that doesn't also exist in the source `projects/`):
   ```bash
   # DANGER — this removes the migrated module. Run only after confirming rollback.
   rm -rf /mnt/rip/vicarius_drive/vicarius/modules/cvr_clip_segmentation
   ```
4. **Leave the data-root copies in place** unless they turned out to be duplicates. `processed/derived_tabular/tcrmp_cvr_all_points.csv` etc. are cheap to keep and may be read by other modules.

---

## 13. Known unknowns — ask the operator before acting

The following decisions materially change the plan. Do not guess; surface them.

1. **Does VICARIUS have a canonical root for bulk derived indexes like `cpc_all/` (67 GB)?** My recommendation is `processed/cpc_index/`. If the operator has a different convention, use theirs.

2. **Should the pretrained YOLO weights land under `models/yolo_pretrained/` or a more structured `models/tcrmp_coral_seg/v0_pretrained/` tree?** My default is the former; the latter is cleaner if VICARIUS uses `{model_family}/{version}/` throughout.

3. **Which study tag do migrated runs inherit?** I default to `S1_historical` everywhere. If a migrated run belongs to a different study (S2_3D_structure, S3_colony_tracking, S4_non_coral, S5_interactions), the operator must tell you which.

4. **Is the Dropbox mount path the same under the VICARIUS-hosted orchestrator?** Same host, same mount, same `config.dropbox_clip_dir`. If a collaborator is running from a different workstation, the path differs and must be surfaced to the operator.

5. **Git repository for the module**: does it live as a standalone GitHub repo (under the operator's GH account or an org), or as a monorepo sub-tree inside VICARIUS? If standalone, initialize `git init` + first commit inside `github_repo/` immediately after §3 finishes. If monorepo, skip.

6. **`master_codes.csv` promotion to a first-class vocabulary**: putting it under `_METADATA/library/definitions/` means OTHER modules may start depending on it. Confirm with the operator that the species-code contract is stable enough to publish as a cross-module vocabulary. If not, keep it alongside `all_points.csv` under `processed/derived_tabular/` and only promote it once the recodeSpecies cycle settles.

---

## 14. One-liner summary for the operator after completion

Post to the operator:

> `cvr_clip_segmentation` landed in VICARIUS. Module lives at
> `modules/cvr_clip_segmentation/`. Source code unchanged — the old repo at
> `hopper/CVR_CLIP_forAI/seg_AI_img_full_april2026/` still works independently.
> Run `./bootstrap.sh` inside `github_repo/` once to build the unified env, then
> `vicarius run cvr_clip_segmentation` launches the orchestrator at
> http://localhost:5050. Raw CVR workbooks are at `raw/tcrmp_tabular/cvr/`
> (chmod a-w). Derived `all_points.csv` at
> `processed/derived_tabular/`. Species vocabulary at
> `_METADATA/library/definitions/tcrmp_species_codes.csv`. Run events stream
> to `vicarius story`. All 10 Commandments verified green.
