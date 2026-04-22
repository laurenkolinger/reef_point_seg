# TCRMPcvr_chooseImages

Balanced image selection for coral segmentation training data.

Given a set of target coral species, selects frames from the TCRMP point-count dataset that provide at least 1,000 frame-instances of each species, distributed **evenly across years** (2014–2025) and balanced across sites and transects to avoid training bias (e.g., repeatedly photographing the same coral colony from the same transect).

## What It Does

1. Loads recoded `all_points.csv` and `master_codes.csv`
2. For each target species, allocates ~equal frame-instances per year (capped by availability)
3. Greedily selects frames year-by-year, preferring frames with multiple target species
4. Verifies each selected frame's source image exists on disk
5. Routes frames to the appropriate pipeline:
   - **CPC** (2014–2019): point coordinates already extracted in `output/cpc_all/`
   - **OCR done** (2020+): already processed by `TCRMPclip_ocrID_batch`
   - **OCR needed** (2020+): needs OCR processing to extract point coordinates
6. Outputs CSVs, a summary, and diagnostic plots

## Input

| File | Source | Description |
|------|--------|-------------|
| `../TCRMPcvr_recodeSpecies/output/all_points_*.csv` | `TCRMPcvr_recodeSpecies` | Recoded point data (auto-detects latest) |
| `../output/master_codes.csv` | `TCRMPcvr_makeAllPoints` | Species code lookup |
| `../input/TCRMP_clip/` | Raw imagery | Source images for path verification |
| `../output/cpc_all/` | `TCRMPclip_cpcID` | Pre-2020 processed point coords |
| `../output/ocr_all/` | `TCRMPclip_ocrID_batch` | Post-2020 processed point coords |

Falls back to `../output/all_points.csv` if no recoded data exists.

## Output

All outputs in `output/`:

| File | Description |
|------|-------------|
| `selected_frames.csv` | Master list: frame IDs, image paths, species present, routing |
| `route_cpc.csv` | Frames with existing CPC point coords (copy from cpc_all) |
| `route_ocr_done.csv` | Frames already OCR-processed |
| `route_ocr_needed.csv` | Frames needing OCR processing via `TCRMPclip_ocrID_batch` |
| `route_cpc_missing.csv` | Pre-2020 frames missing from cpc_all |
| `route_missing.csv` | Frames whose source image was not found on disk |
| `selection_summary.txt` | Human-readable summary of selection |
| `selection_diagnostics.png` | 8-panel distribution plots |
| `config_snapshot.txt` | Parameters used for this run |

### selected_frames.csv columns

| Column | Description |
|--------|-------------|
| `frame_id` | Unique frame identifier (date\|site\|transect\|frame) |
| `basename` | Image filename without extension (e.g., `TCRMP20180501_clip_BWR_T314`) |
| `date`, `year`, `site`, `transect`, `frame` | Metadata |
| `species_present` | Semicolon-separated target species in this frame |
| `n_target_species` | Count of target species present |
| `source_image` | Full path to raw image on disk |
| `pts_image` | Full path to _pts annotated image (if exists) |
| `image_found` | Whether source image was found |
| `route` | `cpc`, `ocr_done`, `ocr_needed`, or `cpc_missing` |
| `coords_source` | Path to point_coords.csv containing this frame's data |

## Setup

```bash
bash setup_env.sh
```

## Quick Start

```bash
# 1. Select images (uses species from config.py)
./env/bin/python src/select_images.py

# 2. Generate diagnostic plots
./env/bin/python src/plot_diagnostics.py
```

## Changing Target Species

Edit `src/config.py`:

```python
TARGET_SPECIES = ["OFRA", "PA", "OA", "OFAV", "AL", "MC", "AA"]
```

Or override via CLI:

```bash
# Different species
./env/bin/python src/select_images.py --species OFRA PA OA

# Different target count
./env/bin/python src/select_images.py --target 500

# Custom input
./env/bin/python src/select_images.py --all-points /path/to/all_points.csv

# Skip image verification (faster, no routing)
./env/bin/python src/select_images.py --skip-image-check
```

Then regenerate plots:
```bash
./env/bin/python src/plot_diagnostics.py
```

## Selection Algorithm

1. **Even year allocation:** For each species, divide the target (1,000) equally across all years (2014–2025). If a year has fewer frames than its share, cap at available and redistribute the shortfall to other years.

2. **Year-stratified greedy selection:** Within each year, greedily pick frames that contribute the most needed species. Frames containing multiple target species are preferred (reducing total frame count).

3. **No year bias:** Unlike temporal weighting (which over-represents recent years), this approach ensures the model sees equal representation from all survey years — critical for a model that must work on both historical and current imagery.

## Downstream Workflow

After running this tool:

1. **CPC-routed frames** (`route_cpc.csv`): Point coordinates already exist in `output/cpc_all/{year}/ids/point_coords.csv`. Copy the raw images and coords to the training set.

2. **OCR-needed frames** (`route_ocr_needed.csv`): These `_pts` images need to be processed through `TCRMPclip_ocrID_batch` to extract point coordinates via GPU OCR.

3. **After OCR processing:** All frames will have (x, y) point coordinates. These become SAM3 click prompts for automated coral segmentation (separate workflow).

## Project Structure

```
TCRMPcvr_chooseImages/
├── README.md
├── setup_env.sh
├── .gitignore
├── src/
│   ├── config.py              ← edit species/params here
│   ├── select_images.py       ← main selection script
│   └── plot_diagnostics.py    ← generate distribution plots
└── output/                    ← created at runtime
```
