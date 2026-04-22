# TCRMPclip_ocrID_batch

GPU-accelerated batch OCR detection for TCRMP coral reef `_pts` annotated images (2020+), with web-based QAQC review and year-organized export compatible with the `cpc_all` output format.

## What It Does

This tool processes **all** `_pts` images from the TCRMP_clip Dropbox directory in a single OCR run, then lets you review and export in manageable batches. It replaces the old per-batch processing model with a two-phase approach:

1. **Phase 1 — Processing**: Scan the entire TCRMP_clip input directory for selected years (2020–2025), run GPU-accelerated EasyOCR on every `_pts` image to detect the 20 red letter annotations (A–T) and their crosshair positions, enrich with species data from `all_points.csv`, and save results to per-year JSON detection files. This phase is **resumable** — if interrupted, it picks up where it left off.

2. **Phase 2 — Review**: Load all detection results and present them in review batches of 5, 10, or 20 images. Use the full-screen QAQC canvas to verify and adjust point positions. Export reviewed batches to year-organized output directories matching the `cpc_all` format.

### How It Differs from TCRMPclip_ocrID (Original)

| Feature | Original (ocrID) | Batch (ocrID_batch) |
|---------|-------------------|---------------------|
| Processing model | Process small batches interactively | Process ALL images first, then review |
| Output organization | Flat export directory | Year-organized (`2020/`, `2021/`, ...) |
| Resume support | Per-batch manifest | Full crash recovery with per-year detection JSON |
| Period handling | N/A | Periods folded into year (Annual + PBL + SCTLD → one dir) |
| Output format | Custom (with pts_image, ocr_confidence fields) | cpc_all compatible (matches existing pipeline) |
| Scale | Designed for ~100 images | Designed for ~29,000+ images |

## System Requirements

- NVIDIA GPU with CUDA 12.8+ driver (tested on 2x RTX 5090, SM 12.0)
- conda (Anaconda or Miniconda)
- ~4GB disk for the conda environment
- Linux (tested on Ubuntu 24.04)

## Setup

```bash
cd TCRMPclip_ocrID_batch
bash setup_env.sh
```

This creates a local conda env at `./env/` with PyTorch (cu130), EasyOCR, Flask, and dependencies.

## Quick Start

```bash
./run.sh
```

Opens the web UI at `http://localhost:5050`. Default port can be changed: `./run.sh 8080`.

## Workflow

### Starting a New Project

1. Launch the app and select **"Start New Processing"**
2. Set the **Import Directory** to the TCRMP_clip root:
   ```
   /home/bizon/UVI Dropbox/SMITH LAB TEAM FOLDER/TCRMP/TCRMP_clip
   ```
   **WARNING**: This directory is READ-ONLY. The app never modifies, deletes, or writes to it.
3. Set the **Export Directory** to where you want output (e.g., `/mnt/rip/.../output/ocr_all`)
4. Click **"Scan Directories"** to see available years and image counts
5. Select which years to process (2020–2025) using the checkboxes
6. Choose a **Review Batch Size** (5, 10, or 20)
7. Click **"Start Processing"**

The processing screen shows a progress bar with ETA. For ~29K images at ~0.5s each, expect ~4 hours. Detection results are auto-saved to disk every 50 images.

### Resuming an Interrupted Project

1. Launch the app and select **"Continue Existing"**
2. Enter the **Export Directory** from a previous session
3. Click **"Load Project"** — the app discovers all year directories and detection data
4. If processing was incomplete, it resumes OCR from where it left off
5. If processing was complete, it goes straight to review mode

### Reviewing and Exporting

After processing completes (or on resume), you enter the QAQC review screen:

- **Green circles**: Detected crosshair positions with high OCR confidence
- **Orange circles/labels**: Unidentified letters (need attention)
- **Red circles**: Missing crosshairs
- **Yellow lines**: Letter-to-crosshair associations

Navigate through images, adjust points if needed, then click **"Export Batch"** to write the current batch to the year-organized output.

#### Keyboard Controls

| Key | Action |
|-----|--------|
| **Space** | Select next label (unidentified first) |
| **Enter** | Lock in current point position |
| **Arrow keys** | Nudge selected point 1px (Shift = 5px) |
| **Mouse drag** | Move selected point freely |
| **N** | Save current image, go to next |
| **B** | Save current image, go to previous |
| **F** | Fit image to viewport |
| **Scroll** | Zoom in/out |
| **Alt+drag** / **Middle-drag** | Pan |
| **Ctrl+S** | Save current image points |
| **Esc** | Deselect point |

## Output Structure

Each year gets its own directory matching the `cpc_all` format:

```
<export_dir>/
├── 2020/
│   ├── raw/                          # Clean images (WITHOUT _pts annotations)
│   ├── ids/
│   │   ├── sam_click_prompts.json    # SAM3 click prompt format
│   │   └── point_coords.csv         # Flat CSV of all points
│   ├── test_pts/                     # Verification overlays (crosshair + label on raw)
│   ├── dataset_summary.txt           # Site/transect/species statistics
│   ├── log.txt                       # Timestamped processing log
│   ├── ocr_detections.json           # Internal: full OCR results (for resume)
│   └── processed_manifest.json       # Internal: tracking (for resume)
├── 2021/ ... 2025/                   # Same structure per year
```

### `ids/sam_click_prompts.json`

Primary output for downstream SAM3 segmentation:

```json
{
  "TCRMP20201022_clip_SWT_T101.jpg": {
    "raw_image": "raw/TCRMP20201022_clip_SWT_T101.jpg",
    "points": [
      {
        "label": "A",
        "species": "SSID",
        "name": "Siderastrea siderea",
        "category": "Coral",
        "x": 164.0,
        "y": 1224.0,
        "point_type": 1
      }
    ]
  }
}
```

### `ids/point_coords.csv`

Flat CSV for analysis (8 columns, matching `cpc_all` format):

```
raw_image,label,species_code,species_name,category,x,y,source
TCRMP20201022_clip_SWT_T101.jpg,A,SSID,Siderastrea siderea,Coral,164.0,1224.0,ocr
```

The `source` column is `"ocr"` to distinguish from CPC-parsed points (`"cpc"`).

### `dataset_summary.txt`

Per-year summary with all periods folded:

```
2020 TCRMP_clip OCR Dataset Summary
======================================================================
Dataset: 28 sites, 168 transects, 6195 images, 123900 points (95% with species)
  Site  Trans  Images  Points  Species
------ ------ ------- ------- --------
   BID      6      89    1780    100%   T1.2.3.4.5.6
   ...
```

## Input Directory Layout

The app handles all TCRMP_clip directory layouts:

| Year | Layout | Periods |
|------|--------|---------|
| 2020 | Period subdirectories | Annual/, PostBL/, SCTLD/ |
| 2021–2023 | Flat date_site directories | (none) |
| 2024–2025 | Period subdirectories | Annual/, PBL/ |

All periods are **folded** into a single year in the output. The scanner recursively finds all `*_pts.jpg`/`*_pts.jpeg` files regardless of the subdirectory structure.

## Adding a New Season or Period

When new data arrives (e.g., TCRMP2026_clip, or a new period added to 2025):

1. Place the new data in the TCRMP_clip Dropbox directory following the existing naming convention:
   - Year directory: `TCRMP2026_clip/`
   - Optional period subdirs: `Annual/`, `PBL/`, etc.
   - Site directories: `TCRMP20261015_clip_BIT/`
   - Files: `TCRMP20261015_clip_BIT_T101.jpg` (raw) and `..._T101_pts.jpg` (annotated)

2. Launch the app and select **"Start New Processing"**

3. Set the import and export directories (use the same export dir as before)

4. Select the new year(s) in the year checkboxes — the app will detect that existing years are already processed and only queue new images

5. Process and review as usual

The app is fully incremental: existing detection data, manifests, CSVs, and JSON files are preserved and appended to.

### If adding a new period to an existing year

If a new period (e.g., `SCTLD/`) is added to an already-processed year (e.g., 2024):

1. The scanner will find the new `_pts` files in the new period subdirectory
2. Existing detections for that year are preserved
3. Only the new (unprocessed) images are queued for OCR
4. On export, the new data is merged into the existing year's output files

## Detection Algorithm

1. **Red mask**: `(R > 150) & (G < 100) & (B < 100)` — captures all red annotations
2. **Connected components**: `scipy.ndimage.label` + `find_objects` (single-pass)
3. **Letter selection**: Top 20 largest components by pixel count
4. **Crosshair search**: For each letter, search 5–100px below for red pixels, compute centroid
5. **OCR**: Crop each letter with 10px padding, scale 4x, threshold, run EasyOCR with `allowlist='ABCDEFGHIJKLMNOPQRST'`
6. **Constraint resolution**: Greedy assignment by confidence (highest first), remaining filled by elimination
7. **Species enrichment**: Join with `all_points.csv` via exact date match, fuzzy date (±7 days), or anagram site code correction

## Project Structure

```
TCRMPclip_ocrID_batch/
├── run.sh              # Launch script
├── setup_env.sh        # Creates conda env with all dependencies
├── README.md           # This file
├── .gitignore
├── env/                # Conda environment (created by setup_env.sh)
└── src/
    ├── app.py          # Flask web app (backend + API routes)
    ├── detect.py       # Letter + crosshair detection engine (EasyOCR)
    ├── scanner.py      # Recursive directory scanner for TCRMP_clip
    ├── species.py      # Species lookup with fuzzy date matching
    ├── export.py       # Year-organized export (raw, ids, test_pts)
    ├── summarize.py    # Dataset summary generation
    ├── speed_test.py   # EasyOCR vs Tesseract benchmark
    └── templates/
        └── index.html  # Single-page web UI
```

## Notes

- The EasyOCR model is loaded once at server startup (~2–3s) and reused for all images
- Detection results are auto-flushed to disk every 50 images during processing
- Species data comes from `../../output/all_points.csv` (generated by the CVR pipeline)
- If no species data is available, points are still detected but species fields will be empty
- The import directory is **never modified** — it is treated as read-only at all times
