# TCRMPcvr_recodeSpecies

Web-based species code recoding tool for TCRMP benthic point-count data.

Coral taxonomy has changed over time (e.g., *Montastraea faveolata* was moved to *Orbicella faveolata*), and the 839 historical CVR spreadsheets used different codes across eras. This tool lets you interactively remap species codes so the unified dataset (`all_points.csv`) uses consistent, current taxonomy, then outputs corrected files for downstream analysis.

## What It Does

1. Loads the master point dataset and species code lookup table
2. Shows every code with its instance count, sorted most-frequent first
3. Lets you edit `new_code`, `new_name`, and `new_category` for any code
4. Lets you **exclude** specific codes (e.g., juvenile coral) from the top-10 coral ranking
5. On save: produces recoded `all_points` + `master_codes` CSVs, an audit log, and the top-10 coral species by point count

## Input

| File | Source | Description |
|------|--------|-------------|
| `../output/all_points.csv` | `TCRMPcvr_makeAllPoints` | ~1.6M point observations (2001–2025) |
| `../output/master_codes.csv` | `TCRMPcvr_makeAllPoints` | ~130 species/substrate codes |

## Output

All outputs go to `output/` within this directory, timestamped so you can run multiple times:

| File | Description |
|------|-------------|
| `all_points_{ts}.csv` | Recoded point dataset with updated codes/names/categories |
| `master_codes_{ts}.csv` | Updated species lookup (merged codes removed) |
| `remap_log_{ts}.json` | Audit trail: every change, affected row counts, excluded codes |

## Setup

```bash
bash setup_env.sh
```

Creates a lightweight conda env with only `flask` and `pandas` (no GPU needed).

## Quick Start

```bash
./run.sh          # opens browser at http://localhost:5055
./run.sh 5060     # custom port
```

Or manually:

```bash
./env/bin/python src/app.py --port 5055
```

Custom input paths:

```bash
./env/bin/python src/app.py \
    --all-points /path/to/all_points.csv \
    --master-codes /path/to/master_codes.csv
```

## Usage

1. Open the UI — codes are sorted by frequency (most points first)
2. Use the **filter box** to search by code, name, or category
3. Edit any row's `new_code`, `new_name`, or `new_category` (changed cells highlight red)
4. Check **exclude** on codes you want omitted from the top-10 coral ranking (e.g., CORJU for juvenile corals)
5. Click **Save & Recode**
6. Review the top-10 coral table and output file names at the bottom

### Example: Merging MFAV → OFAV

1. Filter for "fav"
2. On the MFAV row, change `new_code` to `OFAV`, `new_name` to `Orbicella faveolata`
3. Save — all MFAV points become OFAV, MFAV is removed from master_codes

## How Recoding Works

- Each recode operates on the **original** loaded data, not previous outputs — fully non-destructive
- When `old_code ≠ new_code`: the old code row is removed from master_codes, points are reassigned to the new code
- When only name/category changes: the code stays, metadata is updated in both files
- Chain remaps (A→B and B→C) work correctly — matching uses original codes

## Project Structure

```
TCRMPcvr_recodeSpecies/
├── README.md              this file
├── run.sh                 launcher (port 5055)
├── setup_env.sh           conda env setup (flask + pandas)
├── .gitignore
├── src/
│   ├── app.py             Flask backend
│   └── templates/
│       └── index.html     single-page recode UI
└── output/                created at runtime
    ├── all_points_*.csv
    ├── master_codes_*.csv
    └── remap_log_*.json
```

---

## Planned Next Steps

This tool is **Step 1** of a multi-step pipeline for building coral segmentation training data.

### Step 2 — Balanced Image Selection (next to build)

After recoding, select the **top 3–5 coral species** and generate a balanced training image list:

- **Goal:** ≥1,000 point instances per species
- **Year bias:** 70% post-2018, 30% 2013–2018, nothing before 2013 (older point-count system was unreliable)
- **Site balance:** even representation across TCRMP sites
- **Process:**
  1. From the recoded `all_points`, filter to the selected coral species
  2. Sample frames (each frame has 20 points) to hit the 1,000-instance target per species while balancing site and year
  3. Output an image list CSV mapping each selected frame to its source path in `input/TCRMP_clip/`
  4. For **pre-2020 images**: look up existing point coordinates from `output/cpc_all/{year}/ids/`
  5. For **2020+ images**: copy the `_pts` images and run through `TCRMPclip_ocrID_batch` to extract point coordinates via GPU OCR

### Step 3 — OCR Batch Processing

Run the selected 2020+ images through the existing `TCRMPclip_ocrID_batch` pipeline to produce:
- Raw images, point coordinate CSVs, SAM click prompts
- Year-organized output matching `output/ocr_all/` structure

### Step 4 — SAM3 Segmentation (separate subdirectory)

Use the OCR-extracted (x, y) point coordinates as virtual "click prompts" for SAM3 (Segment Anything Model 3) to automatically generate per-species segmentation masks for each image. This will be implemented as its own workflow subdirectory.
