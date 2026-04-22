# TCRMPclip_cpcID

> **Applicable to TCRMP clip images from 2013-2019 only.**
> Pre-2013 CPC annotations reference original un-clipped source images and cannot be mapped to clip images (see [Appendix](#appendix-year-coverage) below).
> CPCe was retired starting 2020; those images are being re-analyzed with OCR via `TCRMPclip_ocrID`.

Parses CPCe (Coral Point Count with Excel extensions) `.cpc` annotation files, extracts point coordinates (A-T), scales from CPCe canvas space to image pixels, joins species codes from `all_points.csv`, and exports in SAM-click-prompt format for downstream segmentation.

Part of the integrated undersea coral reef survey automation system.

## Requirements

- Python 3.8+
- Pillow (only external dependency)

## Setup

If Pillow is already available in your Python environment, no setup needed.
Otherwise:

```bash
bash setup_env.sh
```

This creates a local `./env/` with just Python + Pillow (~50MB, no GPU needed).

## Usage

### Single year
```bash
./run.sh /path/to/TCRMP2018_clip output/cpc_all/2018 --test-images -1
```

### Generate all test overlay images
```bash
./run.sh --test-pts output/cpc_all/2018 --workers 8
```

### Dataset summary
```bash
./run.sh --summarize output/cpc_all/2018
./run.sh --summarize --source /path/to/TCRMP2019_clip   # shows period breakdown
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--all-points PATH` | auto-detect | Path to `all_points.csv` for species lookup |
| `--test-images N` | 3 | Test overlay images per directory (0=skip, -1=all) |

## Output Structure

```
cpc_all/
├── 2013/
│   ├── raw/                          # Source images (copied)
│   ├── ids/
│   │   ├── sam_click_prompts.json    # SAM3 click prompts (primary output)
│   │   └── point_coords.csv         # Flat CSV of all points
│   ├── test_pts/                     # Visual verification overlays
│   ├── dataset_summary.txt           # Sites, transects, species coverage
│   └── log.txt                       # Processing log
├── 2014/
├── ...
└── 2019/
```

### sam_click_prompts.json format
```json
{
  "TCRMP20181101_clip_BIX_T101.jpg": {
    "raw_image": "raw/TCRMP20181101_clip_BIX_T101.jpg",
    "points": [
      {
        "label": "A",
        "species": "ROD",
        "name": "Sea rod",
        "category": "Gorgonian",
        "x": 3344.1,
        "y": 1242.1,
        "point_type": 1
      }
    ]
  }
}
```

### point_coords.csv format
```
raw_image,label,species_code,species_name,category,x,y,source
TCRMP20181101_clip_BIX_T101.jpg,A,ROD,Sea rod,Gorgonian,3344.1,1242.1,cpc
```

## CPC File Format

CPCe `.cpc` files are plain text:
```
Line 1:  "codefile.txt","image.jpg",canvas_w,canvas_h,...
Lines 2-5: Bounding box corners
Line 6:  Number of points (10, 15, or 20 depending on year)
Lines 7+: X,Y coordinates in canvas space
After coords: "Label","species_code","Notes","note_value"
```

Coordinates are scaled: `pixel = cpc_coord * (image_dim / canvas_dim)`

## Edge Cases Handled

| Issue | Resolution |
|-------|------------|
| ColorCorr subdirectories (2016) | Auto-detects images in parent dir |
| _EDITED subdirectories (2015) | Uses as primary source when parent has no CPC |
| CIG-prefixed files (2012) | Skipped (non-TCRMP naming) |
| JPEG subdirectories (2009-2013) | Skipped (duplicate re-analyses) |
| 4-character site codes (LBPD) | Regex handles variable length |
| Truncated filenames (TCRMP2017120) | Falls back to parent directory date |
| Site code typos (MSR vs MRS) | Anagram matching |
| Date mismatches (3-7 day offset) | Fuzzy match within 7-day window |
| Variable point counts (10/15/20) | Read from each CPC file header |
| Variable canvas sizes | Read from each CPC file header |

## Project Layout

```
TCRMPclip_cpcID/
├── src/
│   ├── app.py                # Main entry point
│   ├── parser.py             # CPC file parsing + directory discovery
│   ├── species.py            # Species lookup with fuzzy date matching
│   ├── process.py            # Core processing pipeline
│   ├── export.py             # SAM prompts, CSV, test image generation
│   ├── generate_test_pts.py  # Parallel test overlay generator
│   └── summarize.py          # Dataset summary (sites, transects, periods)
├── setup_env.sh              # Environment setup (if needed)
├── run.sh                    # Launcher script
├── README.md
└── .gitignore
```

---

## Appendix: Year Coverage

### Why only 2013-2019?

| Year Range | CPC Image Reference | Compatible with Clips? |
|------------|-------------------|----------------------|
| 2009-2012 | Original source images (`Image1.jpg`, etc.) | **No** - CPC coordinates are for the un-clipped originals. The clip images are cropped/resized differently, so point positions don't align. |
| 2013-2019 | Clip images (`TCRMP*_clip_*.jpg`) | **Yes** - CPC annotations were created directly on the clip images. Coordinates map correctly. |
| 2020+ | No CPC files | N/A - CPCe software was retired. These images are analyzed via OCR (`TCRMPclip_ocrID`). |

### Year-specific details

| Year | Points/Image | Canvas Size | Species Match | Notes |
|------|:----------:|:-----------:|:------------:|-------|
| 2013 | 15 (A-O) | 28800x16200 | 98% | Fuzzy date matching needed for some sites |
| 2014 | 20 (A-T) | 28800x16200 | 100% | |
| 2015 | 20 (A-T) | 28800x16200 | 100% | CPC files in `_EDITED/` subdirectories |
| 2016 | 20 (A-T) | 57600x32400 | 100% | Some sites use `_ColorCorr/` subdirectories |
| 2017 | 20 (A-T) | 57600x32400 | 100% | |
| 2018 | 20 (A-T) | 57600x32400 | 100% | Two periods: Annual + WS (Winter/Spring) |
| 2019 | 20 (A-T) | 57600x32400 | 88% | Three periods: Annual + SCTLD + PeakBL (no CPC) |
