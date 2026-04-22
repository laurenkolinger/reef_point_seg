# TCRMPclip_routeChosenImages

Unified QAQC review app for CPC + OCR coral reef images selected by `TCRMPcvr_chooseImages`.

Takes the selected frames list and routes each image through the appropriate pipeline:
- **CPC frames (pre-2020):** loads existing (x,y) point coordinates from `cpc_all`, remaps species codes to the recoded taxonomy (e.g., MFRA → OFRA)
- **OCR frames (2020+):** runs GPU-accelerated EasyOCR detection to extract point coordinates

All frames are presented in the same QAQC review UI (copied from `TCRMPclip_ocrID_batch`) where you can flip through images 10 at a time, verify/edit point positions, add segmentation points, and export a unified dataset for SAM3 segmentation.

## What It Does

1. Reads `selected_frames.csv` from `TCRMPcvr_chooseImages/output/`
2. Filters frames by species (configurable checkboxes in UI)
3. Loads CPC point coordinates for pre-2020 frames, re-enriches with recoded species codes from `TCRMPcvr_recodeSpecies` output
4. Runs GPU OCR detection on 2020+ frames (same engine as `TCRMPclip_ocrID_batch`)
5. Presents all frames in the same review UI for verification
6. Exports unified year-organized output with SAM3 click prompts

## Input

| File | Source | Description |
|------|--------|-------------|
| `../TCRMPcvr_chooseImages/output/selected_frames.csv` | `TCRMPcvr_chooseImages` | Frame selection with routing info |
| `../TCRMPcvr_recodeSpecies/output/all_points_*.csv` | `TCRMPcvr_recodeSpecies` | Recoded species data (latest auto-detected) |
| `../input/TCRMP_clip/` | Raw imagery | Source images (READ-ONLY) |
| `../output/cpc_all/` | `TCRMPclip_cpcID` | Pre-2020 CPC point coordinates |

## Output

```
output/
├── {year}/
│   ├── raw/                        # Copied source images
│   ├── ids/
│   │   ├── sam_click_prompts.json  # SAM3 format
│   │   └── point_coords.csv       # Flat CSV with recoded species codes
│   ├── test_pts/                   # Verification overlay images
│   ├── detections.json             # Internal detection state
│   ├── processed_manifest.json     # Internal tracking
│   ├── dataset_summary.txt
│   └── log.txt
```

## Setup

```bash
bash setup_env.sh
```

Requires NVIDIA GPU with CUDA 12.8+ for OCR detection.

## Quick Start

```bash
./run.sh          # opens browser at http://localhost:5060
./run.sh 5070     # custom port
```

### Filtering by Species

In the startup UI, uncheck species you don't want. Only frames containing at least one checked species will be loaded. This is useful for:
- Testing with a single species first (e.g., just OFRA)
- Processing species in batches

### Review UI

The review UI is identical to `TCRMPclip_ocrID_batch`:

| Key | Action |
|-----|--------|
| **N / B** | Next / Previous image |
| **Space** | Select next unidentified point |
| **Arrows** | Nudge selected point (Shift = 5px) |
| **Ctrl+click** | Place point at click |
| **A+click** | Add segmentation point |
| **1-8** | Assign category to added point |
| **F** | Fit image to window |
| **Z** | Toggle auto-center |
| **Scroll** | Zoom |
| **Alt+drag** | Pan |

## Species Code Remapping

CPC data uses old taxonomy (MFRA, MFAV, MA). This workflow automatically re-enriches all points using the recoded `all_points.csv` from `TCRMPcvr_recodeSpecies`, so all output uses current taxonomy (OFRA, OFAV, OA).

## Project Structure

```
TCRMPclip_routeChosenImages/
├── README.md
├── run.sh                 # Launcher (port 5060)
├── setup_env.sh           # Conda env with CUDA + EasyOCR
├── .gitignore
├── src/
│   ├── app.py             # Flask app (CPC loading + OCR + review)
│   ├── detect.py          # Letter + crosshair OCR detection
│   ├── export.py          # Year-organized export
│   ├── species.py         # Species lookup with fuzzy date matching
│   ├── summarize.py       # Dataset summary generation
│   └── templates/
│       └── index.html     # Full QAQC review UI
└── output/                # Created at runtime
```

## Downstream

The unified output feeds into SAM3 segmentation (separate workflow). The `sam_click_prompts.json` contains all (x,y) coordinates needed as click prompts for automatic coral segmentation mask generation.
