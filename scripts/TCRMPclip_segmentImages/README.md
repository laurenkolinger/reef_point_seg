# TCRMPclip_segmentImages

SAM3-powered segmentation review tool for coral reef images. Takes verified point clicks from `TCRMPclip_routeChosenImages` and generates instance segmentation masks using Meta's Segment Anything Model 3.

Provides a fast review UI for accepting, rejecting, and refining masks with SAM3's interactive click-based editing, then exports to YOLO Segmentation format for training with the [oceankind_CV](https://github.com/laurenkolinger/oceankind_CV) pipeline.

## What It Does

1. Reads `sam_click_prompts.json` + raw images from `TCRMPclip_routeChosenImages/output/`
2. User selects which categories to segment (e.g., only Coral, or Coral + Sponge)
3. Runs SAM3 click-prompt segmentation on each selected point using its (x,y) as a positive click
4. Automatically resolves overlapping masks and merges nearby same-species segments
5. Presents masks in a review UI for verification and editing
6. Exports incrementally to YOLO Segmentation format (`all_images/`, `all_labels/*.txt`, `data.yaml`)

## Input

| File | Source | Description |
|------|--------|-------------|
| `../TCRMPclip_routeChosenImages/output/{year}/ids/sam_click_prompts.json` | `routeChosenImages` | Point clicks with species labels |
| `../TCRMPclip_routeChosenImages/output/{year}/raw/*.jpeg` | `routeChosenImages` | Raw source images |

## Output

```
output/
├── all_images/                    # Symlinks to raw images
├── all_labels/                    # YOLO seg: <class_id> <x1> <y1> ... <xN> <yN>
│   └── *.txt
├── data.yaml                      # YOLO dataset config (path, names)
├── class_map.json                 # species → class_id (persistent)
├── segmentations/{year}/
│   ├── segmentations.json         # Internal mask state
│   └── processed_manifest.json    # Processing tracking
└── export_log.txt
```

## Setup

```bash
bash setup_env.sh
```

Requires NVIDIA GPU with CUDA 12.8+, SAM3 HuggingFace authentication (`hf auth login`).

## Quick Start

```bash
./run.sh          # opens browser at http://localhost:5070
./run.sh 5075     # custom port
```

### Category Filtering

In the startup UI, select which categories to segment. Only points matching selected categories will be processed. Useful for:
- Starting with just Coral species
- Processing categories in stages
- Excluding Non-living points

### Review UI

| Key | Action |
|-----|--------|
| **N / B** | Next / Previous image |
| **Space** | Select next pending mask |
| **Enter** | Accept selected mask |
| **Delete / X** | Reject selected mask |
| **F** | Fit to window |
| **Z** | Auto-center on selected mask |
| **Scroll** | Zoom |
| **Alt+drag** | Pan |
| **O** | Cycle opacity (30/60/90%) |
| **P** | Toggle point markers |
| **M** | Merge mode (click two masks) |

### Mask Editing

| Action | Effect |
|--------|--------|
| **Left-click on mask** | Add region (SAM3 positive click) |
| **Right-click on mask** | Remove region (SAM3 negative click) |
| **Shift+drag box** | Redraw mask from bounding box |
| **Exemplar Scan button** | Find similar regions using selected mask as visual exemplar |

### Incremental Workflow

Designed for annotating 4000+ images over multiple days/weeks:
- Export produces a complete YOLO dataset from the very first batch
- New batches and classes are incrementally added to the same output
- `class_map.json` grows as new species appear
- Resume from where you left off across sessions
- Compatible with oceankind_CV `bal_train_test_split.py` for train/val splitting

## Configuration

Edit `src/config.py` for:
- SAM3 model checkpoint and GPU settings
- Segmentation thresholds (confidence, min area, merge distance)
- Overlap resolution strategy
- Default category selection

## Project Structure

```
TCRMPclip_segmentImages/
├── README.md
├── run.sh                 # Launcher (port 5070)
├── setup_env.sh           # Conda env with CUDA + SAM3
├── .gitignore
├── src/
│   ├── config.py          # Paths, thresholds, model settings
│   ├── app.py             # Flask app (routes, session, persistence)
│   ├── sam_engine.py      # SAM3 model wrapper
│   ├── mask_ops.py        # Overlap resolution, merge, polygon conversion
│   ├── export_yolo.py     # YOLO Segmentation format export
│   └── templates/
│       └── index.html     # Full review UI
└── output/                # Created at runtime
```

## Downstream

The YOLO Segmentation output feeds directly into the [oceankind_CV](https://github.com/laurenkolinger/oceankind_CV) training pipeline:
1. Use `bal_train_test_split.py` to split into train/val/test
2. Run `pre_train.py` to generate training config
3. Run `train.py` for YOLO segmentation model training
