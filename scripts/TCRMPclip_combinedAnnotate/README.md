# Place + Segment annotator (`TCRMPclip_combinedAnnotate`)

Last updated 2026-09-17 15:10 AST (LO).

This is the SAM3 annotator that VEGA (the `reef_point_seg` module) opens for outline annotation. One Flask app, `src/app.py` in this folder (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/TCRMPclip_combinedAnnotate/src/app.py`), serves three sessions:

| Session | Opened from | What it does |
| --- | --- | --- |
| Place + Segment (project) | The Place + Segment tile in the VEGA main window, Open Annotator | Annotates the survey frames Choose Images picked. You click an organism, SAM3 proposes the outline, you correct and accept it |
| Refine (loop) | The Refine (model loop) tile, Seed + Open Review | The same screen, with a trained model's predictions seeded as pending cyan masks for you to accept, fix or reject |
| Annotate Images (standalone) | The Annotate Images dashboard button on the module page, or the Tools rail | The same screen over your own photographs, with no project |

The student manual carries the full procedure and the description of every stage: `/mnt/rip/vicarius_drive/vicarius/_DOCS/modules/reef_point_seg.md` (the Manual button on the VEGA module page). This file is the tool's own reference.

## Launch only from VEGA

Never start this app by hand: not `run.sh` in this folder, not `python src/app.py`. VEGA sets everything a session needs in the process environment (the project and its export folder, the reviewer initials, the session mode, the GPU devices, the SAM3 settings from the Place + Segment panel) and chooses the port. A copy started by hand has none of that and writes to the wrong place or nowhere. `src/config.py` holds only fallback defaults; the settings a session runs with come from the panel in VEGA or the Annotate Images start card, and editing `config.py` is never the way to change them.

Ports: a project or Refine session prefers 5080 and, when that is taken, takes another free port that the tile in VEGA names. Annotate Images runs on 5096, outside the range VEGA sweeps when it starts a project session, so a standalone session is never closed by a project launch or by an orchestrator restart. Edit Masks, a sibling app in `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/TCRMPclip_editMasks/`, prefers 5083. To restart VEGA itself use `bash /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/restart_reef_point_seg.sh`.

Two NVIDIA GPUs are expected: the SAM3 point-prompt tracker on `cuda:1` and the exemplar (look-alike) model on `cuda:0` by default, both set on the Place + Segment panel. A cold model load takes about a minute; VEGA opens the window by itself when the app is ready.

## The screen

Top bar: project or session, your initials (required; every accepted mask is credited to them), Prev, the frame counter, Next (or finish), Unlabeled only, Next unlabeled. Then the frame name and a Pause button for the per-frame timer. Tool row: Opacity, Target points only, Ref points (G), the Edit tools Refine, Merge (M), Add (A), Box (B), Draw (D), the Mask size slider (tight, medium, generous), Undo (U), Merge same ID, Reload IDs. Label row: "Tap a number to arm a label", one button per target label with the full species name and the code as a chip, and a greyed Expert review button. Right panel: Masks (one row per mask with its letter, species, code, area, SAM3 score and a tick box; hide rejected) and Find similar (Find like selected (S), Find all labels, Similarity). Press K for the shortcut sheet, which is the reference the table below is checked against.

## Keys

| Key | Does |
| --- | --- |
| K | Opens and closes the shortcut sheet; Esc also closes it |
| 1 to 9 | Arms that label and enters Add mode; the same number again disarms. Labels past 9 are armed by clicking their button |
| A | Add mode without a label; A again leaves it |
| B | Box mode: drag a rectangle and SAM3 makes a mask from it; B again turns it off |
| D | Draw mode: freehand carve or extend the selected mask; D again leaves it |
| Click, right-click | In Refine: add a region, remove a region |
| Enter, double-click | Accepts the current mask; with no label yet, the label popup opens first |
| X, Delete, Backspace | Rejects the current mask, or every ticked mask. A reject is kept out of the export, never deleted |
| Enter on a rejected row | Accepts it again (untick hide rejected to see the row) |
| C | Opens the label popup for the selected mask: 1 to 9 a target label, s Sponge, m Macroalgae, g Gorgonian, t Turf, 0 removes the label, Esc closes |
| Tab | Selects the next mask |
| Esc | Deselects, closes the popup, ends Add and Box mode |
| M | Merges every ticked mask into one, all parts kept; with none ticked, click mask A then mask B |
| Ctrl+click, Shift+click on a row | Multi-select, range select |
| U | Undoes the last refine edit |
| S | Finds look-alikes of the selected mask (exemplar scan); in the candidate list Enter keeps, X drops |
| Ctrl+S | Saves the frame now |
| Space, N, right arrow | Next frame; the last frame of the batch opens the export dialog |
| Left arrow | Previous frame |
| Y, N | In the export dialog: export the batch, or go back |
| F | Fits the frame to the view |
| O | Cycles mask opacity |
| T | Target masks and points only, on and off |
| G | Reference points (the survey points, read-only), on and off |
| H | Magnifier loupe on and off |
| Z | Auto-centre on the selected mask, on and off |
| P | Pauses and resumes the frame timer |
| Scroll, click-hold and drag, Alt+drag, middle-drag | Zoom, pan, pan, pan |
| Shift+drag | Redraws the selected mask from a rectangle (turn Box mode off first) |

R used to arm the expert-review flag. Flagging has been suspended since 2026-09-06, so R and the REVIEW button only show a refusal; masks flagged before that date are unchanged.

Rules the app enforces: a mask with no label cannot be accepted or exported; a session with no reviewer initials cannot accept a mask; a frame where everything is rejected exports as a negative example; saves are debounced and atomic, and Ctrl+S forces one.

## Where the output goes

| Session | Export folder | Contents |
| --- | --- | --- |
| Place + Segment, Refine | `<project>/step4test_combinedAnnotate/` under `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess/run_*/` | `all_images/`, `all_labels/` (one YOLO text file per frame), `data.yaml`, `class_map.json`, `export_manifest.csv` (one row per frame, per-species counts, outcome), `segmentations/` (working mask state), `label_provenance.csv`, `previews/`, and `routed_input/` (the survey points, read-only). A Refine round adds `loop_rounds/round_*.json` |
| Annotate Images | `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess/_standalone/<session>/` | `session.json`, `custom_imports/`, `segmentations/`, `all_images/`, `all_labels/`, `data.yaml`, `export_manifest.csv` |

Every export records each mask in the cross-project mask registry at `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/inprocess/_mask_registry/manifest.csv`, which the Coverage Matrix reads, and writes an `eml.xml` metadata record beside the bundle. A project export is what the Train Model stage trains on. A standalone session stays out of the Coverage Matrix and STATS; decision record `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/docs/adr/0006-annotate-images-exports-are-a-training-source.md` records that its export is to be offered as a Train Model source, and on 2026-09-17 the Train source list did not yet list sessions.

## For developers

- `src/app.py`: the Flask app, routes and session state. `src/sam_engine.py`: the SAM3 wrapper. `src/mask_ops.py`: overlap, merge, polygon conversion and the review-flag lock (`REVIEW_TAGGING_ENABLED`, held byte-identical with the Edit Masks copy by `tests/test_shared_file_parity.py`). `src/export_yolo.py`: the YOLO export and the registry upsert. `src/seed_from_predictions.py`: the Refine seeder, which reads `predictions.json`, seeds outline entries only (box-only detections are counted and skipped) and never seeds a frame pinned to validation or test. `src/custom_imports.py`: image import for standalone sessions. `src/provenance.py`: the label ledger writer, one copy per sub-tool. `src/templates/index.html`: the whole review UI and its key handler.
- SAM3 loading: `facebook/sam3` ships as a video checkpoint whose single-image tracker weights sit under `tracker_model.*`, so `_load_image_tracker` in `src/sam_engine.py` builds the tracker and backfills those weights, failing loudly if any are missing. Masks that come out as the whole frame at about 0.50 confidence are the signature of a tracker on random weights. Weights load from the Hugging Face cache (`~/.cache/huggingface/hub/models--facebook--sam3`); the loads are offline-first because lab DNS is flaky.
- Tests are house scripts, no pytest: `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python tests/<suite>.py` from this folder for one suite; the full board is `bash /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/run_all_tests.sh`, which redirects the registry to a throwaway folder. Suites that reach a launch route stub `subprocess` at module level; check `ss -ltnp | grep -E ':508[0-9] '` before running the board, because it can kill a live annotator.
- `run.sh` and `setup_env.sh` are the legacy hand launcher and the legacy conda build. The orchestrated flow runs this app from the unified interpreter at `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python`, built by `bash bootstrap.sh` from the repo root.
