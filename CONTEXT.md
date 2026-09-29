# Reef Point Seg

Last updated: 2026-09-21 (LO)

Reef Point Seg turns TCRMP point-count records into SAM3-assisted coral masks and trains YOLO11 segmentation models from them. This glossary fixes the words the orchestrator, its sub-apps, the coverage matrix, the Bounding Boxes page and the training steps share; it is vocabulary only, and procedures live in `README.md`. Decision records live in `docs/adr/` (index at `docs/adr/README.md`): 0001 the non-TCRMP scope rule, 0002 one training surface (the VEGA main window), 0003 runs compare only on the same split and the same metric family, 0004 staged datasets carry their split pins.

## Language

### Projects and steps

**Project**:
A `run_*` directory under `inprocess/` holding `project.json`, which records the current step, each step's status and configuration, and an access log, so a run can stop after any step and resume later.
_Avoid_: job, run folder (as a synonym in prose), session (a session is the standalone thing below); the Metashape project of the TCRMP 3D registry is a different thing with the same word.

**Display step**:
The number an operator sees in the orchestrator sidebar, 1 to 7, where display 4 is the combined annotator and display 5, 6, and 7 are internal steps 6, 7, and 8.
_Avoid_: step number (without saying which kind), tile number.

**Internal step**:
The fixed step key and output directory name in code and on disk (`step1_makeAllPoints` through `step8_inference`, plus `step4test_combinedAnnotate`), which never change.
_Avoid_: renaming these to match display numbers, legacy step.

### Frames, points, labels

**Frame**:
One TCRMP survey still image sampled from transect video, resolved by its globally unique basename.
_Avoid_: photo, still, clip (the clip is the video the frame came from).

**Point**:
One CPC point-count record: a pixel position on a frame with the species code recorded there, which the pipeline uses as a SAM3 click prompt.
_Avoid_: annotation (a mask is the annotation), click, marker.

**Label**:
A species or category code (for example PA) with its full name (Porites astreoides), drawn from the master codes table; the target of a mask and of Step 3 selection. A label can be a coarse category such as Sponge, so "label" is the neutral word.
_Avoid_: species (when the label may be a category), class (the YOLO class id is derived from the label), target species.

**Screened**:
A frame is screened for a label when a project recorded an outcome for that label on it (found by a person, by the model, by an expert, pending an expert, or not found). Step 3 skips a frame another project screened for every target label; a frame screened for only some target labels stays in, and the annotator shows the other project's accepted masks for those labels as locked previews. Built by `scripts/_reefreview/screening.py` on the coverage matrix's cells; archived, standalone, composite and box-training projects never count.
_Avoid_: reviewed (a frame can be reviewed for one label and not another), done.

### Masks and stores

**Mask**:
One reviewed outline of a single organism or patch on one frame, carrying its label, review status, source type, and reviewer initials.
_Avoid_: polygon (the geometry inside a mask), segmentation, instance.

**mask_uid**:
The content-stable identifier of a mask and the key of the mask registry, unchanged across re-exports.
_Avoid_: mask id, uid (unqualified).

**Mask registry**:
The permanent cross-project census at `inprocess/_mask_registry/manifest.csv`, one row per `mask_uid`, upserted by every export from every annotator.
_Avoid_: the TCRMP 3D registry, the module registry, the lock registry (other registries, never this one), census, manifest (unqualified).

**Coverage matrix**:
The image-by-label grid at `/matrix`, derived from the mask registry merged cell by cell with each project's ledger, with its MATRIX, STATS, and TRAINING SETS tabs.
_Avoid_: label matrix, grid, dashboard.

### Review and identity

**REVIEW flag**:
The state of a mask a reviewer could not identify with confidence; a REVIEW-flagged mask leaves the training bundle and waits in the expert-review queue until an expert ID comes back. Setting a new one is suspended since 2026-09-06 (Lauren Olinger's ruling): every annotator refuses a request that would create one, in any of the four ways the tag can be spelled, and shows the REVIEW control greyed out with a tooltip saying so. Reading, displaying and merging an existing flag are unchanged, and the flags already on disk are untouched. The one switch that reverses it is `REVIEW_TAGGING_ENABLED` in each application's `mask_ops.py`, and in `scripts/TCRMPclip_placePoints/src/export.py`.
_Avoid_: pending (a pending mask is one not yet accepted or rejected), flagged (unqualified), uncertain.

**Expert ID**:
An identification assigned to a REVIEW-flagged mask from outside the pipeline and recorded in the expert-ID library with MODE USER or EXPERT; an EXPERT-mode ID locks the mask against relabel.
_Avoid_: expert label, expert review (the round trip, not the identification), verified ID.

### Training and models

**Composite**:
A cross-project YOLO-seg training set under `inprocess/_composite/<name>/`, assembled by the coverage matrix from preferred masks first and the strongest-outcome project otherwise.
_Avoid_: export (unqualified), merged dataset, training set (a composite is one kind of training set).

**Model lab**:
A lab folder under `inprocess/_models/<lab>/` holding its frozen split, its runs and its ledger. Two tools write one: the project-free training page retired on 2026-09-05, which wrote every lab that predates that date, and, since 2026-09-15, the run importer (see Imported lab below). The All trained models disclosure on the Evaluate panel (`GET /api/step/6/all_runs`, built on `list_all_runs` in `scripts/_modellab/labruns.py`) lists every lab run, origin "lab", beside the project runs, and Evaluate, Promote and Continue from act on them.
_Avoid_: lab (unqualified), experiment, project, the training page (it no longer exists).

**Imported lab**:
A model lab written by the run importer (`scripts/_boxes/import_external_runs.py`) from finished detection runs trained outside VEGA, so the main window can list, score and promote them. It holds `lab.json` (the lab name, the task, the staged source, `origin_kind: "imported"`, the note and the imported run names), a `dataset/` built from the staged dataset those runs were trained on, and one folder per run carrying `weights/best.pt`, `args.yaml`, the source `results.csv` when there was one, and a `provenance.json` naming where the run came from. An imported lab carries no score: the importer writes no `rounds.csv` row and no champion record, and a number reaches the ledger only from a VEGA Evaluate pass, which is what records the metric family and the split fingerprint beside it.
_Avoid_: imported run (one run folder inside the lab, not the lab), migrated lab, published score (a number from the source experiment's own paper is not a ledger row).

**Base model**:
One of the YOLO11 checkpoints Ultralytics publishes (nano, small, medium, large, extra large; the `-seg` variants for outlines), trained by Ultralytics on COCO, a set of everyday photographs. It knows nothing about corals or fish. Every VEGA run starts from a base model unless the advanced choice below says otherwise; the Starting model select (`#s6-model` in `scripts/pipeline_orchestrator/templates/index.html`) lists the checkpoints, and the trainer (`scripts/TCRMPtrain_oceankindCV/run_step6.sh`) loads whichever one the `model` value it is given names.
_Avoid_: pretrained model (say base model, and say once that Ultralytics pretrained it on COCO), stock weights, catalog model.

**Train from a base model**:
What VEGA does by default: load the base model's weights, replace its class head with the classes in the training set, and train every layer on that set. The machine-learning literature calls this transfer learning or fine-tuning from COCO; VEGA never calls it fine-tuning, so that the word keeps one meaning. This is the Train Model panel's default state, whenever the "Continue from a VEGA model" disclosure (`#s6-continue-details`) holds no choice: the step's saved config carries no `model_path`, and the trainer receives only `model`, a base checkpoint name.
_Avoid_: full retrain, retrain from scratch (nothing in VEGA starts from random weights), pretrain (VEGA pretrains nothing).

**Continue from a VEGA model**:
The advanced choice: start from the best weights of an earlier VEGA run of the same task (the champion, a run of this project, or a run of another project or lab), with the learning rate lowered to 0.001 and the first ten layers frozen, so a small batch of corrections refines what the model knows without erasing it. This is the only thing VEGA calls fine-tuning, and it is always a deliberate choice, made under the "Continue from a VEGA model (advanced)" disclosure (`#s6-continue-details`, select `#s6-continue-from`); choosing a run there sends `model_path` (that run's `weights/best.pt`) to the trainer alongside `freeze` and `lr0`, and the trainer refuses a run whose task does not match.
_Avoid_: warm start, resume (ultralytics `resume=` only continues a crashed run and is never used), fine-tune preset.

**Champion**:
The promoted run whose weights and metrics are the baseline every gate compares against, written by `POST /api/step/7/promote` (`scripts/pipeline_orchestrator/app.py`) as one family-aware record (`run_dir`, `promoted_at`, `metric_family`, `map50_95_M` or `map50_95_B`, `note`) twice: `champion.json` beside the run's own ledger, where `evaluate_run.py` reads its gate baseline, and `inprocess/_models/champion.json`, which Model Inference and the Continue from a VEGA model disclosure read. It is the model Model Inference uses unless another is picked, and the first option under Continue from a VEGA model; it is not what a run starts from unless it is chosen there.
_Avoid_: best model, production model, baseline (the champion is the baseline; the word alone is ambiguous), set champion (say promote).

**Refine round**:
One turn of the model-assisted loop: infer, seed each above-threshold detection into the annotator as a pending model mask, correct, retrain, evaluate, and append a row to `rounds.csv`.
_Avoid_: iteration, loop (the loop is the cycle; a round is one turn of it), active-learning pass.

**Gate**:
One of the three promotion verdicts on a rounds-ledger row (`gate_map`, `gate_class`, `gate_recall`) comparing the round against the champion in the round's own metric family, written by `scripts/TCRMPtrain_oceankindCV/src/evaluate_run.py`; each reads pass, fail, flag, unpinned (the split is not recorded, so nothing can be compared) or incomparable (the baseline was scored in the other family).
_Avoid_: threshold, check, test.

**Frozen transect holdout**:
The split rule that pins transect 5 to validation, transect 6 to test, and transects 1 to 4 to train (`holdout_mode: "transect"`), so round-over-round metrics compare like with like; `scripts/TCRMPtrain_oceankindCV/src/pinned_split.py` records it in `dataset/split_manifest.json` with the seed, the fractions, per-split counts and the split fingerprint.
_Avoid_: random split, validation set (unqualified), test set (unqualified).

**Split fingerprint**:
The first 16 hex characters of the SHA-256 digest of a dataset's sorted image-to-split assignment, written as `split_fingerprint` with its `split_fingerprint_scheme` (`sha256-16/netstring-utf8-sorted-v1`) into `dataset/split_manifest.json` by both splitters (`scripts/TCRMPtrain_oceankindCV/src/detect_dataset.py`, which defines it, and `src/pinned_split.py`) and read back by `split_fingerprint_for_run` in `scripts/_modellab/labruns.py`; two runs with the same fingerprint were scored on the same held-out images.
_Avoid_: split hash, split id, seed (the seed is an input to the split; the fingerprint is a digest of its result).

**Metric family**:
Which scores a run produces and is judged by: `mask` (outline scores, a segmenter) or `box` (a detector), chosen from the numbers by `metric_family` in `scripts/TCRMPtrain_oceankindCV/src/evaluate_run.py` and written into `metrics.json`, the `metric_family` column of `rounds.csv` and the champion record, so the report, the ledger, the gates and the runs card all read the same family.
_Avoid_: task (the task is what was trained; the family is what was scored), mask metrics (unqualified, when the run is a detector), seg versus detect (those are task names).

**Comparable runs**:
Two runs whose scores may be ranked against each other because their metric family and their split fingerprint both match, decided by `runs_comparable` in `scripts/_modellab/labruns.py`; since 2026-09-17 nothing on screen marks the pair (decision record `docs/adr/0008-the-runs-table-names-training-data-and-drops-comparability-from-the-screen.md`), and the verdict acts in the promote route's 409 answer and in the ledger's gates. A run with no recorded fingerprint is never comparable, itself included.
_Avoid_: same split (necessary, not sufficient), like for like, ranked (a run can be listed without being ranked).

**Absent class**:
A class in the model's class map with no labeled instance in the evaluated split, listed by name under `absent_classes` in `metrics.json` by `scripts/TCRMPtrain_oceankindCV/src/evaluate_run.py` and shown as "not present in this split" in the Step 6 report card and the per-class table, never scored as zero and never allowed to shift another class's row.
_Avoid_: missing class (the class is known; its instances are absent), zero-AP class, dropped class.

**report.html**:
The evaluation report as one HTML file with its stylesheet inlined, written by `scripts/TCRMPtrain_oceankindCV/src/evaluate_run.py` into the evaluation folder beside `report.md` and `metrics.json`, and the exact source headless Chrome prints `report.pdf` from; open it in a browser to see any page the PDF shows.
_Avoid_: the web report (the Step 6 report card is built from `metrics.json` and `GET /api/step/7/report`, not from this file), the PDF source (say report.html).

**Ignored flag**:
A command-line flag that `scripts/TCRMPtrain_oceankindCV/src/run_inference.py` accepts for compatibility and does not act on, recorded by name under `ignored_flags` in `manifest.json` so a reader can see it was passed; the only one currently defined is `--save_crops`, which writes no crop images.
_Avoid_: deprecated flag, no-op (say what the flag does not do), crops (nothing is cropped).

**Training project**:
A `run_*` project minted from a composite (`composite_training`) or a staged box dataset (`bbox_training`) with steps 1 to 4 written as skipped, opening on Train.
_Avoid_: throwaway project, quick-train project, lab.

**Training project handoff**:
The Bounding Boxes page's Train in VEGA action (`POST /boxes/api/create-project` in `scripts/_boxes/blueprint.py`), which mints a `bbox_training` project for a staged box dataset, or reuses the newest one it already has, and opens it in the main window (`/?project_dir=<encoded run dir>`, window name `reefpointseg_orchestrator`) on Train Model; the page itself trains nothing.
_Avoid_: quick-train, remote training, train on Bounding Boxes (nothing trains there).

### Outside imagery

**Standalone session**:
One Annotate Images session under `inprocess/_standalone/<session>/` with no project involved; its registry rows carry `project_id = standalone_<session>`, and the coverage matrix excludes them on purpose.
_Avoid_: project (a session is not one), custom import session, bring-your-own-images.

**Staged box dataset**:
An external box-annotated dataset copied into `supporting_data/bbox_datasets/<name>/` by the Bounding Boxes page's Coral Intel stager (`scripts/_boxes/ingest_coral_intel.py`) or by the generic YOLO stager (`scripts/_boxes/ingest_yolo_boxes.py`, any images folder plus a labels folder plus a class list); non-TCRMP imagery that never enters a project, the mask registry, the coverage matrix, or STATS.
_Avoid_: bbox dataset (the `inprocess/_bbox/` directory holds the SAM3 conversion, a different artifact), external dataset, Coral Intel (one staged dataset, not the kind).

**Split pins**:
A staged dataset's own `split_pins.json`, one train, valid or test assignment per staged image basename, recording the split the dataset's original experiment used. The detection splitter (`scripts/TCRMPtrain_oceankindCV/src/detect_dataset.py`) replays the pins before any seeded draw, so a retrain here is scored on the same held-out images as the model the dataset came with; the build records `holdout_mode: "staged"` in `dataset/split_manifest.json`, and `pins_source` names the file it read them from. A staged dataset with no pins is split exactly as before, by the seeded draw (`holdout_mode: "lab-random"`).
_Avoid_: pinned split (that is the TCRMP transect rule in `src/pinned_split.py`), frozen split, seed (the seed feeds the draw the pins replace).

**Names table**:
A staged dataset's own `names.csv`, one row of class token to full name, written by the generic YOLO stager and copied into the split dataset's `data.yaml` as `names_full`. The evaluation report, the inference overlay and the Step 6 report card read it from there, so a class token is always shown with the name behind it instead of standing alone.
_Avoid_: class map (`class_map.json` maps class ids to tokens), master codes table (the TCRMP species-code table, a different file), labels (a label is the thing a mask carries).
