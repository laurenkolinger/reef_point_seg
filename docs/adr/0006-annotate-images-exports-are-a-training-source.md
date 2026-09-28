# Annotate Images exports are a training source

Status: accepted
Source: Lauren's decision in session, 2026-09-16 01:30 AST, on the VEGA student-ready build; plan
`docs/superpowers/plans/2026-09-16-vega-student-ready.md` decision D8 (at
`/mnt/rip/vicarius_drive/docs/superpowers/plans/2026-09-16-vega-student-ready.md`).
Last updated: 2026-09-17 (LO)

The standalone annotator (the "Annotate Images" dashboard, launched on its own port and writing
under `inprocess/_standalone/<session>/`) already produces a real YOLO export through
`scripts/TCRMPclip_combinedAnnotate/src/export_yolo.py`. The Train step's training-source picker
never offered it: the list held the project's own export, the training sets assembled on the
Coverage Matrix, and the staged box datasets, and nothing else. A person could therefore annotate a
folder of photographs to completion and have no route from that work to a model. We decided that a
standalone session's export is selectable as a training source.

This does not disturb [0001-non-tcrmp-imagery-never-enters-the-stores.md](0001-non-tcrmp-imagery-never-enters-the-stores.md).
That decision protects the numbers the TCRMP program reports about its own survey frames: the
coverage matrix, STATS and the mask registry. It does not protect the trainer, which already reads
non-TCRMP imagery whenever a project trains from a staged bounding-box dataset. A standalone session
stays out of the matrix and out of STATS exactly as before; only the Train step's list of sources
grows.

## Considered options

- Leave the data path alone and relabel the dashboard as a practice sandbox. Rejected: the tool
  produces a complete, correct training export, and telling a student their annotation cannot train
  anything is a worse answer than wiring the one line that lets it.
- Register standalone masks in the mask registry and show them on the coverage matrix. Rejected
  here: that is the one option that genuinely cuts against 0001, and it would need 0001 reversed
  deliberately rather than as a side effect of a usability fix.
- Offer the export as a training source and change nothing else. Chosen.

## Consequences

- The Train step's source list gains a group for standalone sessions, each named by its session and
  carrying its image and label counts, beside the existing project exports, training sets and box
  datasets.
- A run trained from a standalone session records that session as its source, so the run's
  provenance still names where its images came from.
- The coverage matrix and STATS are unchanged: they scan `run_*` directories only
  (`scripts/_matrix/builder.py` line 198), and a standalone session is not one.
- A standalone export writes mask-registry rows under `project_id = _standalone`, as every export does; the coverage matrix and STATS ignore them because they count `run_*` projects only. 0001 is corrected on 2026-09-17 to say so; an earlier line in this record claimed no rows were written, which was wrong.
- A run trained from a session names that session on the Trained runs card: the card reads the source the Train step recorded (`steps.6.config.source_dir`) before it falls back to the project's own export, and the same read names a training set chosen from the picker in an ordinary project.
- The picker groups sessions under their own kind ("standalone") with the same row shape as a training set (`created_at`, `labels`, counts), and the Train note names them as Annotate Images sessions rather than as training sets.
