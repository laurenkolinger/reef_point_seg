# Decision records: Reef Point Seg

Last updated: 2026-09-21 (LO)

This folder holds the decision records (ADRs) for the `reef_point_seg` module. A record is written when a change establishes or reverses a design rule that had real alternatives, is hard to reverse, and would surprise a future reader. Platform-wide decisions go to `/mnt/rip/vicarius_drive/docs/adr/`, not here.

## Index

| Number | Title | Status |
|---|---|---|
| [0001](0001-non-tcrmp-imagery-never-enters-the-stores.md) | Non-TCRMP imagery never enters the stores | accepted |
| [0002](0002-one-training-surface-the-vega-main-window.md) | One training surface: the VEGA main window | accepted |
| [0003](0003-runs-compare-only-on-the-same-split-and-metric-family.md) | Runs compare only on the same split and the same metric family | accepted |
| [0004](0004-staged-datasets-carry-their-split-pins.md) | Staged datasets carry their split pins | accepted |
| [0005](0005-training-recipes-are-chosen-by-a-declared-task-and-subject.md) | Training recipes are chosen by a declared task and subject, never by file order | accepted |
| [0006](0006-annotate-images-exports-are-a-training-source.md) | Annotate Images exports are a training source | accepted |
| [0007](0007-a-project-declares-its-family-and-the-rail-shows-only-its-stages.md) | A project declares its family, and the rail shows only its stages | accepted |
| [0008](0008-the-runs-table-names-training-data-and-drops-comparability-from-the-screen.md) | The runs table names training data and drops comparability from the screen | accepted |
| [0009](0009-a-run-starts-from-a-base-model-and-continuing-from-a-vega-model-is-the-advanced-choice.md) | A run starts from a base model, and continuing from a VEGA model is the advanced choice | accepted |

## Conventions

- Files are `NNNN-slug.md`, numbered from `0001`; scan this folder for the highest number and add one.
- Shape: a title, one to three sentences (context, decision, why), then `Status`, `Considered options`, `Consequences`, or `Cost if wrong` only when they add something a reader needs.
- Every record clears the three-part bar before it is written: hard to reverse, surprising without context, the result of a real trade-off. If any part fails, the changelog entry is enough.
- Every record carries a `Source:` line citing the TODO uid, spec, or README line that prompted it.
- Format reference: `/mnt/rip/vicarius_drive/docs/agents/domain.md`.
