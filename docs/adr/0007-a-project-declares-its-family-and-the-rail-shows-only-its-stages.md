# A project declares its family, and the rail shows only its stages

Status: accepted
Source: Lauren's brief of 2026-09-16 ("not easy to navigate whether using the bounding box or
segmentation model"); plan `docs/superpowers/plans/2026-09-16-vega-student-ready.md` decisions D1 and
D2 (at `/mnt/rip/vicarius_drive/docs/superpowers/plans/2026-09-16-vega-student-ready.md`); audit
findings under `.superpowers/sdd/2026-09-16-vega-student-ready/findings-detailed.md`.
Last updated: 2026-09-17 (LO)

VEGA trains two families of model on one pipeline: a segmenter, from outlines drawn in the annotator
over TCRMP survey frames, and a detector, from a staged bounding-box dataset. Until this decision the
window showed every project the same seven numbered stages, painted the four annotation stages a box
project can never run as "skipped" with the first of them still marked active, counted a box project's
progress against eight internal steps so a finished one read "3 / 8 complete", and headed each panel
with its internal number ("Step 5: Train Model" over a rail whose Train tile was the first stage a
box project had). Internal directory tokens (`{step6_dir}`) rendered as literal text. A student could
not tell which of the two things they were doing, or where they were in it.

We decided that a project declares its family, that the family is read from what the project already
records rather than from a new field a person must set, and that everything the window numbers or
lists is derived from the family: the rail shows only the stages that family has; the panel heading
carries the number the tile carries; progress on the launcher card counts against those stages; and
no internal step key or directory token reaches the screen.

The family is detection when `project.json` carries a `training_source` of kind `bbox`, and
segmentation otherwise. Internal step keys and output directories (`step6_trainModel`,
`step4test_combinedAnnotate`) never change: this is a display decision, and the module's own rule
that pipeline directories are never renamed stands.

## Considered options

- Keep one rail for both families and grey out the stages a box project skips. Rejected: four inert
  tiles are still four things a student reads, and the numbers stay wrong.
- Two separate windows, one per family. Rejected: the stages after data preparation (Train, Evaluate,
  Model Inference, Refine) are the same code and the same controls, and one window that adapts keeps
  the tutorial, the manual and the tests to one surface.
- Ask the person to choose a family when the project is created. Rejected: the project already knows,
  and a wrong answer would be a new way to break a project. Read it.
- Derive everything from the recorded family. Chosen.

## Consequences

- A detection project reads Step 1 Train Model, Step 2 Evaluate Model, Step 3 Model Inference; a
  segmentation project reads its seven. Each panel's number answers to its own element, so opening
  one panel can never renumber another.
- The launcher card counts progress against the family's stages, so a finished detection project
  reads complete rather than 38 percent.
- Copy that is only true for one family renders per family: the Train description, fill opacity's
  default and hint (0.15 for boxes, not the mask 0.45), the inference source list.
- A Refine round started from a detector refuses with a sentence rather than opening a review with
  no masks, because the seeder builds masks from outlines and a detector writes rectangles.
- The tests pin both rails: `scripts/pipeline_orchestrator/tests/test_rail_family.py` (55 checks)
  and the launcher's `vicarius_ui_os/tests/test_launcher_coverage.py`.
