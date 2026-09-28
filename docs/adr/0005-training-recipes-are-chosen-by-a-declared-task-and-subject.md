# Training recipes are chosen by a declared task and subject, never by file order

Status: accepted
Source: Lauren Olinger's ruling of 2026-09-15, "let ME select whether its a bounding box, or segmentation model, or a coral or algae ... this has to be robust". It reverses the rule set on 2026-09-02 under TODO `T20260902-191400` and closes TODO `T20260915-204500-1`. Prior decisions in this folder: `docs/adr/0002-one-training-surface-the-vega-main-window.md`.
Last updated: 2026-09-15 (LO)

From 2026-09-02 the Train panel picked the training recipe a project opened with by reading `config/presets/` in file-name order: the first file carrying a `_meta.default_for_source` flag that matched the project's source kind won, and when no file carried one, the alphabetically first file in the folder won. Nothing on screen said which recipe had been applied or why. The defect this hid was found on 2026-09-15 while a coral detection recipe was being written beside the fish one: `coral_heavy_augment_640.yaml` sorts before `fish_heavy_augment_640.yaml`, so giving the coral recipe the same flag would have switched every box-dataset training run off the fish recipe that scored 0.904 on its own test split, in silence, with the form showing 36 changed fields and no sentence anywhere naming the file they came from. A rule whose answer depends on the first letter of a file name is not a rule about what the recipe is for.

We decided that a recipe declares what it is written for and that the operator declares what the run is, and that a recipe is applied only when those two declarations resolve to exactly one recipe.

- Every recipe in `config/presets/` declares `task` (`detect` or `segment`) and `subject` (`coral`, `fish`, `algae` or `any`) in its `_meta` block. A recipe missing either key is listed as unclassified so it can still be imported by hand, and is never matched.
- The Train panel carries two controls: Training task (`s6-task`), which already existed, and Subject (`s6-subject`), added by this change. The recipe list shows only the recipes written for the pair those two controls hold, plus every recipe written for subject `any`, plus the two built-in refine-loop shortcuts, which suit any run.
- `GET /api/step/6/presets` reports what each file declares and chooses nothing. It answers no `default_id` and no per-entry flag a client could read as "apply this one". Files are listed in name order so the picker is stable between loads, and nothing is decided by that order.
- Exactly one match is offered and, on a project with no saved training settings, applied. More than one match applies nothing at all: the panel names every candidate and the operator picks. No match applies nothing and says so.
- The line under the picker always answers which recipe is in the form and why, including after a recipe has been imported by hand and the controls have moved off it.
- A staged box dataset may record a `subject` key in its `ingest_manifest.json`. It sets the initial value of the Subject control and nothing else: it never overrides a subject restored from a saved project or one the operator picked, and it never selects a recipe.

## Considered options

- Break a tie by file modification time, by file size, or by a `priority` number in the recipe. Rejected: every one of them is the same defect wearing different clothes. The operator still cannot see why one recipe won, and the answer still changes when a file is touched, copied or re-saved. Lauren's word was "robust", and a rule nobody can read off the screen is not.
- Keep one flag, `default_for_source`, and refuse to start when two recipes carry it for the same source kind. Rejected: source kind is a fact about where the project's imagery came from, not about what is in the pictures. It cannot tell a coral detector from a fish detector, which is the distinction that caused the defect, and both of those are box-dataset projects.
- Let the operator pick the recipe by hand every time and apply nothing automatically. Rejected as the everyday cost: a fresh box-dataset project should open on the recipe written for it, and retyping 36 hyperparameters is how a recipe gets typed wrong. Automatic application survives, narrowed to the single unambiguous case.
- Two declared keys, an operator control for each, and a tie handed back to the operator. Chosen.

## Consequences

- `_meta.default_for_source` and `_meta.default` are deleted, not deprecated, along with `STEP6_SOURCE_KINDS`, `_KNOWN_SOURCE_KINDS` and `_project_source_kind` in `scripts/pipeline_orchestrator/app.py`, which existed only to serve the old rule. No preset file, no template, no script and no test reads either flag.
- Two recipes may deliberately claim one pair, and two do today: `tuned_v1_regularized.yaml` and `yolo26l_heavy_seg_aug.yaml` are both written for segmentation plus coral, which is the module's everyday state. A coral segmentation project therefore opens with the platform defaults in the form and both recipe names on screen, and the operator imports one. This is the intended outcome, not a gap: the two recipes answer different questions and no code can know which one today's run is asking.
- The Subject control is saved in the Step 6 configuration as `subject` and restored with every other control, so a reopened project filters its recipe list the way it did when it was closed.
- The two stagers carry the key forward: `scripts/_boxes/ingest_yolo_boxes.py` takes an optional `--subject`, and `scripts/_boxes/subset_by_class_count.py` copies the parent dataset's subject into the subset manifest. A dataset staged without one simply starts the control on its default, coral.
- A recipe added to `config/presets/` without the two keys is listed and never applied, so a half-written file cannot take a project over. The cost is that a contributor must declare what a recipe is for, which is the point.
- The rule lives in the browser, in `step6SortRecipes` and `renderStep6RecipeOptions` in `scripts/pipeline_orchestrator/static/orchestrator.js`, because the two control values are there. The route stays a reporter. Suite `scripts/pipeline_orchestrator/tests/test_step6_preset_default.py` drives that code in Node against the real payload shape, so the rule is executed in the tests rather than read off the source.
