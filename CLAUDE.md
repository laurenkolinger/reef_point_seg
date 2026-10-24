# Reef Point Seg: agent pointer

Last updated: 2026-09-29 (LO)

Module `reef_point_seg`, remote `laurenkolinger/reef_point_seg`, branch `main`. Clone at `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/`.

The rules for every agent live in `/mnt/rip/vicarius_drive/CLAUDE.md`. Read it with the Read tool before editing here; a Bash `cat` does not load it into context.

## Git

- Every git command: `git -C /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo ...`. Stage by explicit path; the working tree carries about 120 in-flight migration paths, and `git add -A` would sweep them in.
- Every gh command: `gh -R laurenkolinger/reef_point_seg ...`.
- Subagents do not commit in this clone. Write the files, verify, and report; Lauren commits from the working tree herself.

## Tests

- House scripts, no pytest. Run one suite with `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python <suite>.py`; the full board is `bash scripts/run_all_tests.sh` from the clone root (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/run_all_tests.sh`).
- The full board can kill a live annotator: any suite that reaches a launch route must stub `subprocess.run` (and `subprocess.Popen`) at module level. Before running the board, check `ss -ltnp | grep -E ':508[0-9] '`; if anything is listening in 5080 to 5089, run only the suites you touched.
- **The full board can take a graphics card, and that outranks the port check.** [scripts/TCRMPclip_combinedAnnotate/tests/test_sam_tracker_cache.py](scripts/TCRMPclip_combinedAnnotate/tests/test_sam_tracker_cache.py) runs on fake loaders by default (no real multi-GB model load); only with `TCRMP_SAM3_REAL_PARITY=1` set does its parity check at [line 138](scripts/TCRMPclip_combinedAnnotate/tests/test_sam_tracker_cache.py#L138) select `cuda:0` and build two real `SAM3Engine` instances, and that suite is on the board at [scripts/run_all_tests.sh:179](scripts/run_all_tests.sh#L179). `bash scripts/run_all_tests.sh` does not set `TCRMP_SAM3_REAL_PARITY`, so an ordinary board run stays on the fake loaders and touches no card; the risk is a human running the suite by hand with that variable set, or a future board change that sets it. Before running the board, run the GPU right-of-way check in [/mnt/rip/vicarius_drive/CLAUDE.md](/mnt/rip/vicarius_drive/CLAUDE.md) (Working style) anyway: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and `pgrep -af "metashape|run_phase1"`. If a VOYAGER reconstruction or a VEGA training run holds a card, run only the suites under `scripts/pipeline_orchestrator/tests/`, which stub every spawn and import nothing that reaches a card, and say in your report that you ran a subset and why.

## Cautions

- Restart the orchestrator only with `bash scripts/restart_reef_point_seg.sh` (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/restart_reef_point_seg.sh`). `vicarius restart reef_point_seg` routes through `run_pipeline.sh`, whose `lsof` sweep hangs on this box. If an annotator is live on 5080, restart by pid instead, as the platform rules file describes.
- **Images are deleted only when a person asks; source images never (Lauren, 2026-10-15; hard rule).** No VEGA code path removes, truncates or overwrites an image on its own: a superseded image copy or image folder is moved to a dated `_superseded/<YYYYMMDD-HHMMSS>/` folder beside it, logged with the count, and never pruned by code. An image copy is deleted only by an explicit request from a person (a holding folder cleared by hand, or the backup retention Lauren set: 14 daily copies plus one a month for 12 months). Source images (the TCRMP clip library, the CPCe raw frames, the raw video archive) are never deleted by anyone through VEGA. Only scratch the code itself made under `/tmp` may be removed freely. Cause: on 2026-10-05 a re-route deleted the image copies behind 556 annotated frames of AT's October 4 project.
- Labels-only renames. The rail numbers stages per project family (a detection project shows three, a segmentation project seven, decision record `docs/adr/0007-a-project-declares-its-family-and-the-rail-shows-only-its-stages.md`); the internal step keys and output directories (`step1_*` through `step8_*`, `step4test_combinedAnnotate`) never change. Rename a label, never a pipeline directory.
- The operator guide at `/mnt/rip/vicarius_drive/vicarius/_DOCS/modules/reef_point_seg.md` is a separate real file, not a symlink to this README. An edit to one does not reach the other.
- `changelog.md` in this clone (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/changelog.md`) is newest-first, one `## YYYY-MM-DD HH:MM AST - Title (LO)` heading per batch, and each bullet carries a `Rationale:` line.
- `scripts/_matrix`, `scripts/_labels`, `scripts/_reefreview`, `scripts/_expertids` and the annotator `src/app.py` are now tracked in git; only their `__pycache__` folders are not. Operate from the on-box working tree, since other sessions carry uncommitted edits there that a fresh clone would not have.

## Model cards

**Model cards.** Every module that trains a model, in any framework, writes a model card beside each weights file it produces, at the end of training, with the shared card writer at `vicarius/_METADATA/model_card/`. It fills every record field it can, asks at training time for the finding tags (subject, training data source) it cannot infer, and refreshes the card's performance part when it evaluates the model. A model without a card is not finished. Format and fields: [the VISION design, section 4](/mnt/rip/vicarius_drive/docs/superpowers/specs/2026-10-15-vision-image-inference-design.md).

VEGA's part is [model_card_hook.py](/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/pipeline_orchestrator/model_card_hook.py): the Train step writes `best.pt.card.md` when a run exits 0 (a card failure is one line in the step log and never fails the step), the Train form's Model subject and Training data rows carry the two tags, and [evaluate_run.py](/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/TCRMPtrain_oceankindCV/src/evaluate_run.py) records the best confidence in `metrics.json` and on the card.

## Where vocabulary and decisions live

- Glossary: `CONTEXT.md` in this clone (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/CONTEXT.md`).
- Decision records: `docs/adr/` in this clone (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/docs/adr/README.md`). Platform-wide decisions go to `/mnt/rip/vicarius_drive/docs/adr/`.
- Backlog: the TODO log at `/mnt/rip/vicarius_drive/vicarius/_METADATA/logs/todo.md`, AREA tag `reef_point_seg`.
