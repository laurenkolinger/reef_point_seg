# Reef Point Seg: agent pointer

Last updated: 2026-09-02 (LO)

Module `reef_point_seg`, remote `laurenkolinger/reef_point_seg`, branch `main`. Clone at `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/`.

The rules for every agent live in `/mnt/rip/vicarius_drive/CLAUDE.md`. Read it with the Read tool before editing here; a Bash `cat` does not load it into context.

## Git

- Every git command: `git -C /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo ...`. Stage by explicit path; the working tree carries about 120 in-flight migration paths, and `git add -A` would sweep them in.
- Every gh command: `gh -R laurenkolinger/reef_point_seg ...`.
- Subagents do not commit in this clone. Write the files, verify, and report; Lauren commits from the working tree herself.

## Tests

- House scripts, no pytest. Run one suite with `/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python <suite>.py`; the full board is `bash scripts/run_all_tests.sh` from the clone root (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/run_all_tests.sh`).
- The full board can kill a live annotator: any suite that reaches a launch route must stub `subprocess.run` (and `subprocess.Popen`) at module level. Before running the board, check `ss -ltnp | grep -E ':508[0-9] '`; if anything is listening in 5080 to 5089, run only the suites you touched.
- **The full board also takes a graphics card, and that outranks the port check.** [scripts/TCRMPclip_combinedAnnotate/tests/test_sam_tracker_cache.py:138](scripts/TCRMPclip_combinedAnnotate/tests/test_sam_tracker_cache.py#L138) selects `cuda:0` whenever CUDA is available and then builds two real `SAM3Engine` instances in process, with nothing mocked, and that suite is on the board at [scripts/run_all_tests.sh:162](scripts/run_all_tests.sh#L162). So `bash scripts/run_all_tests.sh` loads a model onto a card even when every port is quiet. Before running the board, run the GPU right-of-way check in [/mnt/rip/vicarius_drive/CLAUDE.md](/mnt/rip/vicarius_drive/CLAUDE.md) (Working style): `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and `pgrep -af "metashape|run_phase1"`. If a VOYAGER reconstruction or a VEGA training run holds a card, run only the suites under `scripts/pipeline_orchestrator/tests/`, which stub every spawn and import nothing that reaches a card, and say in your report that you ran a subset and why. Found 2026-09-17, when the board would otherwise have loaded SAM3 onto the card a live reconstruction was using.

## Cautions

- Restart the orchestrator only with `bash scripts/restart_reef_point_seg.sh` (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/scripts/restart_reef_point_seg.sh`). `vicarius restart reef_point_seg` routes through `run_pipeline.sh`, whose `lsof` sweep hangs on this box. If an annotator is live on 5080, restart by pid instead, as the platform rules file describes.
- Labels-only renames. The rail numbers stages per project family (a detection project shows three, a segmentation project seven, decision record `docs/adr/0007-a-project-declares-its-family-and-the-rail-shows-only-its-stages.md`); the internal step keys and output directories (`step1_*` through `step8_*`, `step4test_combinedAnnotate`) never change. Rename a label, never a pipeline directory.
- The operator guide at `/mnt/rip/vicarius_drive/vicarius/_DOCS/modules/reef_point_seg.md` is a separate real file, not a symlink to this README. An edit to one does not reach the other.
- `changelog.md` in this clone (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/changelog.md`) is newest-first, one `## YYYY-MM-DD HH:MM AST - Title (LO)` heading per batch, and each bullet carries a `Rationale:` line.
- Large parts of the live orchestrator are untracked in git (`scripts/_matrix`, `scripts/_labels`, `scripts/_reefreview`, `scripts/_expertids`, the annotator `src/app.py`). Operate from the on-box working tree, never from a fresh clone.

## Where vocabulary and decisions live

- Glossary: `CONTEXT.md` in this clone (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/CONTEXT.md`).
- Decision records: `docs/adr/` in this clone (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/docs/adr/README.md`). Platform-wide decisions go to `/mnt/rip/vicarius_drive/docs/adr/`.
- Backlog: the TODO log at `/mnt/rip/vicarius_drive/vicarius/_METADATA/logs/todo.md`, AREA tag `reef_point_seg`.
