# Non-TCRMP imagery never enters the stores

Status: accepted
Source: `README.md` line 241 (the Bounding Boxes scope rule) in this clone (`/mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/README.md`); TODO T20260902-214210.
Last updated: 2026-09-17 (LO)

The module's counts (the coverage matrix, STATS, the mask registry, every `run_*` project's ledger) are the numbers the TCRMP program reports about its own survey frames, and external box datasets such as Coral Intel arrived in 1.6.0 as cheap extra class coverage for training. We decided that non-TCRMP imagery never enters `inprocess/run_*`, the mask registry, the coverage matrix, or STATS, and that the exclusion is enforced in code rather than by convention: the matrix builder and STATS skip every `TRAINING_PROJECT_TYPES` project, a box-sourced training project holds only a pointer to its staged dataset, and the Bounding Boxes page writes only to `supporting_data/bbox_datasets/` and `inprocess/_bbox/`. One outside frame in the census would silently corrupt every per-site and per-year figure the program publishes.

## Considered options

- Enforce by convention (a README warning and operator care). Rejected: the registry is upserted by every export from three tools, and one mistaken export is permanent.
- Tag outside imagery inside the stores and filter it at read time. Rejected: every reader (matrix, STATS, composites, project cards) would need the filter, and a missed one is the same corruption.
- Keep outside imagery out of the stores entirely and give it homes of its own. Chosen.

## Consequences

- Outside imagery has two homes. A staged box dataset under `supporting_data/bbox_datasets/<name>/` (with its SAM3 conversion under `inprocess/_bbox/<name>/`) holds a box-annotated distribution; a standalone session under `inprocess/_standalone/<session>/` holds hand annotation of any folder of photos.
- A standalone session is kept out of the coverage matrix and STATS on purpose, because the matrix scans `run_*` directories only (`scripts/_matrix/builder.py` line 198). The scope rule is about the counts the program reports, not about where a mask may be recorded.
- Amended 2026-09-16 and corrected 2026-09-17 (LO): a standalone session's export does write rows into the mask registry, under `project_id = _standalone` (not `standalone_<session>` as first written here): `export_batch` in `scripts/TCRMPclip_combinedAnnotate/src/export_yolo.py` upserts one row per frame and the standalone launch carries no project id, so `_project_id_from_dir` yields `_standalone`. Those rows never reach the counts this decision protects, because the coverage matrix scans `run_*` directories only (`scripts/_matrix/builder.py` line 198) and STATS counts registry rows whose project id matches a live `run_*` project. The 2026-09-16 amendment stated that no writer records those masks; that was wrong, and this line replaces it. A standalone session's YOLO export is a legitimate training source, decided separately in [0006-annotate-images-exports-are-a-training-source.md](0006-annotate-images-exports-are-a-training-source.md).
- A training project minted from a staged box dataset (`bbox_training`) copies nothing into its run directory; Step 6 reads the staged dataset's recorded path directly.
