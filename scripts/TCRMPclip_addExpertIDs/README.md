# TCRMPclip_addExpertIDs

Optional, **re-runnable** stage that folds an outside expert's IDs back into the
pipeline. It is launched from the VICARIUS orchestrator via the **Add Expert
IDs** tile (placed right after the mask-review step), or standalone:

```bash
./run.sh 5075
```

## What it does

1. **Drag-drop the CSV** the expert emailed back (`uid,code[,confidence,labeler]`,
   produced by the expert-review GitHub-Pages site).
2. For each UID it updates the **permanent cross-project library**
   (`reef_point_seg/inprocess/_expert_id_library/`) — code, name, category,
   confidence, labeler, and **mode = EXPERT** — and relabels the matching mask in
   the project's `segmentations.json`.
3. A new code that isn't in the project's class set is still stored (the library
   always keeps the image + mask + polygon for future training data).
4. **>50%-overlap auto-relabel:** any remaining TO-REVIEW mask whose footprint is
   >50% covered by a now-EXPERT-labeled mask on the same source image inherits
   that expert's ID (mode EXPERT, provenance preserved).
5. Resolved UIDs are **removed from the review folder** and the repo is **pushed**,
   so the expert's queue shrinks.

Idempotent — re-dropping the same CSV is a no-op; new CSVs accumulate. The mode
(`USER` vs `EXPERT`) is recorded for every UID in the library `manifest.csv`.

## Config (env, set by the orchestrator)
- `TCRMP_EXPORT_DIR` — the active project's step-5 export dir
- `TCRMP_REVIEW_DIR` — review-repo working tree (default `/mnt/tear/REVIEW_reefpointseg`)
- `TCRMP_REVIEW_REPO_URL` — remote (default `laurenkolinger/reefpointseg-review`)
- `TCRMP_EXPERT_LIBRARY_DIR` — library dir (default: module `inprocess/_expert_id_library`)
- `TCRMP_REVIEW_GIT_PUSH` — `1` to push (default), `0` to commit locally only
- `TCRMP_REVIEW_OVERLAP_THRESH` — auto-relabel threshold (default 0.5)
