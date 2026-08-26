# Expert-Review Revamp — Cross-Cutting Contracts (LOCKED)

Single source of truth for the data model, CSV, routing, consensus, testing, and
logging that MULTIPLE modules touch. Every implementation agent reads this and
implements against it so the parallel work converges. Do not diverge from these
shapes without updating this file first.

Tracks: A = pipeline flow/UX fixes · B = SAM3/expert-review feature · C = consensus engine.

---

## 1. Canonical project identity  (Track B Phase 0 — DONE)
- `project_id` = `project.json['id']` (= `<name>_<YYYYMMDD>_<uuid>`). `project_name` = `project.json['name']`.
- Orchestrator passes `TCRMP_PROJECT_ID` / `TCRMP_PROJECT_NAME` into every step-5 launch (incl. the review-only SAM3 mode). Step-5 falls back to path-derivation + id-as-name when standalone.
- Resolver: `project_manager.resolve_step_dir(project_id, projects_root, step="5")` and `find_projects(projects_root)`. `projects_root` = `<module>/inprocess` (i.e. `os.path.join(os.path.dirname(REPO_DIR), 'inprocess')`).
- Threaded into `review_export.export_flagged_masks(..., project_id, project_name)` → stamped on BOTH the manifest item and the library record (`MANIFEST_FIELDS` now includes `project_name`).

## 2. Review manifest item schema  (review_repo / review_export)
Each `review_manifest.json` item:
```
{ uid, project_id, project_name, site, site_full, year, transect, frame,
  source_image, featured_codes:[...], candidate_codes:[...], added_at,
  crop, mask, full,                       # relative asset paths
  reviews: [ {reviewer, code, confidence, at} ],   # C5: accumulates, never overwritten
  accepted: null | {code, mode:'EXPERT', labeler, at, basis:'consensus'|'operator'} }
```
- `reviews[]` accumulates one entry PER (reviewer, csv-drop); re-drop by same reviewer UPSERTS that reviewer's row (keyed by `reviewer`).
- `accepted` is set ONLY by the operator in the consensus tool (C6). Until then the item stays on the site showing tentative reviews.
- Top-level manifest: `{ generated_at, contacts:[...], reviewer_names:[...], count, items:[...] }`. `reviewer_names` = distinct reviewers seen (for acknowledgment/credit).
- Idempotent seeds (updated 2026-07-09): `ensure_repo()` and every manifest/codes write compare content
  (ignoring `generated_at`) before writing and SKIP the write when nothing semantic changed. A no-op
  preamble (`ensure_repo` + `set_contacts` with unchanged contacts, e.g. an export that queues zero masks)
  leaves the working tree byte-identical and git-clean; `generated_at` moves only with real content.
  `write_codes()` reads the operator_setup.json sidecar so re-seeds never drop operator candidate extras.

## 3. Library + segmentations mask schema  (per-reviewer IDs)
- Library `MANIFEST_FIELDS`: existing + `project_name`. A per-reviewer detail file `reviews/<uid>.json` = `{uid, reviews:[{reviewer,code,confidence,at}], accepted}` (mirrors the manifest item; survives cross-project).
- Step-5 `segmentations.json` mask gains: `reviews: [{reviewer,code,confidence,at}]` (tentative) and keeps `expert_id` (set only on accept). Step-5 coloring (Phase 1 DONE): expert(green) when `expert_id.mode==='EXPERT'`, pending(amber) when `review===true`/`species==='REVIEW'` and not yet expert. A THIRD state may show "tentative" (reviews present, not accepted) — amber with a count badge.

## 4. CSV format  (viewer export ⇄ importer ingest)  — drag & drop both ways
Header (exact, lower-case): `uid,code,confidence,reviewer,project_id`
- Optional trailing human-readable columns allowed + ignored on import: `project_name,site,frame,name`.
- `reviewer` = the name entered in the viewer's identity gate (C1) — REQUIRED, non-empty.
- `confidence` (updated 2026-07-09): the column is RETAINED in the header for compatibility but the viewer
  ALWAYS emits it BLANK (confidence UI removed end to end; legacy localStorage conf values ignored). The
  importer accepts and ignores any value in the column, so old CSVs carrying high/low still import cleanly.
  Nothing downstream displays confidence anymore (consensus cells + catalog tiles dropped it); stored values
  in `reviews[]` are untouched.
- Viewer download filename: `tcrmp_expert_ids_<reviewer>_<project|all>.csv`.
- Importer accepts a dropped file OR pasted text. Tolerant: header optional, extra cols ignored, blank rows skipped.

## 5. Import routing  (multi-project, multi-expert)
- Group rows by `project_id`. For each: `export_dir = resolve_step_dir(project_id, projects_root)`. If unresolved, fall back to `library.lookup(uid)['project_id']`, else the currently-open project; **report every unrouted UID** (never silent no-op).
- For each row: UPSERT into that uid's `reviews[]` keyed by `reviewer` (rolling). Do NOT prune the review repo, do NOT set `expert_id`. Push the updated manifest (tentative IDs now visible).
- The overlap auto-relabel cascade runs PER resolved `export_dir` only (never globally — identical frame filenames across projects must not cross-contaminate).
- Per-request remote isolation (added 2026-07-09, hammering-pass fix): every mutating `/expertids` route that
  accepts path overrides (import, accept, delete_project, operator_setup, site_preview, site_push) ALSO accepts
  presence-based remote overrides in the same request body: `review_repo_url` (alias `repo_url`; an EXPLICIT ''
  means "no remote") and `git_push` (alias `push`; tolerant bool). ABSENT keys fall back to the provider
  (production remote + push on), so the panel UI is unchanged. Rationale: a `review_dir` override alone
  previously still git-inited the override dir with the PRODUCTION remote and pushed the overridden manifest to
  the live Pages repo; a path-overridden request must never couple to the production remote unless asked to.
  Additionally (2026-07-09, delete_project hammering-pass fix): a LOCAL-ONLY commit (effective push false)
  never creates or rewrites the target directory's git `origin` — `ReviewRepo.commit_push` scrubs/asserts
  origin only on the push path, so `push:false` alone (without `review_repo_url:''`) already guarantees the
  caller's directory is not branded with the production remote in `.git/config`. Origin seeding still happens
  in `ensure_repo` when a fresh tree is initialized with a remote configured.
- Sanctioned removal paths, exactly TWO (updated 2026-07-09):
  1. Acceptance (C6) stays the ONLY path that sets `expert_id`; it removes that single item from the repo and prunes.
  2. `ReviewRepo.remove_project(project_id)`: batch delete of a WHOLE project's pending items in one locked
     pass (single manifest write, count recomputed, each `items/<uid>/` folder deleted, unsafe UIDs skipped
     with a log line). Exposed as POST `/api/delete_project` behind the Expert IDs panel's checkbox/per-row/
     Delete-selected UI with a confirm dialog and a full loader refresh. It NEVER sets `expert_id` and NEVER
     touches the cross-project library, `reviewer_names`, or `contacts`. Unknown project returns
     `{removed: 0, uids: []}`.

## 6. Consensus model  (C6)
- Input: a uid's `reviews[]`. Agreement = all non-empty `code` equal (ignoring blanks/IDK). `consensus` if ≥2 reviewers agree and none conflict; `conflict` if ≥2 distinct codes; `single` if one reviewer; `none` if zero.
- Operator UI: per mask, columns = reviewers, cells = their code (confidence no longer displayed as of 2026-07-09); shows status chip; operator picks the final code (defaults to the consensus code) → `accept(uid, code)` → sets `expert_id` (mode EXPERT, basis consensus|operator) AND `status:'accepted'` on the mask (2026-07-09: required so the accepted mask reaches YOLO training and counts found_expert in the matrix, both of which are status-accepted-only), writes library + segmentations, removes from repo, pushes.
- POST `/api/accept` is stateless like import (added 2026-07-09): the body may override `review_dir` /
  `library_dir` / `export_dir` / `projects_root` plus the remote (`review_repo_url`, and `git_push` or
  `push`), each falling back to the provider's default. The acceptance writes the SAME library/paths the
  caller read its consensus table from; a `push:false` body must never push, and an explicit
  `review_repo_url` of '' detaches the remote. The panel JS sends the library_dir/export_dir it is showing.

## 7. Site names  (C4)
- `supporting_data/site_codes.csv` columns `site,site_code` (full name ⇄ 3-letter). Source = the operator's spreadsheet (Great Pond/GRP, Brewers Bay/BWR, …). Viewer maps `site_code → site` for display; default to the code when unmapped. `site_full` stamped on items at export when resolvable.

## 8. Reviewer identity + acknowledgment  (C1, ack)
- Viewer requires a reviewer name before any labeling (modal gate, stored in localStorage + sent on every row).
- All reviewer names are tracked (manifest `reviewer_names`, library `reviews/<uid>.json`) so the operator can produce an acknowledgment/credits list of everyone who contributed.

## 9. Testing convention  (smoke + unit + adversarial — EVERYWHERE)
- Each tool keeps `tests/` with a NO-pytest harness (mirror `_reefreview/tests/test_reefreview.py`: `check()/run()`, `__main__` exits nonzero on failure), runnable with the unified env python `<repo>/env/bin/python`.
- THREE layers per module: **smoke** (imports + Flask `GET /` renders 200 / `node --check` JS), **unit** (logic: routing, consensus agreement, selection top-up, schema round-trips), **adversarial** (skeptic cases: wrong-project routing, mask shipped without SAM3, identical frame names across projects, re-drop same reviewer, conflict vs consensus edges, already-published items missing new fields, blank reviewer rejected).
- A top-level `scripts/run_all_tests.sh` runs every tool's tests; CI-style summary.

## 10. Change logs + docs
- Append a dated entry per module touched to `_DOCS/expert_review_revamp/CHANGELOG.md` (what/why/files/tests) — feeds the documentation.
- Expert-reviewer viewer gets a "Where your work fits" doc section: Place Points (point drop) → SAM3 segmentation (mask) → expert review (you) → consensus → AI model. Make clear their CSV must come back or the effort is wasted.

## 11. Viewer render order  (added 2026-07-09)
- Cards render GROUPED BY PROJECT with sticky per-project headers carrying live pending counts. Pure,
  unit-tested helpers: `CORE.sortItems` / `CORE.groupByProject` in `viewer.js`.
- Sort: project label ascending; items with NO project fields sort LAST and collapse into one trailing
  "Unknown project" group; duplicate labels split by project key; then site code, then numeric frame
  (missing/non-numeric frames last), then uid. Non-mutating.
- Group headers hide when the project filter narrows to one project, or when the site filter empties a group.
- The 5-bullet "How to review" instructions and the email-back alert are collapsed by default (expandable).

## 12. Email-request text  (operator -> reviewers, added 2026-07-09)
- `/api/email_requests` emits ONE paste-ready message PER project with pending items, shaped exactly:
  `Subject: TCRMP expert IDs requested: <project name>`, a greeting, the pending-mask count for the named
  project, then numbered steps: (1) open the review-site Pages URL, (2) enter your name, (3) choose
  "<project name>" in the Project menu, (4) click a code for each mask (IDK is fine), (5) download the CSV and
  email it back to the return address.
- Return address = the saved operator_setup email, falling back to `lauren.olinger@uvi.edu`. Each project card
  in the Expert IDs panel also gets a mailto "Email draft" link prefilled with this text.

## 13. Coverage-matrix outcome vocabulary  (_matrix + provenance twins, added 2026-07-09)
- `label_provenance.csv` outcome enum, with conflict-resolution strength (higher wins when more than one
  project covers the same image+label pair):
  `found_expert:5` (an ACCEPTED mask of that species carries an expert_id; magenta #c93ad4; beats all) ·
  `found_manual:4` · `found_ai:3` · `found_model:2` ·
  `pending_expert:1` (a non-rejected mask is still flagged for review; steel blue #64809c; beats not_found only) ·
  `not_found:0`.
- Pending-row label rule: each non-rejected REVIEW mask emits one `pending_expert` row per code found in its
  `reviews[]`, EXCLUDING entries from the synthetic `overlap` reviewer, falling back to the literal label
  `REVIEW` when no usable codes remain. Review masks never yield a found_* row; `found_expert` overrides
  manual/ai for the same code. The legacy no-ledger fallback mirrors both rules.
- The matrix UI keeps the legend, the outcome filter, the headline "% of reviewed cells found", and the
  per-row/per-column stats in sync with this vocabulary.

## 14. Orchestrator status note  (2026-07-09)
- The orchestrator sidebar tile "Expert Review I/O" is TEMPORARILY DISABLED: its `data-step` attribute is
  renamed to `data-step-disabled` so `getNavOrder`/`switchStep` skip it, the onclick is removed, and the tile
  renders reduced-opacity with a tooltip documenting the pause. Re-enable by renaming `data-step-disabled`
  back to `data-step`. The `/expertids` blueprint and `panel-expertids` markup stay live for testing.
