# Expert-Review Revamp — Change Log

Detailed, append-only log of changes across the reef_point_seg modules/submodules
for this effort (Tracks A/B/C). Feeds the documentation. Every implementation
step (mine or an agent's) appends an entry. Newest at the bottom of each track.

Format per entry:
```
[YYYY-MM-DD HH:MM AST] <track> <module/file>: what changed + why. Tests: <which ran + result>.
```

---

## Track B — SAM3 / expert-review feature

[2026-06-05] B Phase 0 (canonical project identity): orchestrator passes TCRMP_PROJECT_ID/NAME into the
step-5 launch; review_export.export_flagged_masks gained project_name, stamped on the manifest item +
library record; library MANIFEST_FIELDS gained 'project_name' (additive migration); project_manager gained
find_projects()/resolve_step_dir(). Tests: 4 new unit (test_project_identity.py) + 15 existing reefreview
export/import regression — all PASS.

[2026-06-05] B Phase 1 (step-5 expert/pending colors): TCRMPclip_segmentImages/src/templates/index.html —
maskReviewState() + REVIEW_COLORS; expert=green, pending=amber, targets keep per-id hue; header legend +
per-row review/expert tags. Tests: node --check OK.

## Track A — pipeline flow/UX/correctness

[2026-06-05] A (step-4 menu label): pipeline_orchestrator/templates/index.html — nav title "Route Images
(OCR)" -> "Place Points" (hardcoded nav string, separate from STEP_NAMES); s4 launch button "Open OCR
Review UI" -> "Open Place Points UI". Deployed (orchestrator restarted) + verified served HTML.

[2026-06-05] A-FIX (step-3 forgiving image selection): TCRMPcvr_chooseImages/src/select_images.py —
greedy_select() keeps the year-stratified pass 0, then runs an iterative forgiving top-up (rung 1 drops
year-stratification, rung 2 also drops central-region) until each label reaches target or frames are truly
exhausted; prints per-round status. Root cause: even_allocate() per-year rounding under-allocates lumpy
labels (25 OFRA/9yr, target 10 -> alloc sums to 9). Tests: TCRMPcvr_chooseImages/tests/test_selection_topup.py
(4 cases incl. central-column rung + used-dedupe) — 4/4 PASS (re-run independently). py_compile OK.

[2026-06-05] A-BUG (recoded codes leak — "Montastrea annularis"): root cause = step 5 read the UNRECODED
supporting_data/master_codes.csv because the orchestrator never set TCRMP_MASTER_CODES; codes.json (viewer) +
step-5 labels + library all showed obsolete names. Fix 1 (applied): pipeline_orchestrator/app.py _run_step5
resolves the recoded Step-2 dictionary (steps.2 outputs.master_codes_recoded, else glob step2 dir for
master_codes_recoded.csv / master_codes.csv) and passes it as TCRMP_MASTER_CODES; falls back to canonical
only when Step 2 produced none. py_compile OK. Fix 2 (DEFERRED to post-UI-workflow integration, avoids racing
the Segment agent): TCRMPclip_segmentImages/src/config.py standalone fallback to prefer the recoded dict.
NOTE: takes effect on next orchestrator restart + step-5 relaunch; already-wrong codes.json regenerates on the
next export/push.

[2026-06-05] A (step-4 Place Points labeling UI overhaul): TCRMPclip_placePoints/src/templates/index.html +
src/static/pp_core.js + src/app.py + tests. (1) REVIEW is no longer a special mode — ripped out
reviewMode/setReviewMode/armReviewMode/toggleReviewMode/addReviewPoint + the cyan review-active banner +
the R-arms-mode keybinding; REVIEW is now a normal quick-add label (QUICK_LABELS[REVIEW_IDX], R key/button)
that arms/sticks/places exactly like the number labels and relabels a selected point; REVIEW points keep
species_code 'REVIEW' + review:true so review_export routing is unchanged ("not special until end-of-batch
routing"). (2) Labels pane now sorts ASCENDING alphabetical (pp_core panelSortIndices/compareLabelsAsc).
(3) New points use a MONOTONIC label (pp_core nextLabelMonotonic/labelOrdinal): the label AFTER the current
max, never reusing a freed letter, so additions append to the bottom of the pane. (4) Ctrl/Cmd+click
multi-select in the Labels pane + Del deletes the whole set (deleteSelectedSet). (5) Press-and-hold anywhere
(over points or empty space) pans (HOLD_PAN_MS hold timer + move-past-threshold); a quick tap still
places/selects/deletes; point-drag-to-move preserved. (6) K is a sticky toggle (press to open, press
again/Esc to close) instead of hold-to-show. (7) Enter in reference review confirms the current point,
advances, and always auto-centers (confirmAndAdvance via selectNextLabel+centerOnPoint; no lockPoint symbol
reintroduced). (8) "Done — close & return to main menu" button uses TCRMP_ORCHESTRATOR_URL (now threaded to
the template) else window.close(). On-canvas keyboard legend updated for all of the above. Tests:
pp_core unit/adversarial 53/53 PASS (monotonic, ascending, ordinal inverse, junk-label edges);
verify_apps render+assertions 32/32 PASS; node --check both templates PASS; py_compile PASS;
run_tests.sh = ALL TESTS PASSED.

[2026-06-06] A (orchestrator UI cluster — Track A + Phase-2 shell): pipeline_orchestrator/templates/index.html
+ static/orchestrator.js. Nav reorder to 1,2,3,4,◆SAM3-Review,★Expert-IDs,5,6,7,8; #panel-s4review shell
(Phase-3 launch stub); Prev/Next footers on every panel (getNavOrder/renderStepNavFooters); step-4 + step-5
Advanced <details> collapses + ⓘ tooltips; 10s open-UI loading overlay (openStepService polls /api/step/N/status
then opens); s4/s5 batch readouts. Adversarial-fix pass: batch-readout off-by-one (processed>=total -> all done),
re-gated step-5 Open-Review on segmentation progress, restored window handle so close-on-done works. Tests:
pipeline_orchestrator/tests/test_orchestrator_ui.py 21/21 (smoke+unit+adversarial) — re-run independently.

[2026-06-06] A (Segment UI cluster): TCRMPclip_segmentImages/src/{templates/index.html,app.py,mask_ops.py,config.py}.
Mask list stable alpha+monotonic; ctrl/shift multi-select + multi-delete; sticky-K; always-visible refine/pan hint;
on-demand "Merge same-ID" union (POST /api/merge_same_id via mask_ops.merge_overlapping_same_id). Adversarial-fix:
merge now EXCLUDES placeholder species {REVIEW,'',?} (never unions distinct REVIEW masks -> no data loss), preserves
review_uid/reviews[]/expert_id on the survivor, never downgrades an accepted mask, and REFUSES conflicting expert_ids
(returns refused count surfaced in the UI). Recode Fix 2: config.py standalone MASTER_CODES_CSV prefers the recoded
step2 dict. Also carries the SAM3 tracker-loading fix in sam_engine.py + export_yolo excludes REVIEW from YOLO.
Tests: tests/test_segment_app.py 18/18 (merge + field-merge + smoke) — re-run independently.

[2026-06-06] A (Place Points overhaul + adversarial fixes): TCRMPclip_placePoints (active step-4 tool;
TCRMPclip_routeChosenImages retired). REVIEW demoted from a mode to a normal quick-add label (R); pane now sorts by
label ORDINAL (natural A..Z,AA.. so new monotonic labels land LAST); monotonic non-reused new labels; Ctrl+click pane
multi-select + Del; press-hold pan anywhere; sticky-K; Enter confirm+advance+autocenter in reference review; Done/
return-to-menu. Adversarial-fix: paneSelection.clear() in both single-delete paths (was a wrong-point delete bug),
run_tests.sh export REPO, doneAndExit awaits save. Tests: pp_core 55/55 + verify_apps 32/32 + run_tests.sh ALL PASS
(from non-root CWD).

[2026-06-06] DEPLOY: independent full re-verify before deploy — 12/12 (py_compile all changed py; node --check
orchestrator.js/pp_core.js + all template inline scripts; 6 suites + reefreview regression; Flask GET / smoke for
placepoints+segment). Orchestrator restarted on :5050 and verified serving the new nav/panels/overlay; recode Fix 1
live. Sub-apps (placepoints/segment) relaunch on demand with their fixes. Hard-refresh :5050 to bust cached JS.

## Track C — consensus engine + Phases 4-5 (review-core / viewer / import / consensus)

[2026-06-07] CORE (_reefreview): per-reviewer + multi-project + site-name schema (CONTRACTS §2,3,5,6,7,8).
review_repo.add_review (rolling upsert keyed by reviewer, no prune, tracks reviewer_names[]), accept_item (sets
accepted{mode:EXPERT} then removes — the ONLY removal), pending_by_project(), write_codes emits codes.json
"sites" (from supporting_data/site_codes.csv) + "candidate_codes"; review_export stamps site_full + candidate_codes,
items start reviews:[]/accepted:null; library reviews/<uid>.json; codes.load_sites. Fixed dead candidate-codes
harvest in codes.load_codes. Tests: _reefreview suite green.

[2026-06-07] VIEWER (_reefreview/viewer): reviewer-name gate (required modal, localStorage, stamped on every CSV
row), big "email it back or it's wasted" header (operator + lauren.olinger@uvi.edu, rolling note), project dropdown
+ per-project pending counts, full site names + site filter + bulk-apply, candidate-code quick-picks, tentative
per-reviewer IDs on cards, CSV header uid,code,confidence,reviewer,project_id. Tests: test_viewer_core 73 +
test_viewer_dom 12. (Deploys to the public Pages site on the next review-repo push.)

[2026-06-07] IMPORT + CONSENSUS (_expertids): ROLLING import — group rows by project_id, resolve each to its
step-5 dir (project_manager.resolve_step_dir; fallback library project_id then open export_dir; unrouted UIDs
REPORTED), upsert tentative reviews (never sets expert_id, never prunes), per-project overlap cascade, push.
accept_uid is the ONLY path that sets expert_id on the routed step-5 mask + removes + prunes. Routes:
/api/pending_by_project, /api/consensus (reviewers-as-columns + classify_reviews consensus/conflict/single/none),
/api/accept, /api/operator_setup (operator email + candidate codes), /api/email_requests (per-project text).
Tests: test_expertids 13.

[2026-06-08] TEST MODERNIZATION: rewrote 7 stale test_reefreview.py importer tests to the rolling contract +
5 new accept/consensus tests (45/45); created scripts/run_all_tests.sh (CONTRACTS §9 aggregator). Independent run:
12/12 suites GREEN.

[2026-06-08] INTEGRATION + DEPLOY: orchestrator _expertids_paths now provides projects_root + site_codes (multi-
project routing). Orchestrator restarted on :5050; consensus engine LIVE and reading the real review repo —
/api/pending_by_project shows project test_5june2026 with 19 pending masks; /api/consensus builds the table (19
items, status none). recode Fix 1/2 live. Viewer (reviewer-facing Pages site) deploys on the next review-repo push.

## 2026-07-09 - Expert-review pathway revamp (Tracks A/C + matrix, LO)

[2026-07-09 14:14 AST] LO orchestrator-ui templates/index.html, tests/test_orchestrator_ui.py: disabled the
Expert Review I/O sidebar tile reversibly, data-step renamed to data-step-disabled so getNavOrder/switchStep
skip it, onclick removed, step-disabled class + reduced-opacity no-hover CSS added, tooltip documents the
2026-07-09 pause and the rename-back re-enable path, subtitle gains (disabled); panel-expertids and the
/expertids blueprint stay live for testing; tests updated (nav order drops expertids, new
test_expertids_tile_disabled, step 8 asserted last with disabled Next). Tests: test_orchestrator_ui.py 27/27
PASS; other 7 orchestrator suites PASS; py_compile OK; restart-script HTML assertions simulated ALL PASS.

[2026-07-09 14:22 AST] LO viewer-revamp scripts/_reefreview/viewer/{index.html,viewer.js,viewer.css} +
tests/{test_viewer_core.js,test_viewer_dom.js}: collapsed 5-bullet "How to review" instructions and
two-sentence email-back alert; cards now grouped by project with sticky per-project headers + live pending
counts via new pure CORE.sortItems/groupByProject (unknown project last, site/frame/uid tiebreaks, headers
hide with project filter or when site filter empties a group); confidence UI removed end to end (High/Low
buttons, chips, tooltips) while CSV keeps the exact 5-col header with confidence always blank and legacy
localStorage conf values ignored; tooltip + relic sweep (orphaned .conf/.tip/.alert-note CSS, all em dashes)
across the three files. Tests: node test_viewer_core.js 95 passed 0 failed; node test_viewer_dom.js 45 passed
0 failed; node --check viewer.js clean; _reefreview python suites test_reefreview 45/45, test_review_schema
27/27, test_project_identity 4/4, test_coverage 14/14 all green.

[2026-07-09 14:22 AST] LO expertids-operator scripts/_reefreview/review_repo.py, scripts/_expertids/
{blueprint.py, templates/expertids/panel.html, static/expertids.js, static/expertids.css},
scripts/_expertids/tests/test_expertids.py, scripts/_reefreview/tests/test_review_schema.py: project
batch-delete (ReviewRepo.remove_project one-locked-pass + POST /api/delete_project + checkbox/per-row/
Delete-selected UI with confirm dialog and full loader refresh; library, reviewer_names, contacts untouched),
email carry-through (/api/email_requests now emits subject + greeting + numbered steps + pages URL + return
address from operator_setup falling back to lauren.olinger@uvi.edu; mailto Email-draft link per card; tooltip
fixed to match), confidence display removed from consensus cells + catalog tiles (storage/import parsing
untouched, CSV header contract unchanged), em-dash relic sweep on all touched UI files, tooltips on every new
control. Tests: test_expertids 18/18, test_review_schema 27/27, test_reefreview 45/45 ALL GREEN; py_compile +
node --check clean.

[2026-07-09 14:25 AST] LO matrix scripts/_matrix/{builder.py,static/matrix.js,static/matrix.css,
templates/matrix/index.html,tests/test_matrix.py} + scripts/TCRMPclip_{combinedAnnotate,segmentImages}/src/
provenance.py + scripts/TCRMPclip_combinedAnnotate/tests/test_provenance.py: surfaced expert-review state in
the coverage matrix with two new outcomes, found_expert (accepted expert_id, strength 5, magenta #c93ad4,
beats all) and pending_expert (mask still flagged for review, strength 1, steel blue #64809c, beats not_found
only); provenance twins now emit pending_expert rows for non-rejected review masks (labels from reviews[]
codes excluding the synthetic 'overlap' reviewer, REVIEW fallback) and found_expert overrides manual/ai;
legacy fallback mirrors both rules; legend, outcome filter, headline % found and stats kept in sync. Tests:
test_matrix.py 21/21 PASS, test_provenance.py 22/22 PASS, combinedAnnotate test_segment_app.py 28/28,
segmentImages test_segment_app.py 18/18.

[2026-07-09 14:32 AST] LO tests scripts/_expertids/tests/test_e2e_expert_review.py: new end-to-end
expert-review pathway test (export REVIEW masks from two synthetic projects -> blank-confidence viewer CSV
import for two reviewers -> consensus/conflict/single classification -> accept stamps expert_id + prunes repo
item -> provenance found_expert/pending_expert feeds the coverage matrix -> new ReviewRepo.remove_project
batch delete -> export_yolo trains the accepted code only); temp dirs only, push disabled, house check()/run()
no-pytest pattern. Tests: e2e 7/7 (twice), test_expertids 18/18, test_reefreview 45/45, test_review_schema
27/27, test_coverage 14/14, test_project_identity 4/4, flat_segmentations + step4test_relabel + matrix suites
all green.

[2026-07-09 14:40 AST] LO docs+aggregator _DOCS/expert_review_revamp/CONTRACTS.md, scripts/run_all_tests.sh,
changelog.md: CONTRACTS updated to the new reality (§4 blank-confidence CSV, §5 remove_project as the second
sanctioned removal path, §6 confidence display dropped from consensus cells, new §11 viewer render order,
§12 email-request text, §13 coverage-matrix outcome vocabulary, §14 disabled orchestrator tile note);
run_all_tests.sh SUITES gains _matrix/tests/test_matrix.py and _expertids/tests/test_e2e_expert_review.py;
dictionary descriptors reef_point_seg_expert_review_bundle + reef_point_seg_expert_id_library refreshed
(removal paths, rolling reviews, blank confidence) with schema_version bumps. Tests: bash -n clean;
run_all_tests.sh full aggregator green including the two new suites.

[2026-07-09 15:02 AST] LO importer scripts/_expertids/importer.py + tests/{test_expertids.py,
test_e2e_expert_review.py}: import_rows and accept_uid now regenerate each touched project's
label_provenance.csv ledger themselves (best-effort, never fatal), closing the staleness window where the
coverage matrix ignored a fresh import/accept until the annotator re-touched the frame. New
_refresh_provenance_ledger derives the ledger location exactly like the annotator (_stamp_provenance:
export_dir = parent of the segmentations/ tree, rows keyed splitext(frame), project_id from the run_* path
part else the project dir name), lazily imports the stdlib-only combinedAnnotate provenance module, and
recomputes outcomes with NO target species (not_found stays the annotator's job) for expert-flow frames only
(review-flagged, tentatively reviewed, or expert-stamped masks); stale pending_expert rows whose tentative
code vanished (corrected review, REVIEW placeholder superseded) are pruned so the pending set is exact.
The three seg mutators gained an optional touched= accumulator; refresh runs once per DISTINCT touched file
(never per row) after the overlap cascade and after a successful accept relabel; stats/response gain
provenance_refreshed. e2e stage 50 upgraded to the STRONGER assertion: ledger already shows
found_expert/pending_expert (both projects) straight after stages 20/40 with no manual call, matrix cells +
stats verified (found_expert 1, pending_expert 4), and an annotator-style manual re-stamp is proven
redundant-by-design (idempotent, no cell changes). KNOWN EDGE (pre-existing, unchanged): a REVIEW mask the
annotator never accepted (status pending) yields NO found_expert row after expert accept because provenance
convention counts only accepted non-review masks as found_*; such masks also do not train (export_yolo
accepted-only). Tests: test_expertids 21/21 (new ledger unit + 2 adversarial: broken provenance module /
crashing helper / unwritable ledger path never fail import or accept), e2e 7/7, test_reefreview 45/45,
full run_all_tests.sh aggregator 48/48 green.

[2026-07-09 15:08 AST] LO _expertids/importer.py + tests: _apply_to_mask now also sets status:'accepted' on
expert accept. Closes the KNOWN EDGE above: a REVIEW mask left at the annotator default status 'pending'
previously kept that status through an expert accept, so it never reached YOLO training (export_yolo is
status-accepted-only) and never counted found_expert in the matrix (provenance counts accepted non-review
masks only). e2e fixture m1 now starts 'pending' to prove the flip end to end (m2 keeps the annotator-accepted
pre-state); test_accept_sets_expert_id_and_removes asserts the flip; the provenance ledger unit test starts
from 'pending'. Tests: test_expertids 21/21, e2e 7/7, full run_all_tests.sh 48/48 green.

[2026-07-09 15:14 AST] LO _reefreview/review_repo.py + test_review_schema.py: _gh_token hardened with a
fallback chain: GH_TOKEN/GITHUB_TOKEN env, then `gh auth token`, then parsing gh's stored hosts.yml
(~/.config/gh and ~/snap/gh/*/). Root cause: snap gh cannot exec in capability-restricted process trees
(snap-confine needs cap_dac_override), so an orchestrator launched from such a shell silently skipped every
review-site push ("no gh token; skipped push"). Verified live: /expertids/api/site_push now returns
pushed:true from the running orchestrator. Also pushed the new viewer to the public Pages site (commit
8641d80: project grouping + sort, collapsed slim instructions, confidence removed) and confirmed the
deployed site serves it. Tests: test_review_schema 28/28, full run_all_tests.sh 48/48 green.

[2026-07-09 15:50 AST] LO _expertids/importer.py + blueprint.py + tests: oversized-CSV-field crash fixed
(hammering-pass finding, reproduced first). Any reviewer CSV with a field over csv.field_size_limit's
131072-char default crashed parse_csv_text (uncaught _csv.Error at the reader loop), and because
blueprint.do_import called parse_csv_text OUTSIDE its try/except, POST /api/import answered a raw Flask
HTML 500 instead of the JSON error contract (verified on both the JSON-body and multipart-upload paths).
Two-layer fix: (1) parse_csv_text lifts the csv module's per-field cap to cover the input text (a cell can
never exceed the whole text; the limit is only ever raised, so concurrent parses never see it shrink) —
the giant uid then parses and is reported downstream by is_safe_uid as unsafe_uid per CONTRACTS §4
tolerance, nothing ingested; (2) do_import wraps the parse call in try/except returning a JSON 400 for any
residual parse error so /api/import can never emit an HTML 500 from parsing. Regression tests added to
test_expertids.py: unit test_parse_csv_text_tolerates_oversized_field (1MB uid and 1MB code column both
parse, following rows intact) + adversarial test_adv_oversized_csv_field_returns_json_not_html_500 (both
request paths return JSON with ok:true, unsafe_uid:1, reviews_added:0). Tests: test_expertids 24/24, full
run_all_tests.sh 51/51 green.

[2026-07-09 15:54 AST] LO _reefreview/review_repo.py + _expertids/importer.py + CONTRACTS.md section 2 + tests:
no-op review-repo preamble no longer dirties the production working tree (hammering-pass finding, reproduced
first in a tempdir clone of the production state). Any ReviewRepo.ensure_repo() + set_contacts(<unchanged>)
call, the mandatory preamble of every export_flagged_masks run, rewrote codes.json and review_manifest.json
with pure generated_at churn, rewrote README.md, re-copied the viewer assets, and rewrote .git/config
(unconditional `git config user.name/email`), leaving ' M codes.json / M review_manifest.json' uncommitted on
/mnt/tear/REVIEW_reefpointseg whenever the export queued zero masks (commit_push only runs when something was
exported). That is the cross-session mutation observed live at 15:36:45. Fix makes every seed idempotent:
_json_semantics_match compares content ignoring generated_at and skips the write when nothing semantic
changed (manifest writes via _write_manifest_obj, codes via write_codes), README/viewer writes skip on
identical content (filecmp), and git identity is probed before set. generated_at now moves only when content
moves. write_codes additionally unions candidate extras from the operator_setup.json sidecar so an
ensure_repo re-seed can never drop the operator's extras; importer.save_operator_setup writes the sidecar
BEFORE ensure_repo (removals take effect same-call) and _merge_candidate_codes skips no-change rewrites.
Regression test test_noop_preamble_never_dirties_clean_worktree added to _reefreview/tests/test_reefreview.py:
seeds + commits a clean baseline, reruns the preamble, asserts git status clean + zero inode/mtime churn on
manifest/codes/README/viewer/.git/config + stable generated_at, then proves a REAL contact change still
dirties, and that sidecar extras survive a re-seed. No production paths touched during repro or tests.
Tests: test_reefreview 46/46, test_review_schema 28/28, test_expertids 24/24, e2e 7/7, full
run_all_tests.sh 54/54 green.

[2026-07-09 15:56 AST] LO _expertids/blueprint.py + static/expertids.js + CONTRACTS.md sections 5/6 + tests:
POST /expertids/api/accept could not be isolated and split-brained the panel (hammering-pass finding,
reproduced live over a Flask test client on temp dirs before fixing). The route read only
uid/code/review_dir/basis/labeler from the body: library_dir fell through to the provider's production
_expert_id_library (an overridden accept upserted the EXPERT record + reviews/<uid>.json sidecar into the
LIVE library), git_push fell through to the provider's push-on default (commit_push ran push=True against
the production remote), and projects_root/export_dir fell back to the production inprocess root so the
caller's segmentations were never relabeled (relabeled_seg:false, the mask that feeds YOLO training never
got its expert_id). The panel itself read its consensus table from the cfg-library override (`lib` param)
but accepted into the provider default library. Fix makes accept stateless like import (the blueprint
docstring's own contract): body may override review_dir / library_dir / export_dir / projects_root plus the
remote via the shared _remote_overrides helper (review_repo_url, git_push/push), each falling back to the
provider; api_delete_project now parses its push flag through the shared _parse_push_flag. expertids.js
sends the library_dir/export_dir the panel is showing with each accept. importer.accept_uid already took
every parameter; only the route plumbing was missing. Regression test
test_accept_api_honors_body_overrides_and_push_flag added to _expertids/tests/test_expertids.py: accepts via
the HTTP route with a provider pinned to sentinel "production" paths and body overrides to an isolated env,
asserts the override library gets the EXPERT record, the provider library and sidecar stay untouched,
push:false wins over provider git_push=True, review_repo_url:'' detaches the remote, the relabel routes to
the override export dir, and a second no-override accept still falls back to provider paths/push. No
production paths touched during repro or tests. Tests: test_expertids 26/26, full run_all_tests.sh 54/54
green.

[2026-07-09 15:59 AST] LO _expertids/blueprint.py + tests + CONTRACTS §5: import remote-isolation defect
fixed (hammering-pass finding, reproduced first). POST /api/import honored export_dir/review_dir/library_dir
overrides but repo_url and git_push always came from the paths provider (orchestrator live values: the real
reefpointseg-review.git remote and git_push=True since TCRMP_REVIEW_GIT_PUSH is unset), so an import pointed
at a scratch review_dir was still git-inited with origin = the PRODUCTION Pages URL, committed, and its
scratch manifest pushed to production main - even when the body EXPLICITLY sent review_repo_url:'' and
git_push:false (both keys ignored). Reproduced live through the real HTTP route against a local bare repo
standing in for the production remote (no network, no token: _token_url only rewrites https URLs):
pushed=true, origin wired to the production URL, scratch manifest landed on the remote's main. Fix:
presence-based per-request remote overrides via module-level _remote_overrides/_parse_push_flag - every
mutating route that accepts path overrides (import, accept, delete_project, operator_setup, site_preview,
site_push) now honors review_repo_url (alias repo_url; explicit '' = no remote) and git_push (alias push;
tolerant bool) from the same request body; _repo() takes an optional repo_url; ABSENT keys keep provider
behavior so the panel UI is unchanged (it never sends the remote keys). Contract documented as CONTRACTS §5
"Per-request remote isolation". Regression test
test_adv_import_remote_overrides_isolate_from_provider_repo covers: JSON + multipart-form isolation (no
push, remote detached, production URL never enters the scratch .git/config, provider review_dir untouched),
provider fallback intact without the keys, and accept honoring the same switches. Post-fix the reproduction
script reports pushed=false, no origin remote, nothing on the stand-in production remote. Tests:
test_expertids 26/26, full run_all_tests.sh 54/54 green.

[2026-07-09 16:02 AST] LO _expertids/importer.py + _reefreview/library.py + tests: concurrent-import race on
segmentations.json and reviews/<uid>.json fixed (hammering-pass finding, reproduced first). Two simultaneous
CSV imports for the same project did unlocked read-modify-writes with a FIXED '<path>.tmp' name: both wrote
the same tmp, the first os.replace consumed it, and the second import crashed mid-way with FileNotFoundError
(HTTP 500, partial state); even without the crash the last writer clobbered the other reviewer's freshly
upserted seg-level reviews[] row, so the flock-protected manifest kept both reviewers while segmentations.json
kept only one (silent divergence). Reproduced offline in scratch dirs (never the live library or
/mnt/tear/REVIEW_reefpointseg): 20 iterations of two threaded imports gave 1 FileNotFoundError crash and 2
iterations with a lost seg reviewer (one with NO crash - pure lost update) pre-fix; the same tally is 0/0
post-fix. The regression test then exposed the SAME defect one layer down: Library.save_reviews used a fixed
tmp name and import_rows/accept_uid did unlocked load-filter-append-save on reviews/<uid>.json. Fix, both
layers: (1) importer._seg_locked - per-file flock (hidden sibling .segmentations.json.lock, mirrors
ReviewRepo._locked) held across the WHOLE load-mutate-save in _tentative_in_segmentations,
_relabel_accepted_in_segmentations, and _overlap_tentative_pass, degrading to unlocked on read-only trees
(pre-fix semantics: failures surface only at an actual write); (2) unique mkstemp tmp names for
importer._save, _prune_stale_pending_rows, and Library.save_reviews, replacing via a shared
replace_preserving_mode so mkstemp's 0600 mode never narrows the target file's permissions; (3) new locked
Library.upsert_review / set_review_accepted used by import_rows (real + overlap reviewers) and accept_uid
instead of the manual read-modify-write. Regression test
test_adv_concurrent_imports_same_project_lose_nothing (owning suite _expertids/tests/test_expertids.py):
8 trials x 3 frames of barrier-synced two-reviewer imports; asserts no crash, both reviewers present in
segmentations.json AND the manifest AND reviews/<uid>.json for every uid, and the seg file keeps group/world
read after the atomic replace. Pre-fix it failed 3/3 runs; post-fix 5/5 runs green. Tests: test_expertids
26/26, _reefreview 46/46, e2e 7/7, full run_all_tests.sh 54/54 green.

[2026-07-09 16:08 AST] LO _reefreview/review_repo.py + _expertids/blueprint.py docstring + CONTRACTS.md
section 5 + tests: delete_project push-default finding (hammering pass) reproduced first against a LOCAL
bare repo standing in for the production remote (no network, no production paths touched): POST
/expertids/api/delete_project with a review_dir override and the push key omitted pushed the caller's
directory to the configured remote (provider git_push defaults True live), and even an explicit push:false
still rewrote that directory's git origin to the production URL (commit_push ran _scrub_remote on every
commit). Resolution in two parts. (1) The omitted-push default itself is now governed by the per-request
remote-isolation contract locked earlier today (CONTRACTS section 5, 15:56 entry): absent keys keep provider
behavior because the panel UI sends review_dir on every delete and depends on the default push to update the
public review site; callers wanting isolation send review_repo_url:'' (detach) and/or push:false — verified
working for delete_project and now pinned by regression test
test_delete_project_remote_isolation_never_brands_foreign_dir (real commit_push, no mocks). (2) The residual
defect FIXED: ReviewRepo.commit_push now scrubs/asserts origin ONLY on the push path, so a local-only commit
(push false) never creates origin in a caller-supplied directory and never rewrites an existing origin that
points elsewhere; ensure_repo still seeds origin on fresh init and every pushing commit still re-asserts the
clean token-free URL. Unit pin test_review_repo_local_commit_never_creates_or_rewrites_origin added to
_reefreview/tests/test_reefreview.py (push:false on an origin-less repo attaches nothing; a foreign origin
survives a local commit). Repro rerun post-fix: push:false leaves the scratch tree with no origin; the
sanctioned review_repo_url:'' switch fully isolates delete_project even with push omitted. Tests:
test_expertids 27/27, test_reefreview 47/47, full run_all_tests.sh 54/54 green.

[2026-07-09 16:15 AST] LO _expertids/importer.py + test_expertids.py: two residual hammer findings fixed.
(1) parse_csv_text strips a leading UTF-8 BOM - an Excel round-trip CSV previously failed header detection
and parsed positionally, leaking the header as a phantom data row. (2) Segmentations writes are now
tolerant-and-reported: a read-only/full segmentations tree no longer crashes import_rows with partial state
(manifest+library rows already landed); failures land in stats['seg_write_failed'] per uid, the overlap
cascade is guarded the same way, and accept_uid reports relabel_error instead of raising after the repo item
was already pruned. Tests: test_expertids 29/29 (2 new regressions), full run_all_tests.sh 54/54 green
(suite count grew 48 -> 54 from the hammer fix loop's new isolation/race/churn suites).
