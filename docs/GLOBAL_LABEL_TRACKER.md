# Global Label Tracker & Coverage Planner

Design for a new **reef_point_seg** subsection that turns the existing point
census + the expert-ID mask catalog into a *training-set planner*: choose which
codes you want, how many instances of each, and instantly see coverage, what is
already segmented vs only point-census-known, and the **minimum** extra images to
review to hit your targets — exploiting species co-occurrence so one review pass
yields several classes. Built to scale from "a few coral species" to "many
species, in arbitrary combinations" without re-labeling thousands of images.

Status: **design + validated prototype** (`scripts/_reefreview/coverage.py`,
tested in `scripts/_reefreview/tests/test_coverage.py`, 6/6 green incl. a smoke
test over the real 1.6M-row census: 113,058 images parsed in ~3.3s). UI + the
"build a project from a plan" wiring are **not built yet** — this doc is the map.

---

## 1. Why this works: the data we already have

`supporting_data/all_points.csv` (~1.6M rows) is a **point-level census of every
CPCe/OCR point on every frame**, each row `(date, site, transect, frame,
point_label, species_code, ...)`. Crucially it records *every* point's species —
including species that were never segmented for any given project. So for any
image we already know, with zero new review:

- which species are present, and
- how many point-instances of each.

Measured on the real data (per `coverage.build_census`):

| code | images | point-instances |
|------|-------:|----------------:|
| OFRA | 8,059 | 26,700 |
| PA | 8,577 | 12,911 |
| OA | 2,763 | 7,965 |
| AL | 2,413 | 6,032 |
| MC | 2,715 | 4,951 |
| OFAV | 1,844 | 4,419 |
| AA | 1,975 | 3,062 |

Co-occurrence is real and exploitable: **314** images contain *both* PA and OFAV
(8,263 PA-only, 1,530 OFAV-only). That 314 is the "labeled for X but also
contains Y" set — we can find it from the census instead of re-reviewing
everything.

**Key distinction the planner keeps explicit:**
- *census instances* = upper-bound, planning proxy (points a human dotted).
- *segmented instances* = actual trainable masks in the catalog (the expert-ID
  library). A census point is not a mask until it has gone through route + SAM.

---

## 2. Data model

Two stores, both already defined:

1. **Census** — derived, in-memory/cached, from `all_points.csv`.
   `coverage.build_census(all_points_csv, year_min, year_max, sites)` →
   `{image_key: Counter(code -> n_points)}` where `image_key =
   (date, site, transect, frame)`. ~3s cold; cache to a pickle/JSON keyed by
   (csv mtime, filters) so the UI is instant on repeat.

2. **Global mask catalog** = the existing **expert-ID library**
   (`inprocess/_expert_id_library/manifest.csv` + `polygons/` + `images/` +
   `masks/`), one row per mask UID with `code`, `mode` (USER/EXPERT),
   `source_image`, geometry. This is *the* cross-project label store. The tracker
   adds **no new store** — it reads these two and writes plans.

The census↔catalog join is by **image basename** (`coverage.catalog_index`,
`census_by_basename`). The library stores the clip filename; the census key is a
tuple, so a catalog-aware report must re-key the census to basenames first
(`census_by_basename(census, basename_for_key)`). Caveat documented in code: the
library currently stores `year`, not the full census `date`, so the bridge is
basename-based, not key-based — see §7 open items.

---

## 3. Coverage algorithm (built + tested)

`coverage.coverage_report(census, codes, library=None, target_per_code=None)`:

- per code: `census_images`, `census_instances`, and (with a catalog)
  `segmented_images` vs `gap_images` (present in census, no mask yet);
- pairwise `cooccurrence[a][b]` = images containing both;
- `shortfall[code]` vs an instance target.

`coverage.select_images(census, target_per_code, prefer_codes)` — greedy
set-cover: repeatedly pick the image that contributes the most still-needed
instances across **all** target codes at once (so multi-class frames win first),
until targets are met or candidates run dry. Returns the selected image list,
`reached`, `shortfall`, and `images_reviewed`. This is the "dynamically optimize
the number of images so we're not labeling thousands" engine: minimize images
while covering the requested class mix.

`coverage.gap_images(census, code, library)` — frames that contain `code` per the
census but have no mask for it yet. These are the "review a few extra images"
set: an image labeled for X (has an X mask) that the census says also has Y →
surfaces as a Y gap, so we know to push it through route+SAM to add the Y mask.

Greedy set-cover is the right call here (NP-hard exact; greedy is the standard
≈(1+ln n) approximation) and is O(images × codes) per pick — fine at this scale.

---

## 4. The workflow it enables

```
 choose codes + target instances each
        │
        ▼
 coverage_report  ──►  "you have N segmented, M census-only, K gap images"
        │
        ▼
 select_images (greedy, co-occurrence aware)
        │   minimal image set to hit targets
        ▼
 build a NEW project seeded with:
   • reused mask layers from the catalog (already-segmented frames; no re-work)
   • gap images  ──► route images ──► SAM segmentation ──► REVIEW where unsure
        │                                   │
        │                                   ▼
        │                         new masks saved into the GLOBAL catalog
        ▼                                   │
 project's segmentations + the catalog grow together; the next plan reuses them
```

So a plan does two things: **reuse** existing masks (free) and **queue** only the
gap images for new segmentation, which then flow through the *existing* REVIEW →
expert round-trip and land back in the catalog for every future project.

---

## 5. UI subsection (proposed) — "Coverage Planner"

A new tile in the orchestrator (sibling to "Add Expert IDs": optional,
re-runnable, not a numbered chain step) opening a lightweight Flask sub-app
(`TCRMPclip_coveragePlanner`, ~port 5076, no SAM3). Screen:

- **Filters:** year range, sites (defaults: all). Drives `build_census`.
- **Code picker:** the full master_codes list grouped (reuse `codes.load_codes`
  groups — Coral / Algae / Sponge / Gorgonian / Non-living / …), each with a
  "target instances" number box. Toggle codes in/out; multiselect.
- **Live coverage table** (one row per chosen code): census images, census
  instances, **segmented** (in catalog), **gap** images, shortfall vs target,
  colored bar. Updates on toggle.
- **Co-occurrence heat strip:** for the chosen codes, the pairwise matrix, so you
  see "PA & OFAV share 314 images" at a glance.
- **Plan button:** runs `select_images` → shows *images to review* count, the
  reached/shortfall per code, and a "reused vs new-to-segment" split.
- **Create project from plan:** writes the selected image list as a step-3-style
  `selected_frames.csv` into a new `run_*` project and pre-seeds step 4/5 to point
  at it, plus a manifest of catalog masks to reuse. (Reuses the existing project
  scaffolding in `pipeline_orchestrator/project_manager.py`.)

Every field gets a tooltip (platform rule). Timestamps AST.

---

## 6. Integration points (where it plugs into what exists)

- **Reads:** `all_points.csv` (census) + the expert-ID library (catalog) — both
  already wired via `pipeline.yaml` paths `all_points_csv` and
  `expert_library_dir`.
- **Writes:** a plan JSON + a `selected_frames.csv` into a new project dir; no new
  permanent store.
- **Feeds:** the gap images go through the **existing** step 4 (route) → step 5
  (SAM) → REVIEW round-trip; new masks land in the catalog via the existing
  `review_export` / importer path. The planner does not duplicate any of that.
- **Reuse:** already-segmented frames are referenced from the catalog so a new
  project inherits their masks (the >50%-overlap auto-relabel already handles a
  re-segmented colony inheriting a past expert ID).

---

## 7. Open items / risks (to resolve before building the UI)

1. **Census↔catalog join fidelity.** The library stores `year` not full `date`,
   so the join is by image basename. If two surveys of the same site/transect/
   frame in different years share a basename pattern, confirm basenames are
   globally unique (they encode the full `YYYYMMDD`, so they should be — verify).
2. **Recoded vs raw codes.** The census uses raw `species_code`; a project may
   remap (step 2, e.g. MFAV→OFAV). The planner should apply the active
   `remap_log.json` to the census before counting, or operate on recoded codes,
   so targets line up with what the model will actually train on.
3. **Census instances ≠ trainable masks.** A point-instance is a planning
   upper bound; SAM may merge/split/■reject. Show both numbers; never promise
   census counts as final.
4. **Gap-image cost.** `gap_images` can be large (e.g. 8,263 PA-only frames). The
   planner must cap + let the user dial "how many gap images to add" — that's the
   whole point of `select_images` (target-driven, not "all gaps").
5. **Caching.** Cache the census parse (keyed by csv mtime + filters) so toggling
   codes doesn't re-read 1.6M rows each time.
6. **Scale to many classes.** Greedy set-cover stays linear; the table/heat strip
   are the UI scaling concern (virtualize rows for 130 codes).

---

## 8. Files

- `scripts/_reefreview/coverage.py` — census, report, select_images, gap_images
  (built, pure-stdlib + library).
- `scripts/_reefreview/tests/test_coverage.py` — 6 tests incl. real-data smoke
  (built, green).
- **TODO (next):** `scripts/TCRMPclip_coveragePlanner/` sub-app + an orchestrator
  tile, mirroring the Add-Expert-IDs tile; a census cache; remap application; and
  the "create project from plan" writer.
