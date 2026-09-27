#!/usr/bin/env python3
"""
TCRMPcvr_chooseImages — Balanced image selection for coral training data.

Selects frames evenly across years, sites, and transects to achieve
TARGET_INSTANCES_PER_SPECIES frame-instances for each target coral species.

Outputs:
  output/selected_frames.csv      — master list of selected frames + metadata
  output/route_cpc.csv            — frames with existing CPC point coords (pre-2020)
  output/route_ocr.csv            — frames needing OCR processing (2020+)
  output/route_missing.csv        — frames with no source image found
  output/selection_summary.txt    — human-readable summary
  output/selection_summary.json   : machine-readable per-site and per-species
                                    summary (Step 4 reserve refill and the
                                    orchestrator read this)
  output/selection_diagnostics.png — distribution plots

Usage:
  python src/select_images.py
  python src/select_images.py --all-points /path/to/all_points.csv
  python src/select_images.py --species OFRA PA OA
"""

import os
import sys
import glob
import json
import math
import random
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

# Allow running from repo root or from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_recoded(recode_dir, fallback):
    """Use most recent recoded all_points if available, else fallback."""
    pattern = os.path.join(recode_dir, "all_points_*.csv")
    files = sorted(glob.glob(pattern))
    if files:
        return files[-1]
    return fallback


def build_image_filename(date_str, site, transect, frame):
    """Reconstruct the TCRMP image filename from frame metadata."""
    # date_str is YYYY-MM-DD, need YYYYMMDD
    ymd = date_str.replace("-", "")
    return f"TCRMP{ymd}_clip_{site}_T{int(transect)}{int(frame):02d}"


def ast_timestamp():
    """Local box time is AST; format 24h with an explicit AST suffix."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S AST")


def missing_reason(found, basename, clip_dir):
    """Human-readable reason a frame's image was not resolved on disk."""
    if found:
        return ""
    return (f"no image '{basename}.(jpg|jpeg)' found anywhere under "
            f"{clip_dir} (whole-tree recursive scan)")


def _clip_path_rank(path):
    """Lower rank = more canonical. When the same frame stem exists in more than
    one place (e.g. a high-res JPEG/ re-export or an _edit variant alongside the
    original), prefer the flat original: re-export/edit copies can be cropped or
    re-encoded and would misalign the CPC/OCR points computed for the original.
    Deterministic so Step 3 and Step 4 always resolve to the same file."""
    parts = path.split(os.sep)
    noncanonical = any(p == "JPEG" or p.endswith("_edit") for p in parts[:-1])
    return (1 if noncanonical else 0, len(parts))


def _persisted_file_list(clip_dir):
    """All file paths under clip_dir via the shared persisted index
    (scripts/clip_index.py: one walk, dir-mtime staleness sentinel, auto
    rebuild). Returns None when the module is unavailable or errors, so
    callers keep their original walk as the fallback."""
    try:
        scripts_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if scripts_root not in sys.path:
            sys.path.insert(0, scripts_root)
        import clip_index
        return clip_index.list_files(clip_dir)
    except Exception:
        return None


def build_clip_index(clip_dir):
    """Whole-tree image index of the clip tree.

    Maps each image stem (filename without extension) -> absolute path.
    TCRMP basenames are globally unique per logical frame, so a flat index lets
    us resolve a frame regardless of which TCRMP{season}_clip or period subfolder
    (Annual/, PostBL/, ...) it is filed under. This fixes frames whose survey
    date year differs from their season-folder year (e.g. a 2018-01 frame filed
    under TCRMP2017_clip). Hidden dirs (.AppleDouble, .git) and dot-files are
    skipped; stem collisions resolve to the most canonical copy. jpg/jpeg only.

    Served from the persisted clip index when current (Task 1.4, 2026-08-26);
    falls back to the original one-pass recursive walk otherwise.
    """
    index = {}
    if not clip_dir or not os.path.isdir(clip_dir):
        return index
    paths = _persisted_file_list(clip_dir)
    if paths is None:
        paths = []
        for root, dirs, files in os.walk(clip_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fn in files:
                if not fn.startswith("."):
                    paths.append(os.path.join(root, fn))
    for path in paths:
        fn = os.path.basename(path)
        stem, ext = os.path.splitext(fn)
        if ext.lower() in (".jpg", ".jpeg"):
            prev = index.get(stem)
            if prev is None or _clip_path_rank(path) < _clip_path_rank(prev):
                index[stem] = path
    return index


def find_source_image(basename, clip_index):
    """Resolve the raw image via the prebuilt clip index.
    Returns (path, ext) or (None, None)."""
    path = clip_index.get(basename)
    if path:
        return path, os.path.splitext(path)[1].lstrip(".")
    return None, None


def find_pts_image(basename, clip_index):
    """Resolve the _pts annotated variant via the prebuilt clip index."""
    return clip_index.get(f"{basename}_pts")


def load_cpc_basenames(cpc_dir, years):
    """Load all basenames that have CPC point coords, for given years.
    Returns a set of basenames (without extension)."""
    basenames = set()
    for yr in years:
        csv_path = os.path.join(cpc_dir, str(yr), "ids", "point_coords.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            names = df["raw_image"].str.replace(r"\.(jpg|jpeg)$", "", regex=True)
            basenames.update(names)
        except Exception:
            pass
    return basenames


def frame_eligibility(binary, clip_index, cpc_basenames, name_fn):
    """Return (eligible_index, exclusions) for the candidate pool.

    A frame is eligible iff its source image resolves on disk AND its point
    source resolves: pre-2020 -> basename in cpc_basenames (real CPC coords);
    2020+ -> a `_pts` annotated image resolves (OCR reads it in Step 4).
    Image existence, pre-2020 cpc_basenames membership, and 2020+ _pts
    existence are HARD gates here, so a selected frame can never be dropped
    downstream as image_missing or cpc_missing.

    cpc_basenames is a union across every cpc_all season folder for the
    candidate years (load_cpc_basenames), not a per-frame calendar-year
    check, because a season surveyed late in the year carries dates into the
    next January through June while cpc_all/<folder> is named for the
    season, not the date (cpc_all/2017 holds 2,623 frames dated into 2018).
    The downstream lookup in TCRMPclip_placePoints.load_cpc_points
    (find_cpc_rows_any_season) searches the same set of season folders, so
    this gate and that lookup agree on which frames have real CPC points.
    """
    keep = []
    excl = {"image_missing": 0, "cpc_missing": 0, "pts_missing": 0}
    for fid in binary.index:
        r = binary.loc[fid]
        base = name_fn(str(r["date"]), r["site"], int(r["transect"]), int(r["frame"]))
        if base not in clip_index:
            excl["image_missing"] += 1
            continue
        if int(r["year_int"]) < 2020:
            if base in cpc_basenames:
                keep.append(fid)
            else:
                excl["cpc_missing"] += 1
        else:
            if (base + "_pts") in clip_index:
                keep.append(fid)
            else:
                excl["pts_missing"] += 1
    return keep, excl


def _load_species_remap():
    """Load old_code -> new_code mapping from recodeSpecies remap_log."""
    import json as _json
    remap = {}
    recode_dir = os.path.join(config._REPO_DIR, "TCRMPcvr_recodeSpecies", "output")
    logs = sorted(glob.glob(os.path.join(recode_dir, "remap_log_*.json")))
    if not logs:
        return remap
    with open(logs[-1]) as f:
        data = _json.load(f)
    for entry in data.get("remaps", []):
        old = entry.get("old_code", "")
        new = entry.get("new_code", "")
        if old and new and old != new:
            remap[old] = new
    return remap


def load_cpc_species(cpc_dir, years, remap):
    """Build a DataFrame of (frame_id, species_code, category, x, y) from cpc_all.

    Uses point_coords.csv from cpc_all (correct label-to-species mapping)
    with species remap applied. Returns rows compatible with all_points format.
    """
    import re
    rows = []
    for yr in years:
        csv_path = os.path.join(cpc_dir, str(yr), "ids", "point_coords.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            basename = os.path.splitext(r["raw_image"])[0]
            m = re.match(r"TCRMP(\d{4})(\d{2})(\d{2})_clip_([A-Za-z]+)_T(\d)(\d{2})", basename)
            if not m:
                continue
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            site = m.group(4).upper()
            transect = int(m.group(5))
            frame = int(m.group(6))
            sp = r.get("species_code", "")
            sp = remap.get(sp, sp)  # apply recode
            rows.append({
                "date": date_str,
                "year": int(m.group(1)),
                "site": site,
                "transect": transect,
                "frame": frame,
                "species_code": sp,
                "category": r.get("category", ""),
                "point_label": r.get("label", ""),
                "x": r.get("x", np.nan),
                "y": r.get("y", np.nan),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Selection algorithm
# ─────────────────────────────────────────────────────────────────────────────

def _pick_best(scores):
    """Fresh-random pick among the frames tied at the max score (no fixed seed).
    Preserves the score tiers (central / single-species) — randomness only
    breaks ties, which the binary presence scores produce in bulk."""
    if scores.empty or scores.max() <= 0:
        return None
    top = scores[scores == scores.max()].index.tolist()
    return random.choice(top)


def even_allocate(binary, species_list, years, target):
    """Allocate target instances evenly across years, capped by availability."""
    alloc = {}
    for sp in species_list:
        avail = {yr: int(binary[binary["year_int"] == yr][sp].sum()) for yr in years}
        sp_alloc = {yr: 0 for yr in years}
        remaining = target
        active = list(years)
        for _ in range(10):
            if remaining <= 0 or not active:
                break
            per_year = remaining / len(active)
            settled = []
            for yr in active:
                can_add = avail[yr] - sp_alloc[yr]
                give = min(per_year, can_add)
                sp_alloc[yr] += int(round(give))
                if sp_alloc[yr] >= avail[yr]:
                    settled.append(yr)
            remaining = target - sum(sp_alloc.values())
            for yr in settled:
                active.remove(yr)
        alloc[sp] = sp_alloc
    return alloc


def _water_fill_at_cap(avail, sites, target, cap):
    """One water-fill of `target` across `sites` under a FIXED per-site cap.

    Pure and deterministic. Every site starts at its inclusion floor of 1, then
    the remainder is placed proportional to avail using the largest-remainder
    method, bounded per site by min(avail, cap). Returns {site: count}.

    The cap is fixed for the whole fill (allocate_by_site chooses it), which is
    what makes the fill converge: the previous version recomputed the cap from
    the RUNNING total inside the loop, so the very first round saw
    floor(cap_frac * n_sites) and stalled at zero headroom whenever
    n_sites * cap_frac < 2 (any site count of 7 or fewer at cap_frac 0.25).
    """
    result = {s: 1 for s in sites}
    remaining = target - len(sites)

    for _ in range(50):
        if remaining <= 0:
            break
        headroom = {s: max(0, min(avail[s], cap) - result[s]) for s in sites}
        total_headroom = sum(headroom.values())
        if total_headroom <= 0:
            break  # availability/cap-limited: no site can take more

        total_avail_weight = sum(avail[s] for s in sites if headroom[s] > 0)
        to_place = min(remaining, total_headroom)

        # Largest-remainder proportional split across sites with headroom.
        raw = {}
        for s in sites:
            if headroom[s] <= 0:
                continue
            weight = avail[s] / total_avail_weight if total_avail_weight else 0
            raw[s] = min(headroom[s], to_place * weight)

        floors = {s: int(math.floor(v)) for s, v in raw.items()}
        placed = sum(floors.values())
        leftover = to_place - placed

        # Distribute the leftover (largest-remainder method), deterministic
        # tie-break by (remainder desc, avail desc, site name asc).
        remainders = sorted(
            raw.keys(),
            key=lambda s: (-(raw[s] - floors[s]), -avail[s], s),
        )
        i = 0
        leftover_units = int(round(leftover))
        while leftover_units > 0 and i < len(remainders):
            s = remainders[i]
            if floors[s] < headroom[s]:
                floors[s] += 1
                leftover_units -= 1
            i += 1
            if i >= len(remainders) and leftover_units > 0:
                # second sweep in case earlier sites still have headroom
                remainders = [s for s in remainders if floors[s] < headroom[s]]
                i = 0
                if not remainders:
                    break

        made_progress = False
        for s, add in floors.items():
            if add > 0:
                result[s] += add
                made_progress = True
        remaining = target - sum(result.values())

        if not made_progress:
            break  # nothing placeable this round, stop (converged)

    return result


def _smallest_cap_reaching(avail, sites, target, lo, hi):
    """Smallest per-site cap in [lo, hi] whose fill reaches min(target, avail).

    sum(_water_fill_at_cap(...)) is monotone non-decreasing in cap, so a binary
    search finds the tightest (most balanced) cap that still meets the request.
    Used only when cap_frac itself is unsatisfiable, i.e. when there are too
    few sites for any allocation to hold every site at or under cap_frac.
    """
    goal = min(target, sum(avail[s] for s in sites))
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(_water_fill_at_cap(avail, sites, target, mid).values()) >= goal:
            hi = mid
        else:
            lo = mid + 1
    return lo


def allocate_by_site(avail_by_site, target, cap_frac=0.25):
    """Distribute `target` frame-instances across sites (water-filling).

    Pure, deterministic, no RNG. Guarantees:
      - every site with avail >= 1 gets >= 1 (inclusion is a HARD rule);
      - no site exceeds floor(cap_frac * total_allocated), UNLESS the site
        count makes that cap mathematically unsatisfiable (fewer than
        ceil(1 / cap_frac) sites: 2 or 3 sites can never both clear a 25
        percent share while each holds its floor of 1). In that case the cap
        is documented as unsatisfiable rather than violated, and the tightest
        cap that still meets the request is used instead, which is the most
        even split the data allows (see main()'s WARN print for the runtime
        version of this check);
      - no site exceeds its own avail;
      - allocation is monotonic in avail (higher-abundance sites get >= lower
        ones), modulo the cap and integer rounding;
      - same input -> same output.

    Convergence: the cap is seeded from the intended total, not from the
    post-floor running total, and then shrunk to a fixed point. Because the
    filled total A(T) is monotone non-decreasing in the assumed total T, the
    descent from T = min(target, total_avail) settles on the LARGEST total
    consistent with cap = floor(cap_frac * total). The previous
    running-total seed stalled at zero headroom for any site count of 7 or
    fewer at cap_frac 0.25 and returned exactly n_sites frames.

    Args:
      avail_by_site: {site: available_frame_count} - sites with avail <= 0
        are dropped, not floored.
      target: requested total frame-instances for this species.
      cap_frac: max share of the FINAL allocated total any one site may hold.

    Returns: {site: allocated_count}.
    """
    sites = sorted(s for s, a in avail_by_site.items() if a and a >= 1)
    if not sites:
        return {}

    avail = {s: int(avail_by_site[s]) for s in sites}
    n_sites = len(sites)

    # Step 1 and 2: if target <= n_sites the inclusion floor already satisfies
    # (and exceeds, if target < n_sites) the request. Inclusion beats the
    # numeric target.
    if target <= n_sites:
        return {s: 1 for s in sites}

    total_avail = sum(avail.values())
    budget = min(target, total_avail)

    cap_satisfiable = cap_frac > 0 and n_sites >= math.ceil(1 / cap_frac)

    if not cap_satisfiable:
        # Too few sites for cap_frac to bind on every site at once. The most
        # balanced allocation that still meets the request is the one with the
        # smallest workable per-site ceiling, floored at an even split.
        lo = max(1, math.ceil(budget / n_sites))
        hi = max(lo, max(avail.values()))
        cap = _smallest_cap_reaching(avail, sites, target, lo, hi)
        return _water_fill_at_cap(avail, sites, target, cap)

    # Cap-frac fixed point: start from the intended total so real headroom
    # exists beyond the floor round, then shrink until the cap that produced
    # the allocation is the cap the allocation's own total supports.
    assumed_total = budget
    result = {s: 1 for s in sites}
    for _ in range(64):
        cap = max(1, math.floor(cap_frac * assumed_total))
        result = _water_fill_at_cap(avail, sites, target, cap)
        allocated = sum(result.values())
        if allocated >= assumed_total:
            break
        assumed_total = allocated

    return result


def _spread_alloc_across_years(binary_sub, species_list, site_target_by_sp, sub_years):
    """Spread each species' site-level target across that site's years.

    even_allocate() divides a scalar target by len(active_years) every round
    and int-rounds the quotient; when target is small relative to the number
    of years (the common case here - allocate_by_site's inclusion floor is
    often exactly 1, spread across many years of a long-running site) that
    quotient rounds to 0 in EVERY year and even_allocate returns an all-zero
    alloc, which in turn makes greedy_select recover target=0 (it infers the
    scalar target as max(sum(alloc[sp].values()))) and silently skip the
    site. even_allocate is intentionally left untouched (its own unit tests
    pin its rounding behavior), so this helper uses a largest-remainder
    split instead - capped by each year's true availability, so the full
    site_target is always placed across the years whenever avail supports
    it, and greedy_select still performs the actual frame selection (with
    central-region and single-species preference) inside each year's slice.
    """
    alloc = {sp: {yr: 0 for yr in sub_years} for sp in species_list}
    if not sub_years:
        return alloc
    for sp in species_list:
        target = site_target_by_sp.get(sp, 0)
        if target <= 0:
            continue
        avail = {yr: int(binary_sub[binary_sub["year_int"] == yr][sp].sum()) for yr in sub_years}
        total_avail = sum(avail.values())
        target = min(target, total_avail)
        sp_alloc = {yr: 0 for yr in sub_years}
        remaining = target
        active = [yr for yr in sub_years if avail[yr] > 0]
        # Iteratively water-fill: proportional split by avail, largest-
        # remainder rounding, capped per-year by avail, loop until placed or
        # no year has headroom left (mirrors allocate_by_site's approach).
        for _ in range(50):
            if remaining <= 0 or not active:
                break
            total_active_avail = sum(avail[yr] for yr in active)
            raw = {yr: remaining * (avail[yr] / total_active_avail) for yr in active}
            floors = {yr: min(int(raw[yr]), avail[yr] - sp_alloc[yr]) for yr in active}
            placed_now = sum(floors.values())
            leftover = remaining - placed_now
            order = sorted(active, key=lambda yr: (-(raw[yr] - int(raw[yr])), -avail[yr], yr))
            i = 0
            leftover_units = int(round(leftover))
            while leftover_units > 0 and order:
                yr = order[i % len(order)]
                headroom = avail[yr] - sp_alloc[yr] - floors[yr]
                if headroom > 0:
                    floors[yr] += 1
                    leftover_units -= 1
                i += 1
                if i >= len(order) * 2:
                    break  # no headroom anywhere left, stop
            made_progress = False
            for yr, add in floors.items():
                if add > 0:
                    sp_alloc[yr] += add
                    made_progress = True
            remaining = target - sum(sp_alloc.values())
            active = [yr for yr in active if sp_alloc[yr] < avail[yr]]
            if not made_progress:
                break
        alloc[sp] = sp_alloc
    return alloc


def compute_site_targets(binary, species_list, target, cap_frac=0.25):
    """Per-species site allocation: {species: {site: allocated_count}}.

    Abundance comes from the eligible pool itself (frames at that site carrying
    the label), so an allocation can never exceed what the site can supply.
    Pure and deterministic; site_balanced_select drives the selection from it
    and main() reads the same numbers for the summary, so the reported
    allocation and the allocation actually used are the same object.
    """
    site_target = {}
    for sp in species_list:
        avail_by_site = {
            s: int(cnt) for s, cnt in binary.groupby("site")[sp].sum().items()
            if cnt > 0
        }
        site_target[sp] = allocate_by_site(avail_by_site, target,
                                           cap_frac=cap_frac)
    return site_target


def site_balanced_select(binary, species_list, years, target, cap_frac=0.25):
    """Site-balanced frame selection: no site may exceed cap_frac of the
    selected total, every site carrying a target label is included, and
    frames per site are proportional to that site's label abundance.

    Reuses greedy_select PER SITE so central-region preference, single-
    species preference, and year/transect spread are preserved WITHIN each
    site (they come for free from greedy_select's existing scoring). The
    OUTER site loop enforces the 25% cap and the inclusion floor.

    Returns (selected, achieved) - same shape as greedy_select.
    """
    if "site" not in binary.columns:
        # No site column available - fall back to the original algorithm.
        alloc = even_allocate(binary, species_list, years, target)
        return greedy_select(binary, species_list, alloc, years)

    # Step 1: per-species site allocation from abundance (eligible frames at
    # that site carrying the label - selection-consistent, can't exceed avail).
    site_target = compute_site_targets(binary, species_list, target,
                                       cap_frac=cap_frac)  # [sp][site] = count
    all_sites = set()
    for sp in species_list:
        all_sites.update(site_target[sp].keys())

    selected = []
    achieved = {sp: 0 for sp in species_list}
    used = set()

    # Step 2: iterate sites deterministically (sorted) and select within each
    # site's sub-frame via the existing greedy_select machinery.
    for site in sorted(all_sites):
        sub = binary[(binary["site"] == site) & (~binary.index.isin(used))]
        if sub.empty:
            continue
        sub_years = [y for y in years if y in set(sub["year_int"].unique())]
        if not sub_years:
            continue

        site_sp_target = {sp: site_target[sp].get(site, 0) for sp in species_list}
        if not any(v > 0 for v in site_sp_target.values()):
            continue

        site_alloc = _spread_alloc_across_years(sub, species_list, site_sp_target, sub_years)

        # Each label is topped up to ITS OWN site allocation. Without
        # target_by_species, greedy_select recovers one scalar target (the max
        # allocation-sum across labels) and applies it to every label, so a
        # low-target label at a dominant site was pulled up to the dominant
        # label's number and the site swelled past its combined allocation.
        site_selected, site_achieved = greedy_select(
            sub, species_list, site_alloc, sub_years,
            target_by_species=site_sp_target)

        for fid in site_selected:
            if fid in used:
                continue  # defensive: never double-count a frame
            used.add(fid)
            selected.append(fid)
            for sp in species_list:
                achieved[sp] += int(binary.loc[fid, sp])

    return selected, achieved


def greedy_select(binary, species_list, alloc, years, target_by_species=None):
    """Year-stratified greedy frame selection with a forgiving top-up.

    Pass 0 (year-stratified, unchanged): two sub-passes — first selects
    frames with central target-species points (25-75% of image w/h) to fill
    the central slice of each species' per-year allocation, then fills the
    remaining per-year slots with any eligible frame. Prefers single-species
    frames for diversity — frames with only one needed species score higher
    than frames with multiple, unless the extra species is also still needed.

    Top-up passes (new): the stratified passes fill each year's quota
    independently and stop when a year holds no more frames with a needed
    species, so a label can finish SHORT even though enough instances exist in
    OTHER years. After pass 0, any label still below target is topped up with
    additional passes that progressively RELAX constraints:
        pass 1 — drop year-stratification (pull ANY unused frame with the
                 species, best-first, still preferring single-needed-species
                 frames);
        pass 2 — additionally drop the central-region constraint (a no-op
                 unless an earlier relaxed pass had been central-gated; kept
                 explicit so the relaxation ladder is auditable).
    The loop repeats until every label reaches target OR no eligible unused
    frame remains for any short label (genuinely exhausted). A label only stays
    SHORT when binary[sp].sum() is truly < target.

    Preserves the (selected, achieved) return. `target_by_species` is the only
    added argument and it is optional:
      - omitted (the global path): the per-label target is recovered from the
        allocation exactly as before, as ONE scalar (the even_allocate pass
        distributes the full scalar target across years for any label whose
        total availability >= target, so the max allocation-sum is that
        target). Behavior is byte-for-byte the old behavior.
      - supplied (the per-site path in site_balanced_select): each label is
        topped up to its OWN number. One scalar is wrong per site, because a
        site's allocation differs per label; recovering the max made a
        low-allocation label chase the dominant label's target and inflated
        the dominant site's share of the selection.
    """
    has_central = all(f"{sp}_central" in binary.columns for sp in species_list)

    selected = []
    achieved = {sp: 0 for sp in species_list}
    achieved_central = {sp: 0 for sp in species_list}
    used = set()

    # ── Pass 0: the original year-stratified two-pass selection ─────────────
    for pass_name in (["central", "any"] if has_central else ["any"]):
        for yr in years:
            yr_data = binary[(binary["year_int"] == yr) & (~binary.index.isin(used))]
            yr_rem = {sp: max(0, alloc[sp].get(yr, 0) - sum(
                int(binary.loc[fid, sp]) for fid in selected
                if binary.loc[fid, "year_int"] == yr
            )) for sp in species_list}
            avail = yr_data.copy()

            while any(r > 0 for r in yr_rem.values()) and len(avail) > 0:
                scores = pd.Series(0.0, index=avail.index)
                n_needed = sum(1 for sp in species_list if yr_rem[sp] > 0)
                for sp in species_list:
                    if yr_rem[sp] > 0:
                        scores += avail[sp]

                # Penalise multi-species frames: prefer frames with just
                # one needed species so we get more variety across frames
                if n_needed > 1:
                    n_sp_in_frame = pd.Series(0, index=avail.index)
                    for sp in species_list:
                        if yr_rem[sp] > 0:
                            n_sp_in_frame += avail[sp]
                    # Single-species frames get +0.5 bonus
                    scores += (n_sp_in_frame == 1).astype(float) * 0.5

                if pass_name == "central":
                    central_mask = pd.Series(False, index=avail.index)
                    for sp in species_list:
                        if yr_rem[sp] > 0:
                            central_mask |= avail[f"{sp}_central"] > 0
                    scores = scores.where(central_mask, 0)
                best = _pick_best(scores)
                if best is None:
                    break
                selected.append(best)
                used.add(best)
                for sp in species_list:
                    added = int(avail.loc[best, sp])
                    yr_rem[sp] = max(0, yr_rem[sp] - added)
                    achieved[sp] += added
                    if has_central and added:
                        achieved_central[sp] += int(avail.loc[best, f"{sp}_central"])
                avail = avail.drop(best)

    if has_central:
        print("Central region coverage:")
        for sp in species_list:
            total = achieved[sp]
            central = achieved_central[sp]
            pct = central / total * 100 if total > 0 else 0
            print(f"  {sp}: {central}/{total} central ({pct:.0f}%)")

    # ── Top-up: redistribute deficits across years (relax progressively) ────
    # Per-label target. When the caller supplies one (site_balanced_select
    # does, per site), use it verbatim; otherwise recover the scalar from the
    # allocation. even_allocate spreads the full target across years for any
    # label whose total availability >= target, so the largest allocation-sum
    # is the true target.
    if target_by_species is None:
        scalar_target = max(
            (sum(alloc[sp].values()) for sp in species_list), default=0)
        targets = {sp: scalar_target for sp in species_list}
    else:
        targets = {sp: int(target_by_species.get(sp, 0)) for sp in species_list}
    # True availability per label — used to decide genuine exhaustion (SHORT).
    sp_total = {sp: int(binary[sp].sum()) for sp in species_list}

    # Relaxation ladder. central=True keeps the central-region gate; the final
    # rung drops it. When the matrix has no central columns we only ever run
    # the year-relaxed rung (central gating is meaningless).
    if has_central:
        relax_rungs = [
            (1, "year", True),
            (2, "year + central", False),
        ]
    else:
        relax_rungs = [(1, "year", False)]

    def _short_species():
        return [sp for sp in species_list
                if achieved[sp] < targets[sp] and achieved[sp] < sp_total[sp]]

    for rung, label, keep_central in relax_rungs:
        # Repeat this rung until it can no longer make progress (each iteration
        # pulls at most one frame, so a label needing N more frames loops N
        # times here before either reaching target or exhausting frames).
        while True:
            short = _short_species()
            if not short:
                break

            # Candidate pool: every unused frame that carries a still-short
            # label (no year filter — that is the whole point of the relaxation).
            cand = binary[~binary.index.isin(used)]
            cand_mask = pd.Series(False, index=cand.index)
            for sp in short:
                cand_mask |= cand[sp] > 0
            cand = cand[cand_mask]
            if len(cand) == 0:
                break

            scores = pd.Series(0.0, index=cand.index)
            n_needed = len(short)
            for sp in short:
                scores += cand[sp]
            if n_needed > 1:
                n_sp_in_frame = pd.Series(0, index=cand.index)
                for sp in short:
                    n_sp_in_frame += cand[sp]
                scores += (n_sp_in_frame == 1).astype(float) * 0.5

            if keep_central and has_central:
                central_mask = pd.Series(False, index=cand.index)
                for sp in short:
                    central_mask |= cand[f"{sp}_central"] > 0
                # Prefer central frames but do not hard-zero non-central ones:
                # this rung still allows any frame, central just scores higher.
                scores += central_mask.astype(float) * 0.25
                # If ANY short label has a central candidate, gate to it so the
                # central rung actually pulls central frames first.
                if central_mask.any():
                    scores = scores.where(central_mask, 0)

            best = _pick_best(scores)
            if best is None:
                break

            # Status line: which short labels does this frame help, and by how
            # much are they still short.
            helps = [sp for sp in short if int(binary.loc[best, sp]) > 0]
            deficit_str = ", ".join(
                f"{sp} short by {targets[sp] - achieved[sp]}" for sp in helps
            )
            print(f"Top-up pass {rung} (relaxed: {label}): "
                  f"{deficit_str} -> pulling {best}")

            selected.append(best)
            used.add(best)
            for sp in species_list:
                added = int(binary.loc[best, sp])
                achieved[sp] += added
                if has_central and added:
                    achieved_central[sp] += int(binary.loc[best, f"{sp}_central"])

        # If nothing is short after this rung, no need to relax further.
        if not _short_species():
            break

    # ── Final per-label top-up summary ──────────────────────────────────────
    # The reason is computed, not assumed: a shortfall only reads as pool
    # exhaustion when every frame carrying the label was actually taken.
    print("Top-up summary:")
    for sp in species_list:
        reason = shortfall_reason(achieved[sp], targets[sp], sp_total[sp],
                                  alloc_total=sum(alloc[sp].values()))
        if reason == "ok":
            print(f"  {sp:6s}: OK ({achieved[sp]}/{targets[sp]})")
        elif reason == "pool exhausted":
            print(f"  {sp:6s}: SHORT(pool exhausted) "
                  f"({achieved[sp]}/{targets[sp]}; "
                  f"only {sp_total[sp]} frames exist)")
        else:
            print(f"  {sp:6s}: SHORT({reason}) "
                  f"({achieved[sp]}/{targets[sp]}; "
                  f"{sp_total[sp]} frames exist)")

    return selected, achieved


SHORTFALL_REASONS = ("ok", "cap-limited", "allocation-limited", "pool exhausted")


def shortfall_reason(achieved, target, pool_total, alloc_total=None):
    """Why a label finished where it did. One of SHORTFALL_REASONS.

    Every shortfall used to print "pool exhausted", which is only true when the
    selection actually took every frame carrying the label. The three real
    causes are now separated:

      ok                 - the target was met.
      pool exhausted     - achieved == the whole eligible pool for that label;
                           no frame was left to take.
      cap-limited        - the pool could have supplied the target but the
                           per-site cap held the allocation below it. When
                           pool_total >= target, an allocation below target can
                           only come from the cap: the water-fill runs until
                           either the target is met or every site sits at
                           min(avail, cap), and all-at-avail would mean the
                           allocation equals the pool.
      allocation-limited - the allocation covered the target but the per-site
                           and per-year spread could not place it.

    Args:
      achieved: frame-instances actually selected for the label.
      target: requested frame-instances for the label.
      pool_total: frames in the eligible pool carrying the label.
      alloc_total: summed site allocation for the label, when site balance is
        on. None on the global path, where no site allocation exists.
    """
    if achieved >= target:
        return "ok"
    if pool_total is not None and achieved >= pool_total:
        return "pool exhausted"
    if alloc_total is not None and alloc_total < target:
        return "cap-limited"
    return "allocation-limited"


def build_summary_lines(candidates, eligible, selected, reserve,
                        species_list, achieved, target, exclusions,
                        reasons=None):
    """Return lines for the eligibility-funnel block of selection_summary.txt.

    Pure helper, no I/O, testable in isolation. `reasons` is an optional
    {species: reason} map from shortfall_reason(); omitting it keeps the
    original wording so callers that do not compute reasons are unaffected.
    """
    L = ["=== Eligibility funnel ===",
         f"  candidates : {candidates:,}",
         f"  eligible   : {eligible:,}",
         f"  selected   : {selected:,}",
         f"  reserve    : {reserve:,}",
         "  excluded   : "
         f"image_missing={exclusions.get('image_missing', 0)}, "
         f"cpc_missing={exclusions.get('cpc_missing', 0)}, "
         f"pts_missing={exclusions.get('pts_missing', 0)}",
         "",
         "=== Per-species (achieved / target) ==="]
    for sp in species_list:
        a = achieved.get(sp, 0)
        if a >= target:
            flag = "[OK]"
        else:
            why = (reasons or {}).get(sp) or "pool exhausted"
            flag = f"[SHORT by {target - a}; {why}]"
        L.append(f"  {sp:6s}: {a}/{target}  {flag}")
    L.append("")
    return L


def build_selection_summary(species_list, achieved, target, per_site,
                            pool_totals, alloc_totals, cap_frac, n_selected):
    """Machine-readable twin of selection_summary.txt.

    Written as output/selection_summary.json beside the text summary. Step 4's
    reserve refill reads per_site to steer a replacement toward the site with
    the largest deficit, and the orchestrator UI reads per_species to show why
    a label finished short. Pure helper, no I/O.

    Keys:
      per_site      {site: selected frame count}
      per_species   {species: {target, achieved, reason}}
      cap_frac      the per-site share cap that was requested
      cap_ok        did the largest site actually stay at or under cap_frac
      n_selected    total frames selected
      shortfalls    [{species, target, achieved, reason}] for every label whose
                    reason is not "ok"
    """
    per_species = {}
    shortfalls = []
    for sp in species_list:
        a = int(achieved.get(sp, 0))
        reason = shortfall_reason(
            a, target,
            (pool_totals or {}).get(sp),
            (alloc_totals or {}).get(sp))
        per_species[sp] = {"target": int(target), "achieved": a,
                           "reason": reason}
        if reason != "ok":
            shortfalls.append({"species": sp, "target": int(target),
                               "achieved": a, "reason": reason})

    per_site = {str(s): int(c) for s, c in (per_site or {}).items()}
    total = int(n_selected)
    if total > 0 and per_site:
        cap_ok = bool(max(per_site.values()) / total <= cap_frac + 1e-9)
    else:
        cap_ok = True
    n_sites = len(per_site)
    cap_satisfiable = bool(
        cap_frac > 0 and n_sites >= math.ceil(1 / cap_frac)) if n_sites else True

    return {
        "per_site": per_site,
        "per_species": per_species,
        "cap_frac": float(cap_frac),
        "cap_ok": cap_ok,
        "cap_satisfiable": cap_satisfiable,
        "n_selected": total,
        "shortfalls": shortfalls,
    }


def build_reserve_rows(binary, selected, species_list, name_fn, clip_index):
    """Eligible-but-unselected frames, ranked, tagged by which target species
    they carry. The Step 4 OCR-failure refill draws from this pool.

    Rank order is under-represented sites first: a frame's rank rises with the
    share of the SELECTION its site already holds, and the fresh-random shuffle
    breaks ties within a site. Step 4 filters this list by species and takes
    the lowest rank it can use, so a rank order that ignored site sent
    replacements straight back to whichever site happened to shuffle to the
    top, typically the dominant one. Ordering by site share means that within
    any species bucket Step 4 sees the most under-represented sites first.
    """
    selected_set = set(selected)
    reserve_ids = [fid for fid in binary.index if fid not in selected_set]
    random.shuffle(reserve_ids)  # fresh-random order, no seed

    # Share of the selection each site already holds. A site absent from the
    # selection scores 0 and therefore sorts first.
    n_selected = len(selected_set)
    site_share = {}
    if n_selected and "site" in binary.columns and selected_set:
        counts = binary.loc[sorted(selected_set), "site"].value_counts()
        site_share = {s: c / n_selected for s, c in counts.items()}

    if "site" in binary.columns:
        # One vectorized pull instead of a .loc lookup per comparison: the
        # reserve pool runs to tens of thousands of rows on a real selection.
        site_of = binary["site"].to_dict()
        # list.sort is stable, so the shuffled order survives as the tie-break
        # inside each site.
        reserve_ids.sort(key=lambda fid: site_share.get(site_of.get(fid), 0.0))

    rows = []
    for rank, fid in enumerate(reserve_ids):
        r = binary.loc[fid]
        base = name_fn(str(r["date"]), r["site"], int(r["transect"]), int(r["frame"]))
        carries = [sp for sp in species_list if sp in binary.columns and int(r.get(sp, 0)) == 1]
        rows.append({
            "frame_id": fid, "basename": base, "date": str(r["date"]),
            "year": int(r["year_int"]), "site": r["site"],
            "transect": int(r["transect"]), "frame": int(r["frame"]),
            "species": ";".join(carries), "reserve_rank": rank,
            "route": "cpc" if int(r["year_int"]) < 2020 else "ocr_needed",
            "source_image": clip_index.get(base, ""),
            "pts_image": clip_index.get(base + "_pts", ""),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Balanced image selection")
    parser.add_argument("--all-points", default=None,
                        help="Path to all_points CSV (default: auto-detect recoded or raw)")
    parser.add_argument("--master-codes", default=None)
    parser.add_argument("--species", nargs="+", default=None,
                        help="Target label codes (e.g., --species OFRA PA OA). "
                             "Any label in master_codes; not limited to corals.")
    parser.add_argument("--target", type=int, default=None,
                        help="Frame-instances per label (default: from config)")
    parser.add_argument("--min-year", type=int, default=None,
                        help=f"Earliest year to consider (default: {config.MIN_YEAR})")
    parser.add_argument("--max-year", type=int, default=None,
                        help=f"Latest year to consider (default: {config.MAX_YEAR})")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="Do not emit route_missing.csv. (Image paths are still "
                             "resolved on disk and recorded in selected_frames.csv.)")
    parser.add_argument("--no-site-balance", action="store_true",
                        help="Disable site-balanced selection and fall back to the "
                             "original year-only allocation (even_allocate + "
                             "greedy_select). Site balance is ON by default.")
    parser.add_argument("--max-site-frac", type=float, default=0.25,
                        help="Max share of the selected frames any one site may "
                             "hold when site-balance is on (default: 0.25).")
    args = parser.parse_args()

    species_list = args.species or config.TARGET_SPECIES
    target = args.target or config.TARGET_INSTANCES_PER_SPECIES
    min_year = args.min_year if args.min_year is not None else config.MIN_YEAR
    max_year = args.max_year if args.max_year is not None else config.MAX_YEAR
    site_balance = not args.no_site_balance
    max_site_frac = args.max_site_frac

    # Find input file
    if args.all_points:
        ap_path = args.all_points
    else:
        ap_path = find_latest_recoded(config._RECODE_DIR, config.DEFAULT_ALL_POINTS)
    mc_path = args.master_codes or config.DEFAULT_MASTER_CODES

    print(f"Input:   {ap_path}")
    print(f"Codes:   {mc_path}")
    print(f"Species: {species_list}")
    print(f"Target:  {target} frame-instances each")
    print()

    # Load species remap (e.g. MFRA->OFRA, MA->OA, MFAV->OFAV)
    remap = _load_species_remap()
    if remap:
        print(f"Species remap: {remap}")

    # Load data — use cpc_all for pre-2020, all_points for 2020+
    ap = pd.read_csv(ap_path)
    mc = pd.read_csv(mc_path)
    print(f"Loaded {len(ap):,} rows from all_points, {len(mc)} codes")

    # For pre-2020: replace species data with cpc_all (correct label mapping)
    cpc_years = list(range(min_year, 2020))
    cpc_species = load_cpc_species(config.CPC_ALL_DIR, cpc_years, remap)
    if len(cpc_species) > 0:
        print(f"Loaded {len(cpc_species):,} CPC points from cpc_all (pre-2020, remapped)")
        # Remove pre-2020 rows from all_points, replace with cpc_all data
        ap = ap[ap["year"] >= 2020].copy()
        # Apply remap to post-2020 all_points too
        ap["species_code"] = ap["species_code"].replace(remap)
        ap = pd.concat([cpc_species, ap], ignore_index=True)
        print(f"Combined: {len(ap):,} rows (cpc_all pre-2020 + all_points 2020+)")
    else:
        # No cpc_all data, apply remap to all_points
        ap["species_code"] = ap["species_code"].replace(remap)

    # Year-bounds filter (bounds come from the UI; no category filter — any
    # label in master_codes is selectable, including non-coral phyla).
    ap = ap[(ap["year"] >= min_year) & (ap["year"] <= max_year)].copy()
    ap["frame_id"] = (
        ap["date"].astype(str) + "|" + ap["site"] + "|" +
        ap["transect"].astype(str) + "|" + ap["frame"].astype(str)
    )
    print(f"Year-filtered rows ({min_year}-{max_year}): {len(ap):,}")

    # `coral` retained as the variable name to minimize downstream diff, but
    # it now holds ALL categories, not just Coral.
    coral = ap.copy()

    # Build binary frame × species presence matrix
    target_rows = coral[coral["species_code"].isin(species_list)]
    frame_sp = target_rows.groupby(["frame_id", "species_code"]).size().unstack(fill_value=0)
    for sp in species_list:
        if sp not in frame_sp.columns:
            frame_sp[sp] = 0
    binary = (frame_sp[species_list] > 0).astype(int)

    # ── Central region flags ────────────────────────────────────────────────
    # For each frame+species, check if any target point is in the central
    # 25-75% region of the image. Image dims: 1920x1080 (pre-2017),
    # 3840x2160 (2017+). We use x,y from the data if available.
    if "x" in coral.columns and "y" in coral.columns:
        for sp in species_list:
            sp_pts = target_rows[target_rows["species_code"] == sp].copy()
            if sp_pts.empty or sp_pts["x"].isna().all():
                binary[f"{sp}_central"] = 0
                continue
            # Infer image dims per frame from year
            sp_pts = sp_pts.copy()
            sp_pts["img_w"] = np.where(sp_pts["year"] >= 2017, 3840, 1920)
            sp_pts["img_h"] = np.where(sp_pts["year"] >= 2017, 2160, 1080)
            sp_pts["in_central"] = (
                (sp_pts["x"] >= sp_pts["img_w"] * 0.25) &
                (sp_pts["x"] <= sp_pts["img_w"] * 0.75) &
                (sp_pts["y"] >= sp_pts["img_h"] * 0.25) &
                (sp_pts["y"] <= sp_pts["img_h"] * 0.75)
            )
            central_frames = sp_pts[sp_pts["in_central"]].groupby("frame_id").size()
            binary[f"{sp}_central"] = 0
            for fid in central_frames.index:
                if fid in binary.index:
                    binary.loc[fid, f"{sp}_central"] = 1
        n_central = {sp: int(binary[f"{sp}_central"].sum()) for sp in species_list}
        print(f"Frames with central-region target points: {n_central}")

    # Add metadata
    meta = ap.drop_duplicates("frame_id").set_index("frame_id")[
        ["date", "year", "site", "transect", "frame"]
    ]
    binary = binary.join(meta, how="left")
    binary["year_int"] = binary["year"].astype(int)

    years = sorted(binary["year_int"].unique())
    years = [y for y in years if config.MIN_YEAR <= y <= config.MAX_YEAR]

    print(f"Frames with any target species (before filtering): {len(binary):,}")

    # ── Eligibility gate: keep only frames whose IMAGE and POINT SOURCE both
    # resolve. Pre-2020 needs CPC coords; 2020+ needs a _pts image. This makes
    # downstream image_missing / cpc_missing impossible for a selected frame.
    clip_index = build_clip_index(config.CLIP_DIR)
    print(f"Indexed {len(clip_index):,} clip images under {config.CLIP_DIR}")
    cpc_years = [y for y in years if y < 2020]
    cpc_basenames = load_cpc_basenames(config.CPC_ALL_DIR, cpc_years)
    print(f"  CPC has coords for {len(cpc_basenames):,} frames")
    n_before = len(binary)
    eligible_index, exclusions = frame_eligibility(
        binary, clip_index, cpc_basenames, build_image_filename)
    binary = binary.loc[eligible_index]
    print(f"  Eligibility gate: {n_before:,} candidates -> {len(binary):,} eligible "
          f"(excluded image_missing={exclusions['image_missing']}, "
          f"cpc_missing={exclusions['cpc_missing']}, pts_missing={exclusions['pts_missing']})")

    # Recompute years after filtering
    years = sorted(binary["year_int"].unique())
    years = [y for y in years if config.MIN_YEAR <= y <= config.MAX_YEAR]

    print(f"Eligible frames after filtering: {len(binary):,}")
    print()

    # Availability
    print("Available frame-instances per species:")
    for sp in species_list:
        total = int(binary[sp].sum())
        status = "OK" if total >= target else f"WARNING: only {total} available"
        print(f"  {sp:6s}: {total:>6,} frames  [{status}]")
    print()

    # Allocate and select
    use_site_balance = site_balance and "site" in binary.columns
    alloc_totals = {}
    if use_site_balance:
        print(f"Site-balanced selection: ON (max_site_frac={max_site_frac})")
        # Same pure allocator site_balanced_select drives its selection from,
        # so the allocation reported here is the allocation actually used.
        site_targets = compute_site_targets(
            binary, species_list, target, cap_frac=max_site_frac)
        alloc_totals = {sp: sum(site_targets[sp].values())
                        for sp in species_list}
        selected, achieved = site_balanced_select(
            binary, species_list, years, target, cap_frac=max_site_frac)
    else:
        print("Site-balanced selection: OFF (--no-site-balance)")
        alloc = even_allocate(binary, species_list, years, target)
        selected, achieved = greedy_select(binary, species_list, alloc, years)

    # Per-label pool sizes drive the honest shortfall reason: a label is only
    # reported as exhausted when the selection actually took every frame that
    # carries it. Everything else is cap-limited or allocation-limited.
    pool_totals = {sp: int(binary[sp].sum()) for sp in species_list}
    reasons = {
        sp: shortfall_reason(achieved[sp], target, pool_totals[sp],
                             alloc_totals.get(sp))
        for sp in species_list
    }

    sel = binary.loc[selected].copy()
    n = len(sel)
    print(f"Selected {n:,} frames")
    for sp in species_list:
        if reasons[sp] == "ok":
            status = "OK"
        else:
            status = f"SHORT by {target - achieved[sp]} ({reasons[sp]})"
        print(f"  {sp:6s}: {achieved[sp]:>5,} frame-instances  [{status}]")
    print()

    # ── Build output with image paths and routing ────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # clip_index was built above during the eligibility gate; reuse it here.
    rows_out = []
    for frame_id in selected:
        r = sel.loc[frame_id]
        date_str = str(r["date"])
        site = r["site"]
        transect = int(r["transect"])
        frame = int(r["frame"])
        year_int = int(r["year_int"])
        basename = build_image_filename(date_str, site, transect, frame)

        # Which target species are in this frame?
        spp_present = [sp for sp in species_list if int(r[sp]) == 1]

        row = {
            "frame_id": frame_id,
            "basename": basename,
            "date": date_str,
            "year": year_int,
            "site": site,
            "transect": transect,
            "frame": frame,
            "species_present": ";".join(spp_present),
            "n_target_species": len(spp_present),
        }

        # Routing: simple split — pre-2020 already have CPC coords (we
        # filtered out those without), 2020+ all need OCR processing.
        row["route"] = "cpc" if year_int < 2020 else "ocr_needed"
        row["coords_source"] = os.path.join(
            config.CPC_ALL_DIR, str(year_int), "ids", "point_coords.csv"
        ) if year_int < 2020 else ""

        # Always verify that each selected frame maps to a real image on disk.
        # Downstream steps (4/5) need the resolved paths. Resolution is a
        # season-agnostic whole-tree lookup (clip_index built once above).
        img_path, ext = find_source_image(basename, clip_index)
        pts_path = find_pts_image(basename, clip_index)
        row["source_image"] = img_path or ""
        row["pts_image"] = pts_path or ""
        row["image_found"] = img_path is not None
        row["clip_dir_searched"] = config.CLIP_DIR
        row["missing_reason"] = missing_reason(img_path is not None, basename, config.CLIP_DIR)
        row["point_source"] = "cpc" if year_int < 2020 else "pts"
        row["point_source_path"] = (
            os.path.join(config.CPC_ALL_DIR, str(year_int), "ids", "point_coords.csv")
            if year_int < 2020 else (pts_path or "")
        )

        rows_out.append(row)

    df_out = pd.DataFrame(rows_out)

    # ── Save outputs ─────────────────────────────────────────────────────────

    # Master list
    master_path = os.path.join(config.OUTPUT_DIR, "selected_frames.csv")
    df_out.to_csv(master_path, index=False)
    print(f"Saved: {master_path} ({len(df_out)} rows)")

    reserve_rows = build_reserve_rows(binary, selected, species_list,
                                      build_image_filename, clip_index)
    if reserve_rows:
        reserve_path = os.path.join(config.OUTPUT_DIR, "reserve_frames.csv")
        pd.DataFrame(reserve_rows).to_csv(reserve_path, index=False)
        print(f"Saved: {reserve_path} ({len(reserve_rows)} reserve frames)")

    # Route splits
    for route_name, filename in [
        ("cpc", "route_cpc.csv"),
        ("ocr_needed", "route_ocr_needed.csv"),
    ]:
        subset = df_out[df_out["route"] == route_name]
        if len(subset) > 0:
            p = os.path.join(config.OUTPUT_DIR, filename)
            subset.to_csv(p, index=False)
            print(f"Saved: {p} ({len(subset)} rows)")

    if not args.skip_image_check:
        missing = df_out[df_out["image_found"] == False]  # noqa: E712
        if len(missing) > 0:
            p = os.path.join(config.OUTPUT_DIR, "route_missing.csv")
            missing.to_csv(p, index=False)
            print(f"Saved: {p} ({len(missing)} rows — images not found on disk)")

    # ── Summary ──────────────────────────────────────────────────────────────
    reserve_count = len(reserve_rows)
    summary_lines = [
        f"TCRMPcvr_chooseImages - Selection Summary",
        f"Generated: {ast_timestamp()}",
        f"",
        f"Input:   {os.path.basename(ap_path)}",
        f"Species: {', '.join(species_list)}",
        f"Target:  {target} frame-instances per species",
        f"",
    ]
    summary_lines.extend(build_summary_lines(
        candidates=n_before,
        eligible=len(binary),
        selected=len(df_out),
        reserve=reserve_count,
        species_list=species_list,
        achieved=achieved,
        target=target,
        exclusions=exclusions,
        reasons=reasons,
    ))

    # Year distribution
    yr_dist = sel["year_int"].value_counts().sort_index()
    summary_lines.append("=== Frames per Year ===")
    for y, c in yr_dist.items():
        summary_lines.append(f"  {y}: {c:>5}")
    summary_lines.append("")

    # Site distribution
    site_dist = df_out["site"].value_counts().sort_values(ascending=False)
    n_sel_total = len(df_out)
    summary_lines.append(f"=== Sites ({len(site_dist)}) ===")
    for s, c in site_dist.items():
        share = c / n_sel_total if n_sel_total else 0
        summary_lines.append(f"  {s:5s}: {c:>5}  ({share*100:5.1f}%)")

    if n_sel_total and len(site_dist) > 0:
        top_site = site_dist.index[0]
        top_share = site_dist.iloc[0] / n_sel_total
        # Cap is mathematically unsatisfiable when there are too few sites
        # for max_site_frac to allow every site its inclusion floor of 1
        # (e.g. 2 sites can never both be <=25%). Only WARN in that case;
        # otherwise assert the cap actually held.
        cap_unsatisfiable = len(site_dist) < math.ceil(1 / max_site_frac) if max_site_frac > 0 else False
        if site_balance:
            if top_share <= max_site_frac + 1e-9:
                summary_lines.append(
                    f"  [OK] max site share {top_share*100:.1f}% ({top_site}) "
                    f"<= cap {max_site_frac*100:.0f}%")
            elif cap_unsatisfiable:
                summary_lines.append(
                    f"  [WARN] max site share {top_share*100:.1f}% ({top_site}) "
                    f"exceeds cap {max_site_frac*100:.0f}%, but only {len(site_dist)} "
                    f"site(s) carry this label - the cap is mathematically "
                    f"unsatisfiable with this few sites (each needs its inclusion "
                    f"floor of >=1); the tightest even split is used instead.")
            else:
                summary_lines.append(
                    f"  [WARN] max site share {top_share*100:.1f}% ({top_site}) "
                    f"exceeds cap {max_site_frac*100:.0f}% unexpectedly")
        else:
            summary_lines.append(
                f"  (site-balance OFF: max site share {top_share*100:.1f}% "
                f"({top_site}), no cap enforced)")
    summary_lines.append("")

    # Transect distribution
    trans_dist = df_out["transect"].value_counts().sort_index()
    summary_lines.append("=== Transects ===")
    for t, c in trans_dist.items():
        summary_lines.append(f"  T{int(t)}: {c:>5}")
    summary_lines.append("")

    # Species-per-frame distribution
    sp_count_dist = df_out["n_target_species"].value_counts().sort_index()
    summary_lines.append("=== Target species per frame ===")
    for nsp, c in sp_count_dist.items():
        label = f"{int(nsp)} species"
        summary_lines.append(f"  {label:12s}: {c:>5}")
    summary_lines.append("")

    # Central region coverage
    has_central = all(f"{sp}_central" in sel.columns for sp in species_list)
    if has_central:
        summary_lines.append("=== Central region (25-75% of image) ===")
        for sp in species_list:
            total = int(sel[sp].sum())
            central = int(sel[f"{sp}_central"].sum())
            pct = central / total * 100 if total > 0 else 0
            summary_lines.append(f"  {sp:6s}: {central:>3}/{total} frames with central point ({pct:.0f}%)")
        summary_lines.append("")

    # Routing summary
    route_dist = df_out["route"].value_counts()
    summary_lines.append("=== Routing ===")
    for r, c in route_dist.items():
        summary_lines.append(f"  {r:15s}: {c:>5}")
    summary_lines.append("")

    # Image-resolution reconciliation: found vs missing on disk. route_cpc.csv
    # and route_ocr_needed.csv split by ROUTE and include not-yet-found frames;
    # route_missing.csv is the image_found==False subset (with missing_reason).
    n_found = int(df_out["image_found"].sum())
    n_missing = len(df_out) - n_found
    summary_lines.append("=== Image resolution ===")
    summary_lines.append(f"  clip dir searched : {config.CLIP_DIR}")
    summary_lines.append(f"  images found      : {n_found}/{len(df_out)}")
    summary_lines.append(f"  images missing    : {n_missing}  (see route_missing.csv for per-frame reasons)")
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    summary_path = os.path.join(config.OUTPUT_DIR, "selection_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved: {summary_path}")

    # Machine-readable twin of the Sites and per-species blocks above. Step 4's
    # reserve refill reads per_site to steer a replacement toward the site with
    # the largest deficit, and the orchestrator reads per_species to show why a
    # label finished short instead of guessing "pool exhausted".
    summary_doc = build_selection_summary(
        species_list=species_list,
        achieved=achieved,
        target=target,
        per_site=site_dist.to_dict(),
        pool_totals=pool_totals,
        alloc_totals=alloc_totals,
        cap_frac=max_site_frac if use_site_balance else 1.0,
        n_selected=n_sel_total,
    )
    summary_json_path = os.path.join(config.OUTPUT_DIR, "selection_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_doc, f, indent=2)
    print(f"Saved: {summary_json_path}")

    # Save config snapshot
    cfg_path = os.path.join(config.OUTPUT_DIR, "config_snapshot.txt")
    with open(cfg_path, "w") as f:
        f.write(f"timestamp: {ast_timestamp()}\n")
        f.write(f"species: {species_list}\n")
        f.write(f"target: {target}\n")
        f.write(f"min_year: {config.MIN_YEAR}\n")
        f.write(f"max_year: {config.MAX_YEAR}\n")
        f.write(f"clip_dir: {config.CLIP_DIR}\n")
        f.write(f"all_points: {ap_path}\n")
        f.write(f"master_codes: {mc_path}\n")
        f.write("selection: fresh random, no seed\n")
        f.write(f"eligible_pool: {len(binary)}\n")

    print(f"\nDone. Run 'python src/plot_diagnostics.py' for distribution plots.")


if __name__ == "__main__":
    main()
