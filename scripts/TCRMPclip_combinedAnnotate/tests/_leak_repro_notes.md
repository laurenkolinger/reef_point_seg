# Leak Repro Notes - Step 4 Species Filter / Stale Prompts

**Date:** 2026-06-30
**Branch:** taglab-module
**Task:** Diagnosis only. No production code changed.

---

## Reproduction Command

Run from `github_repo/`:

```bash
env/bin/python - <<'PY'
import os, sys, csv, tempfile, importlib.util
PP = "scripts/TCRMPclip_placePoints/src"
sys.path.insert(0, PP); os.chdir(PP)
tmp = tempfile.mkdtemp(prefix="leak_")
sel = os.path.join(tmp, "selected_frames.csv")
with open(sel, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["basename","year","route","date","site","transect","frame","species_present","source_image","pts_image"])
    w.writerow(["A",2022,"ocr_needed","2022-06-01","FLC",1,1,"PA","",""])      # target
    w.writerow(["B",2022,"ocr_needed","2022-06-01","FLC",1,2,"OFRA","",""])    # target
    w.writerow(["C",2022,"ocr_needed","2022-06-01","FLC",1,3,"XYZ","",""])     # NON-target
spec = importlib.util.spec_from_file_location("pp_leak", os.path.join(os.getcwd(), "app.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cl = m.app.test_client()
for filt in ([], ["PA","OFRA"]):
    r = cl.post("/api/configure", json={"selected_frames": sel,
                "export_dir": os.path.join(tmp,"exp_"+("none" if not filt else "filt")),
                "species_filter": filt, "review_batch_size": 999999})
    print("filter", filt, "->", r.status_code, r.get_data(as_text=True)[:500])
PY
```

## Observed Counts

| species_filter | total_frames kept | species_filter applied (from response) |
|---|---|---|
| `[]` (hardcoded in orchestrator) | 1 | `['PA', 'OFAV']` (auto-filled by TARGET_SPECIES_ONLY=1 fallback) |
| `["PA","OFRA"]` (correct explicit) | 2 | `['PA', 'OFRA']` |

The `[]` run kept **1 frame** (only PA matched OFAV). The explicit `["PA","OFRA"]` run kept **2 frames** (PA + OFRA).

---

## RC#1 - Hardcoded Empty Filter (CONFIRMED)

**File:** `scripts/pipeline_orchestrator/app.py`
**Line:** 938

```python
route_thread = threading.Thread(
    target=_route_drive,
    args=(pp_port, selected_frames, routed_dir, [], current_project),
    daemon=True,
)
```

The `[]` is the `species_filter` argument. `_route_drive` forwards it to placePoints `/api/configure` as `"species_filter": species_filter or []` (line 783).

**Key nuance:** placePoints `TCRMPclip_placePoints/src/app.py` line 877-879 has a fallback:

```python
if getattr(cfg, 'TARGET_SPECIES_ONLY', 0) and not species_filter:
    species_filter = list(ALL_TARGET_SPECIES)
```

`TARGET_SPECIES_ONLY = 1` in `scripts/TCRMPclip_placePoints/src/config.py` (line 19). So the empty filter is NOT passed through blindly -- it gets replaced by `ALL_TARGET_SPECIES` (loaded from env `TCRMP_TARGET_SPECIES` or the project config). This means the production leak occurs when the user's Step 3 target species selection diverges from the `ALL_TARGET_SPECIES` value that placePoints loads at startup (e.g. user filters to `['PA','OFRA']` in the UI panel, but `ALL_TARGET_SPECIES` contains `['PA','OFAV',...]`). The orchestrator's UI-configured species panel filter is silently dropped and the hardcoded default applies instead.

**Recommendation:** Pass the user's actual species filter (from Step 3 config / step4test_cfg) instead of `[]` at line 938.

---

## RC#2 - Stale routed_input Reuse (CONFIRMED REAL LEAK VECTOR)

**File:** `scripts/pipeline_orchestrator/app.py`
**Function:** `_run_step4test` (line 871) + `_routed_prompts_exist` (line 745)

Key code path (lines 893-897):

```python
# Reuse: prior routing already produced prompts -> open instantly.
if not force_reroute and _routed_prompts_exist(routed_dir):
    _route_set(phase="route_ready", ...)
    return _launch_step4test_ui()
```

`_routed_prompts_exist` (line 745-748) returns True if any `routed_input/<year>/ids/sam_click_prompts.json` file exists.

**Why it is a real leak vector:**
1. Only caller of `_run_step4test` is `/api/step/step4test/run` (line 2243-2254), which always passes `force_reroute=False`.
2. There is NO `/reroute` endpoint that calls `_run_step4test(force_reroute=True)`. The comment at line 67 mentions `run|reroute` but there is only a `/run` route registered.
3. The only path that clears `routed_input/` is `/reset` (line 2334), which deletes the ENTIRE `step4test_combinedAnnotate` dir -- a destructive operation that also loses segmentation work.
4. If a user runs Step 4 routing, then changes Step 3 selection (different frames or species), then clicks "Open Step 4" again without hitting Reset, the stale `sam_click_prompts.json` from the prior run is reused without any validation that it matches the current Step 3 selection.

**Evidence of no clearing on fresh run:** `_run_step4test` creates `routed_dir` with `os.makedirs(routed_dir, exist_ok=True)` (line 886) and immediately checks `_routed_prompts_exist` without clearing. No `shutil.rmtree` or equivalent runs on a non-forced launch.

**Recommendation:** Either (a) add a `/reroute` endpoint that calls `_run_step4test(force_reroute=True)` and expose it as a UI "Re-route (force)" button, or (b) invalidate the stale prompt cache when the Step 3 `selected_frames` mtime is newer than the `sam_click_prompts.json` mtime.

---

## Summary

- **RC#1:** Real bug. The orchestrator drops the user's species filter; placePoints falls back to its own hardcoded target species instead of what the user configured. Non-target-for-the-user species can route to SAM3.
- **RC#2:** Real leak vector. Stale `routed_input/` is reused on every non-reset launch with no staleness check. A prior run's (possibly wrong-species) prompts will be reused if the user doesn't manually reset.
