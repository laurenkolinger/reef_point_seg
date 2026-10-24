#!/usr/bin/env bash
# Dedicated restarter for the Reef Point Seg orchestrator ONLY.
#
# Why this exists: `vicarius restart reef_point_seg` routes through
# run_pipeline.sh, whose lsof/pgrep stale-sweep is slow and has hung on this
# box. This script hard-kills the orchestrator + its sub-app ports, starts the
# Flask app directly via run_pipeline.py, then BLOCKS until the server is
# actually serving the current template (verified by a live marker) before it
# reports success. It never touches the VICARIUS UI on :5077.
#
# Usage:  ./restart_reef_point_seg.sh [PORT]   (default PORT=5050)
set -u

PORT="${1:-5050}"
HERE="$(cd "$(dirname "$0")" && pwd)"              # .../github_repo/scripts
REPO_ROOT="$(cd "${HERE}/.." && pwd)"             # .../github_repo
PY="${REPO_ROOT}/env/bin/python"
[ -x "$PY" ] || PY="python3"
LOG="/tmp/reef_point_seg_orch_${PORT}.log"

# Ports owned by this module (orchestrator + its sub-apps). NEVER includes 5077.
# NEVER includes 5096 either: 5096 is the fixed port of the standalone
# "Annotate Images" annotator (runner key "annotate", launched from GET
# /annotate, session dirs under inprocess/_standalone/<session>/). It sits
# outside the 5080-5089 combined-annotator sweep range on purpose, so a
# standalone session in progress survives this restart untouched.
MODULE_PORTS=("$PORT" 5055 5065 5066 5070 5075 5080)

echo "==> Restarting Reef Point Seg orchestrator on port ${PORT}"

# ── 0. Save first (2026-09-27 safety net, D1) ─────────────────────────────────
# Flush any live combined annotator or Edit Masks session before anything
# below can kill it: a best-effort POST to /api/flush, never a stop. Only a
# listener whose /proc/<pid>/cmdline names one of the two annotator entry
# scripts is flushed (the same cmdline check _kill_all_annotators uses in
# pipeline_orchestrator/app.py), so an unrelated service sharing this port
# range is never poked (benthic_species_viewer's serve_edit.py on 5081 and
# vicariusLLM on 5085 have both been seen live here). A standalone "Annotate
# Images" session (port 5096) is included in the candidate list too; this
# script never kills it (see MODULE_PORTS below), so a flush there is only
# ever a save, never a disruption. This script does not set -e, but a failed
# curl is swallowed and printed either way so a future -e stays safe.
ANNOTATOR_FLUSH_PORTS=(5080 5081 5082 5083 5084 5085 5086 5087 5088 5089 5096)
ANNOTATOR_ENTRY_PATTERN='TCRMPclip_combinedAnnotate/src/app.py|TCRMPclip_editMasks/src/app.py'

# POSTs /api/flush and prints one status line, mirroring _flush_annotator's
# reading of the JSON body in pipeline_orchestrator/app.py: a 2xx alone is
# not a clean save. /api/flush (TCRMPclip_combinedAnnotate's force_flush,
# and Edit Masks' own copy) answers 200 with {ok:true, merged:true,
# taken_from_disk:N} for a save that pulled in frame(s) changed only on
# disk, or {ok:true, conflict:true, conflict_file:<name>, both_sides:N} for
# one that also hit a frame edited both here and on disk; an older
# annotator process can still answer the legacy {status:"merged", ...}
# shape, accepted here too. The conflict file and the both_sides/
# taken_from_disk count are named in the printed line either way. python3
# (no new dependency; already required by this repo) parses the body so
# this never depends on grep against arbitrary JSON formatting. Never fails
# the caller: a curl or parse failure just prints "did not confirm".
_flush_and_report() {
    local _fp="$1" _pid="$2"
    curl -s -m 5 -w '\n%{http_code}' -X POST "http://127.0.0.1:${_fp}/api/flush" 2>/dev/null \
        | python3 -c '
import json, sys
port, pid = sys.argv[1], sys.argv[2]
body_text, _, code_text = sys.stdin.read().rpartition("\n")
try:
    code = int(code_text)
except ValueError:
    code = 0
try:
    body = json.loads(body_text) if body_text.strip() else {}
except Exception:
    body = None
ok, note = True, ""
if isinstance(body, dict):
    if body.get("conflict"):
        both_sides = body.get("both_sides")
        detail = f", {both_sides} frame(s) edited on both sides" if both_sides else ""
        note = "conflict: " + str(body.get("conflict_file") or "<unnamed file>") + detail
        if not body.get("ok", False):
            ok = False
    elif body.get("merged") or body.get("status") == "merged":
        taken_from_disk = body.get("taken_from_disk")
        note = "merged" + (f", taken_from_disk={taken_from_disk}" if taken_from_disk is not None else "")
    elif "ok" in body and not body["ok"]:
        ok, note = False, "ok=false"
if 200 <= code < 300 and ok:
    print(f"    flushed port {port} (pid {pid})" + (f": {note}" if note else ""))
else:
    reason = note if note else f"http {code}"
    print(f"    flush on port {port} (pid {pid}) did not confirm ({reason}); continuing")
' "$_fp" "$_pid" \
        || echo "    flush on port ${_fp} (pid ${_pid}) did not confirm (no response); continuing"
}

echo "  flushing any live annotator / Edit Masks session first..."
for fp in "${ANNOTATOR_FLUSH_PORTS[@]}"; do
    fp_ss_line="$(ss -ltnp "sport = :${fp}" 2>/dev/null | grep LISTEN || true)"
    [ -z "$fp_ss_line" ] && continue
    fp_pid="$(printf '%s' "$fp_ss_line" | sed -nE 's/.*pid=([0-9]+).*/\1/p' | head -n1)"
    [ -z "$fp_pid" ] && continue
    fp_cmdline="$(tr '\0' ' ' < "/proc/${fp_pid}/cmdline" 2>/dev/null || true)"
    if ! printf '%s' "$fp_cmdline" | grep -qE "$ANNOTATOR_ENTRY_PATTERN"; then
        continue  # not an annotator/editMasks process, or an unreadable cmdline; skip
    fi
    _flush_and_report "$fp" "$fp_pid"
done

# ── 1. Hard-kill the orchestrator + sub-app processes ─────────────────────────
# NOTE: deliberately NO lsof here. lsof hangs on this box (stale NFS mounts
# under /mnt), which is exactly why `vicarius restart` / run_pipeline.sh stall.
# We use pkill (by process pattern) + fuser (by port), both of which are fast.
echo "  killing orchestrator processes (run_pipeline.py / pipeline_orchestrator)..."
pkill -9 -f "run_pipeline.py" 2>/dev/null || true
pkill -9 -f "pipeline_orchestrator/app.py" 2>/dev/null || true
for p in "${MODULE_PORTS[@]}"; do
    if fuser -k -9 "${p}/tcp" >/dev/null 2>&1; then
        echo "    freed port $p"
    fi
done
sleep 1

# ── 2. Confirm the orchestrator port is free (ss, not lsof) ──────────────────
if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    echo "  ERROR: port ${PORT} still bound after kill; aborting." >&2
    exit 1
fi

# ── 3. Start the orchestrator directly (detached) ─────────────────────────────
echo "  starting: ${PY} ${HERE}/run_pipeline.py --port ${PORT}"
# setsid -> own session, so the server survives this script's caller tearing
# down its process group (it must outlive the restart command).
if command -v setsid >/dev/null 2>&1; then
    setsid "$PY" "${HERE}/run_pipeline.py" --port "$PORT" >"$LOG" 2>&1 < /dev/null &
else
    nohup "$PY" "${HERE}/run_pipeline.py" --port "$PORT" >"$LOG" 2>&1 < /dev/null &
fi
NEW_PID=$!
echo "  launched pid ${NEW_PID}; log: ${LOG}"

# ── 4. Block until it actually serves the CURRENT page ────────────────────────
echo -n "  waiting for http://127.0.0.1:${PORT}/ to serve"
DEADLINE=$(( $(date +%s) + 40 ))
SERVED=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ! kill -0 "$NEW_PID" 2>/dev/null; then
        echo ""
        echo "  ERROR: orchestrator process ${NEW_PID} died on startup. Log tail:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    body="$(curl -s -m 3 "http://127.0.0.1:${PORT}/" 2>/dev/null || true)"
    # Alive when serving either the SPA or the module-lock page (vic-lock-page,
    # served while vicarius/_METADATA/module_locks.json marks this module locked).
    if printf '%s' "$body" | grep -q -e 's3-label-chips' -e 'vic-lock-page'; then
        SERVED="yes"; echo " — up."; break
    fi
    echo -n "."
    sleep 1
done

if [ -z "$SERVED" ]; then
    echo ""
    echo "  ERROR: server did not serve the current page within 40s. Log tail:" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

# ── 4b. Module lock: the SPA markers are deliberately absent while locked ─────
if printf '%s' "$body" | grep -q 'vic-lock-page'; then
    echo "LOCKED: module lock active, HTML self-test skipped"
    echo "==> Reef Point Seg orchestrator restarted on http://localhost:${PORT}/ (module locked)"
    exit 0
fi

# ── 5. Self-test: assert the expected markers are live ────────────────────────
# Note: the page HTML ($body) carries static markup; the footer Reset/Save&Close
# buttons and the picker grouping are injected by orchestrator.js at runtime, so
# those are checked against the served JS, not the HTML.
js="$(curl -s -m 5 "http://127.0.0.1:${PORT}/static/orchestrator.js?v=20261017-cardrows" 2>/dev/null || true)"
echo "  self-test (live markers on :${PORT}):"
fail=0
assert_html_present() { printf '%s' "$body" | grep -q "$1" && echo "    OK  present: $2" || { echo "    FAIL missing: $2"; fail=1; }; }
assert_html_absent()  { printf '%s' "$body" | grep -q "$1" && { echo "    FAIL still there: $2"; fail=1; } || echo "    OK  gone: $2"; }
assert_js_present()   { printf '%s' "$js"   | grep -q "$1" && echo "    OK  present: $2" || { echo "    FAIL missing: $2"; fail=1; }; }
assert_html_present 's3-label-chips'        'Step 3 selection chips container'
assert_html_present 'id="confirm-overlay"'  'reset-confirm modal markup'
assert_html_absent  'Do this:'              'old "Do this:" step text (rewritten in cdb0af5)'
assert_html_present '20261017-cardrows' 'fresh orchestrator.js cache-bust'
assert_html_absent  's3-load-labels'        'old "Load labels from Step 2" button'
assert_html_absent  'What you do'           'old "What you do" phrasing'
assert_html_absent  'data-step="s4review"'  'old SAM3-Review stub tile'
assert_html_absent  'panel-s4review'        'old SAM3-Review stub panel'
assert_html_absent  'Expert Review I/O'     'Expert Review I/O tile (removed in cdb0af5)'
assert_html_present 'data-step="step4test"' 'combined-annotator tile present (route key kept)'
assert_html_present 'panel-step4test'       'combined-annotator panel present'
assert_html_present 'Place + Segment'       'combined-annotator promoted to Step 4 label'
assert_html_present 'data-step="step4loop"' 'Refine model-assist loop tile present'
assert_html_present 'Refine (model loop)'   'Refine loop tile label present'
assert_html_present 'data-step="editmasks"' 'Edit Masks tile present'
assert_html_present 'Edit Masks'            'Edit Masks tile label present'
assert_html_absent  'experimental'          'experimental copy removed (4.test promoted)'
assert_html_absent  'data-step="4"'         'legacy Place Points tile archived from sidebar'
assert_html_absent  'data-step="5"'         'legacy Segment tile archived from sidebar'
assert_js_present   'openStep4test'         'combined-annotator JS launcher'
assert_js_present   'route_status'          '4.test routing-progress poll'
assert_html_absent  'Re-route from Step 3'  'reroute button removed (routing folded into Step 3)'
assert_js_present   'Reset Step'            'Reset button (rendered next to Save and Close)'
assert_js_present   'resetBtn}${saveCloseBtn' 'Reset positioned next to Save and Close'
assert_js_present   'toggleAllStep3Groups'  'grouped/collapsible label picker'
assert_js_present   'LEGACY_S3_DEFAULT'     'zero-selected default on first view'
assert_html_present 's4test-target-only'    '4.test combined Step-4+5 settings block'
# doResetStep builds /api/step/${step}/reset as a template literal, so the live
# JS source carries the footer wiring openResetConfirm("step4test") (the dynamic
# URL string never appears verbatim). Assert on the footer wiring instead.
assert_js_present   'openResetConfirm("step4test")' '4.test Reset Step button wired'
assert_js_present   'name-preview'          'auto project-name preview'
assert_js_present   'function renderProjectName' 'topbar project-name renderer'
assert_js_present   'include_classes_source'   'Step 6 class picker follows the training source'
assert_html_present 'id="s6-source"'       'Step 6 training-source select (composite exports)'
assert_js_present   'function refreshStep6Sources' 'Step 6 training-source refresh from /api/composites'
assert_js_present   'function paintStep4testBoot' 'popup boot painter (single loader)'
assert_js_present   'boot-retry'                'popup retry link'
ann="$(curl -s -m 5 "http://127.0.0.1:${PORT}/annotate" 2>/dev/null || true)"
printf '%s' "$ann" | grep -q 'Annotate Images' && echo "    OK  present: /annotate standalone launch page" || { echo "    FAIL missing: /annotate page"; fail=1; }

# Native Label Coverage Matrix blueprint — prove its full page mounts under
# /matrix and serves its own markup (sentinel: the matrix page <h1>/<title>).
mtx="$(curl -s -m 5 "http://127.0.0.1:${PORT}/matrix/" 2>/dev/null || true)"
assert_url_contains() { printf '%s' "$mtx" | grep -q "$1" && echo "    OK  present: $2" || { echo "    FAIL missing: $2"; fail=1; }; }
assert_url_contains 'Coverage Matrix'       'Label Coverage Matrix page mounted at /matrix'
assert_url_contains 'id="mx-grid"'          'matrix grid table served from blueprint'

# Native Label Manager blueprint — prove its full page mounts under /labels
# and serves its own markup (sentinel: the labels page <h1> + title id).
lbl="$(curl -s -m 5 "http://127.0.0.1:${PORT}/labels/" 2>/dev/null || true)"
assert_lbl_contains() { printf '%s' "$lbl" | grep -q "$1" && echo "    OK  present: $2" || { echo "    FAIL missing: $2"; fail=1; }; }
assert_lbl_contains 'Label Manager'         'Label Manager page mounted at /labels'
assert_lbl_contains 'id="lb-title"'         'labels page served from blueprint'

# Train + Infer (/models) was retired on 2026-09-05 (ADR 0002: one training
# surface, the VEGA main window). The mount must be gone: a 404 proves the
# running process is the new code, a 200 means a stale process is serving.
mdl_status="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:${PORT}/models/" 2>/dev/null || true)"
if [ "$mdl_status" = "404" ]; then
    echo "    OK  retired: /models answers 404 (Train + Infer removed)"
else
    echo "    FAIL stale: /models answered ${mdl_status:-nothing}, expected 404"; fail=1
fi

# Bounding Boxes blueprint stays mounted at /boxes (staging and conversion;
# training is handed to the main window).
bx="$(curl -s -m 5 "http://127.0.0.1:${PORT}/boxes/" 2>/dev/null || true)"
assert_bx_contains() { printf '%s' "$bx" | grep -q "$1" && echo "    OK  present: $2" || { echo "    FAIL missing: $2"; fail=1; }; }
assert_bx_contains 'Bounding Boxes'         'Bounding Boxes page mounted at /boxes'
assert_bx_contains 'Train in VEGA'          'Bounding Boxes hands training to VEGA'

if [ "$fail" -ne 0 ]; then
    echo "==> Restart completed but self-test FAILED (stale assets?). See above." >&2
    exit 2
fi
echo "==> Reef Point Seg orchestrator restarted and verified on http://localhost:${PORT}/"
echo "    (hard-refresh the browser tab: Ctrl+Shift+R)"
