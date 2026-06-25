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
MODULE_PORTS=("$PORT" 5055 5065 5070 5075)

echo "==> Restarting Reef Point Seg orchestrator on port ${PORT}"

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
    if printf '%s' "$body" | grep -q 's3-label-chips'; then
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

# ── 5. Self-test: assert the expected markers are live ────────────────────────
# Note: the page HTML ($body) carries static markup; the footer Reset/Save&Close
# buttons and the picker grouping are injected by orchestrator.js at runtime, so
# those are checked against the served JS, not the HTML.
js="$(curl -s -m 5 "http://127.0.0.1:${PORT}/static/orchestrator.js?v=20260624-review-io" 2>/dev/null || true)"
echo "  self-test (live markers on :${PORT}):"
fail=0
assert_html_present() { printf '%s' "$body" | grep -q "$1" && echo "    OK  present: $2" || { echo "    FAIL missing: $2"; fail=1; }; }
assert_html_absent()  { printf '%s' "$body" | grep -q "$1" && { echo "    FAIL still there: $2"; fail=1; } || echo "    OK  gone: $2"; }
assert_js_present()   { printf '%s' "$js"   | grep -q "$1" && echo "    OK  present: $2" || { echo "    FAIL missing: $2"; fail=1; }; }
assert_html_present 's3-label-chips'        'Step 3 selection chips container'
assert_html_present 'id="confirm-overlay"'  'reset-confirm modal markup'
assert_html_present 'Do this:'              'rewritten user instructions'
assert_html_present 'review-io'             'fresh orchestrator.js cache-bust'
assert_html_absent  's3-load-labels'        'old "Load labels from Step 2" button'
assert_html_absent  'What you do'           'old "What you do" phrasing'
assert_html_absent  'data-step="s4review"'  'old SAM3-Review stub tile'
assert_html_absent  'panel-s4review'        'old SAM3-Review stub panel'
assert_html_present 'Expert Review I/O'     'renamed expert tile + panel'
assert_js_present   'Reset Step'            'Reset button (rendered next to Save and Close)'
assert_js_present   'resetBtn}${saveCloseBtn' 'Reset positioned next to Save and Close'
assert_js_present   'toggleAllStep3Groups'  'grouped/collapsible label picker'
assert_js_present   'LEGACY_S3_DEFAULT'     'zero-selected default on first view'

if [ "$fail" -ne 0 ]; then
    echo "==> Restart completed but self-test FAILED (stale assets?). See above." >&2
    exit 2
fi
echo "==> Reef Point Seg orchestrator restarted and verified on http://localhost:${PORT}/"
echo "    (hard-refresh the browser tab: Ctrl+Shift+R)"
