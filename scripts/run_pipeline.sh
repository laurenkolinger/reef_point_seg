#!/usr/bin/env bash
# Launch the Reef Point Seg Orchestrator.
# Lives inside seg_AI_img_full_april2026/scripts/; the unified env sits one
# level up at seg_AI_img_full_april2026/env/.
set -u
cd "$(dirname "$0")"              # now inside scripts/
REPO_ROOT="$(cd .. && pwd)"       # seg_AI_img_full_april2026 root
PORT="${1:-5050}"

# Prefer the unified env; fall back to system python3.
PYTHON="${REPO_ROOT}/env/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "  Unified env not found at $PYTHON"
    echo "  Run ./bootstrap.sh from ${REPO_ROOT} first, or falling back to system python3"
    PYTHON="python3"
fi

# ── Sweep stale orchestrator + sub-app processes ──────────────────────
# Prior orchestrator runs sometimes leave zombie sub-apps bound to random
# free ports; those hold stale Flask sessions/annotations that leak into
# fresh projects. Kill anything matching our known entry points before
# we start.
echo ""
echo "  Checking for stale orchestrator + sub-app processes..."

STALE_PATTERNS=(
    "run_pipeline.py"
    "TCRMPcvr_recodeSpecies/src/app.py"
    "TCRMPclip_placePoints/src/app.py"
    "TCRMPclip_segmentImages/src/app.py"
    "TCRMPclip_combinedAnnotate/src/app.py"
    "TCRMPclip_addExpertIDs/src/app.py"
)
KILLED=0
for pat in "${STALE_PATTERNS[@]}"; do
    pids=$(pgrep -f "$pat" 2>/dev/null || true)
    for pid in $pids; do
        # Skip our own shell / parent
        [ "$pid" = "$$" ] && continue
        kill "$pid" 2>/dev/null && echo "    killed pid $pid ($pat)" && KILLED=$((KILLED+1))
    done
done

# Free default ports too (catches anything else bound to them)
for p in "$PORT" 5055 5065 5066 5070 5075 5080; do
    pids=$(lsof -ti:"$p" 2>/dev/null || true)
    for pid in $pids; do
        kill "$pid" 2>/dev/null && echo "    killed pid $pid (port $p)" && KILLED=$((KILLED+1))
    done
done

if [ "$KILLED" -gt 0 ]; then
    sleep 2
    # Re-check; escalate to SIGKILL for anything stubborn
    for pat in "${STALE_PATTERNS[@]}"; do
        pgrep -f "$pat" 2>/dev/null | xargs -r kill -9 2>/dev/null
    done
    echo "  Cleaned up $KILLED stale process(es)."
else
    echo "  No stale processes found."
fi

echo ""
echo "  Starting Reef Point Seg Orchestrator on port $PORT..."
echo "  View VICARIUS event stream any time:  vicarius story --days 1"
echo ""

"$PYTHON" run_pipeline.py --port "$PORT" &
PID=$!
sleep 2

# Open browser
xdg-open "http://localhost:$PORT" 2>/dev/null || \
    open "http://localhost:$PORT" 2>/dev/null || \
    echo "  Open http://localhost:$PORT in your browser"

wait $PID
