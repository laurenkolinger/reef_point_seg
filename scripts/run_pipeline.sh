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
echo "  Flushing any live annotator / Edit Masks session before the sweep..."
# Save first (2026-09-27 safety net, D1): the stale-process sweep just below
# pattern-matches TCRMPclip_combinedAnnotate/src/app.py by command line, which
# reaches a standalone "Annotate Images" session on port 5096 too, not only a
# project session in 5080-5089. Only a listener whose /proc/<pid>/cmdline
# names one of the two annotator entry scripts is flushed here (the same
# cmdline check _kill_all_annotators uses in pipeline_orchestrator/app.py),
# so an unrelated service sharing this port range is never poked
# (benthic_species_viewer's serve_edit.py on 5081 and vicariusLLM on 5085
# have both been seen live here). A flush is a best-effort POST to
# /api/flush, never a stop, so it costs nothing when the pid turns out to be
# spared or nothing is actually listening. This script does not set -e; a
# failed curl is swallowed and printed either way so a future -e stays safe.
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
