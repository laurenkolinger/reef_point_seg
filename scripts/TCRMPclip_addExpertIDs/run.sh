#!/usr/bin/env bash
# Launch the TCRMP Add-Expert-IDs importer and open a browser.
# Lightweight (no SAM3) — only needs flask, numpy, pillow.
cd "$(dirname "$0")"

PORT="${1:-5075}"

# Kill any existing server on this port
lsof -ti :"$PORT" 2>/dev/null | xargs -r kill 2>/dev/null && sleep 1

# Resolve python: operator override, then the unified env at the repo root,
# then a sub-tool-local env, then system python3.
UNIFIED="$(cd ../.. && pwd)/env/bin/python"
if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
    :
elif [ -x "$UNIFIED" ]; then
    PYTHON="$UNIFIED"
elif [ -x "./env/bin/python" ]; then
    PYTHON="./env/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" src/app.py --port "$PORT" &
SERVER_PID=$!

sleep 2
xdg-open "http://localhost:$PORT" 2>/dev/null || echo "Open http://localhost:$PORT in your browser"

wait $SERVER_PID
