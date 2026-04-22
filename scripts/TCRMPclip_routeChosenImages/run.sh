#!/usr/bin/env bash
# Launch the TCRMP Route Chosen Images app and open browser
cd "$(dirname "$0")"

PORT="${1:-5065}"

# Kill any existing server on this port
lsof -ti :"$PORT" 2>/dev/null | xargs -r kill 2>/dev/null && sleep 1

# Resolve python: prefer the unified env at the repo root, fall back to a
# sub-tool-local env/, then system python3.
UNIFIED="$(cd ../.. && pwd)/env/bin/python"
if [ -x "$UNIFIED" ]; then
    PYTHON="$UNIFIED"
elif [ -x "./env/bin/python" ]; then
    PYTHON="./env/bin/python"
else
    PYTHON="python3"
fi

# Start server in background
"$PYTHON" src/app.py --port "$PORT" &
SERVER_PID=$!

# Wait for server to be ready, then open browser
sleep 3
xdg-open "http://localhost:$PORT" 2>/dev/null || echo "Open http://localhost:$PORT in your browser"

# Keep running until Ctrl+C
wait $SERVER_PID
