#!/usr/bin/env bash
# Launch the TCRMP Segment Images app and open browser
cd "$(dirname "$0")"

PORT="${1:-5070}"

# Kill any existing server on this port
lsof -ti :"$PORT" 2>/dev/null | xargs -r kill 2>/dev/null && sleep 1

# Resolve python. Preference order:
#   1) $PYTHON env var (operator override)
#   2) the unified env at the repo root (has SAM3 via transformers>=4.51 + torch cu128)
#   3) the documented sam3reef conda env (legacy fallback for pre-unified-env installs)
#   4) a sub-tool-local env/ if someone built one via setup_env.sh
UNIFIED="$(cd ../.. && pwd)/env/bin/python"
SAM3REEF="/home/bizon/anaconda3/envs/sam3reef/bin/python"
if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
    :   # operator-provided, use as-is
elif [ -x "$UNIFIED" ]; then
    PYTHON="$UNIFIED"
elif [ -x "$SAM3REEF" ]; then
    PYTHON="$SAM3REEF"
elif [ -x "./env/bin/python" ]; then
    PYTHON="./env/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" src/app.py --port "$PORT" &
SERVER_PID=$!

# Wait for server to be ready (SAM3 model load takes a few seconds)
sleep 8
xdg-open "http://localhost:$PORT" 2>/dev/null || echo "Open http://localhost:$PORT in your browser"

# Keep running until Ctrl+C
wait $SERVER_PID
