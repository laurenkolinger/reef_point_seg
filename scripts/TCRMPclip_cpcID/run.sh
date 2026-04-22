#!/bin/bash
# TCRMPclip_cpcID - Run Script
#
# Usage:
#   ./run.sh <input_dir> <output_dir> [options]
#   ./run.sh --batch <root_input> <root_output> [options]
#
# Examples:
#   # Single site directory:
#   ./run.sh /path/to/TCRMP20181101_clip_BIX /path/to/output/BIX_2018
#
#   # Batch - all directories under TCRMP_clip:
#   ./run.sh --batch "/home/bizon/UVI Dropbox/SMITH LAB TEAM FOLDER/TCRMP/TCRMP_clip" /path/to/output/cpc_all
#
#   # Generate all test overlay images after a batch run:
#   ./run.sh --test-pts /path/to/output/cpc_all
#
# Options are passed through to the Python scripts. See README.md for details.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNIFIED="$(cd "$SCRIPT_DIR/../.." && pwd)/env/bin/python"

# Determine Python executable. Preference: unified env, then sub-tool-local
# env/, then system python3 (with a Pillow availability check as a guard).
if [ -x "$UNIFIED" ]; then
    PYTHON="$UNIFIED"
elif [ -f "$SCRIPT_DIR/env/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/env/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
    if ! "$PYTHON" -c "from PIL import Image" 2>/dev/null; then
        echo "ERROR: Pillow not found. Run 'bash setup_env.sh' first, or 'pip install Pillow'."
        exit 1
    fi
else
    echo "ERROR: No Python found. Run 'bash setup_env.sh' first."
    exit 1
fi

# Route to the right script
if [ "$1" = "--test-pts" ]; then
    shift
    cd "$SCRIPT_DIR/src"
    exec "$PYTHON" generate_test_pts.py "$@"
elif [ "$1" = "--summarize" ]; then
    shift
    cd "$SCRIPT_DIR/src"
    exec "$PYTHON" summarize.py "$@"
else
    cd "$SCRIPT_DIR/src"
    exec "$PYTHON" app.py "$@"
fi
