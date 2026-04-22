#!/bin/bash
# TCRMPclip_cpcID - Environment Setup
#
# Creates a local Python environment in ./env/ with minimal dependencies.
# Only needed if Pillow is not already available in your system Python.
#
# Usage:
#   bash setup_env.sh
#
# After setup:
#   ./run.sh <input_dir> <output_dir> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$SCRIPT_DIR/env"

echo "=============================================="
echo "TCRMPclip_cpcID - Environment Setup"
echo "=============================================="

# Check if env already exists
if [ -d "$ENV_DIR" ] && [ -f "$ENV_DIR/bin/python" ]; then
    echo "Environment already exists at $ENV_DIR"
    echo "To recreate, remove it first: rm -rf $ENV_DIR"
    "$ENV_DIR/bin/python" -c "from PIL import Image; print('Pillow OK')" 2>/dev/null && exit 0
    echo "Pillow missing, reinstalling..."
fi

# Check for conda
if command -v conda &> /dev/null; then
    echo "Using conda to create environment..."
    conda create -y -p "$ENV_DIR" python=3.11 pillow --no-default-packages
    echo ""
    echo "Environment created at $ENV_DIR"
    echo ""
    "$ENV_DIR/bin/python" -c "from PIL import Image; print('  Pillow: OK')"
    echo ""
    echo "Done! Run with:  ./run.sh <input_dir> <output_dir>"
    exit 0
fi

# Fallback: use system python + venv
if command -v python3 &> /dev/null; then
    echo "conda not found, using python3 venv..."
    python3 -m venv "$ENV_DIR"
    "$ENV_DIR/bin/pip" install --upgrade pip
    "$ENV_DIR/bin/pip" install Pillow
    echo ""
    echo "Environment created at $ENV_DIR"
    echo ""
    "$ENV_DIR/bin/python" -c "from PIL import Image; print('  Pillow: OK')"
    echo ""
    echo "Done! Run with:  ./run.sh <input_dir> <output_dir>"
    exit 0
fi

echo "ERROR: Neither conda nor python3 found. Install one and retry."
exit 1
