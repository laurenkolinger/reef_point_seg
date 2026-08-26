#!/usr/bin/env bash
# TCRMPclip_addExpertIDs — environment setup.
# This tool is lightweight (no SAM3/torch). It reuses the unified env at the
# repo root (../../env) which already has flask, flask-cors, numpy, pillow.
# Build a local env only if you want this tool standalone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"

echo "=== TCRMPclip_addExpertIDs Environment Setup ==="
if [ -x "$(cd "${SCRIPT_DIR}/../.." && pwd)/env/bin/python" ]; then
    echo "Unified repo env found — no local env needed."
    echo "Run with: ./run.sh 5075"
    exit 0
fi

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found and no unified env. Install Anaconda/Miniconda."
    exit 1
fi

if [ -d "${ENV_DIR}" ]; then
    echo "Environment already exists at ${ENV_DIR}"
else
    conda create --prefix "${ENV_DIR}" python=3.12 -y
    "${ENV_DIR}/bin/pip" install flask flask-cors numpy pillow
fi
echo "Done. Run with: ./run.sh 5075"
