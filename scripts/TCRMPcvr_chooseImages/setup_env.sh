#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# TCRMPcvr_chooseImages - Environment Setup
# Creates a local conda env for balanced image selection.
#
# Requirements: conda (no GPU needed)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
PYTHON_VERSION="3.11"

echo "=== TCRMPcvr_chooseImages Environment Setup ==="
echo "Location: ${SCRIPT_DIR}"
echo ""

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

if [ -d "${ENV_DIR}" ]; then
    echo "Environment already exists at ${ENV_DIR}"
    echo "To recreate, delete it first: rm -rf ${ENV_DIR}"
else
    echo "Creating conda env at ${ENV_DIR} (Python ${PYTHON_VERSION})..."
    conda create --prefix "${ENV_DIR}" python="${PYTHON_VERSION}" -y

    "${ENV_DIR}/bin/pip" install pandas matplotlib
fi

"${ENV_DIR}/bin/python" -c "
import pandas; print(f'Pandas: {pandas.__version__}')
import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')
print('All packages verified.')
"

echo ""
echo "=== Setup complete ==="
echo "To run: python src/select_images.py"
echo "To plot: python src/plot_diagnostics.py"
