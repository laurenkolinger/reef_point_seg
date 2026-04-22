#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# TCRMPcvr_recodeSpecies - Environment Setup
# Creates a local conda env for the species code recoding tool.
#
# Requirements: conda (no GPU needed)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="recodespecies"
ENV_DIR="${SCRIPT_DIR}/env"
PYTHON_VERSION="3.11"

echo "=== TCRMPcvr_recodeSpecies Environment Setup ==="
echo "Location: ${SCRIPT_DIR}"
echo ""

# Check for conda
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

# Create env in local folder (portable)
if [ -d "${ENV_DIR}" ]; then
    echo "Environment already exists at ${ENV_DIR}"
    echo "To recreate, delete it first: rm -rf ${ENV_DIR}"
    echo "Activating existing env..."
else
    echo "Creating conda env at ${ENV_DIR} (Python ${PYTHON_VERSION})..."
    conda create --prefix "${ENV_DIR}" python="${PYTHON_VERSION}" -y

    PIP="${ENV_DIR}/bin/pip"

    echo ""
    echo "Installing dependencies..."
    "${PIP}" install flask pandas
fi

PYTHON="${ENV_DIR}/bin/python"

echo ""
echo "=== Verifying installation ==="
"${PYTHON}" -c "
import flask
print(f'Flask: {flask.__version__}')

import pandas
print(f'Pandas: {pandas.__version__}')

print()
print('All packages verified successfully!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run app: ./run.sh"
