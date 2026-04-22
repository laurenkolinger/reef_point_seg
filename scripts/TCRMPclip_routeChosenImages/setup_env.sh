#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# TCRMPclip_routeChosenImages - Environment Setup
# Creates a local conda env for unified CPC+OCR point review.
#
# Requirements: conda, NVIDIA GPU with CUDA 12.8+ driver
# Tested on: 2x RTX 5090 (Blackwell SM 12.0), Driver 580.x
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
PYTHON_VERSION="3.11"

echo "=== TCRMPclip_routeChosenImages Environment Setup ==="
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

    PIP="${ENV_DIR}/bin/pip"

    echo "Installing PyTorch with CUDA 13.0..."
    "${PIP}" install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

    echo "Installing EasyOCR + dependencies..."
    "${PIP}" install \
        easyocr flask flask-cors \
        opencv-python-headless pandas scipy numpy Pillow
fi

PYTHON="${ENV_DIR}/bin/python"

echo ""
echo "=== Verifying installation ==="
"${PYTHON}" -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
import easyocr; print('EasyOCR: OK')
import flask; print(f'Flask: OK')
import pandas; print(f'Pandas: {pandas.__version__}')
print('All packages verified.')
"

echo ""
echo "=== Setup complete ==="
echo "To run: ./run.sh"
