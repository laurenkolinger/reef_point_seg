#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# TCRMPclip_segmentImages - Environment Setup
# Creates a local conda env for SAM3-based segmentation review.
#
# Requirements: conda, NVIDIA GPU with CUDA 12.8+ driver
# Tested on: 2x RTX 5090 (Blackwell SM 12.0), Driver 580.x
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
PYTHON_VERSION="3.12"

echo "=== TCRMPclip_segmentImages Environment Setup ==="
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

    echo "Installing HuggingFace Transformers (SAM3 via facebook/sam3)..."
    "${PIP}" install transformers huggingface_hub

    echo "Installing Flask + utilities..."
    "${PIP}" install \
        flask flask-cors \
        opencv-python-headless numpy Pillow scipy pycocotools
fi

PYTHON="${ENV_DIR}/bin/python"

echo ""
echo "=== Verifying installation ==="
"${PYTHON}" -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
from transformers import Sam3TrackerModel, Sam3Processor; print('SAM3 (transformers): OK')
import flask; print('Flask: OK')
import cv2; print(f'OpenCV: {cv2.__version__}')
print('All packages verified.')
"

echo ""
echo "=== Setup complete ==="
echo "To run: ./run.sh"
