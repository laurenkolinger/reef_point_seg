#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# TCRMPclip_letterID - Environment Setup
# Creates a local conda env for letter/crosshair detection on
# TCRMP coral reef point-annotated images.
#
# Requirements: conda, NVIDIA GPU with CUDA 12.8+ driver
# Tested on: 2x RTX 5090 (Blackwell SM 12.0), Driver 580.x
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="letterid"
ENV_DIR="${SCRIPT_DIR}/env"
PYTHON_VERSION="3.11"

echo "=== TCRMPclip_letterID Environment Setup ==="
echo "Location: ${SCRIPT_DIR}"
echo ""

# Check for conda
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

# Check NVIDIA driver
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. GPU acceleration will not work."
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
    echo "Installing PyTorch with CUDA 13.0 (RTX 5090 / Blackwell compatible)..."
    "${PIP}" install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

    echo ""
    echo "Installing EasyOCR + dependencies..."
    "${PIP}" install \
        easyocr \
        flask \
        flask-cors \
        opencv-python-headless \
        pandas \
        scipy \
        numpy \
        Pillow

    echo ""
    echo "Installing Tesseract OCR (for speed comparison)..."
    "${PIP}" install pytesseract

    # Check if system tesseract binary exists
    if ! command -v tesseract &>/dev/null; then
        echo "NOTE: pytesseract requires the 'tesseract' system binary."
        echo "Install with: sudo apt-get install tesseract-ocr"
    fi
fi

PYTHON="${ENV_DIR}/bin/python"

echo ""
echo "=== Verifying installation ==="
"${PYTHON}" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f'  GPU {i}: {name} (SM {cap[0]}.{cap[1]})')

import easyocr
print(f'EasyOCR: OK')

import cv2
print(f'OpenCV: {cv2.__version__}')

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
echo "To run app:   ./run.sh"
echo "To run tests: ./env/bin/python src/speed_test.py"
