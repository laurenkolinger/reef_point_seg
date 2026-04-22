#!/usr/bin/env bash
# Build the single unified Python environment for all 8 pipeline steps.
# Idempotent: if env/ already exists and has torch CUDA support, re-running
# only ensures pip packages are installed.
#
# Target: Python 3.11 conda env at ../env/ (sibling of scripts/).
# Deps:
#   torch + torchvision cu128 wheels (Blackwell RTX 5090 compatible)
#   ultralytics (editable from scripts/TCRMPtrain_oceankindCV/ultralytics_src/)
#   transformers>=4.51 (SAM3)
#   easyocr==1.7.2 (step 4 OCR)
#   flask, pandas, numpy, pyyaml, matplotlib, scipy, opencv-python, Pillow
#   openpyxl, xlrd (step 1 Excel parsing)
#   scikit-learn (step 6 split)
#   pycocotools, wandb, tqdm, accelerate, safetensors
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
ENV_DIR="${REPO_ROOT}/env"
PY_VERSION="3.11"

find_conda() {
    for c in \
        "/home/bizon/anaconda3/bin/conda" \
        "/home/bizon/miniconda3/bin/conda" \
        "/opt/conda/bin/conda" \
        "$(which conda 2>/dev/null || true)"; do
        if [ -n "$c" ] && [ -x "$c" ]; then
            echo "$c"; return 0
        fi
    done
    return 1
}

CONDA="$(find_conda || true)"
if [ -z "${CONDA}" ]; then
    echo "ERROR: conda not found. Install miniconda/anaconda first."
    exit 1
fi
echo "Using conda: ${CONDA}"

# Create env/ if missing
if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "Creating env at ${ENV_DIR} with Python ${PY_VERSION}..."
    "${CONDA}" create -y -p "${ENV_DIR}" "python=${PY_VERSION}" pip
else
    echo "env/ already exists — skipping conda create"
fi

PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

# Verify torch + CUDA; reinstall cu128 wheels if missing or incompatible
needs_torch_install() {
    "${PY}" - <<'PY' 2>/dev/null || return 0
import torch
assert torch.cuda.is_available(), "no cuda"
try:
    torch.zeros(1).cuda()  # quick smoke
except Exception:
    raise SystemExit(1)
PY
    return $?
}

if ! needs_torch_install; then
    echo "Installing torch + torchvision (cu128 wheels for Blackwell)..."
    "${PIP}" install --upgrade pip
    "${PIP}" install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision
else
    echo "torch/cuda check passed — keeping existing torch"
fi

echo "Installing core pip dependencies..."
"${PIP}" install \
    "flask>=3.0" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "pyyaml>=6.0" \
    "matplotlib>=3.7" \
    "scipy>=1.11" \
    "scikit-learn>=1.3" \
    "Pillow>=10" \
    "opencv-python>=4.8" \
    "openpyxl>=3.1" \
    "xlrd>=2.0" \
    "pycocotools" \
    "tqdm" \
    "wandb" \
    "accelerate" \
    "safetensors" \
    "transformers>=4.51" \
    "easyocr==1.7.2"

# Editable install of the vendored ultralytics source
ULTRA_DIR="${REPO_ROOT}/scripts/TCRMPtrain_oceankindCV/ultralytics_src"
if [ -d "${ULTRA_DIR}" ]; then
    echo "Editable-installing ultralytics from ${ULTRA_DIR}..."
    "${PIP}" install -e "${ULTRA_DIR}"
else
    echo "Vendored ultralytics_src not found — installing from PyPI as fallback"
    "${PIP}" install "ultralytics>=8.4.38"
fi

echo ""
echo "Environment ready at ${ENV_DIR}"
echo "Quick import smoke test:"
"${PY}" -c "import pandas, flask, yaml, torch, ultralytics, transformers; \
print('pandas', pandas.__version__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('ultralytics', ultralytics.__version__); \
print('transformers', transformers.__version__)"
