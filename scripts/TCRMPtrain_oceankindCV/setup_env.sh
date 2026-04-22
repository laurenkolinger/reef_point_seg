#!/usr/bin/env bash
# Create a project-local conda env for step 6 (YOLO segmentation training via
# oceankind_CV / ultralytics). Idempotent: re-running is cheap, it just verifies.
set -e

cd "$(dirname "$0")"
WRAPPER_DIR="$(pwd)"
ENV_DIR="$WRAPPER_DIR/env"
OCEAN_DIR="$WRAPPER_DIR/oceankind_CV"
ULTRA_DIR="$WRAPPER_DIR/ultralytics_src"

# --- conda ----------------------------------------------------------------
if [ -z "$CONDA_EXE" ]; then
    for cand in /home/bizon/anaconda3/bin/conda /opt/conda/bin/conda "$HOME/miniconda3/bin/conda"; do
        [ -x "$cand" ] && export CONDA_EXE="$cand" && break
    done
fi
if [ -z "$CONDA_EXE" ]; then
    echo "ERROR: conda not found. Set CONDA_EXE or install miniconda."; exit 1
fi
CONDA_SH="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$CONDA_SH"

# --- env ------------------------------------------------------------------
if [ ! -d "$ENV_DIR" ]; then
    echo "[setup] creating conda env at $ENV_DIR (this will take a while)..."
    # Go through pip for torch so we can pin a CUDA build that matches the box.
    # Driver detected: 580.x (CUDA 13.0) — cu124 wheels are forward-compat.
    "$CONDA_EXE" create --prefix "$ENV_DIR" python=3.11 pip -y
fi

conda activate "$ENV_DIR"

# --- pytorch (cu128 wheels, Blackwell-compatible) -------------------------
# cu128 wheels bundle sm_120 kernels, which RTX 50-series / Blackwell GPUs
# require. cu124 wheels only cover up to sm_90 and throw "no kernel image"
# on Blackwell. If torch is present but doesn't support the detected GPU's
# compute capability, wipe and reinstall.
NEEDS_TORCH=0
python - <<'PY' || NEEDS_TORCH=1
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("[setup] torch present but CUDA unavailable"); sys.exit(1)
    supported = set(torch.cuda.get_arch_list())
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        tag = f"sm_{cap[0]}{cap[1]}"
        if tag not in supported:
            print(f"[setup] torch lacks kernels for {torch.cuda.get_device_name(i)} ({tag}); will reinstall")
            sys.exit(1)
    print(f"[setup] torch {torch.__version__} OK, CUDA {torch.version.cuda}, archs={sorted(supported)}")
except Exception as e:
    print(f"[setup] torch check failed: {e}"); sys.exit(1)
PY
if [ "${NEEDS_TORCH}" = "1" ]; then
    echo "[setup] installing torch+torchvision from cu128 wheels..."
    pip install --upgrade pip
    pip uninstall -y torch torchvision 2>/dev/null || true
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# --- ultralytics (editable from git, per repo README) ---------------------
if [ ! -d "$ULTRA_DIR" ]; then
    echo "[setup] cloning ultralytics..."
    git clone --depth 1 https://github.com/ultralytics/ultralytics "$ULTRA_DIR"
fi
if ! python -c "import ultralytics" 2>/dev/null; then
    echo "[setup] pip install -e ultralytics..."
    pip install -e "$ULTRA_DIR"
fi

# --- oceankind_CV support libs --------------------------------------------
# Deps the repo scripts actually import: yaml, tqdm, scikit-learn, pandas, Pillow,
# wandb (optional; training script doesn't hard-require it). numpy + opencv come
# in via ultralytics transitively but pin here to be explicit.
pip install --quiet \
    pyyaml tqdm scikit-learn pandas pillow numpy opencv-python wandb

echo "[setup] verifying..."
python "$OCEAN_DIR/tools/test_install.py" || true

echo "[setup] done. env at $ENV_DIR"
