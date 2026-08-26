#!/usr/bin/env bash
# Build the single unified Python environment for all 8 pipeline steps.
# Idempotent: if env/ already exists and has torch CUDA support, re-running
# only ensures pip packages are installed.
#
# Target: Python 3.11 conda env at ../env/ (sibling of scripts/).
#
# Modes:
#   (default, "locked")  Install the exact pinned set from requirements.lock
#                        (the pip freeze of the working env, torch cu130 build
#                        included; see the lockfile header), then the vendored
#                        ultralytics fork as an editable path install with
#                        --no-deps (all its runtime deps are already pinned).
#   --floors             Legacy path: "latest above a floor" for everything
#                        except easyocr, torch via the cu128 special index.
#                        NOT reproducible; the resulting env can drift from
#                        the one the SAM3 weight-backfill was validated on.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
ENV_DIR="${REPO_ROOT}/env"
LOCKFILE="${REPO_ROOT}/requirements.lock"
PY_VERSION="3.11"

MODE="locked"
for arg in "$@"; do
    case "$arg" in
        --floors) MODE="floors" ;;
        --locked) MODE="locked" ;;
        *) echo "Unknown flag: $arg (accepted: --locked [default], --floors)"; exit 2 ;;
    esac
done

if [ "$MODE" = "floors" ]; then
    echo "WARNING: --floors installs unpinned 'latest above a floor' versions."
    echo "WARNING: the result may differ from the validated env and the SAM3"
    echo "WARNING: tracker weight-backfill is version-sensitive. Prefer the"
    echo "WARNING: default locked mode (requirements.lock) for rebuilds."
fi
if [ "$MODE" = "locked" ] && [ ! -f "$LOCKFILE" ]; then
    echo "ERROR: requirements.lock not found at ${LOCKFILE}."
    echo "Run with --floors to build an unpinned env, or restore the lockfile."
    exit 2
fi

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
# The env's pip console script can carry a stale shebang after a module
# rename; python -m pip is immune to that.
PIP="${PY} -m pip"

ULTRA_DIR="${REPO_ROOT}/scripts/TCRMPtrain_oceankindCV/ultralytics_src"

if [ "$MODE" = "locked" ]; then
    echo "Installing the pinned set from requirements.lock..."
    ${PIP} install --upgrade pip
    ${PIP} install -r "${LOCKFILE}"
    if [ -d "${ULTRA_DIR}" ]; then
        echo "Editable-installing vendored ultralytics (path install, --no-deps)..."
        ${PIP} install -e "${ULTRA_DIR}" --no-deps
    else
        echo "ERROR: vendored ultralytics_src not found at ${ULTRA_DIR}."
        echo "Locked mode requires the vendored fork (no PyPI fallback: the"
        echo "lockfile documents ultralytics as a path install)."
        exit 2
    fi
else
    # ── Legacy floors path (unpinned) ──────────────────────────────────
    # Verify torch + CUDA; reinstall cu128 wheels if missing or incompatible
    needs_torch_install() {
        "${PY}" - <<'PYEOF' 2>/dev/null || return 0
import torch
assert torch.cuda.is_available(), "no cuda"
try:
    torch.zeros(1).cuda()  # quick smoke
except Exception:
    raise SystemExit(1)
PYEOF
        return $?
    }

    if ! needs_torch_install; then
        echo "Installing torch + torchvision (cu128 wheels for Blackwell)..."
        ${PIP} install --upgrade pip
        ${PIP} install --index-url https://download.pytorch.org/whl/cu128 \
            torch torchvision
    else
        echo "torch/cuda check passed — keeping existing torch"
    fi

    echo "Installing core pip dependencies (floors, unpinned)..."
    ${PIP} install \
        "flask>=3.0" \
        "flask-cors>=4.0" \
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
    if [ -d "${ULTRA_DIR}" ]; then
        echo "Editable-installing ultralytics from ${ULTRA_DIR}..."
        ${PIP} install -e "${ULTRA_DIR}"
    else
        echo "Vendored ultralytics_src not found — installing from PyPI as fallback"
        ${PIP} install "ultralytics>=8.4.38"
    fi
fi

echo ""
echo "Environment ready at ${ENV_DIR} (mode: ${MODE})"
echo "Quick import smoke test:"
"${PY}" -c "import pandas, flask, yaml, torch, ultralytics, transformers; \
print('pandas', pandas.__version__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('ultralytics', ultralytics.__version__); \
print('transformers', transformers.__version__)"
