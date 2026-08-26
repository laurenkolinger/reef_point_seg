#!/usr/bin/env bash
# Step 6 driver. Orchestrator invokes this with a sub-command.
#   run_step6.sh split <step5_dir> <step6_dir> [valid] [test] [min_samples]
#   run_step6.sh train <step6_dir> <name> [epochs] [imgsz] [model]
#   run_step6.sh all   <step5_dir> <step6_dir> <name> [epochs] [imgsz] [model]
#
# For `all`, the split ratios + every training hyperparameter can be overridden
# via env vars set by the orchestrator:
#
#   TCRMP_STEP6_VALID       TCRMP_STEP6_TEST          TCRMP_STEP6_MIN_SAMPLES
#   TCRMP_STEP6_INCLUDE_CLASSES  (CSV of class IDs to keep; unset = all)
#   TCRMP_STEP6_PIN_SPLIT   ("1" = pinned_split.py (default), frozen by-transect
#                            holdout; "0" = legacy bal_train_test_split.py random split)
#   TCRMP_STEP6_VAL_TRANSECTS TCRMP_STEP6_TEST_TRANSECTS  (CSV transect numbers;
#                            pinned_split.py defaults: val=5, test=6)
#   TCRMP_STEP6_HOLDOUT_MODE  (pinned_split.py fresh-split policy: "transect"
#                            (default) or "pinned-random"; ignored once a
#                            split_manifest.json already exists)
#   TCRMP_STEP6_DEVICE      ("0" single, "0,1" DDP, "cpu", etc.; unset = "0")
#   TCRMP_STEP6_BATCH       TCRMP_STEP6_PATIENCE      TCRMP_STEP6_OPTIMIZER
#   TCRMP_STEP6_LR0         TCRMP_STEP6_LRF           TCRMP_STEP6_MOMENTUM
#   TCRMP_STEP6_WEIGHT_DECAY  TCRMP_STEP6_WARMUP_EPOCHS
#   TCRMP_STEP6_BOX         TCRMP_STEP6_CLS           TCRMP_STEP6_DFL
#   TCRMP_STEP6_HSV_H       TCRMP_STEP6_HSV_S         TCRMP_STEP6_HSV_V
#   TCRMP_STEP6_DEGREES     TCRMP_STEP6_TRANSLATE     TCRMP_STEP6_SCALE
#   TCRMP_STEP6_SHEAR       TCRMP_STEP6_PERSPECTIVE
#   TCRMP_STEP6_FLIPUD      TCRMP_STEP6_FLIPLR
#   TCRMP_STEP6_MOSAIC      TCRMP_STEP6_MIXUP         TCRMP_STEP6_COPY_PASTE
#   TCRMP_STEP6_ERASING     TCRMP_STEP6_AUTO_AUGMENT
#   TCRMP_STEP6_COS_LR      TCRMP_STEP6_CLOSE_MOSAIC
#   TCRMP_STEP6_SEED        TCRMP_STEP6_LABEL_SMOOTHING
#   TCRMP_STEP6_FREEZE      (int; number of leading layers to freeze for
#                            fine-tuning from previous weights; unset = no freeze)
#
# Output dirs inside step6_dir:
#   dataset/   train/valid/test split + data.yaml + test.yaml
#   runs/      ultralytics training runs (each --name is a subdir)
set -e

cd "$(dirname "$0")"
WRAPPER_DIR="$(pwd)"
# Prefer the repo-level unified env (github_repo/env) built by the module
# bootstrap. Fall back to the local per-subtool env only if the unified env
# is missing (legacy standalone installs). The orchestrator may also pin
# the interpreter explicitly via TCRMP_STEP6_PYTHON.
UNIFIED_PY="$WRAPPER_DIR/../../env/bin/python"
LOCAL_PY="$WRAPPER_DIR/env/bin/python"
if [ -n "${TCRMP_STEP6_PYTHON:-}" ] && [ -x "$TCRMP_STEP6_PYTHON" ]; then
    PY="$TCRMP_STEP6_PYTHON"
elif [ -x "$UNIFIED_PY" ]; then
    PY="$UNIFIED_PY"
elif [ -x "$LOCAL_PY" ]; then
    PY="$LOCAL_PY"
else
    echo "ERROR: no usable python env found. Expected unified env at $UNIFIED_PY or local env at $LOCAL_PY. Run the module bootstrap or $WRAPPER_DIR/setup_env.sh." >&2
    exit 1
fi
OCEAN_DIR="$WRAPPER_DIR/oceankind_CV"

cmd="${1:-}"
shift || true

# Helper: forward an env var as a CLI arg if the env var is set + non-empty.
# Usage: add_arg ARGS --lr0 TCRMP_STEP6_LR0
add_arg() {
    local -n _args=$1   # nameref to the array we're appending to
    local flag="$2"
    local var="$3"
    local val="${!var:-}"
    if [ -n "$val" ]; then
        _args+=("$flag" "$val")
    fi
}

run_split() {
    local STEP5="$1" STEP6="$2"
    local VALID="${3:-${TCRMP_STEP6_VALID:-0.2}}"
    local TEST="${4:-${TCRMP_STEP6_TEST:-0.1}}"
    local MIN_SAMPLES="${5:-${TCRMP_STEP6_MIN_SAMPLES:-10}}"
    if [ -z "$STEP5" ] || [ -z "$STEP6" ]; then
        echo "usage: run_step6.sh split <step5_dir> <step6_dir> [valid] [test] [min_samples]" >&2
        exit 2
    fi
    mkdir -p "$STEP6/dataset"
    if [ "${TCRMP_STEP6_PIN_SPLIT:-1}" = "1" ]; then
        echo "[step6] pinned split $STEP5 -> $STEP6/dataset (valid=$VALID test=$TEST min_samples=$MIN_SAMPLES)"
        local -a PIN_ARGS=(
            --src "$STEP5"
            --out "$STEP6/dataset"
            --valid "$VALID"
            --test "$TEST"
            --min_samples "$MIN_SAMPLES"
        )
        add_arg PIN_ARGS --include-classes TCRMP_STEP6_INCLUDE_CLASSES
        add_arg PIN_ARGS --val-transects   TCRMP_STEP6_VAL_TRANSECTS
        add_arg PIN_ARGS --test-transects  TCRMP_STEP6_TEST_TRANSECTS
        add_arg PIN_ARGS --holdout-mode    TCRMP_STEP6_HOLDOUT_MODE
        "$PY" "$WRAPPER_DIR/src/pinned_split.py" "${PIN_ARGS[@]}"
    else
        echo "[step6] legacy random split $STEP5 -> $STEP6/dataset (valid=$VALID test=$TEST min_samples=$MIN_SAMPLES)"
        local -a SPLIT_ARGS=(
            --src "$STEP5"
            --out "$STEP6/dataset"
            --valid "$VALID"
            --test "$TEST"
            --min_samples "$MIN_SAMPLES"
            --yes
        )
        add_arg SPLIT_ARGS --include-classes TCRMP_STEP6_INCLUDE_CLASSES
        "$PY" "$OCEAN_DIR/tools/bal_train_test_split.py" "${SPLIT_ARGS[@]}"
    fi
}

run_train() {
    local STEP6="$1" NAME="$2"
    local EPOCHS="${3:-500}" IMGSZ="${4:-512}" MODEL="${5:-yolo11m-seg.pt}"
    if [ -z "$STEP6" ] || [ -z "$NAME" ]; then
        echo "usage: run_step6.sh train <step6_dir> <name> [epochs] [imgsz] [model]" >&2
        exit 2
    fi
    local DATA_YAML="$STEP6/dataset/data.yaml"
    if [ ! -f "$DATA_YAML" ]; then
        echo "ERROR: $DATA_YAML not found. Run split first." >&2
        exit 3
    fi
    mkdir -p "$STEP6/runs"
    echo "[step6] training on $DATA_YAML -> $STEP6/runs/$NAME (model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ)"

    # Core args — always set
    local -a ARGS=(
        --src "$DATA_YAML"
        --name "$NAME"
        --project "$STEP6/runs"
        --model "$MODEL"
        --epochs "$EPOCHS"
        --imgsz "$IMGSZ"
    )

    # Forward optional env-var overrides as CLI args when set.
    add_arg ARGS --device             TCRMP_STEP6_DEVICE
    add_arg ARGS --batch              TCRMP_STEP6_BATCH
    add_arg ARGS --patience           TCRMP_STEP6_PATIENCE
    add_arg ARGS --optimizer          TCRMP_STEP6_OPTIMIZER
    add_arg ARGS --seed               TCRMP_STEP6_SEED
    add_arg ARGS --cos_lr             TCRMP_STEP6_COS_LR
    add_arg ARGS --close_mosaic       TCRMP_STEP6_CLOSE_MOSAIC
    add_arg ARGS --freeze             TCRMP_STEP6_FREEZE
    # LR
    add_arg ARGS --lr0                TCRMP_STEP6_LR0
    add_arg ARGS --lrf                TCRMP_STEP6_LRF
    add_arg ARGS --momentum           TCRMP_STEP6_MOMENTUM
    add_arg ARGS --weight_decay       TCRMP_STEP6_WEIGHT_DECAY
    add_arg ARGS --warmup_epochs      TCRMP_STEP6_WARMUP_EPOCHS
    # Loss
    add_arg ARGS --box                TCRMP_STEP6_BOX
    add_arg ARGS --cls                TCRMP_STEP6_CLS
    add_arg ARGS --dfl                TCRMP_STEP6_DFL
    add_arg ARGS --label_smoothing    TCRMP_STEP6_LABEL_SMOOTHING
    # Color
    add_arg ARGS --hsv_h              TCRMP_STEP6_HSV_H
    add_arg ARGS --hsv_s              TCRMP_STEP6_HSV_S
    add_arg ARGS --hsv_v              TCRMP_STEP6_HSV_V
    # Geometric
    add_arg ARGS --degrees            TCRMP_STEP6_DEGREES
    add_arg ARGS --translate          TCRMP_STEP6_TRANSLATE
    add_arg ARGS --scale              TCRMP_STEP6_SCALE
    add_arg ARGS --shear              TCRMP_STEP6_SHEAR
    add_arg ARGS --perspective        TCRMP_STEP6_PERSPECTIVE
    add_arg ARGS --flipud             TCRMP_STEP6_FLIPUD
    add_arg ARGS --fliplr             TCRMP_STEP6_FLIPLR
    # Mixing / regularization
    add_arg ARGS --mosaic             TCRMP_STEP6_MOSAIC
    add_arg ARGS --mixup              TCRMP_STEP6_MIXUP
    add_arg ARGS --cutmix             TCRMP_STEP6_CUTMIX
    add_arg ARGS --copy_paste         TCRMP_STEP6_COPY_PASTE
    add_arg ARGS --copy_paste_mode    TCRMP_STEP6_COPY_PASTE_MODE
    add_arg ARGS --erasing            TCRMP_STEP6_ERASING
    add_arg ARGS --auto_augment       TCRMP_STEP6_AUTO_AUGMENT
    add_arg ARGS --bgr                TCRMP_STEP6_BGR

    "$PY" "$WRAPPER_DIR/src/train_wrapper.py" "${ARGS[@]}"
}

case "$cmd" in
    split) run_split "$@" ;;
    train) run_train "$@" ;;
    all)
        STEP5="$1"; STEP6="$2"; NAME="$3"
        run_split "$STEP5" "$STEP6"
        run_train "$STEP6" "$NAME" "${4:-500}" "${5:-512}" "${6:-yolo11m-seg.pt}"
        ;;
    *)
        echo "usage: run_step6.sh {split|train|all} ..." >&2
        exit 2
        ;;
esac
