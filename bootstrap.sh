#!/usr/bin/env bash
# Bootstrap the seg_AI_img_full_april2026 repo on a fresh machine.
#   1. Build the unified Python env at ./env/
#   2. Rewrite absolute paths inside copied projects/*/project.json
#   3. Smoke-test imports
#
# Idempotent: safe to re-run. Does not touch supporting_data/ or projects/
# beyond the one-time path rewrite.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

echo ""
echo "=== [1/3] Building unified environment at ${REPO_ROOT}/env ==="
bash ./scripts/setup_unified_env.sh

echo ""
echo "=== [2/3] Rewriting absolute paths inside projects/ ==="
if [ -d "${REPO_ROOT}/projects" ] && [ -n "$(ls -A "${REPO_ROOT}/projects" 2>/dev/null)" ]; then
    "${REPO_ROOT}/env/bin/python" \
        "${REPO_ROOT}/scripts/rewrite_project_paths.py" \
        --old "/mnt/rip/vicarius_drive/hopper/CVR_CLIP_forAI" \
        --new "${REPO_ROOT}" \
        --apply
else
    echo "  (no projects/ content yet — skipping rewriter)"
fi

echo ""
echo "=== [3/3] Smoke-testing imports for every step ==="
"${REPO_ROOT}/env/bin/python" - <<'PY'
import importlib, sys
mods = ["pandas", "numpy", "flask", "yaml", "matplotlib",
        "openpyxl", "xlrd", "sklearn", "PIL", "cv2", "scipy"]
optional = ["easyocr", "ultralytics", "transformers"]
for m in mods:
    importlib.import_module(m)
    print(f"  [ok] {m}")
for m in optional:
    try:
        importlib.import_module(m)
        print(f"  [ok] {m}")
    except Exception as e:
        print(f"  [WARN] {m} unavailable: {e}")
print("smoke test passed")
PY

echo ""
echo "bootstrap complete. Launch the orchestrator:"
echo "    ./scripts/run_pipeline.sh"
echo ""
echo "Then open http://localhost:5050"
