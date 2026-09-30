#!/usr/bin/env bash
set -euo pipefail
# Paper selector: Policy judge (N=2); policy rubric + scorer; no Kimi requests.
MODEL_PATH="${MODEL_PATH:-}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
if [[ -n "${MODEL_PATH}" ]]; then
  set -- --model-path "${MODEL_PATH}" "$@"
fi
exec "${PYTHON:-python}" "${REPOSITORY}/Evaluation/run.py" \
  --backend TextWorld --best-of-n --selection-mode rubric --branching-factor 2 "$@"
