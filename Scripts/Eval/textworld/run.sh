#!/usr/bin/env bash
set -euo pipefail
# Set MODEL_PATH here, export it, or pass the checkpoint path as the first argument.
MODEL_PATH="${MODEL_PATH:-}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
if [[ -n "${MODEL_PATH}" ]]; then
  set -- --model-path "${MODEL_PATH}" "$@"
fi
exec "${PYTHON:-python}" "${REPOSITORY}/Evaluation/run.py" --backend TextWorld "$@"
