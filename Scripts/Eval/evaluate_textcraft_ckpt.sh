#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CHECKPOINT="${REPOSITORY}/Evaluation/ckpt/TextCraft-step250"

if [[ ! -s "${CHECKPOINT}/config.json" || ! -s "${CHECKPOINT}/model.safetensors.index.json" ]]; then
  echo "TextCraft checkpoint not found in ${CHECKPOINT}. Run: python ${REPOSITORY}/Scripts/Download/download.py --checkpoint textcraft" >&2
  exit 1
fi

exec "${PYTHON:-python}" "${REPOSITORY}/Evaluation/run.py" --backend TextCraft --model-path "${CHECKPOINT}" "$@"
