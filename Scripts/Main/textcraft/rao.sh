#!/usr/bin/env bash
set -euo pipefail
# Paper method: RAO. Training source: 142; evaluated checkpoint project: 110.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/../../../Training" && pwd)
exec "${PYTHON:-python}" "${TRAINING_DIR}/launch.py" \
  --environment textcraft --experiment rao \
  --set seed=343 \
  --set total_train_steps=250 \
  --set 'execution_reward="rao"' \
  --set 'stage_schedule="execution:1"' \
  --set stage_cycles=-1 \
  --set rollout.max_concurrent_rollouts=8 \
  --set rollout.max_head_offpolicyness=3 \
  "$@"
