#!/usr/bin/env bash
set -euo pipefail
# Paper method: RAO. Training/evaluated-checkpoint source: 85.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/../../../Training" && pwd)
exec "${PYTHON:-python}" "${TRAINING_DIR}/launch.py" \
  --environment textworld --experiment rao \
  --set seed=367 \
  --set 'execution_reward="rao"' \
  --set 'stage_schedule="execution:1"' \
  --set stage_cycles=-1 \
  --set rollout.max_concurrent_rollouts=32 \
  --set rollout.max_head_offpolicyness=0 \
  --set 'judge.model="kimi-k2.6"' \
  --set 'judge.endpoint=""' \
  --set 'judge.api_key_env="KIMI_API_KEY"' \
  --set judge.context_length=10240 \
  --set judge.max_completion_tokens=1024 \
  --set judge.temperature=1.0 \
  --set judge.max_concurrency=16 \
  --set judge.request_timeout_seconds=1800 \
  --set delegation_lambda=0.0 \
  "$@"
