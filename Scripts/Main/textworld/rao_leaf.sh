#!/usr/bin/env bash
set -euo pipefail
# Paper method: RAO + D (Decomposition reward). Source: 86.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/../../../Training" && pwd)
exec "${PYTHON:-python}" "${TRAINING_DIR}/launch.py" \
  --environment textworld --experiment rao_leaf \
  --set seed=367 \
  --set 'execution_reward="rao"' \
  --set 'stage_schedule="execution:12,delegation:4"' \
  --set stage_cycles=-1 \
  --set rollout.max_concurrent_rollouts=32 \
  --set rollout.max_head_offpolicyness=0 \
  --set leaf_credit.filter_zero_variance_groups=false \
  --set leaf_credit.invalid_delegation_reward=0.0 \
  --set 'leaf_credit.launch_advantage_mode="reward_first_root_balanced"' \
  --set 'leaf_credit.launch_credit_mode="leaf"' \
  --set leaf_credit.subagent_success_gate=false \
  --set leaf_credit.workload_weight_cap=1.0 \
  --set 'leaf_credit.output_dir="leaf_reward_artifacts"' \
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
