#!/usr/bin/env bash
set -euo pipefail
# Paper ablation: Generator=Policy (global), Scorer=Policy, Training=--. Source: 4.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/../../../Training" && pwd)
exec "${PYTHON:-python}" "${TRAINING_DIR}/launch.py" \
  --environment textcraft --experiment global_rubric_policy_scorer \
  --set seed=343 \
  --set total_train_steps=250 \
  --set 'execution_reward="rubric"' \
  --set 'rubric_scope="global"' \
  --set 'scorer_provider="policy"' \
  --set 'global_rubric_template_path="assets/textcraft_global_rubric.json"' \
  --set 'stage_schedule="execution:1"' \
  --set stage_cycles=-1 \
  --set rollout.max_concurrent_rollouts=8 \
  --set rollout.max_head_offpolicyness=3 \
  --set rubric_subagent_reward.enabled=true \
  --set 'rubric_subagent_reward.provider="policy"' \
  --set 'rubric_subagent_reward.model="active-policy-snapshot"' \
  --set 'rubric_subagent_reward.endpoint=""' \
  --set 'rubric_subagent_reward.api_key_env="KIMI_API_KEY"' \
  --set rubric_subagent_reward.max_concurrency=16 \
  --set rubric_subagent_reward.max_retries=5 \
  --set rubric_subagent_reward.timeout_seconds=1800 \
  --set rubric_subagent_reward.rubric_temperature=1.0 \
  --set rubric_subagent_reward.scoring_temperature=0.0 \
  --set rubric_subagent_reward.max_rubric_tokens=1024 \
  --set rubric_subagent_reward.max_scoring_tokens=1024 \
  --set rubric_subagent_reward.min_rubric_criteria=2 \
  --set rubric_subagent_reward.cache=true \
  --set 'rubric_subagent_reward.cache_dir="global_rubric_reward_cache"' \
  --set 'rubric_subagent_reward.artifact_dir="global_rubric_reward_artifacts"' \
  --set 'rubric_subagent_reward.failure_policy="drop_trajectory"' \
  --set 'rubric_subagent_reward.extra_body.reasoning_effort="none"' \
  "$@"
