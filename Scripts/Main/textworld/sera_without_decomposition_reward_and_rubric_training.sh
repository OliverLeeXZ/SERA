#!/usr/bin/env bash
set -euo pipefail
# Paper method: Self-Evaluating Recursive Agents without Decomposition Reward and Rubric Training. Training/evaluated-checkpoint source: 82.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/../../../Training" && pwd)
exec "${PYTHON:-python}" "${TRAINING_DIR}/launch.py" \
  --environment textworld --experiment sera_without_decomposition_reward_and_rubric_training \
  --set seed=362 \
  --set 'execution_reward="rubric"' \
  --set 'stage_schedule="execution:1"' \
  --set stage_cycles=-1 \
  --set rollout.max_concurrent_rollouts=32 \
  --set rollout.max_head_offpolicyness=0 \
  --set rubric_subagent_reward.enabled=true \
  --set 'rubric_subagent_reward.provider="policy"' \
  --set 'rubric_subagent_reward.model="active-policy-snapshot"' \
  --set rubric_subagent_reward.max_concurrency=32 \
  --set rubric_subagent_reward.max_retries=5 \
  --set rubric_subagent_reward.timeout_seconds=1800 \
  --set rubric_subagent_reward.rubric_temperature=1.0 \
  --set rubric_subagent_reward.scoring_temperature=0.0 \
  --set rubric_subagent_reward.max_rubric_tokens=1024 \
  --set rubric_subagent_reward.max_scoring_tokens=1024 \
  --set rubric_subagent_reward.min_rubric_criteria=2 \
  --set rubric_subagent_reward.cache=true \
  --set 'rubric_subagent_reward.cache_dir="rubric_reward_cache"' \
  --set 'rubric_subagent_reward.artifact_dir="rubric_reward_artifacts"' \
  --set 'rubric_subagent_reward.failure_policy="drop_trajectory"' \
  --set 'rubric_subagent_reward.extra_body.reasoning_effort="none"' \
  --set 'rubric_subagent_reward.endpoint=""' \
  "$@"
