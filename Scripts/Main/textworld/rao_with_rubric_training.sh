#!/usr/bin/env bash
set -euo pipefail
# Paper method: Recursive Agent Optimization with Rubric Training. Source: 172.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/../../../Training" && pwd)
exec "${PYTHON:-python}" "${TRAINING_DIR}/launch.py" \
  --environment textworld --experiment rao_with_rubric_training \
  --set seed=362 \
  --set 'execution_reward="rao"' \
  --set 'stage_schedule="execution:16,rubric_generation:2"' \
  --set stage_cycles=-1 \
  --set rollout.max_concurrent_rollouts=24 \
  --set rollout.max_head_offpolicyness=0 \
  --set rubric_generation_ranking.branching_factor=8 \
  --set rubric_generation_ranking.max_counterfactual_envs_per_rollout=16 \
  --set rubric_generation_ranking.margin=0.2 \
  --set rubric_generation_ranking.rubric_temperature=1.0 \
  --set rubric_generation_ranking.scoring_temperature=0.0 \
  --set rubric_generation_ranking.max_rubric_tokens=1024 \
  --set rubric_generation_ranking.max_scoring_tokens=512 \
  --set rubric_generation_ranking.min_rubric_criteria=2 \
  --set rubric_generation_ranking.max_policy_concurrency=24 \
  --set 'rubric_generation_ranking.output_dir="rubric_generation_ranking_artifacts"' \
  --set 'rubric_generation_ranking.judge_model="kimi-k2.6"' \
  --set 'rubric_generation_ranking.judge_endpoint=""' \
  --set 'rubric_generation_ranking.judge_api_key_env="KIMI_API_KEY"' \
  --set rubric_generation_ranking.judge_max_concurrency=16 \
  --set rubric_generation_ranking.judge_max_prompt_tokens=10240 \
  --set rubric_generation_ranking.judge_max_completion_tokens=1024 \
  --set rubric_generation_ranking.judge_temperature=1.0 \
  --set rubric_generation_ranking.judge_timeout_seconds=1800 \
  --set rubric_generation_ranking.judge_max_retries=2 \
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
