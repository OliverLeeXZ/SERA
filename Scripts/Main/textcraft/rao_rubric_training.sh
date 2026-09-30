#!/usr/bin/env bash
set -euo pipefail
# Paper method: RAO + RT (Rubric Training). Source: 171.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/../../../Training" && pwd)
exec "${PYTHON:-python}" "${TRAINING_DIR}/launch.py" \
  --environment textcraft --experiment rao_rubric_training \
  --set seed=353 \
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
  --set rubric_generation_ranking.max_scoring_tokens=1024 \
  --set rubric_generation_ranking.min_rubric_criteria=2 \
  --set rubric_generation_ranking.max_policy_concurrency=24 \
  --set 'rubric_generation_ranking.output_dir="rubric_generation_ranking_artifacts"' \
  "$@"
