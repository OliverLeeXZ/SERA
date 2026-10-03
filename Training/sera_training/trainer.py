"""Build environment/reward-specific workflows around one common stage router."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import os
from pathlib import Path

from areal.api.cli_args import load_expr_config
from datasets import Dataset
from rubric_subagent_reward import RubricSubagentRewardConfig, RubricSubagentRewardProcessor
from rubric_generation_ranking_textworld.config import RubricGenerationRankingConfig
from three_stage_train.config import OptimizationConfig, RubricGenerationConfig, StageScheduleItem, ThreeStageTrainConfig
from two_stage_rao_leaf.config import TwoStageRaoLeafConfig

from .config import TrainingConfig, JudgeConfig
from .stage_kernel import StageSchedule
from .stage_workflow import SharedStageWorkflow
from .rubric_clients import create_rubric_client, prepare_rubric_config
from .external_models import preflight_training






def normalize(config):
    if config.environment not in {"textcraft", "textworld"} or config.execution_reward not in {"rao", "rubric"}:
        raise ValueError("Unknown environment or execution reward")
    for key, cls in (("rubric_subagent_reward", RubricSubagentRewardConfig),
                     ("rubric_generation_ranking", RubricGenerationRankingConfig),
                     ("leaf_credit", TwoStageRaoLeafConfig), ("judge", JudgeConfig)):
        value = getattr(config, key)
        if isinstance(value, Mapping):
            setattr(config, key, cls(**dict(value)))
    if not config.depth_aware or config.step_credit_enabled or config.reward_mode != "rao":
        raise ValueError("These experiments use depth-aware recursion and unmodified binary root reward")
    schedule = StageSchedule.parse(config.stage_schedule, config.train_dataset.batch_size, config.stage_cycles)
    if "execution" not in {item.stage for item in schedule.schedule}:
        raise ValueError("An execution stage is required")
    if len(schedule.schedule) > 1:
        # Preserve the exact-version rule used by the source two/three-stage runs.
        config.rollout.max_head_offpolicyness = 0
    if config.execution_reward == "rubric":
        prepare_rubric_config(config)
    if config.rubric_generation_ranking.objective == "order_discrimination":
        if config.environment != "textcraft" or "rubric_generation" not in {item.stage for item in schedule.schedule}:
            raise ValueError("The order/discrimination ablation requires TextCraft rubric-generation training")
        config.rubric_generation_ranking.validate()
    leaf = config.leaf_credit
    leaf.environment = config.environment
    leaf.root_group_size = config.workflow_config.group_size
    leaf.max_subagent_depth = config.workflow_config.rollout_config.max_subagent_depth
    leaf.depth_level_weighting = config.workflow_config.depth_level_weighting
    leaf.depth_level_discount_gamma = config.workflow_config.depth_level_discount_gamma
    for obj, fields in ((leaf, ("output_dir",)),
                        (config.rubric_subagent_reward, ("cache_dir", "artifact_dir")),
                        (config.rubric_generation_ranking, ("output_dir",))):
        for name in fields:
            value = Path(getattr(obj, name))
            if not value.is_absolute():
                setattr(obj, name, str(Path(config.cluster.fileroot) / value))
    if schedule.finite_total_steps is not None:
        config.total_train_steps = schedule.finite_total_steps
        size = len(datasets(config)[0])
        steps_per_epoch = max(1, math.ceil(size / config.train_dataset.batch_size))
        config.total_train_epochs = max(config.total_train_epochs, math.ceil(config.total_train_steps / steps_per_epoch))
    return schedule


def stage_config(config, stage):
    if stage == "delegation":
        result = config.leaf_credit.build_leaf_stage_config()
        # One processor only: stage switching is exclusively the outer shared router.
        result.schedule = [StageScheduleItem("delegation", 1, config.workflow_config.group_size)]
        return result
    rank = config.rubric_generation_ranking
    return ThreeStageTrainConfig(
        environment=config.environment, root_group_size=config.workflow_config.group_size,
        max_subagent_depth=config.workflow_config.rollout_config.max_subagent_depth,
        max_policy_concurrency=rank.max_policy_concurrency, cycles=-1,
        schedule=[StageScheduleItem("rubric_generation", 1, config.workflow_config.group_size)],
        rubric_generation=RubricGenerationConfig(
            branching_factor=rank.branching_factor, max_counterfactual_envs_per_rollout=rank.max_counterfactual_envs_per_rollout,
            commit_policy="branch_0", num_rubric_criteria_min=rank.min_rubric_criteria,
            rubric_temperature=rank.rubric_temperature, scoring_temperature=rank.scoring_temperature,
            max_rubric_tokens=rank.max_rubric_tokens, max_scoring_tokens=rank.max_scoring_tokens),
        optimization=OptimizationConfig(depth_level_weighting=config.workflow_config.depth_level_weighting,
                                        depth_level_discount_gamma=config.workflow_config.depth_level_discount_gamma,
                                        filter_zero_variance_groups=False, checkpoint_at_stage_boundary=False),
        output_dir=rank.output_dir)


def external_endpoint(config):
    endpoint = config.judge.endpoint or os.getenv("JUDGE_API_URL", "")
    key = os.getenv(config.judge.api_key_env, "")
    if not endpoint or not key:
        raise ValueError("This TextWorld objective needs JUDGE_API_URL and the judge API key on every trainer worker")
    return endpoint, key


def build_workflows(config, trainer, schedule):
    from platoon.train.areal.workflows import StepWiseArealWorkflow
    from rubric_subagent_reward.workflow import RubricRewardArealWorkflow
    from three_stage_train.workflow import ThreeStageArealWorkflow
    stages = {item.stage for item in schedule.schedule}
    is_craft = config.environment == "textcraft"
    if is_craft:
        from platoon.textcraft.synth_tasks import get_synth_task as task_loader
        from platoon.textcraft.synth_rollout import run_synth_depth_aware_rollout as rollout
        from platoon.textcraft.reward import build_reward_processor
        from three_stage_train.adapters.textcraft import TextCraftThreeStageAdapter
        from three_stage_train.textcraft_rollout import run_textcraft_three_stage_rollout as ranking_rollout
        from reward_first_launch.stage import RewardFirstLaunchDelegationStage as LeafStage
        from rubric_generation_ranking.stage import RubricGenerationRankingStage as RankStage
        reward = build_reward_processor(config)
        adapter = TextCraftThreeStageAdapter()
        execution_class = RubricRewardArealWorkflow
        binary_class = StepWiseArealWorkflow
        binary_kwargs = {}
        leaf_rollout = ranking_rollout
    else:
        from textworld_continuous_policy.tasks import get_textworld_task as task_loader
        from textworld_continuous_policy.rollout import run_textworld_depth_aware_rollout as rollout
        from textworld_continuous_policy.reward import textworld_root_reward as reward
        from textworld_continuous_policy.adapter import TextWorldContinuousPolicyAdapter
        from textworld_continuous_policy.workflow import TextWorldRubricRewardArealWorkflow
        from textworld_rao_policy_judge.rollout import run_textworld_rao_policy_judge_rollout
        from textworld_rao_policy_judge.reward import textworld_rao_reward
        from textworld_rao_policy_judge.workflow import TextWorldRAOPolicyJudgeWorkflow
        from textworld_rao_policy_judge.judge import TextWorldKimiJudge
        from textworld_two_stage.adapter import TextWorldDelegationAdapter
        from textworld_two_stage.leaf_stage import TextWorldRewardFirstLaunchDelegationStage as LeafStage
        from rubric_generation_ranking_textworld.stage import RubricGenerationRankingStage as RankStage
        from textworld_two_stage_rollout import run_textworld_two_stage_rollout as ranking_rollout
        adapter = TextWorldContinuousPolicyAdapter()
        execution_class = TextWorldRubricRewardArealWorkflow
        binary_class = TextWorldRAOPolicyJudgeWorkflow
        binary_kwargs = {}
        # Leaf coverage depends on the verified root outcome and tree topology,
        # not on binary judgments of the child trajectories.
        leaf_rollout = rollout
        if config.execution_reward == "rao":
            endpoint, key = external_endpoint(config)
            judge = TextWorldKimiJudge(model_name=os.getenv("JUDGE_MODEL", config.judge.model), endpoint=endpoint, api_key=key,
                                      max_concurrency=config.judge.max_concurrency, max_prompt_tokens=config.judge.context_length,
                                      max_completion_tokens=config.judge.max_completion_tokens, temperature=config.judge.temperature,
                                      request_timeout_seconds=config.judge.request_timeout_seconds,
                                      artifact_dir=str(Path(config.cluster.fileroot) / "binary_judge_artifacts"))
            binary_kwargs["policy_judge"] = judge

    shared = dict(config=config.workflow_config, proxy_server=trainer.proxy_server,
                  stats_scope="train_rollout", device=trainer.actor.device, filter_errors=is_craft, reward_processor=reward)
    workflows = {}
    if config.execution_reward == "rubric":
        rubric_client = create_rubric_client(config, trainer.proxy_server)
        processor = RubricSubagentRewardProcessor(config.rubric_subagent_reward, adapter=adapter, client=rubric_client)
        workflows["execution"] = execution_class(rollout, task_loader, **shared, rubric_processor=processor)
    else:
        binary_shared = dict(shared)
        if not is_craft:
            binary_shared["reward_processor"] = textworld_rao_reward
            execution_rollout = run_textworld_rao_policy_judge_rollout
        else:
            execution_rollout = rollout
        workflows["execution"] = binary_class(execution_rollout, task_loader, **binary_shared, **binary_kwargs)

    if "delegation" in stages:
        leaf_adapter = adapter if is_craft else TextWorldDelegationAdapter()
        leaf_config = stage_config(config, "delegation")
        workflow = ThreeStageArealWorkflow(leaf_rollout, task_loader, config.workflow_config, leaf_config,
                                           trainer.proxy_server, "train_rollout", trainer.actor.device, leaf_adapter)
        workflow.stage_processors["delegation"] = LeafStage(leaf_config, leaf_adapter, config.leaf_credit)
        workflows["delegation"] = workflow

    if "rubric_generation" in stages:
        rank = config.rubric_generation_ranking
        rank.validate()
        if rank.objective == "order_discrimination":
            from rubric_generation_discrimination.stage import RubricGenerationDiscriminationStage
            RankStage = RubricGenerationDiscriminationStage
        rank_config = stage_config(config, "rubric_generation")
        workflow = ThreeStageArealWorkflow(ranking_rollout, task_loader, config.workflow_config, rank_config,
                                           trainer.proxy_server, "train_rollout", trainer.actor.device, adapter)
        judge_args = ()
        if not is_craft:
            from kimi_judge import KimiJudgeClient
            endpoint = rank.judge_endpoint or external_endpoint(config)[0]
            key = os.getenv(rank.judge_api_key_env, "")
            judge_args = (KimiJudgeClient(endpoint=endpoint, api_key=key, model=os.getenv("JUDGE_MODEL", rank.judge_model),
                                          max_concurrency=rank.judge_max_concurrency, max_prompt_tokens=rank.judge_max_prompt_tokens,
                                          max_completion_tokens=rank.judge_max_completion_tokens, temperature=rank.judge_temperature,
                                          timeout_seconds=rank.judge_timeout_seconds, max_retries=rank.judge_max_retries),)
        workflow.stage_processors["rubric_generation"] = RankStage(rank_config, adapter, rank, *judge_args)
        workflows["rubric_generation"] = workflow

    evaluation = deepcopy(config.workflow_config)
    evaluation.group_size = 1
    if not is_craft:
        evaluation.rollout_config.inference_params.temperature = 0.0
        evaluation.rollout_config.inference_params.max_prompt_tokens = 10240
        evaluation.rollout_config.inference_params.max_completion_tokens = 3072
    eval_workflow = StepWiseArealWorkflow(rollout, task_loader, evaluation, trainer.eval_proxy_server,
                                         "eval_rollout", trainer.actor.device, filter_errors=False,
                                         reward_processor=reward)
    return SharedStageWorkflow(schedule, workflows, audit_dir=Path(config.cluster.fileroot) / "stage_routes"), eval_workflow


def datasets(config):
    if config.environment == "textcraft":
        from platoon.textcraft.synth_tasks import get_synth_task_ids, get_synth_task_ids_by_difficulty, Difficulty
        def get_filtered_task_ids(split, difficulties, **counts):
            if not difficulties:
                return get_synth_task_ids(split, **counts)
            return [task_id for difficulty in difficulties for task_id in
                    get_synth_task_ids_by_difficulty(split, Difficulty(difficulty), **counts)]
        train_ids = get_filtered_task_ids("train", config.train_difficulties, num_samples_train=2522)
        val_ids = get_filtered_task_ids("val", config.eval_difficulties, num_samples_val=632)[:100]
    else:
        from textworld_continuous_policy.tasks import get_textworld_task_ids
        train_ids, val_ids = get_textworld_task_ids("train"), get_textworld_task_ids("dev")
    return Dataset.from_list([{"task_id": x} for x in train_ids]), Dataset.from_list([{"task_id": x} for x in val_ids])


def main(args):
    config, _ = load_expr_config(args, TrainingConfig)
    # Every trainer process checks its own credentials/network before GPU initialization.
    preflight_training(config)
    schedule = normalize(config)
    from platoon.train.areal import PlatoonArealRLTrainer
    from safety.sync_safe_batch import (install_schema_safe_trajectory_concat,
                                        install_schema_safe_distributed_rollout, install_sync_safe_prepare_batch)
    install_schema_safe_trajectory_concat()
    install_schema_safe_distributed_rollout()
    class SafeTrainer(PlatoonArealRLTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_sync_safe_prepare_batch(self)
    train, validation = datasets(config)
    with SafeTrainer(config=config, train_dataset=train, val_dataset=validation) as trainer:
        workflow, eval_workflow = build_workflows(config, trainer, schedule)
        trainer.train(workflow=workflow, eval_workflow=eval_workflow)
