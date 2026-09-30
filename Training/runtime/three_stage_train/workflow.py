from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.proxy import ProxyServer
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors

from platoon.envs.base import Task
from platoon.train.areal.config_defs import WorkflowConfig
from platoon.train.areal.proxy import ArealProxySession
from platoon.train.areal.workflows.timeouts import (
    RolloutHardTimeout,
    await_with_hard_timeout,
)

from .adapters.base import ThreeStageEnvironmentAdapter
from .config import ThreeStageTrainConfig
from .policy_client import ArealPolicyClient
from .runtime_context import (
    active_config,
    active_stage,
    root_rollout_index,
    shared_environment_budget,
)
from .schedule import StageController, StagePosition
from .stages import DelegationStage, RubricGenerationStage, SubagentExecutionStage
from .stages.common import RawRollout, StageBatchResult, root_reward


class ThreeStageArealWorkflow(RolloutWorkflow):
    """AReaL workflow whose accepted batches correspond to real optimizer updates."""

    def __init__(
        self,
        rollout_fn: Callable[[Task, Any], Any],
        get_task_fn: Callable[[str], Task],
        config: WorkflowConfig,
        three_stage_config: ThreeStageTrainConfig,
        proxy_server: ProxyServer,
        stats_scope: str,
        device: torch.device,
        adapter: ThreeStageEnvironmentAdapter,
    ) -> None:
        self.config = deepcopy(config)
        self.config.group_size = three_stage_config.root_group_size
        self.config.rollout_config.return_dict = True
        self.config.rollout_config.train = True
        self.three_stage_config = three_stage_config
        self.three_stage_config.validate()
        self.proxy_server = proxy_server
        self.proxy_url = f"{proxy_server.public_addr}/v1"
        self.stats_scope = stats_scope
        self.device = device
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn
        self.adapter = adapter
        self.controller = StageController(three_stage_config)
        self.policy_client = ArealPolicyClient(
            proxy_server=proxy_server,
            model_name=self.config.rollout_config.model_name,
            max_concurrency=three_stage_config.max_policy_concurrency,
            max_prompt_tokens=(
                self.config.rollout_config.inference_params.max_prompt_tokens
            ),
            default_completion_tokens=(
                self.config.rollout_config.inference_params.max_completion_tokens
            ),
        )
        self.stage_processors = {
            "subagent_execution": SubagentExecutionStage(
                three_stage_config, adapter
            ),
            "delegation": DelegationStage(three_stage_config, adapter),
            "rubric_generation": RubricGenerationStage(
                three_stage_config, adapter
            ),
        }
        self.output_dir = Path(three_stage_config.output_dir)
        self._artifact_lock = asyncio.Lock()
        self.config.rollout_config.output_dir = os.path.join(
            self.config.rollout_config.output_dir,
            self.stats_scope,
        )

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> dict[str, torch.Tensor] | None:
        task_id = data["task_id"]
        timeout_seconds = float(
            os.environ.get(
                "R3AO_ROLLOUT_GROUP_HARD_TIMEOUT_SECONDS",
                self.config.rollout_config.timeout or 900,
            )
        )
        cancellation_grace = float(
            os.environ.get("R3AO_ROLLOUT_CANCEL_GRACE_SECONDS", "30")
        )
        try:
            return await await_with_hard_timeout(
                self._arun_episode_impl(engine, data),
                timeout_seconds=timeout_seconds,
                cancellation_grace_seconds=cancellation_grace,
                label=f"task={task_id} scope={self.stats_scope}",
            )
        except RolloutHardTimeout as exc:
            stats_tracker.get(self.stats_scope).scalar(
                rollout_group_hard_timeout=1.0
            )
            print(f"[ThreeStageWorkflow] {exc}")
            return None

    async def _arun_episode_impl(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> dict[str, torch.Tensor] | None:
        global_step = int(engine.get_version())
        position = self.controller.position(global_step)
        raw_results = await asyncio.gather(
            *[
                self._run_raw_rollout(
                    engine,
                    data,
                    rollout_index,
                    position,
                )
                for rollout_index in range(self.config.group_size)
            ]
        )
        raw_rollouts = [result for result in raw_results if result is not None]
        if len(raw_rollouts) < 2:
            result = StageBatchResult(
                metrics={"workflow/skipped_insufficient_root_rollouts": 1.0}
            )
            return await self._finalize_result(
                data["task_id"], position, result, raw_rollouts
            )

        processor = self.stage_processors[position.stage]
        result = await processor.process(raw_rollouts, self.policy_client)
        return await self._finalize_result(
            data["task_id"], position, result, raw_rollouts
        )

    async def _finalize_result(
        self,
        task_id: str,
        position: StagePosition,
        result: StageBatchResult,
        raw_rollouts: list[RawRollout],
    ) -> dict[str, torch.Tensor] | None:
        train_data: dict[str, torch.Tensor] | None = None
        trainable = True
        if result.datums:
            candidate = concat_padded_tensors(result.datums)
            rewards = candidate["rewards"]
            if rewards.numel() == 0:
                result.metrics["workflow/skipped_empty_reward_tensor"] = 1.0
            elif rewards.max() == rewards.min():
                result.metrics["workflow/zero_variance_reward_group"] = 1.0
                if not self.three_stage_config.optimization.filter_zero_variance_groups:
                    train_data = candidate
                elif position.stage == "rubric_generation":
                    train_data = candidate
                    trainable = False
            else:
                train_data = candidate

        # Keep zero-gradient Rubric tasks in the sampled batch so AReaL does
        # not prefetch replacement tasks. The trainer removes these datums
        # before the optimizer update.
        if train_data is None and position.stage == "rubric_generation":
            train_data = self._rubric_noop_batch()
            trainable = False
            result.metrics["workflow/noop_zero_gradient_step"] = 1.0

        result.metrics["workflow/sampled_tasks"] = 1.0
        result.metrics["workflow/skipped_tasks"] = float(not trainable)
        await self._write_artifact(
            task_id, position, result, raw_rollouts
        )
        self._record_stage_metrics(position, result, raw_rollouts)
        if train_data is None:
            return None

        train_data["task_reward"] = torch.tensor(
            [root_reward(raw.collection) for raw in raw_rollouts],
            dtype=torch.float32,
        )
        train_data["trainable_datums"] = torch.full_like(
            train_data["rewards"], trainable, dtype=torch.bool
        )
        return train_data

    @staticmethod
    def _rubric_noop_batch() -> dict[str, torch.Tensor]:
        count = 1
        return {
            "input_ids": torch.zeros((count, 1), dtype=torch.long),
            "loss_mask": torch.zeros((count, 1), dtype=torch.long),
            "logprobs": torch.zeros((count, 1), dtype=torch.float32),
            "versions": torch.full((count, 1), -1, dtype=torch.long),
            "attention_mask": torch.ones((count, 1), dtype=torch.bool),
            "num_input_tokens": torch.zeros(count, dtype=torch.float32),
            "num_output_tokens": torch.zeros(count, dtype=torch.float32),
            "num_steps": torch.zeros(count, dtype=torch.float32),
            "rewards": torch.zeros(count, dtype=torch.float32),
            "token_rewards": torch.zeros((count, 1), dtype=torch.float32),
            "traj_depth": torch.zeros(count, dtype=torch.float32),
            "traj_start": torch.ones(count, dtype=torch.float32),
        }

    async def _run_raw_rollout(
        self,
        engine: InferenceEngine,
        data: dict,
        rollout_index: int,
        position: StagePosition,
    ) -> RawRollout | None:
        config = deepcopy(self.config)
        task_id = data["task_id"]
        stage_token = active_stage.set(position.stage)
        config_token = active_config.set(self.three_stage_config)
        rollout_token = root_rollout_index.set(rollout_index)
        budget_token = shared_environment_budget.set(None)
        try:
            task = self.get_task_fn(task_id)
            if config.rollout_config.max_steps is not None:
                task.max_steps = config.rollout_config.max_steps
            async with ArealProxySession(base_url=self.proxy_url) as session:
                config.rollout_config.model_endpoint = session.session_base_url
                config.rollout_config.model_name = (
                    "openai/" + config.rollout_config.model_name
                )
                config.rollout_config.model_api_key = "None"
                config.rollout_config.output_dir = os.path.join(
                    config.rollout_config.output_dir,
                    str(engine.get_version()),
                )
                collection = await asyncio.create_task(
                    self.rollout_fn(task, config.rollout_config)
                )
                completions = dict(
                    self.proxy_server.session_cache[
                        session.session_id
                    ].completions
                )
                if not collection or not collection.get("trajectories"):
                    return None
                return RawRollout(
                    rollout_index=rollout_index,
                    collection=collection,
                    completions=completions,
                )
        except Exception as exc:
            print(
                f"[ThreeStageWorkflow] rollout failed for task={task_id} "
                f"rollout={rollout_index}: {type(exc).__name__}: {exc}"
            )
            return None
        finally:
            shared_environment_budget.reset(budget_token)
            root_rollout_index.reset(rollout_token)
            active_config.reset(config_token)
            active_stage.reset(stage_token)

    def _record_stage_metrics(
        self,
        position: StagePosition,
        result: StageBatchResult,
        raw_rollouts: list[RawRollout],
    ) -> None:
        tracker = stats_tracker.get(self.stats_scope)
        for key, value in result.metrics.items():
            tracker.scalar(**{key: float(value)})
        tracker.scalar(
            **{
                "three_stage/global_step": float(position.global_step),
                "three_stage/cycle_index": float(position.cycle_index),
                "three_stage/schedule_index": float(position.schedule_index),
                "three_stage/step_in_stage": float(position.step_in_stage),
                "three_stage/root_rollouts": float(len(raw_rollouts)),
                "three_stage/stage_batch_size": float(position.batch_size),
            }
        )
        if not raw_rollouts:
            return
        task_rewards = torch.tensor(
            [root_reward(raw.collection) for raw in raw_rollouts],
            dtype=torch.float32,
            device=self.device,
        )
        task_reward_mask = torch.ones_like(task_rewards, dtype=torch.bool)
        tracker.denominator(task_reward_mask=task_reward_mask)
        tracker.stat(
            task_reward=task_rewards,
            denominator="task_reward_mask",
        )

    async def _write_artifact(
        self,
        task_id: str,
        position: StagePosition,
        result: StageBatchResult,
        raw_rollouts: list[RawRollout],
    ) -> None:
        counterfactual_counts = [
            int((raw.collection.get("_three_stage") or {}).get(
                "counterfactual_environment_count", 0
            ))
            for raw in raw_rollouts
        ]
        complete_fork_groups = sum(
            1
            for raw in raw_rollouts
            for group in (raw.collection.get("_three_stage") or {}).get(
                "fork_groups", []
            )
            if group.get("complete")
        )
        payload = {
            "task_id": task_id,
            "global_step": position.global_step,
            "cycle_index": position.cycle_index,
            "schedule_index": position.schedule_index,
            "stage": position.stage,
            "step_in_stage": position.step_in_stage,
            "stage_steps": position.stage_steps,
            "stage_batch_size": position.batch_size,
            "root_rollouts": len(raw_rollouts),
            "root_rewards": [
                root_reward(raw.collection) for raw in raw_rollouts
            ],
            "counterfactual_environments": (
                sum(counterfactual_counts)
            ),
            "counterfactual_environments_per_rollout": counterfactual_counts,
            "max_counterfactual_environments_per_rollout": (
                max(counterfactual_counts) if counterfactual_counts else 0
            ),
            "complete_fork_groups": (
                complete_fork_groups
            ),
            "metrics": result.metrics,
            "records": result.records,
        }
        async with self._artifact_lock:
            records_dir = self.output_dir / "stage_records"
            records_dir.mkdir(parents=True, exist_ok=True)
            path = records_dir / f"records-{os.getpid()}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.controller.write_state(
                self.output_dir / "schedule_state.json",
                position.global_step,
            )
