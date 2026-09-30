from __future__ import annotations

import asyncio
import os
import traceback
from copy import deepcopy
from typing import Any

from areal.api.engine_api import InferenceEngine
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors

from platoon.episode.context import subagent_launcher_override
from platoon.train.areal.proxy import ArealProxySession
from platoon.train.areal.workflows import StepWiseArealWorkflow
from platoon.utils.areal_data_processing import get_train_data_for_trajectory_collection

from .async_pipeline import AsyncRubricRolloutCoordinator
from .processor import RubricSubagentRewardProcessor


class RubricRewardArealWorkflow(StepWiseArealWorkflow):
    """RAO workflow with delegation-time asynchronous rubric scoring."""

    def __init__(
        self,
        *args: Any,
        rubric_processor: RubricSubagentRewardProcessor,
        **kwargs: Any,
    ) -> None:
        self.rubric_processor = rubric_processor
        kwargs.pop("group_trajectory_processor", None)
        super().__init__(
            *args,
            group_trajectory_processor=self._unused_group_processor,
            **kwargs,
        )
        if self.config.use_subprocesses:
            raise ValueError(
                "Delegation-time rubric scoring requires "
                "workflow_config.use_subprocesses=false"
            )

    @staticmethod
    async def _unused_group_processor(
        trajectories: list[dict], task_id: str
    ) -> list[dict]:
        del task_id
        return trajectories

    async def _arun_episode_raw(
        self,
        engine: InferenceEngine,
        data: dict,
        rollout_number: int,
    ) -> tuple[dict | None, dict, dict[str, float]] | None:
        config = deepcopy(self.config)
        task_id = data["task_id"]
        coordinator = AsyncRubricRolloutCoordinator(
            processor=self.rubric_processor,
            task_id=task_id,
            rollout_index=rollout_number,
        )
        launcher_token = subagent_launcher_override.set(
            coordinator.launch_subagent
        )
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
                trajectory_data = await asyncio.create_task(
                    self.rollout_fn(task, config.rollout_config)
                )
                completions = dict(
                    self.proxy_server.session_cache[
                        session.session_id
                    ].completions
                )

            if not isinstance(trajectory_data, dict):
                trajectory_data = trajectory_data.to_dict()
            processed = await coordinator.finalize(trajectory_data)
            return (
                processed.collections.get(rollout_number),
                completions,
                processed.metrics,
            )
        except asyncio.CancelledError:
            await coordinator.cancel()
            raise
        except Exception as exc:
            await coordinator.cancel()
            print(
                f"[RubricRewardWorkflow] Error in raw rollout for task "
                f"{task_id} and rollout {rollout_number}: {exc}"
            )
            traceback.print_exc()
            return None
        finally:
            subagent_launcher_override.reset(launcher_token)

    def _process_trajectory_result(
        self,
        trajectory_data: dict | None,
        session: ArealProxySession | None,
        task_id: str,
        rollout_number: int,
        *,
        completions: dict | None = None,
    ) -> dict | None:
        """Convert a scored raw rollout into AReaL training data.

        Rubric scoring is finalized after the proxy session closes, so the
        grouped path already has a copied completion map and intentionally
        passes ``session=None``.  The base RAO implementation only accepts a
        session and reads the cache itself, which is invalid for this path.
        """
        if trajectory_data is None:
            print(
                f"[RubricRewardWorkflow] Rollout {rollout_number} returned None "
                f"for task {task_id}"
            )
            return None
        if not trajectory_data.get("trajectories"):
            print(
                f"[RubricRewardWorkflow] No trajectories for task {task_id}, "
                f"rollout {rollout_number}"
            )
            return None

        if completions is None:
            if session is None:
                raise ValueError(
                    "Either completions or an active ArealProxySession is required"
                )
            completions = self.proxy_server.session_cache[
                session.session_id
            ].completions

        use_depth_weighting = self.config.depth_level_weighting
        use_depth_discount = self.config.depth_level_discount_gamma is not None
        train_data = get_train_data_for_trajectory_collection(
            trajectory_data,
            completions,
            task_id,
            self.filter_errors,
            self.reward_processor,
            self.merge_prefixes,
            concat_fn=concat_padded_tensors,
            include_traj_depth=use_depth_weighting or use_depth_discount,
            include_traj_start=use_depth_weighting,
        )
        if train_data is None:
            print(
                f"[RubricRewardWorkflow] No train data for task {task_id}, "
                f"rollout {rollout_number}"
            )
        return train_data

    async def _arun_episode_group(
        self, engine: InferenceEngine, data: dict
    ) -> list[dict | None]:
        raw_results = await asyncio.gather(
            *[
                self._arun_episode_raw(engine, data, rollout_number)
                for rollout_number in range(self.config.group_size)
            ]
        )
        results: list[dict | None] = [None] * self.config.group_size
        metric_values: dict[str, list[float]] = {}

        for rollout_number, raw_result in enumerate(raw_results):
            if raw_result is None:
                continue
            trajectory_data, completions, metrics = raw_result
            for key, value in metrics.items():
                metric_values.setdefault(key, []).append(float(value))
            if trajectory_data is None:
                continue
            results[rollout_number] = self._process_trajectory_result(
                trajectory_data,
                None,
                data["task_id"],
                rollout_number,
                completions=completions,
            )

        tracker = stats_tracker.get(self.stats_scope)
        additive_suffixes = (
            "root_rollouts",
            "expected_subagent_trajectories",
            "subagent_candidates",
            "valid_rubrics",
            "valid_scores",
            "judge_failed_root_rollouts",
            "skipped_root_rollouts",
            "dropped_trajectories",
            "async_rubrics_started",
            "async_scores_started",
        )
        for key, values in metric_values.items():
            aggregate = (
                sum(values)
                if key.endswith(additive_suffixes)
                else sum(values) / len(values)
            )
            tracker.scalar(**{key: float(aggregate)})
        return results
