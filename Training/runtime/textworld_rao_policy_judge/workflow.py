from __future__ import annotations

from typing import Any

import torch
from areal.utils.data import concat_padded_tensors
from platoon.train.areal.workflows import StepWiseArealWorkflow
from platoon.utils.areal_data_processing import get_train_data_for_trajectory_collection

from .judge import (
    TextWorldPolicyJudge,
    active_policy_judge,
    mask_failed_judge_trajectories,
    set_active_policy_judge,
)


class TextWorldRAOPolicyJudgeWorkflow(StepWiseArealWorkflow):
    """TextWorld group/depth workflow with a binary Policy Judge."""

    def __init__(self, *args: Any, policy_judge: TextWorldPolicyJudge, **kwargs: Any):
        self.policy_judge = policy_judge
        super().__init__(*args, **kwargs)

    async def arun_episode(self, engine, data):
        """Keep zero-variance groups as real optimizer batches.

        The shared StepWise workflow returns a zero ``trainable_datums`` mask
        for an all-equal reward group even when
        ``filter_zero_variance_groups=false``.  AReaL then discards that
        batch, so a low-success TextWorld run can sample forever without
        advancing ``global_step``.  The project setting deliberately keeps
        these groups in the fixed-size batch; their centered advantages are
        zero, but the optimizer step must still be allowed to complete.
        """
        result = await super().arun_episode(engine, data)
        if (
            result is not None
            and not self.config.filter_zero_variance_groups
            and "rewards" in result
            and result["rewards"].numel() > 0
            and result["rewards"].max() == result["rewards"].min()
        ):
            result["trainable_datums"] = torch.ones_like(result["rewards"], dtype=torch.bool)
        return result

    async def _arun_episode_raw(self, engine, data, rollout_number):
        token = set_active_policy_judge(self.policy_judge)
        try:
            return await super()._arun_episode_raw(engine, data, rollout_number)
        finally:
            active_policy_judge.reset(token)

    async def _arun_episode_single(self, engine, data, rollout_number):
        """Make the Judge visible on the normal in-process rollout path.

        ``StepWiseArealWorkflow`` uses this path when subprocess rollouts and a
        group trajectory processor are disabled.  The raw-path hook above is
        therefore insufficient for the default 67 configuration.
        """
        token = set_active_policy_judge(self.policy_judge)
        try:
            return await super()._arun_episode_single(engine, data, rollout_number)
        finally:
            active_policy_judge.reset(token)

    def _process_trajectory_result(
        self,
        trajectory_data: dict | None,
        session: Any,
        task_id: str,
        rollout_number: int,
    ) -> dict | None:
        """Skip only Judge-request failures from PPO, keeping the full tree."""

        if trajectory_data is None:
            print(
                f"[TextWorldPolicyJudgeWorkflow] Rollout {rollout_number} "
                f"returned None for task {task_id}"
            )
            return None
        if not trajectory_data.get("trajectories"):
            print(
                f"[TextWorldPolicyJudgeWorkflow] No trajectories for task "
                f"{task_id}, rollout {rollout_number}"
            )
            return None

        completions = self.proxy_server.session_cache[session.session_id].completions
        config = self.config
        train_data = get_train_data_for_trajectory_collection(
            mask_failed_judge_trajectories(trajectory_data),
            completions,
            task_id,
            self.filter_errors,
            self.reward_processor,
            self.merge_prefixes,
            concat_fn=concat_padded_tensors,
            include_traj_depth=(
                config.depth_level_weighting
                or config.depth_level_discount_gamma is not None
            ),
            include_traj_start=config.depth_level_weighting,
        )
        if train_data is None:
            print(
                f"[TextWorldPolicyJudgeWorkflow] No train data for task "
                f"{task_id}, rollout {rollout_number}"
            )
        return train_data
