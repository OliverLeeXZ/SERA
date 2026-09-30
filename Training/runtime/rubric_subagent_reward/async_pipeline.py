from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, cast

from platoon.agents.base import ForkableAgent
from platoon.envs.base import ForkableEnv
from platoon.episode.context import (
    budget_tracker,
    current_agent,
    current_codeact_action,
    current_env,
    current_trajectory,
    current_trajectory_collection,
    episode_step_timeout,
)
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import BudgetExceededError, Trajectory
from platoon.train.areal.workflows.timeouts import cancel_tasks
from platoon.utils.span_profile import profile_span
from Runtime.rubric.prompts import (
    build_rubric_messages,
    build_scoring_messages,
)

from .client import ParsedGeneration
from .processor import RubricRewardBatchResult, RubricSubagentRewardProcessor


@dataclass
class _PendingAssessment:
    trajectory_id: str
    parent_trajectory_id: str
    parent_step_index: int
    depth: int
    child_goal: str
    rubric_messages: list[dict[str, str]]
    scoring_messages: list[dict[str, str]]
    rubric_result: ParsedGeneration
    score_result: ParsedGeneration | None


class AsyncRubricRolloutCoordinator:
    """Pipeline KIMI rubric work with one recursive Root rollout."""

    def __init__(
        self,
        *,
        processor: RubricSubagentRewardProcessor,
        task_id: str,
        rollout_index: int,
    ) -> None:
        self.processor = processor
        self.task_id = task_id
        self.rollout_index = rollout_index
        self._pending: list[asyncio.Task[_PendingAssessment]] = []
        self.rubric_requests_started = 0
        self.score_requests_started = 0

    async def launch_subagent(
        self,
        goal: str,
        max_steps: int = 15,
        task_misc: dict | None = None,
        verbose: bool = True,
    ) -> Any:
        """Start the rubric before the child rollout and score after it returns."""
        agent = cast(ForkableAgent, current_agent.get())
        env = cast(ForkableEnv, current_env.get())
        task = env.task
        parent_trajectory = current_trajectory.get()
        parent_step_index = len(parent_trajectory.steps)
        child_depth = self._trajectory_depth(parent_trajectory) + 1
        rubric_messages = await self._build_rubric_messages(
            goal=goal,
            max_steps=max_steps,
            child_depth=child_depth,
            parent_trajectory=parent_trajectory,
            env=env,
            agent=agent,
        )

        # Start KIMI as soon as the Parent has committed to this delegation.
        rubric_task = asyncio.create_task(
            self.processor.client.generate_rubric(rubric_messages)
        )
        self.rubric_requests_started += 1

        async with profile_span(
            "launch_subagent_rubric_pipeline",
            metadata={
                "goal_len": len(goal),
                "max_steps": max_steps,
                "parent_task_id": getattr(task, "id", None),
                "parent_trajectory_id": parent_trajectory.id,
            },
        ):
            forked_agent = None
            forked_env = None
            budget_reserved = False
            try:
                subtask = task.fork(goal, max_steps, task_misc=task_misc)
                forked_agent = await agent.fork(subtask)
                forked_env = await env.fork(subtask)
                budget_tracker.get().reserve_budget(
                    max_steps + 1, raise_on_failure=True
                )
                budget_reserved = True
            except (BudgetExceededError, ValueError) as exc:
                rubric_task.cancel()
                with suppress(asyncio.CancelledError):
                    await rubric_task
                if forked_agent is not None:
                    await forked_agent.close()
                if forked_env is not None:
                    await forked_env.close()
                guidance = getattr(exc, "guidance", "")
                message = f"Not enough budget to launch subagent for goal {goal}. {exc}"
                if guidance:
                    message += " " + guidance
                return message
            except BaseException:
                rubric_task.cancel()
                with suppress(asyncio.CancelledError):
                    await rubric_task
                if forked_agent is not None:
                    await forked_agent.close()
                if forked_env is not None:
                    await forked_env.close()
                raise

            try:
                child_trajectory = await asyncio.create_task(
                    run_episode(
                        forked_agent,
                        forked_env,
                        timeout=episode_step_timeout.get(),
                    )
                )
            except BaseException:
                rubric_task.cancel()
                with suppress(asyncio.CancelledError):
                    await rubric_task
                raise
            finally:
                if budget_reserved:
                    budget_tracker.get().release_budget(max_steps + 1)

            trajectory_dump = asdict(child_trajectory)
            assessment_task = asyncio.create_task(
                self._score_after_child_return(
                    trajectory_id=child_trajectory.id,
                    parent_trajectory_id=parent_trajectory.id,
                    parent_step_index=parent_step_index,
                    depth=child_depth,
                    child_goal=goal,
                    rubric_messages=rubric_messages,
                    trajectory_dump=trajectory_dump,
                    rubric_task=rubric_task,
                )
            )
            self._pending.append(assessment_task)

            used_recursive = int(
                budget_tracker.get().used_budget_for(child_trajectory.id)
            )
            remaining_total = int(budget_tracker.get().remaining_budget())
            budget_message = (
                f"\n\nBudget used by subagent: {used_recursive}/{max_steps} steps. "
                f"Total remaining budget for the current task is "
                f"{remaining_total} steps.\n"
            )
            result = (
                child_trajectory.finish_message
                or child_trajectory.error_message
                or ""
            )
            return result + budget_message if verbose else result

    async def finalize(
        self, collection: dict[str, Any]
    ) -> RubricRewardBatchResult:
        """Drain pending Judge calls and apply scores to the serialized tree."""
        result = RubricRewardBatchResult(
            collections={self.rollout_index: collection}
        )
        assessments = (
            await asyncio.gather(*self._pending) if self._pending else []
        )
        judge_failed = False
        skip_root = False
        dropped_trajectories = 0
        scores: list[float] = []
        original_rewards: list[float] = []
        trajectories = collection.get("trajectories", {})

        for assessment in assessments:
            trajectory = trajectories.get(assessment.trajectory_id)
            if trajectory is None:
                judge_failed = True
                skip_root = (
                    self.processor.config.failure_policy == "skip_rollout"
                )
                result.records.append(
                    self._failure_record(
                        assessment,
                        "trajectory_lookup",
                        "Child trajectory missing from serialized collection",
                    )
                )
                continue
            if not assessment.rubric_result.valid:
                judge_failed = True
                if self.processor.config.failure_policy == "skip_rollout":
                    skip_root = True
                elif self.processor.config.failure_policy == "drop_trajectory":
                    self.processor._mark_gradient_skip(
                        trajectory,
                        stage="rubric_generation",
                        error=assessment.rubric_result.error,
                    )
                    dropped_trajectories += 1
                result.records.append(
                    self._failure_record(
                        assessment,
                        "rubric_generation",
                        assessment.rubric_result.error,
                    )
                )
                continue
            if (
                assessment.score_result is None
                or not assessment.score_result.valid
            ):
                judge_failed = True
                score_error = (
                    assessment.score_result.error
                    if assessment.score_result is not None
                    else "score request not started"
                )
                if self.processor.config.failure_policy == "skip_rollout":
                    skip_root = True
                elif self.processor.config.failure_policy == "drop_trajectory":
                    self.processor._mark_gradient_skip(
                        trajectory,
                        stage="trajectory_scoring",
                        error=score_error,
                    )
                    dropped_trajectories += 1
                result.records.append(
                    self._failure_record(
                        assessment,
                        "trajectory_scoring",
                        (
                            score_error
                        ),
                    )
                )
                continue

            score, score_payload = assessment.score_result.parsed
            score = float(score)
            original_reward = self.processor._original_success(trajectory)
            rubric_success = bool(
                score_payload.get("success", score > 0.0)
            )
            self.processor._apply_reward(
                trajectory,
                score=score,
                rubric_success=rubric_success,
                original_reward=original_reward,
                rubric=assessment.rubric_result.parsed,
                score_payload=score_payload,
            )
            scores.append(score)
            original_rewards.append(original_reward)
            result.records.append(
                {
                    "task_id": self.task_id,
                    "rollout_index": self.rollout_index,
                    "trajectory_id": assessment.trajectory_id,
                    "parent_trajectory_id": assessment.parent_trajectory_id,
                    "depth": assessment.depth,
                    "child_goal": assessment.child_goal,
                    "status": "scored",
                    "pipeline": "delegation_async",
                    "original_binary_reward": original_reward,
                    "rubric_reward": score,
                    "rubric_success": rubric_success,
                    "rubric": assessment.rubric_result.parsed,
                    "score_payload": score_payload,
                    "rubric_request_id": (
                        assessment.rubric_result.request_id
                    ),
                    "score_request_id": assessment.score_result.request_id,
                    "rubric_cache_hit": (
                        assessment.rubric_result.cache_hit
                    ),
                    "score_cache_hit": assessment.score_result.cache_hit,
                    "rubric_messages": assessment.rubric_messages,
                    "scoring_messages": assessment.scoring_messages,
                }
            )

        skipped = skip_root
        if skipped:
            result.collections[self.rollout_index] = None
        result.metrics = {
            "rubric_reward/root_rollouts": 1.0,
            "rubric_reward/expected_subagent_trajectories": float(
                len(assessments)
            ),
            "rubric_reward/subagent_candidates": float(len(assessments)),
            "rubric_reward/valid_rubrics": float(
                sum(item.rubric_result.valid for item in assessments)
            ),
            "rubric_reward/valid_scores": float(len(scores)),
            "rubric_reward/judge_failed_root_rollouts": float(judge_failed),
            "rubric_reward/skipped_root_rollouts": float(skipped),
            "rubric_reward/dropped_trajectories": float(
                dropped_trajectories
            ),
            "rubric_reward/mean_score": mean(scores) if scores else 0.0,
            "rubric_reward/min_score": min(scores) if scores else 0.0,
            "rubric_reward/max_score": max(scores) if scores else 0.0,
            "rubric_reward/original_success_rate": (
                mean(original_rewards) if original_rewards else 0.0
            ),
            "rubric_reward/async_rubrics_started": float(
                self.rubric_requests_started
            ),
            "rubric_reward/async_scores_started": float(
                self.score_requests_started
            ),
        }
        await self.processor._write_artifacts(self.task_id, result.records)
        return result

    async def cancel(self) -> None:
        await cancel_tasks(
            self._pending,
            grace_seconds=5.0,
        )

    async def _score_after_child_return(
        self,
        *,
        trajectory_id: str,
        parent_trajectory_id: str,
        parent_step_index: int,
        depth: int,
        child_goal: str,
        rubric_messages: list[dict[str, str]],
        trajectory_dump: dict[str, Any],
        rubric_task: asyncio.Task[ParsedGeneration],
    ) -> _PendingAssessment:
        rubric_result = await rubric_task
        if not rubric_result.valid:
            return _PendingAssessment(
                trajectory_id=trajectory_id,
                parent_trajectory_id=parent_trajectory_id,
                parent_step_index=parent_step_index,
                depth=depth,
                child_goal=child_goal,
                rubric_messages=rubric_messages,
                scoring_messages=[],
                rubric_result=rubric_result,
                score_result=None,
            )
        scoring_messages = build_scoring_messages(
            rubric=rubric_result.parsed,
            child_goal=child_goal,
            trajectory=self.processor.adapter.serialize_agent_trajectory(
                trajectory_dump
            ),
            final_environment_state=(
                self.processor.adapter.final_state_from_trajectory(
                    trajectory_dump
                )
            ),
        )
        self.score_requests_started += 1
        score_result = await self.processor.client.score_trajectory(
            scoring_messages
        )
        return _PendingAssessment(
            trajectory_id=trajectory_id,
            parent_trajectory_id=parent_trajectory_id,
            parent_step_index=parent_step_index,
            depth=depth,
            child_goal=child_goal,
            rubric_messages=rubric_messages,
            scoring_messages=scoring_messages,
            rubric_result=rubric_result,
            score_result=score_result,
        )

    async def _build_rubric_messages(
        self,
        *,
        goal: str,
        max_steps: int,
        child_depth: int,
        parent_trajectory: Trajectory,
        env: Any,
        agent: Any,
    ) -> list[dict[str, str]]:
        action = current_codeact_action.get()
        parent_prefix = (
            getattr(action, "_request_messages", None)
            if action is not None
            else None
        )
        if parent_prefix is None:
            observation = await env.observe()
            parent_prefix = agent.prompt_builder.build_messages(observation)
        parent_action = (
            str(action.action or str(action)) if action is not None else ""
        )
        try:
            environment_state = self.processor.adapter.snapshot(env)
        except Exception:
            environment_state = {}
        return build_rubric_messages(
            parent_prefix=[dict(message) for message in parent_prefix],
            parent_action=parent_action,
            child_goal=goal,
            environment_state=environment_state,
            available_tools=getattr(
                self.processor.adapter,
                "available_tools",
                "get_info, view_inventory, craft, finish, launch_subagent",
            ),
            execution_budget=max_steps,
            min_criteria=self.processor.config.min_rubric_criteria,
            parent_task=getattr(env.task, "goal", None),
            parent_metadata={
                "parent_trajectory_id": parent_trajectory.id,
                "parent_depth": child_depth - 1,
                "child_depth": child_depth,
                "fork_step": len(parent_trajectory.steps),
            },
        )

    @staticmethod
    def _trajectory_depth(trajectory: Trajectory) -> int:
        collection = current_trajectory_collection.get()
        depth = 0
        current = trajectory
        while current.parent_info is not None:
            depth += 1
            current = collection.trajectories[current.parent_info.id]
        return depth

    def _failure_record(
        self,
        assessment: _PendingAssessment,
        stage: str,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollout_index": self.rollout_index,
            "trajectory_id": assessment.trajectory_id,
            "parent_trajectory_id": assessment.parent_trajectory_id,
            "depth": assessment.depth,
            "child_goal": assessment.child_goal,
            "status": "failed",
            "pipeline": "delegation_async",
            "failure_stage": stage,
            "error": error,
        }
