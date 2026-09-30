from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import mean
from typing import Any

from ..config import ThreeStageTrainConfig
from ..data import completion_text, completion_to_datum
from Runtime.rubric.prompts import (
    build_rubric_messages,
    build_scoring_messages,
    build_teacher_messages,
)
from Runtime.rubric.scoring import (
    leave_one_out_agreement_credits,
    parse_policy_score,
    parse_rubric,
)
from ..teacher import TeacherClient
from .common import RawRollout, StageBatchResult, step_completion_id


@dataclass
class _BranchWorkItem:
    raw: RawRollout
    group: dict[str, Any]
    branch: dict[str, Any]
    trajectory: dict[str, Any]
    rubric_messages: list[dict[str, str]]
    rubric_generation: Any | None = None
    rubric: dict[str, Any] | None = None
    rubric_valid: bool = False
    scoring_messages: list[dict[str, str]] | None = None
    scoring_generation: Any | None = None
    policy_score: float = 0.0
    policy_score_valid: bool = False


class RubricGenerationStage:
    """Stage 3: train independent rubric generation and rubric-based scoring."""

    def __init__(self, config: ThreeStageTrainConfig, adapter: Any) -> None:
        self.config = config
        self.stage_config = config.rubric_generation
        self.adapter = adapter
        self._teacher_semaphore = asyncio.Semaphore(
            self.stage_config.max_teacher_concurrency
        )

    async def process(
        self,
        raw_rollouts: list[RawRollout],
        policy_client: Any,
    ) -> StageBatchResult:
        groups = self._complete_groups(raw_rollouts)
        if not groups:
            return StageBatchResult(
                metrics={"stage3/skipped_no_complete_fork_groups": 1.0}
            )

        items = [
            item
            for raw, group in groups
            for item in self._build_group_items(raw, group)
        ]
        expected_items = len(groups) * self.stage_config.branching_factor
        if len(items) != expected_items:
            return StageBatchResult(
                metrics={
                    "stage3/complete_fork_groups": float(len(groups)),
                    "stage3/skipped_incomplete_training_records": 1.0,
                }
            )

        grouped_items = self._group_items(items)
        teacher_task = asyncio.create_task(self._teacher_scores(grouped_items))
        try:
            # Context A: each R_i is sampled from a fresh rubric-generation request.
            rubric_outputs = await policy_client.generate_batch(
                [item.rubric_messages for item in items],
                temperature=self.stage_config.rubric_temperature,
                max_completion_tokens=self.stage_config.max_rubric_tokens,
            )
            for item, generation in zip(items, rubric_outputs, strict=True):
                item.rubric_generation = generation
                if not generation.valid:
                    continue
                try:
                    item.rubric = parse_rubric(
                        generation.text,
                        self.stage_config.num_rubric_criteria_min,
                    )
                    item.rubric_valid = True
                except (TypeError, ValueError):
                    item.rubric = {"invalid_rubric_output": generation.text}

                # Context B is a new request. It serializes R_i as frozen data and
                # never inherits the rubric-generation assistant message or state.
                item.scoring_messages = build_scoring_messages(
                    rubric=item.rubric,
                    child_goal=str(item.group.get("child_goal", "")),
                    trajectory=self.adapter.serialize_agent_trajectory(
                        item.trajectory
                    ),
                    final_environment_state=(item.branch.get("final_state") or {}),
                )

            if any(item.scoring_messages is None for item in items):
                teacher_task.cancel()
                await asyncio.gather(teacher_task, return_exceptions=True)
                return StageBatchResult(
                    metrics={
                        "stage3/complete_fork_groups": float(len(groups)),
                        "stage3/skipped_policy_rubric_request_failure": 1.0,
                    }
                )

            scoring_outputs = await policy_client.generate_batch(
                [item.scoring_messages for item in items],
                temperature=self.stage_config.scoring_temperature,
                max_completion_tokens=self.stage_config.max_scoring_tokens,
            )
            for item, generation in zip(items, scoring_outputs, strict=True):
                item.scoring_generation = generation
                if not generation.valid:
                    continue
                try:
                    parsed_score, _ = parse_policy_score(generation.text)
                    if item.rubric_valid:
                        item.policy_score = parsed_score
                        item.policy_score_valid = True
                    else:
                        item.policy_score = 0.0
                except (TypeError, ValueError):
                    # Invalid score formatting is observable policy failure.
                    item.policy_score = 0.0
            teacher_results = await teacher_task
        except BaseException:
            if not teacher_task.done():
                teacher_task.cancel()
                await asyncio.gather(teacher_task, return_exceptions=True)
            raise

        datums: list[dict] = []
        records: list[dict[str, Any]] = []
        agreement_values: list[float] = []
        teacher_failures = 0
        invalid_rubrics = 0
        invalid_policy_scores = 0

        for (raw, group), group_items, teacher_result in zip(
            groups,
            grouped_items,
            teacher_results,
            strict=True,
        ):
            del raw
            if teacher_result is None:
                teacher_failures += 1
                continue
            if any(
                item.rubric_generation is None
                or not item.rubric_generation.valid
                or item.scoring_generation is None
                or not item.scoring_generation.valid
                for item in group_items
            ):
                continue

            teacher_scores, teacher_payload, teacher_request_id = teacher_result
            policy_scores = [item.policy_score for item in group_items]
            raw_credits, advantages, agreement = leave_one_out_agreement_credits(
                policy_scores,
                teacher_scores,
                self.stage_config.agreement_alpha,
            )
            agreement_values.append(agreement["agreement_score"])
            depth = (
                int(group.get("parent_depth", 0))
                if self._include_depth_metadata()
                else None
            )

            for item, teacher_score, raw_credit, advantage in zip(
                group_items,
                teacher_scores,
                raw_credits,
                advantages,
                strict=True,
            ):
                invalid_rubrics += int(not item.rubric_valid)
                invalid_policy_scores += int(not item.policy_score_valid)
                datums.append(
                    completion_to_datum(
                        item.rubric_generation.completion_entry,
                        advantage,
                        trajectory_depth=depth,
                        trajectory_start=True,
                    )
                )
                datums.append(
                    completion_to_datum(
                        item.scoring_generation.completion_entry,
                        advantage,
                        trajectory_depth=depth,
                        trajectory_start=False,
                    )
                )
                records.append(
                    {
                        "fork_group_id": group.get("id"),
                        "branch_index": int(item.branch.get("branch_index", -1)),
                        "trajectory_id": item.branch.get("trajectory_id"),
                        "rubric_completion_id": item.rubric_generation.completion_id,
                        "scoring_completion_id": item.scoring_generation.completion_id,
                        "teacher_request_id": teacher_request_id,
                        "rubric_input_messages": item.rubric_messages,
                        "rubric_output": item.rubric_generation.text,
                        "rubric": item.rubric,
                        "scoring_input_messages": item.scoring_messages,
                        "scoring_output": item.scoring_generation.text,
                        "rubric_valid": item.rubric_valid,
                        "policy_score_valid": item.policy_score_valid,
                        "policy_score": item.policy_score,
                        "teacher_score": teacher_score,
                        "raw_credit": raw_credit,
                        "advantage": advantage,
                        "agreement": agreement,
                        "teacher_payload": (
                            teacher_payload
                            if int(item.branch.get("branch_index", -1)) == 0
                            else None
                        ),
                    }
                )

        return StageBatchResult(
            datums=datums,
            records=records,
            metrics={
                "stage3/complete_fork_groups": float(len(groups)),
                "stage3/scored_fork_groups": float(len(agreement_values)),
                "stage3/teacher_failures": float(teacher_failures),
                "stage3/invalid_rubrics": float(invalid_rubrics),
                "stage3/invalid_policy_scores": float(invalid_policy_scores),
                "stage3/mean_agreement": (
                    mean(agreement_values) if agreement_values else 0.0
                ),
            },
        )

    def _complete_groups(
        self,
        raw_rollouts: list[RawRollout],
    ) -> list[tuple[RawRollout, dict[str, Any]]]:
        groups: list[tuple[RawRollout, dict[str, Any]]] = []
        for raw in raw_rollouts:
            metadata = raw.collection.get("_three_stage") or {}
            for group in metadata.get("fork_groups") or []:
                branches = group.get("branches") or []
                if not group.get("complete"):
                    continue
                if len(branches) != self.stage_config.branching_factor:
                    continue
                if any(not branch.get("trajectory_id") for branch in branches):
                    continue
                groups.append((raw, group))
        return groups

    def _build_group_items(
        self,
        raw: RawRollout,
        group: dict[str, Any],
    ) -> list[_BranchWorkItem]:
        trajectories = raw.collection.get("trajectories") or {}
        parent_id = group.get("parent_trajectory_id")
        parent = trajectories.get(parent_id)
        parent_step_index = int(group.get("parent_step", -1))
        if parent is None:
            return []
        parent_steps = parent.get("steps") or []
        if not 0 <= parent_step_index < len(parent_steps):
            return []
        completion_id = step_completion_id(parent_steps[parent_step_index])
        if completion_id is None or completion_id not in raw.completions:
            return []
        parent_completion = raw.completions[completion_id]

        branches = sorted(
            group.get("branches") or [],
            key=lambda branch: int(branch.get("branch_index", -1)),
        )
        child_goal = str(group.get("child_goal", ""))
        initial_state = branches[0].get("initial_state") or {}
        items: list[_BranchWorkItem] = []
        for branch in branches:
            trajectory = trajectories.get(branch.get("trajectory_id"))
            if trajectory is None:
                return []
            task = trajectory.get("task") or {}
            execution_budget = task.get("max_steps") if isinstance(task, dict) else None
            messages = build_rubric_messages(
                parent_prefix=parent_completion.messages,
                parent_action=completion_text(parent_completion),
                child_goal=child_goal,
                environment_state=initial_state,
                available_tools=(
                    "get_info, view_inventory, craft, finish, launch_subagent"
                ),
                execution_budget=execution_budget,
                min_criteria=self.stage_config.num_rubric_criteria_min,
                parent_task=self._parent_task(raw, group),
                parent_metadata={
                    "fork_group_id": group.get("id"),
                    "parent_trajectory_id": parent_id,
                    "parent_depth": group.get("parent_depth"),
                    "child_depth": group.get("child_depth"),
                    "fork_step": parent_step_index,
                    "branch_index": branch.get("branch_index"),
                },
            )
            items.append(
                _BranchWorkItem(
                    raw=raw,
                    group=group,
                    branch=branch,
                    trajectory=trajectory,
                    rubric_messages=messages,
                )
            )
        return items

    def _group_items(
        self,
        items: list[_BranchWorkItem],
    ) -> list[list[_BranchWorkItem]]:
        size = self.stage_config.branching_factor
        return [items[index : index + size] for index in range(0, len(items), size)]

    async def _teacher_scores(
        self,
        grouped_items: list[list[_BranchWorkItem]],
    ) -> list[tuple[list[float], dict[str, Any], str] | None]:
        teacher = TeacherClient(self.stage_config.teacher)

        async def score_group(group_items: list[_BranchWorkItem]):
            child_goal = str(group_items[0].group.get("child_goal", ""))
            messages = build_teacher_messages(
                child_goal=child_goal,
                initial_environment_state=(
                    group_items[0].branch.get("initial_state") or {}
                ),
                trajectories=[
                    self.adapter.serialize_agent_trajectory(item.trajectory)
                    for item in group_items
                ],
                final_environment_states=[
                    item.branch.get("final_state") or {} for item in group_items
                ],
                parent_task=self._parent_task(group_items[0].raw, group_items[0].group),
                branch_metadata=[
                    {
                        "branch_index": item.branch.get("branch_index"),
                        "initial_state": item.branch.get("initial_state") or {},
                        "reward": item.branch.get("reward"),
                        "finish_message": item.branch.get("finish_message"),
                        "error_message": item.branch.get("error_message"),
                    }
                    for item in group_items
                ],
            )
            try:
                async with self._teacher_semaphore:
                    return await teacher.score(messages, len(group_items))
            except Exception:
                return None

        try:
            return await asyncio.gather(
                *(score_group(group_items) for group_items in grouped_items)
            )
        finally:
            await teacher.close()

    def _include_depth_metadata(self) -> bool:
        optimization = self.config.optimization
        return (
            optimization.depth_level_weighting
            or optimization.depth_level_discount_gamma is not None
        )

    @staticmethod
    def _parent_task(raw: RawRollout, group: dict[str, Any]) -> str | None:
        parent = (raw.collection.get("trajectories") or {}).get(
            group.get("parent_trajectory_id")
        )
        if not isinstance(parent, dict):
            return None
        task = parent.get("task") or {}
        return task.get("goal") if isinstance(task, dict) else None
