from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from three_stage_train.adapters.textcraft import TextCraftThreeStageAdapter
from three_stage_train.agent_tree import build_agent_tree
from Runtime.rubric.prompts import (
    build_rubric_messages,
    build_scoring_messages,
)
from three_stage_train.stages.common import (
    RawRollout,
    SubagentCandidate,
    build_subagent_candidates,
    rubric_context,
)

from .client import KimiRubricClient, ParsedGeneration
from .config import RubricSubagentRewardConfig


@dataclass
class RubricRewardBatchResult:
    collections: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)


class RubricSubagentRewardProcessor:
    """Assign the configured client's rubric scores to non-root trajectories."""

    def __init__(
        self,
        config: RubricSubagentRewardConfig,
        *,
        adapter: TextCraftThreeStageAdapter | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or KimiRubricClient(config)
        self.adapter = adapter or TextCraftThreeStageAdapter()
        self.artifact_dir = Path(config.artifact_dir)
        self._artifact_lock = asyncio.Lock()

    async def process(
        self,
        raw_rollouts: list[RawRollout],
        task_id: str,
    ) -> RubricRewardBatchResult:
        result = RubricRewardBatchResult(
            collections={
                raw.rollout_index: raw.collection for raw in raw_rollouts
            }
        )
        candidates: list[SubagentCandidate] = []
        malformed_rollouts: set[int] = set()
        expected_candidates = 0

        for raw in raw_rollouts:
            try:
                tree = build_agent_tree(raw.collection)
            except ValueError:
                malformed_rollouts.add(raw.rollout_index)
                continue
            expected = sum(node.depth > 0 for node in tree.nodes.values())
            rollout_candidates = build_subagent_candidates(raw, self.adapter)
            expected_candidates += expected
            if len(rollout_candidates) != expected:
                malformed_rollouts.add(raw.rollout_index)
            candidates.extend(rollout_candidates)

        rubric_requests: list[list[dict[str, str]]] = []
        for candidate in candidates:
            candidate.rubric_messages = build_rubric_messages(
                **rubric_context(candidate),
                min_criteria=self.config.min_rubric_criteria,
            )
            rubric_requests.append(candidate.rubric_messages)
        rubric_results = await asyncio.gather(
            *(self.client.generate_rubric(messages) for messages in rubric_requests)
        )

        scored_inputs: list[
            tuple[SubagentCandidate, ParsedGeneration, list[dict[str, str]]]
        ] = []
        failed_rollouts = set(malformed_rollouts)
        failed_candidates: list[tuple[SubagentCandidate, str, str | None]] = []
        for candidate, rubric_result in zip(
            candidates, rubric_results, strict=True
        ):
            if not rubric_result.valid:
                failed_rollouts.add(candidate.raw.rollout_index)
                failed_candidates.append(
                    (candidate, "rubric_generation", rubric_result.error)
                )
                result.records.append(
                    self._failure_record(
                        task_id, candidate, "rubric_generation", rubric_result.error
                    )
                )
                continue
            scoring_messages = build_scoring_messages(
                rubric=rubric_result.parsed,
                child_goal=candidate.child_goal,
                trajectory=self.adapter.serialize_agent_trajectory(
                    candidate.trajectory
                ),
                final_environment_state=self.adapter.final_state_from_trajectory(
                    candidate.trajectory
                ),
            )
            scored_inputs.append((candidate, rubric_result, scoring_messages))

        score_results = await asyncio.gather(
            *(
                self.client.score_trajectory(scoring_messages)
                for _, _, scoring_messages in scored_inputs
            )
        )
        scores: list[float] = []
        original_rewards: list[float] = []
        for (
            candidate,
            rubric_result,
            scoring_messages,
        ), score_result in zip(scored_inputs, score_results, strict=True):
            if not score_result.valid:
                failed_rollouts.add(candidate.raw.rollout_index)
                failed_candidates.append(
                    (candidate, "trajectory_scoring", score_result.error)
                )
                result.records.append(
                    self._failure_record(
                        task_id, candidate, "trajectory_scoring", score_result.error
                    )
                )
                continue
            score, score_payload = score_result.parsed
            score = float(score)
            original_reward = self._original_success(candidate.trajectory)
            rubric_success = bool(score_payload.get("success", score > 0.0))
            self._apply_reward(
                candidate.trajectory,
                score=score,
                rubric_success=rubric_success,
                original_reward=original_reward,
                rubric=rubric_result.parsed,
                score_payload=score_payload,
            )
            scores.append(score)
            original_rewards.append(original_reward)
            result.records.append(
                {
                    "task_id": task_id,
                    "rollout_index": candidate.raw.rollout_index,
                    "trajectory_id": candidate.trajectory_id,
                    "parent_trajectory_id": candidate.parent_id,
                    "depth": candidate.depth,
                    "child_goal": candidate.child_goal,
                    "status": "scored",
                    "original_binary_reward": original_reward,
                    "rubric_reward": score,
                    "rubric_success": rubric_success,
                    "rubric": rubric_result.parsed,
                    "score_payload": score_payload,
                    "rubric_request_id": rubric_result.request_id,
                    "score_request_id": score_result.request_id,
                    "rubric_cache_hit": rubric_result.cache_hit,
                    "score_cache_hit": score_result.cache_hit,
                    "rubric_messages": candidate.rubric_messages,
                    "scoring_messages": scoring_messages,
                }
            )

        judge_failed_rollouts = set(failed_rollouts)
        skipped_rollouts: set[int] = set()
        dropped_trajectories = 0
        if failed_rollouts:
            if self.config.failure_policy == "skip_rollout":
                for rollout_index in failed_rollouts:
                    result.collections[rollout_index] = None
                skipped_rollouts.update(failed_rollouts)
            elif self.config.failure_policy == "drop_trajectory":
                for candidate, stage, error in failed_candidates:
                    self._mark_gradient_skip(
                        candidate.trajectory,
                        stage=stage,
                        error=error,
                    )
                    dropped_trajectories += 1
                # A malformed tree has no safely identifiable child trajectory.
                for rollout_index in malformed_rollouts:
                    result.collections[rollout_index] = None
                skipped_rollouts.update(malformed_rollouts)
            else:
                # Failed candidates retain their original environment reward.
                failed_rollouts.clear()

        result.metrics = {
            "rubric_reward/root_rollouts": float(len(raw_rollouts)),
            "rubric_reward/expected_subagent_trajectories": float(
                expected_candidates
            ),
            "rubric_reward/subagent_candidates": float(len(candidates)),
            "rubric_reward/valid_rubrics": float(
                sum(item.valid for item in rubric_results)
            ),
            "rubric_reward/valid_scores": float(len(scores)),
            "rubric_reward/judge_failed_root_rollouts": float(
                len(judge_failed_rollouts)
            ),
            "rubric_reward/skipped_root_rollouts": float(len(skipped_rollouts)),
            "rubric_reward/dropped_trajectories": float(
                dropped_trajectories
            ),
            "rubric_reward/mean_score": mean(scores) if scores else 0.0,
            "rubric_reward/min_score": min(scores) if scores else 0.0,
            "rubric_reward/max_score": max(scores) if scores else 0.0,
            "rubric_reward/original_success_rate": (
                mean(original_rewards) if original_rewards else 0.0
            ),
        }
        await self._write_artifacts(task_id, result.records)
        return result

    @staticmethod
    def _original_success(trajectory: dict[str, Any]) -> float:
        for step in reversed(trajectory.get("steps") or []):
            reward_misc = step.get("misc", {}).get("reward_misc", {})
            if "reward/success" in reward_misc:
                return float(reward_misc["reward/success"])
        return float(trajectory.get("reward", 0.0))

    @staticmethod
    def _apply_reward(
        trajectory: dict[str, Any],
        *,
        score: float,
        rubric_success: bool,
        original_reward: float,
        rubric: dict[str, Any],
        score_payload: dict[str, Any],
    ) -> None:
        trajectory["reward"] = score
        trajectory.setdefault("misc", {})["rubric_subagent_reward"] = {
            "score": score,
            "success": rubric_success,
            "original_binary_reward": original_reward,
            "rubric": rubric,
            "score_payload": score_payload,
        }
        steps = trajectory.get("steps") or []
        if not steps:
            return
        reward_misc = steps[-1].setdefault("misc", {}).setdefault(
            "reward_misc", {}
        )
        reward_misc["reward/code_success"] = original_reward
        reward_misc["reward/success"] = score
        reward_misc["reward/rubric_score"] = score
        reward_misc["reward/filter_success"] = float(rubric_success)

    @staticmethod
    def _mark_gradient_skip(
        trajectory: dict[str, Any],
        *,
        stage: str,
        error: str | None,
    ) -> None:
        reason = (
            "rubric_parse_failure"
            if stage == "rubric_generation"
            else "score_parse_failure"
        )
        trajectory.setdefault("misc", {})["skip_gradient_update"] = {
            "reason": reason,
            "stage": stage,
            "error": error,
        }

    @staticmethod
    def _failure_record(
        task_id: str,
        candidate: SubagentCandidate,
        stage: str,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "rollout_index": candidate.raw.rollout_index,
            "trajectory_id": candidate.trajectory_id,
            "parent_trajectory_id": candidate.parent_id,
            "depth": candidate.depth,
            "child_goal": candidate.child_goal,
            "status": "failed",
            "failure_stage": stage,
            "error": error,
        }

    async def _write_artifacts(
        self, task_id: str, records: list[dict[str, Any]]
    ) -> None:
        if not records:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / f"{task_id}.jsonl"
        async with self._artifact_lock:
            with path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

    async def close(self) -> None:
        await self.client.close()
