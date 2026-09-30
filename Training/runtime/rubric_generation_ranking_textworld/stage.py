from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import mean
from typing import Any, Sequence

from kimi_judge import KimiJudgeClient
from three_stage_train.data import completion_to_datum
from Runtime.rubric.prompts import build_scoring_messages
from Runtime.rubric.scoring import parse_policy_score, parse_rubric
from three_stage_train.stages.common import RawRollout, StageBatchResult
from three_stage_train.stages.rubric_generation import RubricGenerationStage, _BranchWorkItem

from .config import RubricGenerationRankingConfig


@dataclass(frozen=True)
class RankingResult:
    group_reward: float
    pair_count: int
    pairwise_accuracy: float
    margin_satisfaction_rate: float
    mean_score_gap: float
    mean_hinge_loss: float


def pairwise_margin_rewards(scores: Sequence[float], successes: Sequence[bool], margin: float) -> RankingResult | None:
    positives = [i for i, value in enumerate(successes) if value]
    negatives = [i for i, value in enumerate(successes) if not value]
    if not positives or not negatives:
        return None
    gaps, losses = [], []
    correct = satisfied = 0
    for positive in positives:
        for negative in negatives:
            gap = float(scores[positive]) - float(scores[negative])
            loss = max(0.0, margin - gap)
            gaps.append(gap)
            losses.append(loss)
            correct += int(gap > 0.0)
            satisfied += int(gap >= margin)
    count = len(losses)
    return RankingResult(-mean(losses), count, correct / count, satisfied / count, mean(gaps), mean(losses))


class RubricGenerationRankingStage(RubricGenerationStage):
    """Train rubric generation using Policy scores and KIMI binary labels."""

    def __init__(self, three_stage_config: Any, adapter: Any,
                 ranking_config: RubricGenerationRankingConfig,
                 judge_client: KimiJudgeClient) -> None:
        super().__init__(three_stage_config, adapter)
        ranking_config.validate()
        self.ranking_config = ranking_config
        self.judge_client = judge_client

    async def process(self, raw_rollouts: list[RawRollout], policy_client: Any) -> StageBatchResult:
        groups = self._complete_groups(raw_rollouts)
        grouped_items = []
        for raw, group in groups:
            items = self._build_group_items(raw, group)
            if len(items) == self.ranking_config.branching_factor:
                grouped_items.append(items)
        if not grouped_items:
            return StageBatchResult(metrics={
                "kimi_rubric/skipped_no_complete_fork_groups": 1.0,
                "kimi_rubric/complete_fork_groups": float(len(groups)),
            })

        rubric_outputs = await policy_client.generate_batch(
            [items[0].rubric_messages for items in grouped_items],
            temperature=self.ranking_config.rubric_temperature,
            max_completion_tokens=self.ranking_config.max_rubric_tokens,
        )
        valid_groups = []
        invalid_rubrics = 0
        for items, generation in zip(grouped_items, rubric_outputs, strict=True):
            if not generation.valid:
                invalid_rubrics += 1
                continue
            try:
                rubric = parse_rubric(generation.text, self.ranking_config.min_rubric_criteria)
            except (TypeError, ValueError):
                invalid_rubrics += 1
                continue
            for item in items:
                item.rubric_generation = generation
                item.rubric = rubric
                item.rubric_valid = True
                item.scoring_messages = build_scoring_messages(
                    rubric=rubric,
                    child_goal=str(item.group.get("child_goal", "")),
                    trajectory=self.adapter.serialize_agent_trajectory(item.trajectory),
                    final_environment_state=item.branch.get("final_state") or {},
                )
            valid_groups.append(items)
        score_items = [item for items in valid_groups for item in items]
        if not score_items:
            return StageBatchResult(metrics={
                "kimi_rubric/complete_fork_groups": float(len(groups)),
                "kimi_rubric/invalid_rubrics": float(invalid_rubrics),
            })

        # The active Policy scores branches while KIMI independently supplies
        # binary success labels.
        score_task = policy_client.generate_batch(
            [item.scoring_messages for item in score_items],
            temperature=self.ranking_config.scoring_temperature,
            max_completion_tokens=self.ranking_config.max_scoring_tokens,
        )
        judge_task = self.judge_client.judge_batch([item.trajectory for item in score_items])
        score_outputs, judge_outputs = await asyncio.gather(score_task, judge_task)
        for item, score_generation, judge_result in zip(score_items, score_outputs, judge_outputs, strict=True):
            item.scoring_generation = score_generation
            item.policy_score_valid = False
            if score_generation.valid:
                try:
                    item.policy_score, _ = parse_policy_score(score_generation.text)
                    item.policy_score_valid = True
                except (TypeError, ValueError):
                    pass
            item.judge_result = judge_result
            item.judge_success = judge_result.success
            item.judge_valid = judge_result.valid

        ranked_groups: list[tuple[list[_BranchWorkItem], RankingResult]] = []
        invalid_scores = invalid_judges = pair_count = 0
        accuracies: list[float] = []
        margin_rates: list[float] = []
        gaps: list[float] = []
        losses: list[float] = []
        for items in valid_groups:
            valid_items = []
            for item in items:
                if not item.policy_score_valid:
                    invalid_scores += 1
                elif not getattr(item, "judge_valid", False):
                    invalid_judges += 1
                else:
                    valid_items.append(item)
            if len(valid_items) < 2:
                continue
            ranking = pairwise_margin_rewards(
                [item.policy_score for item in valid_items],
                [bool(item.judge_success) for item in valid_items],
                self.ranking_config.margin,
            )
            if ranking is None:
                continue
            ranked_groups.append((valid_items, ranking))
            pair_count += ranking.pair_count
            accuracies.append(ranking.pairwise_accuracy)
            margin_rates.append(ranking.margin_satisfaction_rate)
            gaps.append(ranking.mean_score_gap)
            losses.append(ranking.mean_hinge_loss)

        rewards = [ranking.group_reward for _, ranking in ranked_groups]
        if len(rewards) >= 2:
            total = sum(rewards)
            advantages = [reward - (total - reward) / (len(rewards) - 1) for reward in rewards]
        else:
            advantages = [0.0] * len(rewards)
        datums: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for (items, ranking), advantage in zip(ranked_groups, advantages, strict=True):
            generation = items[0].rubric_generation
            if generation is None or not generation.valid:
                continue
            depth = int(items[0].group.get("child_depth", 0)) if self._include_depth_metadata() else None
            datums.append(completion_to_datum(generation.completion_entry, advantage, trajectory_depth=depth, trajectory_start=True))
            for item in items:
                result = item.judge_result
                records.append({
                    "fork_group_id": item.group.get("id"),
                    "branch_index": int(item.branch.get("branch_index", -1)),
                    "trajectory_id": item.branch.get("trajectory_id"),
                    "code_success": bool(item.branch.get("success")),
                    "kimi_success": item.judge_success,
                    "kimi_reason": result.reason,
                    "kimi_output": result.text,
                    "policy_score": item.policy_score,
                    "group_ranking_reward": ranking.group_reward,
                    "advantage": advantage,
                    "rubric_completion_id": generation.completion_id,
                    "rubric_output": generation.text,
                    "scoring_output": item.scoring_generation.text if item.scoring_generation else None,
                    "trained_action": "rubric_generation",
                })
        return StageBatchResult(
            datums=datums, records=records,
            metrics={
                "kimi_rubric/complete_fork_groups": float(len(groups)),
                "kimi_rubric/valid_rubric_groups": float(len(valid_groups)),
                "kimi_rubric/mixed_groups": float(len(ranked_groups)),
                "kimi_rubric/pairs": float(pair_count),
                "kimi_rubric/invalid_rubrics": float(invalid_rubrics),
                "kimi_rubric/invalid_policy_scores": float(invalid_scores),
                "kimi_rubric/invalid_judges": float(invalid_judges),
                "kimi_rubric/judge_candidates": float(len(score_items)),
                "kimi_rubric/judge_valid": float(sum(getattr(item, "judge_valid", False) for item in score_items)),
                "kimi_rubric/pairwise_accuracy": mean(accuracies) if accuracies else 0.0,
                "kimi_rubric/margin_satisfaction_rate": mean(margin_rates) if margin_rates else 0.0,
                "kimi_rubric/mean_score_gap": mean(gaps) if gaps else 0.0,
                "kimi_rubric/mean_hinge_loss": mean(losses) if losses else 0.0,
                "kimi_rubric/trained_rubric_actions": float(len(datums)),
            },
        )
