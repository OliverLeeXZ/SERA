from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Sequence

from three_stage_train.data import completion_to_datum
from Runtime.rubric.prompts import build_scoring_messages
from Runtime.rubric.scoring import parse_policy_score, parse_rubric
from three_stage_train.stages.common import RawRollout, StageBatchResult
from three_stage_train.stages.rubric_generation import (
    RubricGenerationStage,
    _BranchWorkItem,
)

from .config import RubricGenerationRankingConfig


@dataclass(frozen=True)
class RankingResult:
    group_reward: float
    pair_count: int
    pairwise_accuracy: float
    margin_satisfaction_rate: float
    mean_score_gap: float
    mean_hinge_loss: float


def pairwise_margin_rewards(
    scores: Sequence[float],
    successes: Sequence[bool],
    margin: float,
) -> RankingResult | None:
    """Compute the call-group reward induced by pairwise margin loss."""

    if len(scores) != len(successes):
        raise ValueError("scores and successes must have equal length")
    positive = [index for index, value in enumerate(successes) if value]
    negative = [index for index, value in enumerate(successes) if not value]
    if not positive or not negative:
        return None

    correct = 0
    margin_satisfied = 0
    gaps: list[float] = []
    losses: list[float] = []
    for positive_index in positive:
        for negative_index in negative:
            gap = float(scores[positive_index]) - float(scores[negative_index])
            loss = max(0.0, margin - gap)
            gaps.append(gap)
            losses.append(loss)
            correct += int(gap > 0.0)
            margin_satisfied += int(gap >= margin)
    pair_count = len(losses)
    return RankingResult(
        group_reward=-mean(losses),
        pair_count=pair_count,
        pairwise_accuracy=correct / pair_count,
        margin_satisfaction_rate=margin_satisfied / pair_count,
        mean_score_gap=mean(gaps),
        mean_hinge_loss=mean(losses),
    )


class RubricGenerationRankingStage(RubricGenerationStage):
    """Train rubric-generation actions from fork-8 ranking supervision.

    The fork-8 trajectories and code-success labels are exactly the same as in
    Project 41's fork protocol is reused. A rubric is generated once per
    SubAgent-call node; the policy
    then scores the eight frozen trajectories in a separate context. The
    scores are evaluation-only. The pairwise ranking reward is attached to the
    single rubric-generation completion for that call node, so scorer
    completions never enter the optimizer batch.
    """

    def __init__(
        self,
        three_stage_config: Any,
        adapter: Any,
        ranking_config: RubricGenerationRankingConfig,
    ) -> None:
        super().__init__(three_stage_config, adapter)
        ranking_config.validate()
        self.ranking_config = ranking_config

    def _ranking_result(self, scores, successes):
        return pairwise_margin_rewards(scores, successes, self.ranking_config.margin)

    def _ranking_record(self, ranking):
        return {}

    def _ranking_metrics(self, ranked_groups):
        return {}

    async def process(
        self,
        raw_rollouts: list[RawRollout],
        policy_client: Any,
    ) -> StageBatchResult:
        groups = self._complete_groups(raw_rollouts)
        if not groups:
            return StageBatchResult(
                metrics={"rubric_generation_ranking/skipped_no_complete_fork_groups": 1.0}
            )

        grouped_items: list[list[_BranchWorkItem]] = []
        for raw, group in groups:
            items = self._build_group_items(raw, group)
            if len(items) == self.ranking_config.branching_factor:
                grouped_items.append(items)
        if not grouped_items:
            return StageBatchResult(
                metrics={"rubric_generation_ranking/skipped_incomplete_records": 1.0}
            )

        # One rubric is generated per call node. This is the action trained by
        # this experiment, so retain its completion entry for the final datum.
        rubric_outputs = await policy_client.generate_batch(
            [items[0].rubric_messages for items in grouped_items],
            temperature=self.ranking_config.rubric_temperature,
            max_completion_tokens=self.ranking_config.max_rubric_tokens,
        )

        valid_groups: list[list[_BranchWorkItem]] = []
        invalid_rubrics = 0
        for items, generation in zip(grouped_items, rubric_outputs, strict=True):
            if not generation.valid:
                invalid_rubrics += 1
                continue
            try:
                rubric = parse_rubric(
                    generation.text,
                    self.ranking_config.min_rubric_criteria,
                )
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
                    trajectory=self.adapter.serialize_agent_trajectory(
                        item.trajectory
                    ),
                    final_environment_state=item.branch.get("final_state") or {},
                )
            valid_groups.append(items)

        score_items = [item for items in valid_groups for item in items]
        if not score_items:
            return StageBatchResult(
                metrics={
                    "rubric_generation_ranking/complete_fork_groups": float(len(groups)),
                    "rubric_generation_ranking/invalid_rubrics": float(invalid_rubrics),
                }
            )

        # Scoring is a detached evaluator pass. Its completions are deliberately
        # never converted to training datums in this Stage 2 variant.
        score_outputs = await policy_client.generate_batch(
            [item.scoring_messages for item in score_items],
            temperature=self.ranking_config.scoring_temperature,
            max_completion_tokens=self.ranking_config.max_scoring_tokens,
        )
        for item, generation in zip(score_items, score_outputs, strict=True):
            item.scoring_generation = generation
            if not generation.valid:
                continue
            try:
                item.policy_score, _ = parse_policy_score(generation.text)
                item.policy_score_valid = True
            except (TypeError, ValueError):
                item.policy_score_valid = False

        datums: list[dict] = []
        records: list[dict[str, Any]] = []
        ranked_groups: list[
            tuple[list[_BranchWorkItem], RankingResult]
        ] = []
        pair_count = 0
        mixed_groups = 0
        zero_variance_groups = 0
        homogeneous_groups = 0
        invalid_scores = 0
        accuracies: list[float] = []
        margin_rates: list[float] = []
        score_gaps: list[float] = []
        hinge_losses: list[float] = []

        for items in valid_groups:
            valid_items = [
                item
                for item in items
                if item.scoring_generation is not None
                and item.scoring_generation.valid
                and item.policy_score_valid
            ]
            invalid_scores += len(items) - len(valid_items)
            if len(valid_items) < 2:
                continue
            scores = [item.policy_score for item in valid_items]
            successes = [bool(item.branch.get("success")) for item in valid_items]
            ranking = self._ranking_result(scores, successes)
            if ranking is None:
                # Homogeneous groups contain no success/failure pair and thus no
                # rubric-generation ranking supervision.
                homogeneous_groups += 1
                continue
            mixed_groups += 1
            pair_count += ranking.pair_count
            accuracies.append(ranking.pairwise_accuracy)
            margin_rates.append(ranking.margin_satisfaction_rate)
            score_gaps.append(ranking.mean_score_gap)
            hinge_losses.append(ranking.mean_hinge_loss)
            ranked_groups.append((valid_items, ranking))

        group_rewards = [ranking.group_reward for _, ranking in ranked_groups]
        if len(group_rewards) >= 2:
            total_reward = sum(group_rewards)
            group_advantages = [
                reward - (total_reward - reward) / (len(group_rewards) - 1)
                for reward in group_rewards
            ]
        else:
            group_advantages = [0.0] * len(group_rewards)
        if group_rewards and max(group_rewards) == min(group_rewards):
            zero_variance_groups = len(group_rewards)

        # Each call node has one rubric-generation action. Attach its group
        # advantage exactly once; do not duplicate it over the eight judge calls.
        for (valid_items, ranking), advantage in zip(
            ranked_groups,
            group_advantages,
            strict=True,
        ):
            rubric_generation = valid_items[0].rubric_generation
            if rubric_generation is None or not rubric_generation.valid:
                continue
            depth = (
                int(valid_items[0].group.get("child_depth", 0))
                if self._include_depth_metadata()
                else None
            )
            datums.append(
                completion_to_datum(
                    rubric_generation.completion_entry,
                    advantage,
                    trajectory_depth=depth,
                    trajectory_start=True,
                )
            )
            for item in valid_items:
                records.append(
                    {
                        "fork_group_id": item.group.get("id"),
                        "branch_index": int(item.branch.get("branch_index", -1)),
                        "trajectory_id": item.branch.get("trajectory_id"),
                        "code_success": bool(item.branch.get("success")),
                        "policy_score": item.policy_score,
                        "group_ranking_reward": ranking.group_reward,
                        **self._ranking_record(ranking),
                        "advantage": advantage,
                        "rubric_completion_id": rubric_generation.completion_id,
                        "rubric_input_messages": item.rubric_messages,
                        "rubric_output": rubric_generation.text,
                        "scoring_input_messages": item.scoring_messages,
                        "scoring_output": (
                            item.scoring_generation.text
                            if item.scoring_generation is not None
                            else None
                        ),
                        "trained_action": "rubric_generation",
                    }
                )

        return StageBatchResult(
            datums=datums,
            records=records,
            metrics={
                "rubric_generation_ranking/complete_fork_groups": float(len(groups)),
                "rubric_generation_ranking/valid_rubric_groups": float(len(valid_groups)),
                "rubric_generation_ranking/mixed_groups": float(mixed_groups),
                "rubric_generation_ranking/homogeneous_groups": float(homogeneous_groups),
                **self._ranking_metrics(ranked_groups),
                "rubric_generation_ranking/pairs": float(pair_count),
                "rubric_generation_ranking/invalid_rubrics": float(invalid_rubrics),
                "rubric_generation_ranking/invalid_scores": float(invalid_scores),
                "rubric_generation_ranking/zero_variance_groups": float(
                    zero_variance_groups
                ),
                "rubric_generation_ranking/pairwise_accuracy": (
                    mean(accuracies) if accuracies else 0.0
                ),
                "rubric_generation_ranking/margin_satisfaction_rate": (
                    mean(margin_rates) if margin_rates else 0.0
                ),
                "rubric_generation_ranking/mean_score_gap": (
                    mean(score_gaps) if score_gaps else 0.0
                ),
                "rubric_generation_ranking/mean_hinge_loss": (
                    mean(hinge_losses) if hinge_losses else 0.0
                ),
                "rubric_generation_ranking/trained_rubric_actions": float(
                    len(datums)
                ),
            },
        )
