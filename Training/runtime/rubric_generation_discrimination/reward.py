from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class RubricRewardResult:
    """Reward and diagnostics for one mixed success/failure fork group."""

    group_reward: float
    pair_count: int
    pairwise_accuracy: float
    margin_satisfaction_rate: float
    mean_score_gap: float
    mean_hinge_loss: float
    order_reward: float
    discrimination_reward: float
    score_std: float


def calculate_rubric_reward(
    scores: Sequence[float],
    successes: Sequence[bool],
    margin: float,
    *,
    order_weight: float = 0.5,
    discrimination_weight: float = 0.5,
    discrimination_scale: float = 2.0,
) -> RubricRewardResult | None:
    """Combine normalized success ordering and score discrimination.

    ``scores`` are rubric scores in [0, 1]. ``successes`` are binary labels
    supplied by the environment in TextCraft. The TextWorld project supplies
    the corresponding KIMI labels before calling this function.

    A group with no positive/negative pair returns None by design. This keeps
    the agreed all-success/all-failure filtering behavior explicit.
    """

    if len(scores) != len(successes):
        raise ValueError("scores and successes must have equal length")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    if order_weight < 0.0 or discrimination_weight < 0.0:
        raise ValueError("reward weights must be non-negative")
    if abs(order_weight + discrimination_weight - 1.0) > 1e-6:
        raise ValueError("reward weights must sum to one")
    if discrimination_scale < 0.0:
        raise ValueError("discrimination_scale must be non-negative")
    if not scores:
        return None

    checked_scores = [float(score) for score in scores]
    if any(score != score or score < 0.0 or score > 1.0 for score in checked_scores):
        raise ValueError("all rubric scores must be finite values in [0, 1]")

    positive = [index for index, value in enumerate(successes) if value]
    negative = [index for index, value in enumerate(successes) if not value]
    if not positive or not negative:
        return None

    gaps: list[float] = []
    hinge_losses: list[float] = []
    order_scores: list[float] = []
    correct = 0
    margin_satisfied = 0
    for positive_index in positive:
        for negative_index in negative:
            gap = checked_scores[positive_index] - checked_scores[negative_index]
            hinge_loss = max(0.0, margin - gap)
            gaps.append(gap)
            hinge_losses.append(hinge_loss)
            order_scores.append(max(0.0, min(1.0, gap / margin)))
            correct += int(gap > 0.0)
            margin_satisfied += int(gap >= margin)

    pair_count = len(gaps)
    score_std = pstdev(checked_scores)
    discrimination_reward = max(
        0.0, min(1.0, discrimination_scale * score_std)
    )
    order_reward = mean(order_scores)
    group_reward = (
        order_weight * order_reward
        + discrimination_weight * discrimination_reward
    )
    return RubricRewardResult(
        group_reward=group_reward,
        pair_count=pair_count,
        pairwise_accuracy=correct / pair_count,
        margin_satisfaction_rate=margin_satisfied / pair_count,
        mean_score_gap=mean(gaps),
        mean_hinge_loss=mean(hinge_losses),
        order_reward=order_reward,
        discrimination_reward=discrimination_reward,
        score_std=score_std,
    )
