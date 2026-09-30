"""Reward processing utilities for TextCraft training and evaluation."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from platoon.textcraft.step_credit import StepCreditConfig, compute_step_penalties


TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP = 0.0
RewardProcessorResult = tuple[float, dict[str, float]] | tuple[float, dict[str, float], list[float]]
RewardProcessor = Callable[[dict[str, Any]], RewardProcessorResult]


def reward_processor(
    traj: dict[str, Any],
    step_credit_config: StepCreditConfig | None = None,
) -> RewardProcessorResult:
    """Process trajectory rewards and optionally produce step-level rewards."""
    rewards_dict: dict[str, float] = {}
    for step in traj["steps"]:
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith("reward/"):
                if reward_key not in rewards_dict:
                    rewards_dict[reward_key] = 0.0
                rewards_dict[reward_key] += float(reward_value)

    # Keep request failures (for example a context-length overflow) as
    # explicit zero-reward trajectories. The partial steps remain available
    # to the GRPO batch, while an earlier reward field cannot leak through.
    rollout_error_zero_reward = bool(traj.get("misc", {}).get("rollout_error_zero_reward", False))
    success_reward = 0.0 if rollout_error_zero_reward else rewards_dict.get("reward/success", 0.0)
    score = success_reward
    launched = rewards_dict.get("reward/subagent_launched", 0.0)
    if launched > 0:
        subagent_success_rate = rewards_dict.get("reward/subagent_succeeded", 0.0) / launched
        score += TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP * subagent_success_rate
    if not rewards_dict and not rollout_error_zero_reward:
        score = float(traj.get("reward", 0.0))

    if step_credit_config and step_credit_config.enabled and step_credit_config.lambda_ > 0:
        penalties = compute_step_penalties(traj, step_credit_config)
        raw_penalties = [penalty.total for penalty in penalties]
        step_rewards = [score - step_credit_config.lambda_ * penalty for penalty in raw_penalties]
        total_penalty = sum(raw_penalties)
        rewards_dict.setdefault("reward/step_credit_caused_unreachable", 0.0)
        rewards_dict["reward/step_penalty"] = total_penalty
        rewards_dict["reward/step_penalty/avg"] = total_penalty / len(raw_penalties) if raw_penalties else 0.0
        rewards_dict["reward/step_credit_score"] = (
            sum(step_rewards) / len(step_rewards) if step_rewards else score
        )
        return score, rewards_dict, step_rewards
    return score, rewards_dict


def _step_credit_config_from(config: Any) -> StepCreditConfig:
    return StepCreditConfig(
        enabled=True,
        lambda_=float(getattr(config, "step_credit_lambda", 0.0)),
        shortage=float(getattr(config, "step_credit_shortage_penalty", 1.0)),
        excess=float(getattr(config, "step_credit_excess_penalty", 1.0)),
        unnecessary=float(getattr(config, "step_credit_unnecessary_penalty", 1.0)),
        invalid=float(getattr(config, "step_credit_invalid_penalty", 1.0)),
        unreachable=float(getattr(config, "step_credit_unreachable_penalty", 1.0)),
    )


def build_reward_processor(config: Any) -> RewardProcessor:
    mode = getattr(config, "reward_mode", None)
    if mode is None:
        mode = "step_credit" if getattr(config, "step_credit_enabled", False) else "rao"
    mode = str(mode).strip().lower().replace("-", "_")
    if mode == "auto":
        mode = "step_credit" if getattr(config, "step_credit_enabled", False) else "rao"

    if mode in {"rao", "official", "original"}:
        return reward_processor
    if mode in {"step_credit", "r3ao"}:
        return partial(reward_processor, step_credit_config=_step_credit_config_from(config))
    raise ValueError(f"Unsupported reward_mode: {mode}. Expected one of: rao, step_credit.")
