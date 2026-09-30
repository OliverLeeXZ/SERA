"""Group-relative reward transforms shared by rollout workflows."""

import torch


def normalize_group_relative_rewards(
    rewards: torch.Tensor,
    task_rewards: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return standard GRPO rewards using population-standard-deviation scaling."""

    if epsilon <= 0:
        raise ValueError(f"group_advantage_epsilon must be positive, got {epsilon}")
    if task_rewards.numel() == 0:
        raise ValueError("Cannot normalize an empty GRPO reward group")

    group_std = task_rewards.float().std(unbiased=False)
    denominator = group_std.clamp_min(epsilon).to(
        device=rewards.device,
        dtype=rewards.dtype,
    )
    return rewards / denominator
