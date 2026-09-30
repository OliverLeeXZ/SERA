from __future__ import annotations

import os
from typing import Any


DELEGATION_LAMBDA = float(os.getenv("TEXTWORLD_DELEGATION_LAMBDA", "0.0"))


def _is_root(trajectory: dict[str, Any]) -> bool:
    parent_info = trajectory.get("parent_info")
    return not isinstance(parent_info, dict) or not parent_info.get("id")


def _root_environment_success(trajectory: dict[str, Any]) -> float:
    for step in reversed(trajectory.get("steps") or []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        if "reward/code_success" in reward_misc:
            return float(bool(float(reward_misc["reward/code_success"])))
        if "reward/success" in reward_misc:
            return float(bool(float(reward_misc["reward/success"])))
    return float(bool(float(trajectory.get("reward", 0.0))))


def _direct_subagent_stats(trajectory: dict[str, Any]) -> tuple[int, float]:
    misc = trajectory.get("misc", {})
    return (
        int(misc.get("rao_direct_subagent_launched", 0)),
        float(misc.get("rao_direct_subagent_succeeded", 0.0)),
    )


def textworld_rao_reward(
    trajectory: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Compute the RAO reward for one node of a TextWorld agent tree.

    Root success is obtained from the executable TextWorld environment. A
    non-root node gets the strict binary result produced by the Policy Judge.
    Every node may receive a configurable delegation bonus from its direct
    children. Projects 85/86 set this lambda to zero for a pure Judge swap.
    """

    if _is_root(trajectory):
        own_success = _root_environment_success(trajectory)
    else:
        own_success = float(
            bool(trajectory.get("misc", {}).get("policy_judge_success", False))
        )

    launched, succeeded = _direct_subagent_stats(trajectory)
    success_rate = succeeded / launched if launched else 0.0
    reward = own_success + DELEGATION_LAMBDA * success_rate
    metrics = {
        "reward/success": reward,
        "reward/code_success": own_success if _is_root(trajectory) else 0.0,
        "reward/policy_judge_success": (
            own_success if not _is_root(trajectory) else 0.0
        ),
        "reward/subagent_launched": float(launched),
        "reward/subagent_succeeded": succeeded,
        "reward/subagent_success_rate": success_rate,
        "reward/rao_delegation_bonus": DELEGATION_LAMBDA * success_rate,
    }
    return reward, metrics


def textworld_root_eval_reward(
    trajectory: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Evaluation processor: report only executable Root Task success."""

    success = _root_environment_success(trajectory)
    return success, {
        "reward/code_success": success,
        "root_success": success,
    }
