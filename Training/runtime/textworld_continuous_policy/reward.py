from __future__ import annotations

from typing import Any


def textworld_root_reward(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Return the executable TextWorld success reward for the root trajectory.

    SubAgent rewards are replaced by the continuous rubric processor. The root
    trajectory deliberately keeps the environment's binary task-success signal.
    """
    for step in reversed(trajectory.get("steps") or []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        if "reward/code_success" in reward_misc:
            value = float(reward_misc["reward/code_success"])
            return value, {
                "reward/success": value,
                "reward/code_success": value,
                "root_success": value,
            }
    value = float(trajectory.get("reward", 0.0))
    return value, {"reward/success": value, "root_success": value}
