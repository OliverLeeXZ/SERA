from __future__ import annotations

import json
from typing import Any

from .base import AgentRuntime, ThreeStageEnvironmentAdapter


class TextCraftThreeStageAdapter(ThreeStageEnvironmentAdapter):
    name = "textcraft"

    def canonical_subtask_key(self, trajectory: dict[str, Any]) -> str | None:
        task = trajectory.get("task") or {}
        misc = (task.get("misc") or {}) if isinstance(task, dict) else {}
        targets = misc.get("target_items")
        if not isinstance(targets, dict) or not targets:
            return None
        normalized = sorted((str(item), int(count)) for item, count in targets.items())
        return json.dumps(normalized, separators=(",", ":"))

    def serialize_agent_trajectory(
        self, trajectory: dict[str, Any], mode: str = "judge"
    ) -> dict[str, Any]:
        task = trajectory.get("task") or {}
        steps = []
        for index, step in enumerate(trajectory.get("steps") or []):
            entry = {
                "step": index,
                "thought": step.get("thought"),
                "code": step.get("code"),
                "output": step.get("output"),
                "error": step.get("error"),
            }
            if mode == "audit":
                entry["reward_misc"] = step.get("misc", {}).get("reward_misc", {})
            steps.append(entry)
        return {
            "task_goal": task.get("goal") if isinstance(task, dict) else None,
            "steps": steps,
            "finish_message": trajectory.get("finish_message"),
            "error_message": trajectory.get("error_message"),
        }

    def final_state_from_trajectory(
        self, trajectory: dict[str, Any]
    ) -> dict[str, Any]:
        for step in reversed(trajectory.get("steps") or []):
            reward_misc = step.get("misc", {}).get("reward_misc", {})
            final_inventory = reward_misc.get("final_inventory")
            if isinstance(final_inventory, dict):
                return {"inventory": dict(final_inventory)}
        return {}

    async def fork_isolated(
        self, parent_agent: Any, parent_env: Any, subtask: Any
    ) -> AgentRuntime:
        initial_state = self.snapshot(parent_env)
        child_agent = await parent_agent.fork(subtask)
        child_env = await parent_env.fork(subtask)
        child_env.code_executor.inventory = dict(initial_state["inventory"])
        return AgentRuntime(agent=child_agent, env=child_env)

    def snapshot(self, env: Any) -> dict[str, Any]:
        return {"inventory": dict(env.code_executor.inventory)}

    def commit(self, parent_env: Any, state: dict[str, Any]) -> None:
        inventory = parent_env.code_executor.inventory
        inventory.clear()
        inventory.update(state["inventory"])

    def is_success(self, trajectory: Any) -> bool:
        for step in reversed(trajectory.steps):
            reward_misc = step.misc.get("reward_misc", {})
            if float(reward_misc.get("reward/success", 0.0)) > 0:
                return True
        return float(trajectory.reward) > 0
