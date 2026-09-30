from __future__ import annotations

from typing import Any

from three_stage_train.adapters.base import AgentRuntime, ThreeStageEnvironmentAdapter


class TextWorldDelegationAdapter(ThreeStageEnvironmentAdapter):
    """Minimal adapter for the launch-only stage.

    Project 68 uses the already-tested TextWorld rollout to construct the
    complete agent tree. The delegation credit processor only needs the tree
    and completion spans, but keeping the adapter complete makes the stage
    compatible with the generic three-stage workflow API.
    """

    name = "textworld"

    def canonical_subtask_key(self, trajectory: dict[str, Any]) -> str | None:
        task = trajectory.get("task") or {}
        goal = task.get("goal") if isinstance(task, dict) else None
        return str(goal) if goal else None

    def serialize_agent_trajectory(
        self, trajectory: dict[str, Any], mode: str = "judge"
    ) -> dict[str, Any]:
        del mode
        task = trajectory.get("task") or {}
        return {
            "task_goal": task.get("goal") if isinstance(task, dict) else None,
            "steps": [
                {
                    "step": index,
                    "thought": step.get("thought"),
                    "code": step.get("code"),
                    "output": step.get("output"),
                    "error": step.get("error"),
                }
                for index, step in enumerate(trajectory.get("steps") or [])
            ],
            "finish_message": trajectory.get("finish_message"),
            "error_message": trajectory.get("error_message"),
        }

    def final_state_from_trajectory(
        self, trajectory: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "finished": bool(trajectory.get("finish_message")),
            "error": trajectory.get("error_message"),
        }

    async def fork_isolated(
        self, parent_agent: Any, parent_env: Any, subtask: Any
    ) -> AgentRuntime:
        del parent_agent, parent_env, subtask
        raise NotImplementedError(
            "Project 68 consumes TextWorld's existing recursive rollout tree"
        )

    def snapshot(self, env: Any) -> dict[str, Any]:
        del env
        return {}

    def commit(self, parent_env: Any, state: dict[str, Any]) -> None:
        del parent_env, state

    def is_success(self, trajectory: Any) -> bool:
        if isinstance(trajectory, dict):
            for step in reversed(trajectory.get("steps") or []):
                reward_misc = (step.get("misc") or {}).get("reward_misc") or {}
                if float(reward_misc.get("reward/success", 0.0)) > 0:
                    return True
            return float(trajectory.get("reward", 0.0)) > 0
        return float(getattr(trajectory, "reward", 0.0)) > 0
