from __future__ import annotations

import uuid
from typing import Any

from three_stage_train.adapters.base import AgentRuntime, ThreeStageEnvironmentAdapter


class TextWorldContinuousPolicyAdapter(ThreeStageEnvironmentAdapter):
    name = "textworld"
    available_tools = (
        "TextWorld environment actions, inventory lookup, finish, and launch_subagent"
    )

    def canonical_subtask_key(self, trajectory: dict[str, Any]) -> str | None:
        task = trajectory.get("task") or {}
        if not isinstance(task, dict):
            return None
        return str(task.get("goal", "")).strip() or None

    def snapshot(self, env: Any, *, include_commit_token: bool = True) -> dict[str, Any]:
        state = env.snapshot()
        if include_commit_token and getattr(env, "_counterfactual_branch", False):
            token = uuid.uuid4().hex
            self._commit_states[token] = (
                env._coordinator,
                str(env._agent_id),
                env,
            )
            state["_commit_token"] = token
        return state

    def __init__(self) -> None:
        self._commit_states: dict[str, tuple[Any, str, Any]] = {}

    async def fork_isolated(
        self, parent_agent: Any, parent_env: Any, subtask: Any
    ) -> AgentRuntime:
        return AgentRuntime(
            agent=await parent_agent.fork(subtask),
            env=await parent_env.fork_counterfactual(subtask),
        )

    def commit(self, parent_env: Any, state: dict[str, Any]) -> None:
        token = state.get("_commit_token")
        if not token:
            return
        payload = self._commit_states.pop(str(token), None)
        if payload is None:
            return
        coordinator, branch_agent_id, branch_env = payload
        parent_agent_id = parent_env._agent_id
        get_location = getattr(coordinator, "get_agent_location", None)
        if callable(get_location):
            branch_location = get_location(branch_agent_id)
        else:
            locations = getattr(coordinator, "_agent_locations", {})
            branch_location = locations.get(branch_agent_id)
        if branch_location is not None:
            coordinator._agent_locations[parent_agent_id] = branch_location
        activate_agent = getattr(coordinator, "_activate_agent", None)
        if callable(activate_agent):
            activate_agent(parent_agent_id)
        # The coordinator is transferred to the parent.  Closing the selected
        # branch after launch must not shut down the newly committed parent.
        branch_env._owns_coordinator = False
        parent_env._coordinator = coordinator
        parent_env._view = coordinator.fork_shared(parent_agent_id)
        parent_env._owns_coordinator = True

    def discard_commit_token(self, state: dict[str, Any]) -> None:
        token = state.get("_commit_token")
        if token:
            self._commit_states.pop(str(token), None)

    def serialize_agent_trajectory(
        self, trajectory: dict[str, Any], mode: str = "judge"
    ) -> dict[str, Any]:
        task = trajectory.get("task") or {}
        steps = []
        for index, step in enumerate(trajectory.get("steps") or []):
            misc = step.get("misc", {}) if isinstance(step, dict) else {}
            entry = {
                "step": index,
                "raw_response": misc.get("action_misc", {}).get(
                    "raw_response", step.get("raw_response")
                ),
                "action": step.get("action"),
                "observation": step.get("observation"),
                "error": step.get("error"),
            }
            if mode == "audit":
                entry["reward_misc"] = misc.get("reward_misc", {})
                entry["textworld"] = misc.get("textworld", {})
            steps.append(entry)
        return {
            "task_goal": task.get("goal") if isinstance(task, dict) else None,
            "steps": steps,
            "finish_message": trajectory.get("finish_message"),
            "error_message": trajectory.get("error_message"),
            "reward": trajectory.get("reward", 0.0),
        }

    def final_state_from_trajectory(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        for step in reversed(trajectory.get("steps") or []):
            state = step.get("misc", {}).get("textworld", {})
            if isinstance(state, dict) and state:
                return {
                    "observation": state.get("observation", ""),
                    "infos": state.get("infos", {}),
                    "location": state.get("location"),
                }
        return {}

    def is_success(self, trajectory: Any) -> bool:
        if isinstance(trajectory, dict):
            for step in reversed(trajectory.get("steps") or []):
                reward_misc = step.get("misc", {}).get("reward_misc", {})
                if float(reward_misc.get("reward/code_success", 0.0)) > 0:
                    return True
            return float(trajectory.get("reward", 0.0)) > 0
        return float(getattr(trajectory, "reward", 0.0)) > 0
