from __future__ import annotations

import asyncio
import copy
import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent
from platoon.envs.base import Observation, Task
from platoon.envs.codeact import CodeActAction
from platoon.episode.context import (
    current_codeact_action,
    current_trajectory,
    current_trajectory_collection,
    error_message,
    finish_message,
)
from platoon.episode.trajectory import TrajectoryStep
from Runtime.environments.textworld.composite_cooking import (
    CompositeAgentView as SharedAgentView,
    CompositeCookingWorldCoordinator as SharedEnvironmentCoordinator,
)

try:
    from .prompts import is_inventory_action
except ImportError:  # Support the project's path-loaded smoke test.
    from textworld_continuous_policy.prompts import is_inventory_action


def _composite_runtime():
    return SharedEnvironmentCoordinator


def _composite_task(task: Task) -> Any:
    """Adapt the Platoon Task metadata to the V9 coordinator contract."""
    from types import SimpleNamespace

    return SimpleNamespace(
        task_description=str(task.goal or ""),
        game_params=str(task.misc.get("textworld_game_params", "{}")),
        generation_properties=dict(task.misc.get("generation_properties", {})),
    )


@dataclass
class TextWorldStep(TrajectoryStep):
    raw_response: str = ""
    action: str = ""
    code: str = ""
    output: str = ""
    observation: str = ""
    error: str | None = None
    reward: float = 0.0
    thought: str | None = None


@dataclass
class TextWorldObservation(Observation):
    action_space: str = ""
    history: list[TextWorldStep] = field(default_factory=list)
    initial_observation: str = ""
    initial_infos: dict[str, Any] = field(default_factory=dict)


def _success(infos: dict[str, Any]) -> bool:
    return bool(infos.get("tasksuccess", False)) or float(infos.get("score", 0.0)) >= 1.0


def _done(infos: dict[str, Any]) -> bool:
    return bool(infos.get("done", False)) or bool(infos.get("taskfailure", False))


def _child_result_summary(results: list[Any]) -> str:
    lines = ["[SubAgent Results]"]
    for index, result in enumerate(results):
        lines.append(f"- SubAgent {index}: {str(result).strip() or '(no explicit return)'}")
    return "\n".join(lines)


class TextWorldEnv:
    """AReaL-compatible Env over one shared TextWorldExpress coordinator.

    Root environments own the canonical simulator. Forked environments only own
    a logical Agent view, so closing a child never closes the shared world.
    """

    def __init__(
        self,
        task: Task,
        *,
        coordinator: SharedEnvironmentCoordinator | None = None,
        view: SharedAgentView | None = None,
        agent_id: str | None = None,
        owns_coordinator: bool = True,
        action_space: str = "",
    ) -> None:
        self._task = task
        self._coordinator = coordinator
        self._view = view
        self._agent_id = agent_id or f"root-{uuid.uuid4().hex}"
        self._owns_coordinator = owns_coordinator
        self._action_space = action_space
        self._state = TextWorldObservation(task=task, action_space=action_space)

    @property
    def task(self) -> Task:
        return self._task

    @property
    def coordinator(self) -> SharedEnvironmentCoordinator:
        return self._active_coordinator()

    def _active_coordinator(self) -> SharedEnvironmentCoordinator:
        """Recover a live coordinator retained by a view after wrapper cleanup.

        Composite environments do not close their coordinator when a branch
        wrapper is cleaned up.  The view therefore remains the authoritative
        owner of the live state even if the wrapper's cached reference was
        cleared by ``close``.
        """
        coordinator = self._coordinator
        if coordinator is None and self._view is not None:
            coordinator = getattr(self._view, "coordinator", None)
            if coordinator is not None:
                self._coordinator = coordinator
        if coordinator is None:
            raise RuntimeError("TextWorld environment has not been reset")
        return coordinator

    async def reset(self) -> TextWorldObservation:
        if self._coordinator is None:
            env_step_limit = int(self._task.misc.get("shared_environment_max_steps", 100))
            if self._task.misc.get("textworld_game") == "cookingworld_multidish":
                coordinator_type = _composite_runtime()
                self._coordinator = coordinator_type(
                    _composite_task(self._task),
                    env_step_limit=env_step_limit,
                )
                self._coordinator.reset()
            else:
                raise ValueError("Only fixed TextWorld-Sync composite tasks are supported")
            self._view = self._coordinator.fork_shared(self._agent_id)
        elif self._view is None:
            self._view = self._coordinator.fork_shared(self._agent_id)

        observation, infos = await self._view.observe_async()
        self._state = TextWorldObservation(
            task=self._task,
            action_space=self._action_space,
            history=[],
            initial_observation=observation,
            initial_infos=dict(infos),
            misc={"infos": dict(infos), "observation": observation},
        )
        trajectory = current_trajectory.get()
        collection = current_trajectory_collection.get()
        collection.set_trajectory_task(trajectory.id, self._task)
        trajectory.reward = 0.0
        return await self.observe()

    async def observe(self, return_copy: bool = True) -> TextWorldObservation:
        if self._view is None:
            raise RuntimeError("TextWorld environment has not been reset")
        observation, infos = await self._view.observe_async()
        self._state.misc["infos"] = dict(infos)
        self._state.misc["observation"] = observation
        self._state.finished = _success(infos) or _done(infos) or finish_message.get(None) is not None
        if return_copy:
            from copy import deepcopy

            return deepcopy(self._state)
        return self._state

    async def step(self, action: CodeActAction) -> TextWorldObservation:
        if self._view is None:
            raise RuntimeError("TextWorld environment has not been reset")
        action_token = current_codeact_action.set(action)
        try:
            decision = action.misc.get("decision", "invalid")
            output = ""
            error: str | None = action.misc.get("parse_error")
            reward = 0.0
            infos: dict[str, Any]
            observation: str

            if decision == "delegate":
                delegations = action.misc.get("delegations", [])
                try:
                    results = await asyncio.gather(
                        *(
                            launch_subagent(
                                goal=str(item["goal"]),
                                max_steps=int(item.get("max_steps", 20)),
                                task_misc=dict(self._task.misc),
                            )
                            for item in delegations
                        )
                    )
                    output = _child_result_summary(list(results))
                except Exception as exc:
                    error = f"SubAgent launch failed: {type(exc).__name__}: {exc}"
                    output = error
                observation, infos = await self._view.observe_async()
            elif decision == "finish":
                output = str(action.misc.get("finish_message", ""))
                observation, infos = await self._view.observe_async()
            elif decision == "action":
                command = str(action.parsed_code or "").strip()
                if is_inventory_action(command):
                    observation, infos = await self._view.observe_async()
                    output = (
                        "[Inventory Lookup]\nCurrent shared inventory:\n"
                        + str(infos.get("inventory", "(empty)"))
                        + "\n\nCurrent environment observation:\n"
                        + observation
                    )
                else:
                    observation, reward, done, infos = await self._view.step(command)
                    output = observation
                    events = self._active_coordinator().get_events()
                    latest_event = events[-1] if events else {}
                    if latest_event and not bool(latest_event.get("accepted", True)):
                        error = error or "Action was not accepted by the shared environment."
                    self._state.finished = bool(done) or _success(infos) or _done(infos)
            else:
                observation, infos = await self._view.observe_async()
                error = error or "The model response did not contain a valid TextWorld action block."
                output = observation

            success = float(_success(infos))
            self._state.finished = self._state.finished or success > 0 or _done(infos) or finish_message.get(None) is not None
            step = TextWorldStep(
                raw_response=str(action.misc.get("raw_response", action.action or "")),
                action=str(action.parsed_code or ""),
                code=str(action.parsed_code or ""),
                output=str(output),
                observation=str(output),
                error=error,
                reward=float(reward),
                misc={
                    "action_misc": dict(action.misc),
                    "reward_misc": {
                        "reward/success": success,
                        "reward/code_success": success,
                        "textworld_score": float(infos.get("score", 0.0)),
                        "task_done": bool(infos.get("done", False)),
                        "task_failure": bool(infos.get("taskfailure", False)),
                    },
                    "textworld": {
                        "observation": str(observation),
                        "infos": dict(infos),
                        "agent_id": self._agent_id,
                        "location": self._agent_location(),
                    },
                },
            )
            self._state.history.append(step)
            self._state.reward += float(reward)
            self._state.misc["infos"] = dict(infos)
            self._state.misc["observation"] = observation
            trajectory = current_trajectory.get()
            collection = current_trajectory_collection.get()
            collection.add_trajectory_step(trajectory.id, step)
            if self._state.finished:
                trajectory.reward = success
            return await self.observe()
        finally:
            current_codeact_action.reset(action_token)

    async def close(self) -> None:
        if self._owns_coordinator and self._coordinator is not None:
            self._coordinator.close()
            self._coordinator = None
            self._view = None

    async def fork(self, task: Task) -> "TextWorldEnv":
        coordinator = self._active_coordinator()
        agent_id = f"{self._agent_id}/subagent-{uuid.uuid4().hex[:10]}"
        view = coordinator.fork_shared(agent_id)
        child = TextWorldEnv(
            task,
            coordinator=coordinator,
            view=view,
            agent_id=agent_id,
            owns_coordinator=False,
            action_space=self._action_space,
        )
        return child

    async def fork_counterfactual(self, task: Task) -> "TextWorldEnv":
        """Fork the complete current simulator state for an independent branch."""
        source = self._active_coordinator()
        source_location = self._agent_location()
        # CompositeCookingWorldCoordinator is a pure-Python state machine;
        # copy mutable fields while creating a fresh asyncio lock.
        cloned = copy.copy(source)
        cloned._lock = asyncio.Lock()
        for name in (
            "_events", "_agent_locations", "_properties", "_params",
            "_dishes", "_gated_dishes", "_tool_cooldowns", "_inventory",
            "_prepared_on_counter", "_meal_prepared", "_meal_eaten",
            "_items", "_distractors",
        ):
            if hasattr(source, name):
                setattr(cloned, name, copy.deepcopy(getattr(source, name)))
        cloned._last = copy.deepcopy(getattr(source, "_last", None))

        child_id = f"{self._agent_id}/cf-{uuid.uuid4().hex[:10]}"
        locations = getattr(cloned, "_agent_locations", {})
        locations[child_id] = source_location or locations.get("root", "kitchen")
        view = cloned.fork_shared(child_id)
        child = TextWorldEnv(
            task,
            coordinator=cloned,
            view=view,
            agent_id=child_id,
            owns_coordinator=True,
            action_space=self._action_space,
        )
        child._counterfactual_branch = True
        return child

    def snapshot(self) -> dict[str, Any]:
        observation, infos = self._snapshot_observation()
        return {
            "game": self._task.misc.get("textworld_game"),
            "agent_id": self._agent_id,
            "location": self._agent_location(),
            "observation": observation,
            "infos": infos,
        }

    def _snapshot_observation(self) -> tuple[str, dict[str, Any]]:
        """Read a synchronous snapshot from both native and composite views."""
        coordinator = self._coordinator
        if coordinator is None and self._view is not None:
            coordinator = getattr(self._view, "coordinator", None)
            if coordinator is not None:
                self._coordinator = coordinator
        if coordinator is None:
            return "", {}
        observer = getattr(coordinator, "observe", None)
        if callable(observer):
            try:
                return observer(self._agent_id)
            except TypeError:
                return observer()
        view_observer = getattr(self._view, "observe", None)
        if callable(view_observer):
            return view_observer()
        raise AttributeError(
            f"{type(self._view).__name__} has no synchronous snapshot observer"
        )

    def _agent_location(self) -> str | None:
        coordinator = self._coordinator
        if coordinator is None and self._view is not None:
            coordinator = getattr(self._view, "coordinator", None)
            if coordinator is not None:
                self._coordinator = coordinator
        if coordinator is None:
            return None
        getter = getattr(coordinator, "get_agent_location", None)
        if callable(getter):
            return getter(self._agent_id)
        locations = getattr(coordinator, "_agent_locations", {})
        return locations.get(self._agent_id)


def create_textworld_env(task: Task, action_space: str) -> TextWorldEnv:
    task.misc.setdefault("shared_environment_max_steps", 100)
    return TextWorldEnv(task, action_space=action_space)
