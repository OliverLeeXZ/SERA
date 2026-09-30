from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from platoon.agents.actions.subagent import launch_subagent_default
from platoon.episode.context import (
    current_agent,
    current_env,
    current_trajectory,
    episode_step_timeout,
)
from platoon.episode.loop import run_episode

from .adapters.base import ThreeStageEnvironmentAdapter
from .records import BranchResult, ForkGroup
from .runtime_context import BranchContext, current_branch


@dataclass
class CounterfactualEnvironmentBudget:
    maximum: int
    root_environments: int
    created: int = field(init=False)
    complete_groups: int = 0
    _next_id: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.maximum < self.root_environments:
            raise ValueError("Environment budget must include every root rollout")
        self.created = self.root_environments

    @property
    def remaining(self) -> int:
        return max(0, self.maximum - self.created)

    async def reserve(self, branching_factor: int) -> list[str]:
        """Reserve a complete group while the rollout is below the hard cap.

        The cap is checked before every delegation. If the current rollout has
        not reached the cap, we still reserve a complete branch group even when
        that group crosses the cap. Once it has crossed the cap, subsequent
        delegations fall back to the normal single SubAgent call.
        """
        async with self._lock:
            if self.created >= self.maximum:
                return []
            count = branching_factor
            self.complete_groups += 1
            ids = [
                f"cf-env-{index:06d}"
                for index in range(self._next_id, self._next_id + count)
            ]
            self._next_id += count
            self.created += count
            return ids


class CounterfactualCollector:
    def __init__(
        self,
        *,
        adapter: ThreeStageEnvironmentAdapter,
        budget: CounterfactualEnvironmentBudget,
        branching_factor: int,
        max_subagent_depth: int,
        task_id: str,
        rollout_index: int,
    ) -> None:
        self.adapter = adapter
        self.budget = budget
        self.branching_factor = branching_factor
        self.max_subagent_depth = max_subagent_depth
        self.task_id = task_id
        self.rollout_index = rollout_index
        self.fork_groups: list[ForkGroup] = []

    async def launch_subagent(
        self,
        goal: str,
        max_steps: int = 15,
        task_misc: dict | None = None,
        verbose: bool = True,
    ) -> Any:
        parent_agent = current_agent.get()
        parent_env = current_env.get()
        parent_trajectory = current_trajectory.get()
        branch_context = current_branch.get()
        parent_depth = branch_context.depth if branch_context is not None else 0
        child_depth = parent_depth + 1

        if child_depth > self.max_subagent_depth:
            return (
                f"Cannot launch subagent beyond max_subagent_depth="
                f"{self.max_subagent_depth}; complete the task in the current agent."
            )

        environment_ids = await self.budget.reserve(self.branching_factor)
        if not environment_ids:
            return await launch_subagent_default(
                goal=goal,
                max_steps=max_steps,
                task_misc=task_misc,
                verbose=verbose,
            )

        group_id = str(uuid.uuid4())
        initial_state = self.adapter.snapshot(parent_env)
        digest = hashlib.sha256(
            json.dumps(initial_state, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        parent_misc = task_misc if task_misc is not None else parent_env.task.misc

        async def run_branch(branch_index: int, environment_id: str) -> BranchResult:
            token = current_branch.set(
                BranchContext(
                    environment_id=environment_id,
                    depth=child_depth,
                    fork_group_id=group_id,
                    branch_index=branch_index,
                )
            )
            runtime = None
            trajectory = None
            try:
                subtask = parent_env.task.fork(
                    goal,
                    max_steps=max_steps,
                    task_misc=deepcopy(parent_misc),
                )
                runtime = await self.adapter.fork_isolated(
                    parent_agent, parent_env, subtask
                )
                trajectory = await asyncio.create_task(
                    run_episode(
                        runtime.agent,
                        runtime.env,
                        timeout=episode_step_timeout.get(),
                    )
                )
                return BranchResult(
                    branch_index=branch_index,
                    trajectory_id=trajectory.id,
                    depth=child_depth,
                    reward=float(trajectory.reward),
                    success=self.adapter.is_success(trajectory),
                    initial_state=deepcopy(initial_state),
                    final_state=self.adapter.snapshot(runtime.env),
                    finish_message=trajectory.finish_message,
                    error_message=trajectory.error_message,
                )
            except Exception as exc:
                return BranchResult(
                    branch_index=branch_index,
                    trajectory_id=getattr(trajectory, "id", None),
                    depth=child_depth,
                    reward=float(getattr(trajectory, "reward", 0.0)),
                    success=False,
                    initial_state=deepcopy(initial_state),
                    final_state=(
                        self.adapter.snapshot(runtime.env)
                        if runtime is not None
                        else None
                    ),
                    finish_message=getattr(trajectory, "finish_message", None),
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            finally:
                current_branch.reset(token)

        branches = await asyncio.gather(
            *[
                run_branch(index, environment_id)
                for index, environment_id in enumerate(environment_ids)
            ]
        )
        selected = branches[0]
        selected.selected = True
        if selected.final_state is not None:
            self.adapter.commit(parent_env, selected.final_state)

        self.fork_groups.append(
            ForkGroup(
                id=group_id,
                task_id=self.task_id,
                root_rollout_index=self.rollout_index,
                parent_trajectory_id=parent_trajectory.id,
                parent_step=len(parent_trajectory.steps),
                parent_depth=parent_depth,
                child_depth=child_depth,
                child_goal=goal,
                initial_state_digest=digest,
                requested_branching_factor=self.branching_factor,
                branches=branches,
                skipped_due_to_budget=len(branches) != self.branching_factor,
            )
        )

        message = (
            selected.finish_message
            or selected.error_message
            or f"Subagent branch {selected.branch_index} completed."
        )
        if not verbose:
            return message
        return (
            f"{message}\n\nSelected branch: 0; "
            f"counterfactual branches collected: {len(branches)}.\n"
        )

    def serialized_fork_groups(self) -> list[dict[str, Any]]:
        return [asdict(group) | {"complete": group.complete} for group in self.fork_groups]
