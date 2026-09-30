from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from platoon.agents.actions.subagent import launch_subagent_default
from platoon.episode.context import (
    current_agent,
    current_env,
    current_trajectory,
    episode_step_timeout,
    current_trajectory_collection,
    subagent_launcher_override,
    budget_tracker,
)
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from three_stage_train.counterfactual import CounterfactualEnvironmentBudget
from three_stage_train.records import BranchResult, ForkGroup
from three_stage_train.runtime_context import (
    BranchContext,
    active_config,
    active_stage,
    current_branch,
    root_rollout_index,
    shared_environment_budget,
)
from textworld_continuous_policy.adapter import TextWorldContinuousPolicyAdapter
from textworld_continuous_policy.agent import TextWorldAgent
from textworld_continuous_policy.env import TextWorldEnv, create_textworld_env
from textworld_continuous_policy.prompts import TextWorldPromptBuilder


class TextWorldCounterfactualCollector:
    """Fork-8 collector that closes branch runtimes and commits one branch."""

    def __init__(self, *, adapter: Any, budget: CounterfactualEnvironmentBudget,
                 branching_factor: int, max_subagent_depth: int, task_id: str,
                 rollout_index: int) -> None:
        self.adapter = adapter
        self.budget = budget
        self.branching_factor = branching_factor
        self.max_subagent_depth = max_subagent_depth
        self.task_id = task_id
        self.rollout_index = rollout_index
        self.fork_groups: list[ForkGroup] = []

    async def launch_subagent(self, goal: str, max_steps: int = 20,
                              task_misc: dict | None = None,
                              verbose: bool = True) -> Any:
        parent_agent = current_agent.get()
        parent_env = current_env.get()
        parent_trajectory = current_trajectory.get()
        branch_context = current_branch.get()
        parent_depth = branch_context.depth if branch_context is not None else 0
        child_depth = parent_depth + 1
        if child_depth > self.max_subagent_depth:
            return f"Cannot launch subagent beyond max_subagent_depth={self.max_subagent_depth}."

        environment_ids = await self.budget.reserve(self.branching_factor)
        if not environment_ids:
            return await launch_subagent_default(
                goal=goal, max_steps=max_steps, task_misc=task_misc, verbose=verbose
            )

        group_id = str(uuid.uuid4())
        initial_state = self.adapter.snapshot(parent_env)
        digest_payload = {key: value for key, value in initial_state.items() if key != "_commit_token"}
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        parent_misc = task_misc if task_misc is not None else parent_env.task.misc
        runtimes: dict[int, Any] = {}

        async def run_branch(branch_index: int, environment_id: str) -> BranchResult:
            token = current_branch.set(BranchContext(
                environment_id=environment_id, depth=child_depth,
                fork_group_id=group_id, branch_index=branch_index,
            ))
            runtime = None
            trajectory = None
            original_owns_coordinator = None
            try:
                subtask = parent_env.task.fork(
                    goal, max_steps=max_steps, task_misc=deepcopy(parent_misc)
                )
                runtime = await self.adapter.fork_isolated(parent_agent, parent_env, subtask)
                runtimes[branch_index] = runtime
                # run_episode closes its environment in a finally block. Keep
                # the isolated coordinator alive until terminal state capture
                # and branch selection are complete.
                if hasattr(runtime.env, "_owns_coordinator"):
                    original_owns_coordinator = runtime.env._owns_coordinator
                    runtime.env._owns_coordinator = False
                trajectory = await asyncio.create_task(
                    run_episode(runtime.agent, runtime.env, timeout=episode_step_timeout.get())
                )
                final_state = self.adapter.snapshot(runtime.env, include_commit_token=False)
                return BranchResult(
                    branch_index=branch_index, trajectory_id=trajectory.id,
                    depth=child_depth, reward=float(trajectory.reward),
                    success=self.adapter.is_success(trajectory),
                    initial_state=deepcopy(initial_state),
                    final_state=final_state,
                    finish_message=trajectory.finish_message,
                    error_message=trajectory.error_message,
                )
            except Exception as exc:
                return BranchResult(
                    branch_index=branch_index,
                    trajectory_id=getattr(trajectory, "id", None),
                    depth=child_depth, reward=float(getattr(trajectory, "reward", 0.0)),
                    success=False, initial_state=deepcopy(initial_state),
                    final_state=(self.adapter.snapshot(runtime.env) if runtime else None),
                    finish_message=getattr(trajectory, "finish_message", None),
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if runtime is not None and original_owns_coordinator is not None:
                    runtime.env._owns_coordinator = original_owns_coordinator
                current_branch.reset(token)

        branches = await asyncio.gather(*(
            run_branch(index, environment_id)
            for index, environment_id in enumerate(environment_ids)
        ))
        selected = branches[0]
        selected.selected = True
        if selected.final_state is not None:
            selected_runtime = runtimes.get(selected.branch_index)
            if selected_runtime is not None:
                commit_state = self.adapter.snapshot(selected_runtime.env)
                self.adapter.commit(parent_env, commit_state)
            discard = getattr(self.adapter, "discard_commit_token", None)
            if callable(discard):
                for index, branch in enumerate(branches):
                    if index != selected.branch_index and branch.final_state is not None:
                        discard(branch.final_state)
        for index, runtime in list(runtimes.items()):
            try:
                await runtime.env.close()
            except Exception:
                pass
            try:
                await runtime.agent.close()
            except Exception:
                pass
        self.fork_groups.append(ForkGroup(
            id=group_id, task_id=self.task_id, root_rollout_index=self.rollout_index,
            parent_trajectory_id=parent_trajectory.id,
            parent_step=len(parent_trajectory.steps), parent_depth=parent_depth,
            child_depth=child_depth, child_goal=goal,
            initial_state_digest=digest, requested_branching_factor=self.branching_factor,
            branches=branches, skipped_due_to_budget=len(branches) != self.branching_factor,
        ))
        message = selected.finish_message or selected.error_message or (
            f"Subagent branch {selected.branch_index} completed."
        )
        return message if not verbose else (
            f"{message}\n\nSelected branch: 0; counterfactual branches collected: {len(branches)}.\n"
        )

    def serialized_fork_groups(self) -> list[dict[str, Any]]:
        return [asdict(group) | {"complete": group.complete} for group in self.fork_groups]


async def run_textworld_two_stage_rollout(task: Any, config: Any) -> dict[str, Any]:
    three_stage_config = active_config.get()
    stage = active_stage.get()
    if three_stage_config is None or stage != "rubric_generation":
        # This path is intentionally the normal Project 82 RAO rollout.
        from textworld_continuous_policy.rollout import run_textworld_depth_aware_rollout
        return await run_textworld_depth_aware_rollout(task, config)

    max_steps = int(config.subagent_max_steps or config.max_steps or 20)
    max_depth = int(three_stage_config.max_subagent_depth)
    task.max_steps = max_steps
    builder = TextWorldPromptBuilder(
        max_prompt_tokens=int(config.inference_params.max_prompt_tokens or 10240),
        max_depth=max_depth, max_subagent_steps=max_steps, allow_subagents=True,
    )
    client = LiteLLMClient(
        model=config.model_name, base_url=config.model_endpoint, api_key=config.model_api_key
    )
    env: TextWorldEnv | None = None
    agent: TextWorldAgent | None = None
    collection = TrajectoryCollection()
    current_trajectory_collection.set(collection)
    budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=max_depth))
    shared_budget = shared_environment_budget.get()
    if not isinstance(shared_budget, CounterfactualEnvironmentBudget):
        shared_budget = CounterfactualEnvironmentBudget(
            maximum=three_stage_config.rubric_generation.max_counterfactual_envs_per_rollout,
            root_environments=1,
        )
    rollout_index = root_rollout_index.get()
    collector = TextWorldCounterfactualCollector(
        adapter=TextWorldContinuousPolicyAdapter(), budget=shared_budget,
        branching_factor=three_stage_config.rubric_generation.branching_factor,
        max_subagent_depth=max_depth, task_id=str(task.id), rollout_index=rollout_index,
    )
    branch_token = current_branch.set(BranchContext(
        environment_id=f"root-{rollout_index}", depth=0
    ))
    launcher_token = subagent_launcher_override.set(collector.launch_subagent)
    events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{collection.id}.jsonl")
    collection.register_event_handlers(
        JsonlFileSink(events_path, collection_id=collection.id, process_id=os.getpid())
    )
    try:
        env = create_textworld_env(task, action_space=builder._action_space(True))
        agent = TextWorldAgent(
            llm_client=client, inference_params=config.inference_params, prompt_builder=builder
        )
        await asyncio.wait_for(
            asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout)),
            timeout=config.timeout,
        )
        result = collection.to_dict()
        result["_three_stage"] = {
            "stage": stage, "root_rollout_index": rollout_index,
            "fork_groups": collector.serialized_fork_groups(),
            "counterfactual_environment_count": shared_budget.created,
        }
        return result
    finally:
        subagent_launcher_override.reset(launcher_token)
        current_branch.reset(branch_token)
        if agent is not None:
            await agent.close()
        elif client is not None:
            await client.aclose()
        if env is not None:
            await env.close()
