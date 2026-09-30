from __future__ import annotations

import asyncio
import os

from platoon.episode.context import (
    budget_tracker,
    current_trajectory_collection,
    subagent_launcher_override,
)
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.textcraft.agent import TextCraftDepthAwareAgent
from platoon.textcraft.env import create_synth_depth_aware_env
from platoon.textcraft.synth_rollout import run_synth_depth_aware_rollout
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .adapters.textcraft import TextCraftThreeStageAdapter
from .counterfactual import CounterfactualCollector, CounterfactualEnvironmentBudget
from .runtime_context import (
    BranchContext,
    active_config,
    active_stage,
    current_branch,
    root_rollout_index,
    shared_environment_budget,
)


async def run_textcraft_three_stage_rollout(task, config):
    three_stage_config = active_config.get()
    stage = active_stage.get()
    if three_stage_config is None or stage != "rubric_generation":
        max_depth = (
            three_stage_config.max_subagent_depth
            if three_stage_config is not None
            else config.max_subagent_depth
        )
        per_agent_steps = config.subagent_max_steps or config.max_steps or 25
        return await run_synth_depth_aware_rollout(
            task,
            config,
            per_agent_max_steps=per_agent_steps,
            max_depth=max_depth,
        )

    adapter = TextCraftThreeStageAdapter()
    llm_client = LiteLLMClient(
        model=config.model_name,
        base_url=config.model_endpoint,
        api_key=config.model_api_key,
    )
    per_agent_steps = config.subagent_max_steps or config.max_steps or 25
    task.max_steps = per_agent_steps
    env = create_synth_depth_aware_env(
        task,
        subagent_max_steps=per_agent_steps,
        skip_subagent_reward_computation=config.skip_subagent_reward_computation,
    )
    agent = TextCraftDepthAwareAgent(
        llm_client=llm_client,
        inference_params=config.inference_params,
    )
    collection = TrajectoryCollection()
    current_trajectory_collection.set(collection)
    budget_tracker.set(
        DepthAwareStepBudgetTracker(
            max_depth=three_stage_config.max_subagent_depth
        )
    )

    shared_budget = shared_environment_budget.get()
    if not isinstance(shared_budget, CounterfactualEnvironmentBudget):
        shared_budget = CounterfactualEnvironmentBudget(
            maximum=three_stage_config.rubric_generation.max_counterfactual_envs_per_rollout,
            root_environments=1,
        )
    rollout_index = root_rollout_index.get()
    collector = CounterfactualCollector(
        adapter=adapter,
        budget=shared_budget,
        branching_factor=three_stage_config.rubric_generation.branching_factor,
        max_subagent_depth=three_stage_config.max_subagent_depth,
        task_id=str(task.id),
        rollout_index=rollout_index,
    )
    branch_token = current_branch.set(
        BranchContext(environment_id=f"root-{rollout_index}", depth=0)
    )
    launcher_token = subagent_launcher_override.set(collector.launch_subagent)

    events_path = os.path.join(
        config.output_dir,
        "events",
        f"events_{task.id}_{collection.id}.jsonl",
    )
    collection.register_event_handlers(
        JsonlFileSink(events_path, collection_id=collection.id, process_id=os.getpid())
    )

    try:
        rollout_task = asyncio.create_task(
            run_episode(agent, env, timeout=config.step_timeout)
        )
        await asyncio.wait_for(rollout_task, timeout=config.timeout)
        result = collection.to_dict()
        result["_three_stage"] = {
            "stage": stage,
            "root_rollout_index": rollout_index,
            "fork_groups": collector.serialized_fork_groups(),
            "counterfactual_environment_count": shared_budget.created,
        }
        return result
    finally:
        subagent_launcher_override.reset(launcher_token)
        current_branch.reset(branch_token)
        await agent.close()
        await env.close()
