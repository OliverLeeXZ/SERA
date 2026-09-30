from __future__ import annotations

import asyncio
import os

from platoon.config_defs import RolloutConfig
from platoon.episode.context import budget_tracker, current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .agent import TextWorldAgent
from .env import TextWorldEnv, create_textworld_env
from Runtime.prompts.textworld_training import TextWorldPromptBuilder


async def run_textworld_depth_aware_rollout(
    task,
    config: RolloutConfig,
) -> dict | TrajectoryCollection:
    max_steps = int(config.subagent_max_steps or config.max_steps or 20)
    max_depth = int(config.max_subagent_depth or 3)
    task.max_steps = max_steps
    action_space = TextWorldPromptBuilder(
        max_prompt_tokens=int(config.inference_params.max_prompt_tokens or 9728),
        max_depth=max_depth,
        max_subagent_steps=max_steps,
    )._action_space(True)
    client = LiteLLMClient(
        model=config.model_name,
        base_url=config.model_endpoint,
        api_key=config.model_api_key,
    )
    builder = TextWorldPromptBuilder(
        max_prompt_tokens=int(config.inference_params.max_prompt_tokens or 9728),
        max_depth=max_depth,
        max_subagent_steps=max_steps,
        allow_subagents=True,
    )
    env: TextWorldEnv | None = None
    agent: TextWorldAgent | None = None
    collection = TrajectoryCollection()
    current_trajectory_collection.set(collection)
    budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=max_depth))
    events_path = os.path.join(
        config.output_dir,
        "events",
        f"events_{task.id}_{collection.id}.jsonl",
    )
    collection.register_event_handlers(
        JsonlFileSink(events_path, collection_id=collection.id, process_id=os.getpid())
    )
    try:
        env = create_textworld_env(task, action_space=action_space)
        agent = TextWorldAgent(
            llm_client=client,
            inference_params=config.inference_params,
            prompt_builder=builder,
        )
        episode = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        await asyncio.wait_for(episode, timeout=config.timeout)
        result: dict | TrajectoryCollection = collection.to_dict() if config.return_dict else collection
        return result
    finally:
        if agent is not None:
            await agent.close()
        elif client is not None:
            await client.aclose()
        if env is not None:
            await env.close()
