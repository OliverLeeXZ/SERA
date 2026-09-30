from __future__ import annotations

import asyncio
import os

from platoon.config_defs import RolloutConfig
from platoon.episode.context import budget_tracker, current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from textworld_continuous_policy.agent import TextWorldAgent
from textworld_continuous_policy.env import TextWorldEnv, create_textworld_env
from textworld_continuous_policy.prompts import TextWorldPromptBuilder

from .judge import active_policy_judge


async def run_textworld_rao_policy_judge_rollout(
    task,
    config: RolloutConfig,
) -> dict | TrajectoryCollection:
    max_steps = int(config.subagent_max_steps or config.max_steps or 20)
    max_depth = int(config.max_subagent_depth or 3)
    task.max_steps = max_steps
    builder = TextWorldPromptBuilder(
        max_prompt_tokens=int(config.inference_params.max_prompt_tokens or 9728),
        max_depth=max_depth,
        max_subagent_steps=max_steps,
        allow_subagents=True,
    )
    action_space = builder._action_space(True)
    client = LiteLLMClient(
        model=config.model_name,
        base_url=config.model_endpoint,
        api_key=config.model_api_key,
    )
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
    env: TextWorldEnv | None = None
    agent: TextWorldAgent | None = None
    try:
        env = create_textworld_env(task, action_space=action_space)
        agent = TextWorldAgent(
            llm_client=client,
            inference_params=config.inference_params,
            prompt_builder=builder,
        )
        episode = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        await asyncio.wait_for(episode, timeout=config.timeout)

        result = collection.to_dict()
        judge = active_policy_judge.get()
        if judge is not None:
            await judge.judge_collection(
                result,
                task_id=str(task.id),
                collection_id=collection.id,
            )
        return result if config.return_dict else collection
    finally:
        if agent is not None:
            await agent.close()
        else:
            await client.aclose()
        if env is not None:
            await env.close()
