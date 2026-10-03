from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from platoon.config_defs import InferenceParams, RolloutConfig
from platoon.episode.context import budget_tracker, current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.inference import (
    DefaultInferenceGroupWorkflow,
    InferenceBenchmarkRunner,
    InferenceWorkflowConfig,
)
from platoon.textcraft.agent import TextCraftAgent, TextCraftDepthAwareAgent
from platoon.textcraft.env import create_synth_env, create_synth_depth_aware_env
from platoon.textcraft.synth_tasks import get_synth_task
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from parallel_fulltest_textcraft.sharding import ShardManifest
from .summary import summarize_records, write_summary


LOGGER = logging.getLogger("sera.textcraft_fulltest")


def release_trajectory_payload(record):
    """Keep rollout metrics in memory after the full trajectory is durable."""
    record.trajectory_collection = None
    return record


@dataclass(frozen=True)
class EvaluationSettings:
    temperature: float = 0.0
    context_length: int = 10240
    max_prompt_tokens: int = 9728
    max_completion_tokens: int = 512
    max_steps: int = 20
    subagent_max_steps: int = 20
    max_depth: int = 3
    num_rollouts_per_task: int = 1
    concurrency: int = 64
    step_timeout: int = 1800
    task_timeout: int = 3600
    task_retries: int = 1

    def __post_init__(self) -> None:
        if self.max_prompt_tokens + self.max_completion_tokens > self.context_length:
            raise ValueError("prompt and completion caps exceed context length")
        positive = (
            self.context_length,
            self.max_prompt_tokens,
            self.max_completion_tokens,
            self.max_steps,
            self.subagent_max_steps,
            self.num_rollouts_per_task,
            self.concurrency,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("evaluation limits must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")


def build_rollout_config(
    output_dir: str | Path,
    settings: EvaluationSettings,
) -> RolloutConfig:
    return RolloutConfig(
        max_steps=settings.max_steps,
        subagent_max_steps=settings.subagent_max_steps,
        max_subagent_depth=settings.max_depth,
        max_env_steps=None,
        output_dir=str(Path(output_dir) / "rollouts"),
        verbose=False,
        timeout=settings.task_timeout,
        step_timeout=settings.step_timeout,
        propogate_root_success=False,
        skip_subagent_reward_computation=False,
        inference_params=InferenceParams(
            temperature=settings.temperature,
            top_p=None,
            max_completion_tokens=settings.max_completion_tokens,
            max_prompt_tokens=settings.max_prompt_tokens,
        ),
    )


def reward_processor(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for step in trajectory.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        if not isinstance(reward_misc, dict):
            continue
        for key, value in reward_misc.items():
            if key.startswith("reward/"):
                components[key] = components.get(key, 0.0) + float(value)
    if components:
        return components.get("reward/success", 0.0), components
    return float(trajectory.get("reward", 0.0)), components


def load_task(task_id: str):
    return deepcopy(get_synth_task(task_id))


async def run_local_depth_aware_rollout(
    task,
    config: RolloutConfig,
) -> dict[str, Any] | TrajectoryCollection:
    agent = None
    env = None
    collection_token = None
    budget_token = None
    try:
        single_agent = config.max_subagent_depth == 0
        per_agent_max_steps = (config.max_steps if single_agent else config.subagent_max_steps) or 20
        llm_client = LiteLLMClient(
            model=str(config.model_name),
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        task.max_steps = per_agent_max_steps
        env = (create_synth_env(task, skip_subagent_reward_computation=config.skip_subagent_reward_computation)
               if single_agent else create_synth_depth_aware_env(
                   task, subagent_max_steps=per_agent_max_steps,
                   skip_subagent_reward_computation=config.skip_subagent_reward_computation))
        agent = (TextCraftAgent if single_agent else TextCraftDepthAwareAgent)(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )

        collection = TrajectoryCollection()
        collection_token = current_trajectory_collection.set(collection)
        if not single_agent:
            budget_token = budget_tracker.set(
                DepthAwareStepBudgetTracker(max_depth=config.max_subagent_depth)
            )
        events_path = (
            Path(config.output_dir)
            / "events"
            / f"events_{task.id}_{collection.id}.jsonl"
        )
        collection.register_event_handlers(
            JsonlFileSink(
                events_path,
                collection_id=collection.id,
                process_id=os.getpid(),
            )
        )

        rollout_task = asyncio.create_task(
            run_episode(agent, env, timeout=config.step_timeout)
        )
        try:
            await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            rollout_task.cancel()
            try:
                await asyncio.wait_for(rollout_task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            raise
        return collection.to_dict() if config.return_dict else collection
    finally:
        if budget_token is not None:
            budget_tracker.reset(budget_token)
        if collection_token is not None:
            current_trajectory_collection.reset(collection_token)
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()


class RetryingWorkflow(DefaultInferenceGroupWorkflow):
    def __init__(self, *args, task_retries: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_retries = task_retries

    async def arun_rollout(self, data, rollout_index, output_dir):
        last_record = None
        for attempt in range(self.task_retries + 1):
            last_record = await super().arun_rollout(
                data=data,
                rollout_index=rollout_index,
                output_dir=output_dir,
            )
            if last_record.error is None:
                return last_record
            if attempt < self.task_retries:
                LOGGER.warning(
                    "retry task=%s attempt=%d/%d: %s",
                    data["task_id"],
                    attempt + 1,
                    self.task_retries,
                    last_record.error,
                )
                await asyncio.sleep(min(30, 2**attempt))
        return last_record


class ProgressInferenceBenchmarkRunner(InferenceBenchmarkRunner):
    def __init__(self, *args, manifest: ShardManifest, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest = manifest

    def _load_record_from_artifacts(self, task_id, rollout_index, collection_path, metadata_path):
        # The parent derives all metrics before returning. Keep only those
        # metrics and the durable source path, including during final reporting:
        # retaining every task's full payload would inflate memory and serialize
        # a second copy of the entire benchmark into the summary report.
        record = super()._load_record_from_artifacts(
            task_id, rollout_index, collection_path, metadata_path)
        return release_trajectory_payload(record) if record is not None else None

    def _publish_summary(self, records) -> dict[str, Any]:
        summary = summarize_records(records, self.manifest)
        write_summary(self.output_dir / "progress_summary.json", summary)
        return summary

    async def run_rollout_stage(
        self,
        dataset: list[dict[str, Any]],
        resume: bool = True,
    ) -> dict[str, Any]:
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.workflow.config.num_concurrent_workers)

        async def guarded(data, rollout_index):
            async with semaphore:
                return await self._arun_single_rollout(
                    data=data,
                    rollout_index=rollout_index,
                    resume=resume,
                )

        jobs = [
            asyncio.create_task(guarded(data, rollout_index))
            for data in dataset
            for rollout_index in range(self.workflow.config.num_rollouts_per_task)
        ]
        started = time.perf_counter()
        records = []
        for completed, future in enumerate(asyncio.as_completed(jobs), start=1):
            record = await future
            release_trajectory_payload(record)
            records.append(record)
            summary = self._publish_summary(records)
            overall = summary["overall"]
            difficulty_text = " ".join(
                f"{name}={stats['successful']}/{stats['valid']}"
                for name, stats in summary["by_difficulty"].items()
            )
            elapsed = time.perf_counter() - started
            throughput = completed / elapsed * 60 if elapsed else 0.0
            print(
                "[progress] "
                f"{completed}/{len(jobs)} "
                f"success={overall['successful']}/{overall['valid']} "
                f"({overall['accuracy']:.2%}) errors={overall['errored']} "
                f"tasks_per_min={throughput:.2f} {difficulty_text} "
                f"last={record.task_id} last_wall={record.wall_time_seconds or 0:.1f}s",
                flush=True,
            )

        self._last_rollout_stage_elapsed_seconds = time.perf_counter() - started
        if self.workflow.config.fail_fast:
            first_error = next(
                (record.error for record in records if record.error is not None),
                None,
            )
            if first_error is not None:
                raise RuntimeError(first_error)
        return {
            "num_rollouts_requested": len(jobs),
            "num_rollouts_completed": len(records),
            "num_rollouts_with_errors": sum(
                record.error is not None for record in records
            ),
            "elapsed_seconds": self._last_rollout_stage_elapsed_seconds,
        }


async def evaluate_checkpoint(
    *,
    manifest: ShardManifest,
    output_dir: str | Path,
    model: str,
    base_url: str,
    api_key: str,
    settings: EvaluationSettings,
    resume: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rollout_config = build_rollout_config(output_path, settings)
    workflow_config = InferenceWorkflowConfig(
        num_rollouts_per_task=settings.num_rollouts_per_task,
        num_concurrent_workers=settings.concurrency,
        success_threshold=1.0,
        fail_fast=False,
        use_subprocesses=False,
        rollout_config=rollout_config,
    )
    workflow = RetryingWorkflow(
        rollout_fn=run_local_depth_aware_rollout,
        get_task_fn=load_task,
        config=workflow_config,
        model_name=model,
        model_endpoint=base_url,
        model_api_key=api_key,
        reward_processor=reward_processor,
        task_retries=settings.task_retries,
    )
    runner = ProgressInferenceBenchmarkRunner(
        workflow=workflow,
        output_dir=str(output_path),
        manifest=manifest,
    )
    resolved = {
        "manifest": manifest.to_dict(),
        "settings": asdict(settings),
        "model": model,
        "base_url": base_url,
    }
    (output_path / "evaluation_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n",
        encoding="utf-8",
    )

    result = await runner.arun(
        dataset=[{"task_id": task.task_id} for task in manifest.tasks],
        resume=resume,
        run_rollouts=True,
        generate_report=True,
    )
    records = runner._collect_records_from_disk()
    difficulty_report = summarize_records(records, manifest)
    write_summary(output_path / "reports" / "difficulty_report.json", difficulty_report)
    return {"benchmark": result, "difficulty": difficulty_report}
