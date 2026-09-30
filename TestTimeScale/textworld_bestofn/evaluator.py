from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from Runtime.environments.textworld.composite_cooking import CompositeAgentView, CompositeCookingWorldCoordinator

from Runtime.clients.openai_chat import VLLMChatClient
from Runtime.environments.textworld.manifest import TaskManifest, TaskSpec
from Runtime.prompts.textworld_evaluation import (
    Decision,
    TextWorldPromptBuilder,
    Turn,
    extract_action,
    extract_decision,
    is_inventory_action,
)
from .summary import summarize_records, write_json_atomic
from .rubric_selector import ForkBudget, RubricSelectionConfig, select_delegated_trajectory
from .kimi_oracle import KimiTextWorldJudge, ExternalJudgeConfig


@dataclass(frozen=True)
class EvaluationSettings:
    temperature: float = 0.0
    context_length: int = 13312
    max_prompt_tokens: int = 10240
    max_completion_tokens: int = 3072
    max_steps: int = 20
    subagent_max_steps: int = 20
    max_depth: int = 3
    concurrency: int = 64
    task_retries: int = 1
    request_timeout: float = 1800.0
    enable_subagents: bool = True
    enable_reasoning: bool = True
    shared_environment_max_steps: int = 100

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.task_retries < 0 or self.request_timeout <= 0:
            raise ValueError("Temperature/retries must be non-negative and request timeout positive")
        if self.max_prompt_tokens + self.max_completion_tokens > self.context_length:
            raise ValueError("prompt and completion caps exceed context length")
        if any(
            value <= 0
            for value in (
                self.context_length,
                self.max_prompt_tokens,
                self.max_completion_tokens,
                self.max_steps,
                self.subagent_max_steps,
                self.max_depth,
                self.concurrency,
                self.shared_environment_max_steps,
            )
        ):
            raise ValueError("evaluation limits must be positive")


def _write_task(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def _is_completed(path: Path, task_id: str, manifest_sha256: str, *, retain_errors: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("task_id") == task_id
        and payload.get("manifest_sha256") == manifest_sha256
        and payload.get("status") == "completed"
        and (retain_errors or payload.get("error") is None)
    )


def _environment_success(infos: dict[str, Any]) -> bool:
    return bool(infos.get("tasksuccess", False)) or float(infos.get("score", 0.0)) >= 1.0


def _environment_done(infos: dict[str, Any]) -> bool:
    return bool(infos.get("done", False)) or bool(infos.get("taskfailure", False))


def _delegation_summary(children: list[dict[str, Any]]) -> str:
    lines = ["[SubAgent Results]"]
    for child in children:
        lines.extend(
            [
                f"- Agent {child['agent_id']}: returned={child['returned_to_parent']}, "
                f"environment_success={child['environment_success']}, steps={child['steps']}",
                f"  Budget used by subagent: {child['steps']}/{child.get('max_steps', child['steps'])} steps. "
                f"Total remaining budget for this SubAgent task is {child.get('remaining_steps', 0)} steps.",
                f"  result: {child.get('finish_message') or '(no explicit finish message)'}",
                f"  final observation: {str(child.get('final_observation', ''))[:1200]}",
            ]
        )
    return "\n".join(lines)


def _subagent_tree_stats(agent_tree: dict[str, Any]) -> dict[str, int | bool]:
    """Summarize recursive child calls for task-level logs and reports."""
    children = list(agent_tree.get("children", []))
    nodes = 0
    successful = 0

    def visit(node: dict[str, Any]) -> None:
        nonlocal nodes, successful
        nodes += 1
        if bool(node.get("environment_success") or node.get("final_tasksuccess")):
            successful += 1
        for child in node.get("children", []):
            visit(child)

    for child in children:
        visit(child)
    return {
        "subagent_called": bool(nodes),
        "subagent_direct_calls": len(children),
        "subagent_nodes": nodes,
        "subagent_successful_nodes": successful,
    }


async def _run_agent(
    *,
    task: TaskSpec,
    goal: str,
    coordinator: CompositeCookingWorldCoordinator,
    view: CompositeAgentView,
    client: VLLMChatClient,
    settings: EvaluationSettings,
    builder: TextWorldPromptBuilder,
    agent_id: str,
    parent_agent_id: str | None,
    depth: int,
    selector: RubricSelectionConfig,
    kimi_judge: KimiTextWorldJudge | None,
    fork_budget: ForkBudget,
) -> dict[str, Any]:
    """Run one Root/SubAgent while keeping only its own prompt history."""
    agent_task = replace(task, task_id=f"{task.task_id}:{agent_id}", task_description=goal)
    turns: list[Turn] = []
    step_records: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    selector_groups: list[dict[str, Any]] = []
    prompt_tokens_peak = 0
    truncated_turns = 0
    returned_to_parent = False
    finish_message = ""
    started = time.perf_counter()
    agent_max_steps = settings.max_steps if depth == 0 else settings.subagent_max_steps
    observation, infos = await view.observe_async()
    initial_observation = observation
    initial_infos = dict(infos)

    for step_index in range(settings.max_steps if depth == 0 else settings.subagent_max_steps):
        observation, infos = await view.observe_async()
        if _environment_success(infos) or _environment_done(infos):
            break
        messages, estimated_prompt_tokens, truncated = builder.build_messages(
            agent_task,
            observation,
            initial_infos,
            turns,
            agent_id=agent_id,
            depth=depth,
            agent_max_steps=agent_max_steps,
        )
        prompt_tokens_peak = max(prompt_tokens_peak, estimated_prompt_tokens)
        truncated_turns += int(truncated)
        completion = await asyncio.to_thread(
            client.complete,
            messages,
            temperature=settings.temperature,
            max_completion_tokens=settings.max_completion_tokens,
        )
        decision: Decision = extract_decision(completion.content)
        common = {
            "step": step_index,
            "raw_response": completion.content,
            "reasoning_content": completion.reasoning_content,
            "decision": decision.kind,
            "parse_ok": decision.parse_ok,
            "parse_error": decision.error,
            "usage": completion.usage,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "response_model": completion.model,
            "response_id": completion.response_id,
        }

        if decision.kind == "finish":
            finish_message = decision.finish_message
            returned_to_parent = depth > 0
            step_records.append({**common, "action_type": "finish", "finish_message": finish_message})
            break

        if decision.kind == "delegate":
            if not settings.enable_subagents or depth >= settings.max_depth:
                rejection = (
                    f"SubAgent delegation rejected: current depth {depth} reached the "
                    f"maximum allowed depth {settings.max_depth}."
                )
                observation_after, infos_after = await view.observe_async()
                turns.append(
                    Turn(completion.content, "", False, f"{rejection}\n{observation_after}", dict(infos_after))
                )
                step_records.append({**common, "action_type": "delegation_rejected", "result": rejection})
                continue

            delegation_children: list[dict[str, Any]] = []
            for child_index, request in enumerate(decision.delegations):
                async def run_candidate(
                    branch_coordinator: Any,
                    child_id: str,
                    branch_index: int,
                ) -> dict[str, Any]:
                    del branch_index
                    return await _run_agent(
                        task=task,
                        goal=request.goal,
                        coordinator=branch_coordinator,
                        view=branch_coordinator.fork_shared(child_id),
                        client=client,
                        settings=replace(
                            settings,
                            temperature=selector.candidate_temperature,
                            subagent_max_steps=min(
                                request.max_steps, settings.subagent_max_steps
                            ),
                        ),
                        builder=builder,
                        agent_id=child_id,
                        parent_agent_id=agent_id,
                        depth=depth + 1,
                        selector=selector,
                        kimi_judge=kimi_judge,
                        fork_budget=fork_budget,
                    )

                selected_child, selection_group = await select_delegated_trajectory(
                    task=task,
                    child_goal=request.goal,
                    child_max_steps=min(request.max_steps, settings.subagent_max_steps),
                    coordinator=coordinator,
                    client=client,
                    config=selector,
                    parent_messages=messages,
                    parent_action=completion.content,
                    parent_task=goal,
                    parent_agent_id=agent_id,
                    parent_depth=depth,
                    parent_step=step_index * 100 + child_index,
                    run_candidate=run_candidate,
                    kimi_judge=kimi_judge,
                    fork_budget=fork_budget,
                )
                delegation_children.append(selected_child)
                selector_groups.append(selection_group)
            children.extend(delegation_children)
            observation_after, infos_after = await view.observe_async()
            parent_observation = _delegation_summary(delegation_children) + "\n\n[Current Environment Observation]\n" + observation_after
            turns.append(Turn(completion.content, "", decision.parse_ok, parent_observation, dict(infos_after)))
            step_records.append({**common, "action_type": "delegation", "children": delegation_children})
            if _environment_success(infos_after) or _environment_done(infos_after):
                break
            continue

        # A normal environment action. For malformed responses, retain the old
        # evaluator's best-effort fallback and let the environment decide validity.
        action = decision.action if decision.kind == "action" else extract_action(completion.content)[0]
        if is_inventory_action(action):
            # Match TextCraft's view_inventory(): the lookup is a model-visible action,
            # but it must not consume a canonical TextWorld state transition.
            current_observation, current_infos = await view.observe_async()
            inventory = TextWorldPromptBuilder.format_inventory(dict(current_infos))
            inventory_observation = (
                "[Inventory Lookup]\n"
                "Current shared inventory (visible to all Agents):\n"
                f"{inventory}\n\n"
                "Current environment observation:\n"
                f"{current_observation}"
            )
            step_records.append(
                {
                    **common,
                    "action_type": "inventory_lookup",
                    "action": action,
                    "inventory": inventory,
                    "observation": inventory_observation,
                }
            )
            turns.append(
                Turn(
                    raw_response=completion.content,
                    action=action,
                    parse_ok=decision.parse_ok,
                    observation=inventory_observation,
                    infos=dict(current_infos),
                )
            )
            continue
        next_observation, reward, done, next_infos = await view.step(action)
        step_records.append(
            {
                **common,
                "action_type": "environment_action",
                "action": action,
                "observation": next_observation,
                "reward": reward,
                "score": next_infos.get("score", 0.0),
                "done": bool(done),
                "tasksuccess": bool(next_infos.get("tasksuccess", False)),
                "taskfailure": bool(next_infos.get("taskfailure", False)),
                "valid_actions": list(next_infos.get("validActions", [])),
            }
        )
        turns.append(
            Turn(
                raw_response=completion.content,
                action=action,
                parse_ok=decision.parse_ok,
                observation=next_observation,
                infos=dict(next_infos),
            )
        )
        if _environment_success(next_infos) or done:
            break

    final_observation, final_infos = await view.observe_async()
    return {
        "agent_id": agent_id,
        "parent_agent_id": parent_agent_id,
        "depth": depth,
        "goal": goal,
        "returned_to_parent": returned_to_parent,
        "finish_message": finish_message,
        "environment_success": _environment_success(final_infos),
        "environment_done": _environment_done(final_infos),
        "steps": len(step_records),
        "max_steps": agent_max_steps,
        "remaining_steps": max(agent_max_steps - len(step_records), 0),
        "final_score": float(final_infos.get("score", 0.0)),
        "final_tasksuccess": bool(final_infos.get("tasksuccess", False)),
        "final_taskfailure": bool(final_infos.get("taskfailure", False)),
        "peak_estimated_prompt_tokens": prompt_tokens_peak,
        "truncated_context_steps": truncated_turns,
        "wall_time_seconds": time.perf_counter() - started,
        "initial_observation": initial_observation,
        "initial_infos": initial_infos,
        "final_observation": final_observation,
        "final_infos": dict(final_infos),
        "steps_detail": step_records,
        "children": children,
        "rubric_selection_groups": selector_groups,
    }


def _run_task_sync(
    task: TaskSpec,
    manifest_sha256: str,
    client: VLLMChatClient,
    settings: EvaluationSettings,
    selector: RubricSelectionConfig,
    kimi_judge: KimiTextWorldJudge | None,
) -> dict[str, Any]:
    return asyncio.run(
        _run_task_async(task, manifest_sha256, client, settings, selector, kimi_judge)
    )


async def _run_task_async(
    task: TaskSpec,
    manifest_sha256: str,
    client: VLLMChatClient,
    settings: EvaluationSettings,
    selector: RubricSelectionConfig,
    kimi_judge: KimiTextWorldJudge | None,
) -> dict[str, Any]:
    if task.game == "cookingworld_multidish":
        parallelism = dict(
            (task.generation_properties or {}).get("parallelism", {}) or {}
        )
        task_step_limit = int(
            parallelism.get(
                "shared_environment_max_steps",
                settings.shared_environment_max_steps,
            )
        )
        coordinator = CompositeCookingWorldCoordinator(
            task,
            env_step_limit=task_step_limit,
        )
    else:
        raise ValueError("Only fixed V9 cookingworld_multidish tasks are supported")
    builder = TextWorldPromptBuilder(
        settings.max_prompt_tokens,
        max_depth=settings.max_depth,
        max_subagent_steps=settings.subagent_max_steps,
        allow_subagents=settings.enable_subagents,
    )
    started = time.perf_counter()
    try:
        if task.game == "cookingworld_multidish":
            coordinator.reset(seed=task.seed, game_fold="test")
        root = await _run_agent(
            task=task,
            goal=task.task_description,
            coordinator=coordinator,
            view=coordinator.fork_shared("root"),
            client=client,
            settings=settings,
            builder=builder,
            agent_id="root",
            parent_agent_id=None,
            depth=0,
            selector=selector,
            kimi_judge=kimi_judge,
            fork_budget=ForkBudget(selector.max_forked_environments),
        )
        subagent_stats = _subagent_tree_stats(root)
        return {
            "schema_version": 2,
            "status": "completed",
            "error": None,
            "manifest_sha256": manifest_sha256,
            "task_id": task.task_id,
            "game": task.game,
            "fold": task.fold,
            "difficulty": task.difficulty,
            "seed": task.seed,
            "game_params": task.game_params,
            "task_description": task.task_description,
            "success": bool(root["environment_success"]),
            "steps": root["steps"],
            "final_score": root["final_score"],
            "final_tasksuccess": root["final_tasksuccess"],
            "final_taskfailure": root["final_taskfailure"],
            **subagent_stats,
            "peak_estimated_prompt_tokens": root["peak_estimated_prompt_tokens"],
            "truncated_context_steps": root["truncated_context_steps"],
            "wall_time_seconds": time.perf_counter() - started,
            "initial_observation": root["initial_observation"],
            "initial_infos": root["initial_infos"],
            "steps_detail": root["steps_detail"],
            "agent_tree": root,
            "shared_environment_events": coordinator.get_events(),
        }
    finally:
        coordinator.close()


def _error_record(task: TaskSpec, manifest_sha256: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed",
        "error": f"{type(error).__name__}: {error}",
        "manifest_sha256": manifest_sha256,
        "task_id": task.task_id,
        "game": task.game,
        "fold": task.fold,
        "difficulty": task.difficulty,
        "seed": task.seed,
        "success": False,
        "steps": 0,
        "wall_time_seconds": 0.0,
    }


async def evaluate_manifest(
    *,
    manifest: TaskManifest,
    output_dir: str | Path,
    model: str,
    base_url: str,
    api_key: str,
    settings: EvaluationSettings,
    selector: RubricSelectionConfig,
    judge_config: ExternalJudgeConfig | None = None,
    resume: bool = True,
    resume_retain_errors: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rollouts = output / "rollouts"
    rollouts.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output / "evaluation_config.json",
        {
            "dataset": manifest.dataset,
            "split": manifest.split,
            "manifest_sha256": manifest.source_sha256,
            "task_count": len(manifest.tasks),
            "model": model,
            "base_url": base_url,
            "settings": asdict(settings),
            "rubric_selection": asdict(selector),
            "external_judge": asdict(judge_config) if selector.selection_mode == "oracle" and judge_config else None,
        },
    )
    client = VLLMChatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=settings.request_timeout,
        enable_reasoning=settings.enable_reasoning,
    )
    kimi_judge: KimiTextWorldJudge | None = None
    if selector.selection_mode == "oracle":
        if judge_config is None:
            raise ValueError("oracle selection requires explicit external judge configuration")
        judge_config.validate()
        kimi_judge = KimiTextWorldJudge(
            endpoint=judge_config.endpoint, api_key=os.getenv(judge_config.api_key_env, ""),
            model=judge_config.model, timeout=judge_config.timeout, retries=judge_config.retries,
            max_prompt_tokens=judge_config.max_prompt_tokens,
            max_completion_tokens=judge_config.max_completion_tokens,
            temperature=judge_config.temperature,
        )
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def run_one(task: TaskSpec) -> dict[str, Any]:
        destination = rollouts / f"{task.task_id}.json"
        if resume and _is_completed(
            destination, task.task_id, manifest.source_sha256,
            retain_errors=resume_retain_errors,
        ):
            return json.loads(destination.read_text(encoding="utf-8"))
        last_error: Exception | None = None
        async with semaphore:
            for attempt in range(settings.task_retries + 1):
                try:
                    result = await asyncio.to_thread(
                        _run_task_sync,
                        task,
                        manifest.source_sha256,
                        client,
                        settings,
                        selector,
                        kimi_judge,
                    )
                    _write_task(destination, result)
                    return result
                except Exception as error:  # Keep one bad task/API request from killing a shard.
                    last_error = error
                    if attempt < settings.task_retries:
                        await asyncio.sleep(min(8.0, 2.0**attempt))
        result = _error_record(task, manifest.source_sha256, last_error or RuntimeError("unknown error"))
        _write_task(destination, result)
        return result

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    tasks = [asyncio.create_task(run_one(task)) for task in manifest.tasks]
    for index, future in enumerate(asyncio.as_completed(tasks), start=1):
        records.append(await future)
        last_record = records[-1]
        print(
            f"[task] task_id={last_record['task_id']} success={bool(last_record.get('success'))} "
            f"subagent_called={bool(last_record.get('subagent_called', False))} "
            f"subagent_direct_calls={int(last_record.get('subagent_direct_calls', 0))} "
            f"subagent_nodes={int(last_record.get('subagent_nodes', 0))} "
            f"subagent_successful_nodes={int(last_record.get('subagent_successful_nodes', 0))}",
            flush=True,
        )
        summary = summarize_records(records, manifest)
        write_json_atomic(output / "progress_summary.json", summary)
        overall = summary["overall"]
        print(
            f"[progress] {index}/{len(tasks)} success={overall['successful']}/{overall['completed']} "
            f"({overall['accuracy']:.2%}) errors={overall['errored']} "
            f"last={records[-1]['task_id']}",
            flush=True,
        )
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(rollouts.glob("*.json"))
        if path.is_file()
    ]
    summary = summarize_records(records, manifest)
    write_json_atomic(output / "progress_summary.json", summary)
    write_json_atomic(
        output / "reports" / "difficulty_report.json",
        summary,
    )
    write_json_atomic(
        output / "reports" / "final_report.json",
        {
            "dataset": manifest.dataset,
            "split": manifest.split,
            "manifest_sha256": manifest.source_sha256,
            "model": model,
            "settings": asdict(settings),
            "rubric_selection": asdict(selector),
            "external_judge": asdict(judge_config) if selector.selection_mode == "oracle" and judge_config else None,
            "summary": summary["overall"],
            "by_game": summary["by_game"],
            "by_difficulty": summary["by_difficulty"],
            "tasks": sorted(records, key=lambda record: record["task_id"]),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return summary
