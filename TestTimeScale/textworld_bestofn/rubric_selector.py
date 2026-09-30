from __future__ import annotations

import asyncio
import copy
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from Runtime.rubric.prompts import build_rubric_messages, build_scoring_messages
from Runtime.rubric.scoring import parse_policy_score, parse_rubric

from .kimi_oracle import KimiTextWorldJudge


@dataclass(frozen=True)
class RubricSelectionConfig:
    branching_factor: int = 2
    candidate_temperature: float = 1.0
    rubric_temperature: float = 0.0
    scoring_temperature: float = 0.0
    max_rubric_tokens: int = 1024
    max_scoring_tokens: int = 512
    min_rubric_criteria: int = 2
    selection_mode: str = "rubric"
    max_forked_environments: int = 32

    def __post_init__(self) -> None:
        if self.branching_factor < 1:
            raise ValueError("branching_factor must be positive")
        if self.selection_mode not in {"rubric", "oracle", "first", "random"}:
            raise ValueError(
                "TextWorld selection_mode must be rubric, oracle, first, or random"
            )
        if self.max_forked_environments < self.branching_factor:
            raise ValueError("max_forked_environments must be at least branching_factor")
        if min(self.candidate_temperature, self.rubric_temperature, self.scoring_temperature) < 0:
            raise ValueError("Selector temperatures must be non-negative")
        if min(self.max_rubric_tokens, self.max_scoring_tokens, self.min_rubric_criteria) < 1:
            raise ValueError("Rubric/scoring token budgets and criteria count must be positive")


@dataclass
class ForkBudget:
    maximum: int
    used: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def reserve(self, amount: int) -> bool:
        async with self.lock:
            if self.used + amount > self.maximum:
                return False
            self.used += amount
            return True


def clone_coordinator(source: Any) -> Any:
    """Clone the complete V9 pure-Python coordinator state."""
    cloned = copy.copy(source)
    for name, value in vars(source).items():
        if name == "_lock":
            continue
        setattr(cloned, name, copy.deepcopy(value))
    cloned._lock = asyncio.Lock()
    return cloned


def commit_coordinator(destination: Any, source: Any) -> None:
    """Commit a selected clone without invalidating existing Agent views."""
    lock = destination._lock
    for name, value in vars(source).items():
        if name == "_lock":
            continue
        setattr(destination, name, copy.deepcopy(value))
    destination._lock = lock


def coordinator_state(
    coordinator: Any, agent_id: str, *, game: str | None = None
) -> dict[str, Any]:
    observation, infos = coordinator.observe(agent_id)
    return {
        "game": game,
        "agent_id": agent_id,
        "location": getattr(coordinator, "_agent_locations", {}).get(agent_id),
        "observation": observation,
        "infos": infos,
    }


def serialize_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Match Project 172's TextWorld adapter scoring payload."""
    steps = []
    for index, step in enumerate(trajectory.get("steps_detail") or []):
        steps.append(
            {
                "step": index,
                "raw_response": step.get("raw_response"),
                "action": step.get("action"),
                "observation": step.get("observation"),
                "error": step.get("error", step.get("parse_error")),
            }
        )
    return {
        "task_goal": trajectory.get("goal"),
        "steps": steps,
        "finish_message": trajectory.get("finish_message"),
        "error_message": trajectory.get("error_message"),
        "reward": trajectory.get("final_score", 0.0),
    }


async def _complete(
    client: Any,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str | None]:
    try:
        completion = await asyncio.to_thread(
            client.complete,
            messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return completion.content, None
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


async def select_delegated_trajectory(
    *,
    task: Any,
    child_goal: str,
    child_max_steps: int,
    coordinator: Any,
    client: Any,
    config: RubricSelectionConfig,
    parent_messages: list[dict[str, str]],
    parent_action: str,
    parent_task: str,
    parent_agent_id: str,
    parent_depth: int,
    parent_step: int,
    run_candidate: Callable[[Any, str, int], Awaitable[dict[str, Any]]],
    kimi_judge: KimiTextWorldJudge | None = None,
    fork_budget: ForkBudget | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finish candidate subtrees, then select with policy rubric or KIMI."""
    group_id = str(uuid.uuid4())
    if fork_budget is not None and not await fork_budget.reserve(config.branching_factor):
        return (
            {
                "agent_id": f"{parent_agent_id}/fork-budget-exhausted-{group_id[:8]}",
                "parent_agent_id": parent_agent_id,
                "depth": parent_depth + 1,
                "goal": child_goal,
                "returned_to_parent": False,
                "environment_success": False,
                "environment_done": False,
                "steps": 0,
                "max_steps": child_max_steps,
                "remaining_steps": child_max_steps,
                "finish_message": "Fork budget exhausted; complete this delegation locally.",
                "final_observation": "",
                "children": [],
            },
            {
                "group_id": group_id,
                "child_goal": child_goal,
                "branching_factor": config.branching_factor,
                "selection_mode": config.selection_mode,
                "branches": [],
                "selected_branch_index": None,
                "fallback_reason": f"fork budget exhausted ({fork_budget.used}/{fork_budget.maximum})",
                "forked_environment_count": fork_budget.used,
                "max_forked_environments": fork_budget.maximum,
                "oracle_judge": config.selection_mode == "oracle",
            },
        )
    initial_state = coordinator_state(
        coordinator, parent_agent_id, game=getattr(task, "game", None)
    )
    rubric_messages: list[dict[str, str]] = []
    rubric_output = ""
    rubric_error: str | None = None
    rubric = None
    if config.selection_mode == "rubric":
        rubric_messages = build_rubric_messages(
            parent_prefix=parent_messages,
            parent_action=parent_action,
            child_goal=child_goal,
            environment_state=initial_state,
            available_tools=(
                "TextWorld environment actions, inventory lookup, finish, and "
                "launch_subagent"
            ),
            execution_budget=child_max_steps,
            min_criteria=config.min_rubric_criteria,
            parent_task=parent_task,
            parent_metadata={
                "parent_agent_id": parent_agent_id,
                "parent_depth": parent_depth,
                "child_depth": parent_depth + 1,
                "fork_step": parent_step,
            },
        )
        rubric_output, rubric_error = await _complete(
            client,
            rubric_messages,
            temperature=config.rubric_temperature,
            max_tokens=config.max_rubric_tokens,
        )
        if rubric_error is None:
            try:
                rubric = parse_rubric(rubric_output, config.min_rubric_criteria)
            except (TypeError, ValueError) as exc:
                rubric_error = f"{type(exc).__name__}: {exc}"
    elif config.selection_mode == "oracle" and kimi_judge is None:
        rubric_error = "oracle selection requires a configured KIMI Judge"

    async def one(branch_index: int) -> dict[str, Any]:
        branch_coordinator = clone_coordinator(coordinator)
        child_id = (
            f"{parent_agent_id}/subagent-{parent_step}-{branch_index}-"
            f"{uuid.uuid4().hex[:6]}"
        )
        try:
            result = await run_candidate(branch_coordinator, child_id, branch_index)
            return {
                "branch_index": branch_index,
                "coordinator": branch_coordinator,
                "trajectory": result,
                "score": None,
                "score_valid": False,
                "scoring_output": "",
                "scoring_error": None,
                "oracle_success": None,
                "oracle_reason": "",
                "oracle_output": "",
                "oracle_error": None,
                "selected": False,
            }
        except Exception as exc:
            return {
                "branch_index": branch_index,
                "coordinator": branch_coordinator,
                "trajectory": {
                    "agent_id": child_id,
                    "parent_agent_id": parent_agent_id,
                    "depth": parent_depth + 1,
                    "goal": child_goal,
                    "returned_to_parent": False,
                    "environment_success": False,
                    "steps": 0,
                    "max_steps": child_max_steps,
                    "remaining_steps": child_max_steps,
                    "finish_message": f"SubAgent crashed: {type(exc).__name__}: {exc}",
                    "final_observation": "",
                    "children": [],
                },
                "score": None,
                "score_valid": False,
                "scoring_output": "",
                "scoring_error": f"{type(exc).__name__}: {exc}",
                "oracle_success": None,
                "oracle_reason": "",
                "oracle_output": "",
                "oracle_error": None,
                "selected": False,
            }

    # Candidate recursion is complete before this gather returns, so scoring
    # and selection are bottom-up at every node.
    branches = list(
        await asyncio.gather(*(one(index) for index in range(config.branching_factor)))
    )

    if rubric is not None:
        async def score(branch: dict[str, Any]) -> None:
            trajectory = branch["trajectory"]
            state = coordinator_state(
                branch["coordinator"],
                str(trajectory["agent_id"]),
                game=getattr(task, "game", None),
            )
            messages = build_scoring_messages(
                rubric=rubric,
                child_goal=child_goal,
                trajectory=serialize_trajectory(trajectory),
                final_environment_state=state,
            )
            output, error = await _complete(
                client,
                messages,
                temperature=config.scoring_temperature,
                max_tokens=config.max_scoring_tokens,
            )
            branch["scoring_messages"] = messages
            branch["scoring_output"] = output
            branch["scoring_error"] = error
            if error is not None:
                return
            try:
                value, payload = parse_policy_score(output)
            except (TypeError, ValueError) as exc:
                branch["scoring_error"] = f"{type(exc).__name__}: {exc}"
                return
            branch["score"] = value
            branch["score_payload"] = payload
            branch["score_valid"] = True

        await asyncio.gather(*(score(branch) for branch in branches))

    if config.selection_mode == "oracle" and kimi_judge is not None:
        async def judge(branch: dict[str, Any]) -> None:
            trajectory = dict(branch["trajectory"])
            trajectory["root_goal"] = parent_task
            state = coordinator_state(
                branch["coordinator"],
                str(trajectory["agent_id"]),
                game=getattr(task, "game", None),
            )
            result = await asyncio.to_thread(
                kimi_judge.judge,
                child_goal=child_goal,
                trajectory=trajectory,
                final_environment_state=state,
            )
            branch["oracle_success"] = result.success
            branch["oracle_reason"] = result.reason
            branch["oracle_output"] = result.raw_output
            branch["oracle_error"] = result.error

        await asyncio.gather(*(judge(branch) for branch in branches))

    fallback_reason = None
    if config.selection_mode == "first":
        selected_index = 0
    elif config.selection_mode == "random":
        selected_index = random.randrange(len(branches))
    elif config.selection_mode == "oracle":
        valid = [branch for branch in branches if branch["oracle_success"] is True]
        if valid:
            selected_index = int(valid[0]["branch_index"])
        else:
            selected_index = 0
            fallback_reason = (
                rubric_error
                or next(
                    (
                        branch.get("oracle_error")
                        for branch in branches
                        if branch.get("oracle_error")
                    ),
                    "no candidate was judged successful by KIMI",
                )
            )
    else:
        valid = [branch for branch in branches if branch["score_valid"]]
        if valid:
            selected_index = int(
                max(valid, key=lambda branch: float(branch["score"]))[
                    "branch_index"
                ]
            )
        else:
            selected_index = 0
            fallback_reason = rubric_error or "all candidate scores invalid"

    selected = branches[selected_index]
    selected["selected"] = True
    commit_coordinator(coordinator, selected["coordinator"])

    serialized_branches = []
    for branch in branches:
        serialized = {key: value for key, value in branch.items() if key != "coordinator"}
        serialized_branches.append(serialized)
    group = {
        "group_id": group_id,
        "child_goal": child_goal,
        "branching_factor": config.branching_factor,
        "selection_mode": config.selection_mode,
        "rubric_messages": rubric_messages,
        "rubric_output": rubric_output,
        "rubric": rubric,
        "rubric_error": rubric_error,
        "branches": serialized_branches,
        "selected_branch_index": selected_index,
        "fallback_reason": fallback_reason,
        "oracle_judge": config.selection_mode == "oracle",
        "forked_environment_count": fork_budget.used if fork_budget is not None else None,
        "max_forked_environments": fork_budget.maximum if fork_budget is not None else None,
    }
    return selected["trajectory"], group
