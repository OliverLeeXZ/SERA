from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agent_tree import AgentTree, build_agent_tree
from ..data import completion_text


@dataclass
class RawRollout:
    rollout_index: int
    collection: dict[str, Any]
    completions: dict[str, Any]


@dataclass
class StageBatchResult:
    datums: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SubagentCandidate:
    raw: RawRollout
    tree: AgentTree
    trajectory_id: str
    trajectory: dict[str, Any]
    parent_id: str
    parent_step_index: int
    parent_step: dict[str, Any]
    parent_completion_id: str
    parent_completion: Any
    depth: int
    child_goal: str
    subtask_key: str | None
    rubric_messages: list[dict[str, str]] | None = None
    rubric_generation: Any | None = None
    rubric: dict[str, Any] | None = None
    scoring_messages: list[dict[str, str]] | None = None
    scoring_generation: Any | None = None
    score: float | None = None


def root_trajectory(
    collection: dict[str, Any], tree: AgentTree | None = None
) -> tuple[str, dict[str, Any]]:
    tree = tree or build_agent_tree(collection)
    if len(tree.roots) != 1:
        raise ValueError(f"Expected one root trajectory, found {len(tree.roots)}")
    root_id = tree.roots[0]
    return root_id, collection["trajectories"][root_id]


def root_reward(collection: dict[str, Any], tree: AgentTree | None = None) -> float:
    _, root = root_trajectory(collection, tree)
    for step in reversed(root.get("steps") or []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        if "reward/success" in reward_misc:
            return float(reward_misc["reward/success"])
    return float(root.get("reward", 0.0))


def step_completion_id(step: dict[str, Any]) -> str | None:
    return step.get("misc", {}).get("action_misc", {}).get("completion_id")


def build_subagent_candidates(raw: RawRollout, adapter: Any) -> list[SubagentCandidate]:
    collection = raw.collection
    tree = build_agent_tree(collection)
    candidates: list[SubagentCandidate] = []
    for trajectory_id, node in tree.nodes.items():
        if node.depth == 0 or node.parent_trajectory_id is None or node.fork_step is None:
            continue
        trajectory = collection["trajectories"][trajectory_id]
        parent = collection["trajectories"].get(node.parent_trajectory_id)
        if parent is None:
            continue
        parent_steps = parent.get("steps") or []
        if not 0 <= node.fork_step < len(parent_steps):
            continue
        parent_step = parent_steps[node.fork_step]
        completion_id = step_completion_id(parent_step)
        if completion_id is None or completion_id not in raw.completions:
            continue
        candidates.append(
            SubagentCandidate(
                raw=raw,
                tree=tree,
                trajectory_id=trajectory_id,
                trajectory=trajectory,
                parent_id=node.parent_trajectory_id,
                parent_step_index=node.fork_step,
                parent_step=parent_step,
                parent_completion_id=completion_id,
                parent_completion=raw.completions[completion_id],
                depth=node.depth,
                child_goal=node.task_goal,
                subtask_key=adapter.canonical_subtask_key(trajectory),
            )
        )
    return candidates


def rubric_context(candidate: SubagentCandidate) -> dict[str, Any]:
    task = candidate.trajectory.get("task") or {}
    task_misc = (task.get("misc") or {}) if isinstance(task, dict) else {}
    parent = candidate.raw.collection.get("trajectories", {}).get(candidate.parent_id) or {}
    parent_task = parent.get("task") or {}
    action_space = "get_info, view_inventory, craft, finish, launch_subagent"
    return {
        "parent_prefix": candidate.parent_completion.messages,
        "parent_action": completion_text(candidate.parent_completion),
        "child_goal": candidate.child_goal,
        "environment_state": {
            "inventory": task_misc.get("initial_inventory", {})
        },
        "available_tools": action_space,
        "execution_budget": task.get("max_steps") if isinstance(task, dict) else None,
        "parent_task": parent_task.get("goal") if isinstance(parent_task, dict) else None,
        "parent_metadata": {
            "parent_trajectory_id": candidate.parent_id,
            "child_trajectory_id": candidate.trajectory_id,
            "parent_depth": candidate.depth - 1,
            "child_depth": candidate.depth,
            "fork_step": candidate.parent_step_index,
        },
    }
