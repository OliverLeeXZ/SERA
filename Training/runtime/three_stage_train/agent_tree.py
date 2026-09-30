from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .records import AgentNode, DelegationEvent


@dataclass(frozen=True)
class AgentTree:
    nodes: dict[str, AgentNode]
    children: dict[str, tuple[str, ...]]
    roots: tuple[str, ...]

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(node_id for node_id in self.nodes if not self.children.get(node_id))

    def descendant_leaves(self, node_id: str) -> tuple[str, ...]:
        leaves: list[str] = []
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                raise ValueError(f"Cycle detected in trajectory tree at {current}")
            seen.add(current)
            child_ids = self.children.get(current, ())
            if not child_ids:
                leaves.append(current)
            else:
                stack.extend(reversed(child_ids))
        return tuple(leaves)


def _parent_info(trajectory: dict[str, Any]) -> tuple[str | None, int | None]:
    parent = trajectory.get("parent_info")
    if not isinstance(parent, dict):
        return None, None
    parent_id = parent.get("id")
    fork_step = parent.get("fork_step")
    return (
        str(parent_id) if parent_id is not None else None,
        int(fork_step) if fork_step is not None else None,
    )


def build_agent_tree(collection: dict[str, Any]) -> AgentTree:
    trajectories = collection.get("trajectories") or {}
    if not isinstance(trajectories, dict) or not trajectories:
        raise ValueError("Trajectory collection contains no trajectories")

    parents: dict[str, str | None] = {}
    fork_steps: dict[str, int | None] = {}
    for trajectory_id, trajectory in trajectories.items():
        parent_id, fork_step = _parent_info(trajectory)
        parents[str(trajectory_id)] = parent_id
        fork_steps[str(trajectory_id)] = fork_step

    depth_cache: dict[str, int] = {}

    def depth_for(trajectory_id: str, active: set[str] | None = None) -> int:
        if trajectory_id in depth_cache:
            return depth_cache[trajectory_id]
        active = set() if active is None else active
        if trajectory_id in active:
            raise ValueError(f"Cycle detected in parent links at {trajectory_id}")
        active.add(trajectory_id)
        parent_id = parents[trajectory_id]
        if parent_id is None or parent_id not in parents:
            depth = 0
        else:
            depth = depth_for(parent_id, active) + 1
        active.remove(trajectory_id)
        depth_cache[trajectory_id] = depth
        return depth

    children_lists: dict[str, list[str]] = defaultdict(list)
    nodes: dict[str, AgentNode] = {}
    roots: list[str] = []
    for trajectory_id, trajectory in trajectories.items():
        trajectory_id = str(trajectory_id)
        parent_id = parents[trajectory_id]
        if parent_id in trajectories:
            children_lists[parent_id].append(trajectory_id)
        else:
            roots.append(trajectory_id)
        task = trajectory.get("task") or {}
        nodes[trajectory_id] = AgentNode(
            trajectory_id=trajectory_id,
            parent_trajectory_id=parent_id,
            depth=depth_for(trajectory_id),
            fork_step=fork_steps[trajectory_id],
            task_goal=str(task.get("goal", "")) if isinstance(task, dict) else "",
            reward=float(trajectory.get("reward", 0.0)),
        )

    children = {
        parent_id: tuple(sorted(child_ids))
        for parent_id, child_ids in children_lists.items()
    }
    return AgentTree(nodes=nodes, children=children, roots=tuple(sorted(roots)))


def build_delegation_events(
    collection: dict[str, Any],
    tree: AgentTree | None = None,
) -> list[DelegationEvent]:
    tree = tree or build_agent_tree(collection)
    trajectories = collection["trajectories"]
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for child_id, node in tree.nodes.items():
        if node.parent_trajectory_id is None or node.fork_step is None:
            continue
        grouped[(node.parent_trajectory_id, node.fork_step)].append(child_id)

    root_leaves: dict[str, set[str]] = {
        root_id: set(tree.descendant_leaves(root_id)) for root_id in tree.roots
    }

    def owning_root(node_id: str) -> str:
        current = node_id
        while tree.nodes[current].parent_trajectory_id in tree.nodes:
            current = tree.nodes[current].parent_trajectory_id or current
        return current

    events: list[DelegationEvent] = []
    for (parent_id, parent_step), child_ids in sorted(grouped.items()):
        if parent_id not in trajectories:
            continue
        leaves = sorted(
            {
                leaf
                for child_id in child_ids
                for leaf in tree.descendant_leaves(child_id)
            }
        )
        denominator = len(root_leaves[owning_root(parent_id)])
        structural_weight = len(leaves) / denominator if denominator else 0.0
        completion_id = None
        parent_steps = trajectories[parent_id].get("steps") or []
        if 0 <= parent_step < len(parent_steps):
            action_misc = (
                parent_steps[parent_step].get("misc", {}).get("action_misc", {})
            )
            completion_id = action_misc.get("completion_id")
        events.append(
            DelegationEvent(
                parent_trajectory_id=parent_id,
                parent_step=parent_step,
                child_trajectory_ids=tuple(sorted(child_ids)),
                descendant_leaf_ids=tuple(leaves),
                structural_weight=structural_weight,
                completion_id=completion_id,
            )
        )
    return events
