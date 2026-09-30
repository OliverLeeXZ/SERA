from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentNode:
    trajectory_id: str
    parent_trajectory_id: str | None
    depth: int
    fork_step: int | None
    task_goal: str
    reward: float


@dataclass(frozen=True)
class DelegationEvent:
    parent_trajectory_id: str
    parent_step: int
    child_trajectory_ids: tuple[str, ...]
    descendant_leaf_ids: tuple[str, ...]
    structural_weight: float
    completion_id: str | None = None
    source_char_spans: tuple[tuple[int, int], ...] = ()


@dataclass
class BranchResult:
    branch_index: int
    trajectory_id: str | None
    depth: int
    reward: float
    success: bool
    initial_state: dict[str, Any]
    final_state: dict[str, Any] | None
    finish_message: str | None = None
    error_message: str | None = None
    selected: bool = False


@dataclass
class ForkGroup:
    id: str
    task_id: str
    root_rollout_index: int
    parent_trajectory_id: str
    parent_step: int
    parent_depth: int
    child_depth: int
    child_goal: str
    initial_state_digest: str
    requested_branching_factor: int
    branches: list[BranchResult] = field(default_factory=list)
    skipped_due_to_budget: bool = False

    @property
    def complete(self) -> bool:
        return (
            not self.skipped_due_to_budget
            and len(self.branches) == self.requested_branching_factor
            and all(branch.trajectory_id for branch in self.branches)
        )


@dataclass
class RubricScorePair:
    fork_group_id: str
    branch_index: int
    trajectory_id: str
    rubric_input_messages: list[dict[str, str]]
    rubric_completion_id: str
    rubric: dict[str, Any]
    scoring_input_messages: list[dict[str, str]]
    scoring_completion_id: str
    policy_score: float
    teacher_score: float | None = None
    raw_credit: float | None = None
    advantage: float | None = None
    valid: bool = True
