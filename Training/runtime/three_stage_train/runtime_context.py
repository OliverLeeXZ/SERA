from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from .config import StageName, ThreeStageTrainConfig


@dataclass(frozen=True)
class BranchContext:
    environment_id: str
    depth: int
    fork_group_id: str | None = None
    branch_index: int | None = None


active_stage: ContextVar[StageName | None] = ContextVar(
    "three_stage_active_stage", default=None
)
active_config: ContextVar[ThreeStageTrainConfig | None] = ContextVar(
    "three_stage_active_config", default=None
)
root_rollout_index: ContextVar[int] = ContextVar(
    "three_stage_root_rollout_index", default=0
)
current_branch: ContextVar[BranchContext | None] = ContextVar(
    "three_stage_current_branch", default=None
)
current_counterfactual_collector: ContextVar[object | None] = ContextVar(
    "three_stage_counterfactual_collector", default=None
)
shared_environment_budget: ContextVar[object | None] = ContextVar(
    "three_stage_shared_environment_budget", default=None
)
