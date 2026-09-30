from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


StageName = str

VALID_STAGES: set[str] = {
    "rubric_generation",
    "subagent_execution",
    "delegation",
}


@dataclass
class StageScheduleItem:
    stage: StageName = "subagent_execution"
    train_steps: int = 20
    batch_size: int = 8

    def validate(self) -> None:
        if self.stage not in VALID_STAGES:
            raise ValueError(f"Unknown three-stage stage: {self.stage}")
        if self.train_steps <= 0:
            raise ValueError("Stage train_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("Stage batch_size must be positive")


@dataclass
class SubagentExecutionConfig:
    min_valid_subagent_trajectories_per_task: int = 8
    max_subagent_trajectories_per_task: int | None = None
    exclude_root: bool = True
    include_delegation_tokens: bool = False
    scorer_snapshot: str = "batch_start"
    rubric_temperature: float = 1.0
    scoring_temperature: float = 0.0
    max_rubric_tokens: int = 1024
    max_scoring_tokens: int = 1024


@dataclass
class DelegationConfig:
    invalid_delegation_reward: float = 0.0
    leaf_definition: str = "terminal_agent_trajectory"


@dataclass
class TeacherConfig:
    provider: str = "kimi"
    model: str = "kimi-k2.6"
    endpoint: str | None = None
    api_key_env: str = "KIMI_API_KEY"
    max_retries: int = 5
    timeout_seconds: int = 1800
    cache: bool = True
    cache_dir: str = "outputs/three_stage_teacher_cache"
    extra_body: dict[str, Any] = field(
        default_factory=lambda: {"reasoning_effort": "none"}
    )

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


@dataclass
class RubricGenerationConfig:
    branching_factor: int = 8
    max_counterfactual_envs_per_rollout: int = 16
    commit_policy: str = "branch_0"
    num_rubric_criteria_min: int = 2
    agreement_alpha: float = 0.5
    rubric_temperature: float = 1.0
    scoring_temperature: float = 0.0
    max_rubric_tokens: int = 1024
    max_scoring_tokens: int = 1024
    max_teacher_concurrency: int = 8
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    def __post_init__(self) -> None:
        if isinstance(self.teacher, Mapping):
            self.teacher = TeacherConfig(**dict(self.teacher))


@dataclass
class OptimizationConfig:
    depth_level_weighting: bool = True
    depth_level_discount_gamma: float | None = None
    filter_zero_variance_groups: bool = True
    checkpoint_at_stage_boundary: bool = True


@dataclass
class ThreeStageTrainConfig:
    enabled: bool = True
    environment: str = "textcraft"
    root_group_size: int = 8
    max_subagent_depth: int = 3
    max_policy_concurrency: int = 128
    cycles: int = 1
    schedule: list[StageScheduleItem] = field(
        default_factory=lambda: [
            StageScheduleItem("rubric_generation", 20, 8),
            StageScheduleItem("subagent_execution", 20, 8),
            StageScheduleItem("delegation", 20, 8),
        ]
    )
    subagent_execution: SubagentExecutionConfig = field(
        default_factory=SubagentExecutionConfig
    )
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    rubric_generation: RubricGenerationConfig = field(
        default_factory=RubricGenerationConfig
    )
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    output_dir: str = "outputs/three_stage_train"

    def __post_init__(self) -> None:
        self.schedule = [
            item if isinstance(item, StageScheduleItem) else StageScheduleItem(**item)
            for item in self.schedule
        ]
        if isinstance(self.subagent_execution, Mapping):
            self.subagent_execution = SubagentExecutionConfig(
                **dict(self.subagent_execution)
            )
        if isinstance(self.delegation, Mapping):
            self.delegation = DelegationConfig(**dict(self.delegation))
        if isinstance(self.rubric_generation, Mapping):
            self.rubric_generation = RubricGenerationConfig(
                **dict(self.rubric_generation)
            )
        if isinstance(self.optimization, Mapping):
            self.optimization = OptimizationConfig(**dict(self.optimization))

    @property
    def steps_per_cycle(self) -> int:
        return sum(item.train_steps for item in self.schedule)

    @property
    def finite_total_steps(self) -> int | None:
        if self.cycles == -1:
            return None
        return self.steps_per_cycle * self.cycles

    def validate(self) -> None:
        if not self.schedule:
            raise ValueError("three_stage_train.schedule must not be empty")
        for item in self.schedule:
            item.validate()
        if self.cycles == 0 or self.cycles < -1:
            raise ValueError("cycles must be positive or -1 for infinite cycling")
        if self.root_group_size < 2:
            raise ValueError("root_group_size must be at least 2")
        if self.max_subagent_depth < 1:
            raise ValueError("max_subagent_depth must be positive")
        if self.max_policy_concurrency < 1:
            raise ValueError("max_policy_concurrency must be positive")
        if self.subagent_execution.min_valid_subagent_trajectories_per_task < 2:
            raise ValueError("Stage 1 needs at least two trajectories for comparison")
        rubric = self.rubric_generation
        if rubric.branching_factor < 3:
            raise ValueError(
                "Stage 3 branching_factor must be at least 3 for leave-one-out credit"
            )
        if rubric.max_counterfactual_envs_per_rollout < 1:
            raise ValueError(
                "max_counterfactual_envs_per_rollout must include the root environment"
            )
        if rubric.commit_policy != "branch_0":
            raise ValueError("The first implementation only permits commit_policy=branch_0")
        if rubric.max_teacher_concurrency < 1:
            raise ValueError("max_teacher_concurrency must be positive")
        if not 0.0 <= rubric.agreement_alpha <= 1.0:
            raise ValueError("agreement_alpha must be in [0, 1]")
        if (
            self.optimization.depth_level_weighting
            and self.optimization.depth_level_discount_gamma is not None
        ):
            raise ValueError(
                "depth_level_weighting and depth_level_discount_gamma are mutually exclusive"
            )
