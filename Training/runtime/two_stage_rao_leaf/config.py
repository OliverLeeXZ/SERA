from __future__ import annotations

from dataclasses import dataclass

from three_stage_train.config import (
    DelegationConfig,
    OptimizationConfig,
    StageScheduleItem,
    ThreeStageTrainConfig,
)




@dataclass
class TwoStageRaoLeafConfig:
    enabled: bool = True
    environment: str = "textcraft"
    root_group_size: int = 8
    max_subagent_depth: int = 3
    filter_zero_variance_groups: bool = True
    depth_level_weighting: bool = True
    depth_level_discount_gamma: float | None = None
    checkpoint_at_stage_boundary: bool = False
    invalid_delegation_reward: float = 0.0
    launch_advantage_mode: str = "credit_weighted_root"
    launch_credit_mode: str = "leaf"
    subagent_success_gate: bool = False
    workload_weight_cap: float = 1.0
    output_dir: str = "outputs/two_stage_rao_leaf"



    def validate(self) -> None:
        if self.environment not in {"textcraft", "textworld"}:
            raise ValueError("environment must be textcraft or textworld")
        if self.root_group_size < 2:
            raise ValueError("root_group_size must be at least 2 for LOO advantages")
        if self.max_subagent_depth < 1:
            raise ValueError("max_subagent_depth must be positive")
        if self.launch_credit_mode not in {"leaf", "workload"}:
            raise ValueError("launch_credit_mode must be leaf or workload")
        if self.launch_advantage_mode not in {
            "credit_weighted_root",
            "reward_first_flat",
            "reward_first_root_balanced",
        }:
            raise ValueError(
                "launch_advantage_mode must be credit_weighted_root, "
                "reward_first_flat, or reward_first_root_balanced"
            )
        if self.workload_weight_cap <= 0:
            raise ValueError("workload_weight_cap must be positive")
        if (
            self.depth_level_weighting
            and self.depth_level_discount_gamma is not None
        ):
            raise ValueError(
                "depth_level_weighting and depth_level_discount_gamma are mutually exclusive"
            )


    def build_leaf_stage_config(self) -> ThreeStageTrainConfig:
        """Build the existing delegation processor's minimal runtime config."""
        self.validate()
        return ThreeStageTrainConfig(
            enabled=True,
            environment=self.environment,
            root_group_size=self.root_group_size,
            max_subagent_depth=self.max_subagent_depth,
            cycles=-1,
            schedule=[
                StageScheduleItem(
                    stage="delegation",
                    train_steps=1,
                    batch_size=self.root_group_size,
                )
            ],
            delegation=DelegationConfig(
                invalid_delegation_reward=self.invalid_delegation_reward,
                leaf_definition="terminal_agent_trajectory",
            ),
            optimization=OptimizationConfig(
                depth_level_weighting=self.depth_level_weighting,
                depth_level_discount_gamma=self.depth_level_discount_gamma,
                filter_zero_variance_groups=self.filter_zero_variance_groups,
                checkpoint_at_stage_boundary=False,
            ),
            output_dir=self.output_dir,
        )
