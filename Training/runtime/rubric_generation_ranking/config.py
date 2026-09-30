from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RubricGenerationRankingConfig:
    branching_factor: int = 8
    max_counterfactual_envs_per_rollout: int = 16
    margin: float = 0.2
    rubric_temperature: float = 1.0
    scoring_temperature: float = 0.0
    max_rubric_tokens: int = 1024
    max_scoring_tokens: int = 1024
    min_rubric_criteria: int = 2
    max_policy_concurrency: int = 16
    output_dir: str = "outputs/rubric_generation_ranking"

    def validate(self) -> None:
        if self.branching_factor != 8:
            raise ValueError(
                "The rubric-generation ranking experiment requires fork-8"
            )
        if self.max_counterfactual_envs_per_rollout < 1:
            raise ValueError("The environment budget must include the root")
        if not 0.0 < self.margin <= 1.0:
            raise ValueError("margin must be in (0, 1]")
        if self.min_rubric_criteria < 1:
            raise ValueError("min_rubric_criteria must be positive")
        if self.max_policy_concurrency < 1:
            raise ValueError("max_policy_concurrency must be positive")
