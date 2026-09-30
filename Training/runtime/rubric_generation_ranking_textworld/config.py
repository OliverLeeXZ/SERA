from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RubricGenerationRankingConfig:
    objective: str = "ranking"
    order_reward_weight: float = 0.5
    discrimination_reward_weight: float = 0.5
    discrimination_scale: float = 2.0
    branching_factor: int = 8
    max_counterfactual_envs_per_rollout: int = 96
    margin: float = 0.2
    rubric_temperature: float = 1.0
    scoring_temperature: float = 0.0
    max_rubric_tokens: int = 1024
    max_scoring_tokens: int = 512
    min_rubric_criteria: int = 2
    max_policy_concurrency: int = 32
    output_dir: str = "outputs/rubric_generation_ranking"
    judge_model: str = "kimi-k2.6"
    judge_endpoint: str = ""
    judge_api_key_env: str = "KIMI_API_KEY"
    judge_max_concurrency: int = 16
    judge_max_prompt_tokens: int = 10240
    judge_max_completion_tokens: int = 1024
    judge_temperature: float = 1.0
    judge_timeout_seconds: float = 900.0
    judge_max_retries: int = 2

    def validate(self) -> None:
        if self.objective not in {"ranking", "order_discrimination"}:
            raise ValueError("Unknown rubric-generation objective")
        if self.order_reward_weight < 0 or self.discrimination_reward_weight < 0:
            raise ValueError("Reward weights must be non-negative")
        if abs(self.order_reward_weight + self.discrimination_reward_weight - 1.0) > 1e-6:
            raise ValueError("Reward weights must sum to one")
        if self.discrimination_scale < 0:
            raise ValueError("Discrimination scale must be non-negative")
        if self.branching_factor != 8:
            raise ValueError("Rubric-generation training requires fork-8")
        if self.max_counterfactual_envs_per_rollout < 1:
            raise ValueError("max_counterfactual_envs_per_rollout must include root")
        if not 0.0 < self.margin <= 1.0:
            raise ValueError("margin must be in (0, 1]")
        if self.min_rubric_criteria < 1:
            raise ValueError("min_rubric_criteria must be positive")
        if self.max_policy_concurrency < 1 or self.judge_max_concurrency < 1:
            raise ValueError("judge and policy concurrency must be positive")
