from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricSubagentRewardConfig:
    """Replace binary subagent rewards with policy or external rubric scores."""

    enabled: bool = True
    provider: str = "policy"
    model: str = "kimi-k2.6"
    endpoint: str = ""
    api_key_env: str = "KIMI_API_KEY"
    max_concurrency: int = 16
    max_retries: int = 5
    timeout_seconds: int = 1800
    rubric_temperature: float = 1.0
    scoring_temperature: float = 0.0
    max_rubric_tokens: int = 1024
    max_scoring_tokens: int = 1024
    min_rubric_criteria: int = 2
    cache: bool = True
    cache_dir: str = "rubric_reward_cache"
    artifact_dir: str = "rubric_reward_artifacts"
    failure_policy: str = "drop_trajectory"
    extra_body: dict[str, Any] = field(
        default_factory=lambda: {"reasoning_effort": "none"}
    )

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("Rubric-as-SubAgent-Reward must be enabled")
        if self.provider not in {"kimi", "policy"}:
            raise ValueError("provider must be kimi or policy")
        if self.provider == "kimi" and not self.endpoint:
            raise ValueError("KIMI endpoint is required")
        if self.provider == "kimi" and not self.api_key:
            raise ValueError(
                f"KIMI API key is missing from environment variable {self.api_key_env}"
            )
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.min_rubric_criteria < 1:
            raise ValueError("min_rubric_criteria must be positive")
        if self.failure_policy not in {
            "skip_rollout",
            "drop_trajectory",
            "fallback_binary",
        }:
            raise ValueError(
                "failure_policy must be skip_rollout, drop_trajectory, "
                "or fallback_binary"
            )
