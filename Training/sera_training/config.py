"""Training schemas; importing them does not load CUDA model/server engines."""
from dataclasses import dataclass, field
from platoon.train.areal.config_defs import PlatoonArealRLTrainerConfig
from rubric_subagent_reward.config import RubricSubagentRewardConfig
from rubric_generation_ranking_textworld.config import RubricGenerationRankingConfig
from two_stage_rao_leaf.config import TwoStageRaoLeafConfig


@dataclass
class JudgeConfig:
    model: str = "kimi-k2.6"
    endpoint: str = ""
    api_key_env: str = "KIMI_API_KEY"
    context_length: int = 10240
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    max_concurrency: int = 16
    request_timeout_seconds: float = 1800.0


@dataclass
class TrainingConfig(PlatoonArealRLTrainerConfig):
    train_difficulties: list[str] | None = None
    eval_difficulties: list[str] | None = None
    recursive: bool = False
    depth_aware: bool = True
    reward_mode: str = "rao"
    step_credit_enabled: bool = False
    environment: str = "textcraft"
    execution_reward: str = "rao"
    scorer_provider: str = "policy"
    rubric_scope: str = "per_subtask"
    global_rubric_template_path: str = "assets/textcraft_global_rubric.json"
    stage_schedule: str = "execution:1"
    stage_cycles: int = -1
    delegation_lambda: float = 0.0
    rubric_subagent_reward: RubricSubagentRewardConfig = field(default_factory=RubricSubagentRewardConfig)
    rubric_generation_ranking: RubricGenerationRankingConfig = field(default_factory=RubricGenerationRankingConfig)
    leaf_credit: TwoStageRaoLeafConfig = field(default_factory=TwoStageRaoLeafConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
