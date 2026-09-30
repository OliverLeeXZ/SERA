"""Select rubric generation/scoring backends independently of stage routing."""
from dataclasses import replace
import os
from pathlib import Path

from rubric_subagent_reward.client import KimiRubricClient, PolicyRubricClient

TRAINING_ROOT = Path(__file__).resolve().parents[1]


def validate_rubric_mode(generator, scorer, scope, environment):
    if (generator, scorer) not in {("policy", "policy"), ("kimi", "policy"), ("kimi", "kimi")}:
        raise ValueError("Supported rubric generator/scorer pairs: policy/policy, kimi/policy, kimi/kimi")
    if scope not in {"per_subtask", "global"}:
        raise ValueError("rubric_scope must be per_subtask or global")
    if scope == "global" and (environment != "textcraft" or (generator, scorer) != ("policy", "policy")):
        raise ValueError("The global-rubric ablation uses the TextCraft template and policy scoring")


def prepare_rubric_config(config):
    reward = config.rubric_subagent_reward
    validate_rubric_mode(reward.provider, config.scorer_provider, config.rubric_scope, config.environment)
    if reward.provider == "kimi":
        reward.endpoint = reward.endpoint or os.getenv("KIMI_BASE_URL", "") or os.getenv("JUDGE_API_URL", "")
        reward.model = os.getenv("KIMI_MODEL", reward.model)
    reward.validate()
    if config.rubric_scope == "global":
        from global_rubric_reward.client import _prepare_global_rubric
        path = Path(config.global_rubric_template_path).expanduser()
        if not path.is_absolute():
            path = TRAINING_ROOT / path
        config.global_rubric_template_path = str(_prepare_global_rubric(path)[0])


class KimiRubricPolicyScorerClient:
    """Generate with Kimi and score with the active policy, as in source 159."""

    def __init__(self, config, **policy_kwargs):
        self.kimi_client = KimiRubricClient(config)
        self.policy_client = PolicyRubricClient(replace(config, provider="policy"), **policy_kwargs)

    async def generate_rubric(self, messages):
        return await self.kimi_client.generate_rubric(messages)

    async def score_trajectory(self, messages):
        return await self.policy_client.score_trajectory(messages)

    async def close(self):
        await self.kimi_client.close()
        await self.policy_client.close()


def create_rubric_client(config, proxy_server):
    reward = config.rubric_subagent_reward
    inference = config.workflow_config.rollout_config.inference_params
    kwargs = dict(proxy_server=proxy_server, model_name=config.workflow_config.rollout_config.model_name,
                  max_prompt_tokens=inference.max_prompt_tokens,
                  default_completion_tokens=inference.max_completion_tokens)
    if config.rubric_scope == "global":
        from global_rubric_reward.client import GlobalRubricPolicyClient
        return GlobalRubricPolicyClient(reward, config.global_rubric_template_path, **kwargs)
    if reward.provider == "policy":
        return PolicyRubricClient(reward, **kwargs)
    if config.scorer_provider == "policy":
        return KimiRubricPolicyScorerClient(reward, **kwargs)
    return KimiRubricClient(reward)
