"""Resolve only the external backends used by the selected training objectives."""
from collections.abc import Mapping
import os

from Runtime.clients.external_model import ExternalModelSpec, preflight_external_models
from .stage_kernel import StageSchedule


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _set(config, name, value):
    if isinstance(config, Mapping):
        config[name] = value
    else:
        setattr(config, name, value)


def required_external_models(config):
    """Fill effective endpoints/model names so preflight and runtime use the same values."""
    stages = {item.stage for item in StageSchedule.parse(_get(config, "stage_schedule")).schedule}
    specs = []
    if _get(config, "execution_reward") == "rubric":
        reward = _get(config, "rubric_subagent_reward")
        if _get(reward, "provider") == "kimi":
            endpoint = _get(reward, "endpoint", "") or os.getenv("KIMI_BASE_URL", "") or os.getenv("JUDGE_API_URL", "")
            model = os.getenv("KIMI_MODEL", _get(reward, "model", "kimi-k2.6"))
            _set(reward, "endpoint", endpoint)
            _set(reward, "model", model)
            if _get(reward, "failure_policy", "drop_trajectory") == "fallback_binary":
                raise ValueError("External rubric failures must be recorded, not replaced by fallback_binary rewards")
            for kind in (["rubric", "score"] if _get(config, "scorer_provider", "policy") == "kimi" else ["rubric"]):
                specs.append(ExternalModelSpec(
                    role=f"rubric {kind}", endpoint=endpoint, model=model,
                    endpoint_setting="rubric_subagent_reward.endpoint or KIMI_BASE_URL/JUDGE_API_URL",
                    model_setting="rubric_subagent_reward.model or KIMI_MODEL",
                    api_key_env=_get(reward, "api_key_env", "KIMI_API_KEY"), output_kind=kind,
                    temperature=_get(reward, "rubric_temperature" if kind == "rubric" else "scoring_temperature", 0.0),
                    max_tokens=_get(reward, "max_rubric_tokens" if kind == "rubric" else "max_scoring_tokens", 1024),
                    completion_token_field="max_completion_tokens",
                    min_criteria=_get(reward, "min_rubric_criteria", 2),
                    extra_body=dict(_get(reward, "extra_body", {"reasoning_effort": "none"}))))
    if _get(config, "environment") != "textworld":
        return specs
    judge = _get(config, "judge", {})
    default_endpoint = _get(judge, "endpoint", "") or os.getenv("JUDGE_API_URL", "")
    extras = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    if _get(config, "execution_reward") == "rao":
        model = os.getenv("JUDGE_MODEL", _get(judge, "model", "kimi-k2.6"))
        _set(judge, "endpoint", default_endpoint)
        _set(judge, "model", model)
        specs.append(ExternalModelSpec(
            role="TextWorld binary execution judge", endpoint=default_endpoint, model=model,
            endpoint_setting="judge.endpoint or JUDGE_API_URL", model_setting="judge.model or JUDGE_MODEL",
            api_key_env=_get(judge, "api_key_env", "KIMI_API_KEY"),
            temperature=_get(judge, "temperature", 1.0), max_tokens=_get(judge, "max_completion_tokens", 1024),
            completion_token_field="max_completion_tokens",
            extra_body=extras | {"max_prompt_tokens": _get(judge, "context_length", 10240)}))
    if "rubric_generation" in stages:
        rank = _get(config, "rubric_generation_ranking")
        endpoint = _get(rank, "judge_endpoint", "") or default_endpoint
        model = os.getenv("JUDGE_MODEL", _get(rank, "judge_model", "kimi-k2.6"))
        _set(rank, "judge_endpoint", endpoint)
        _set(rank, "judge_model", model)
        specs.append(ExternalModelSpec(
            role="TextWorld rubric-training judge", endpoint=endpoint, model=model,
            endpoint_setting="rubric_generation_ranking.judge_endpoint or JUDGE_API_URL",
            model_setting="rubric_generation_ranking.judge_model or JUDGE_MODEL",
            api_key_env=_get(rank, "judge_api_key_env", "KIMI_API_KEY"),
            temperature=_get(rank, "judge_temperature", 1.0), max_tokens=_get(rank, "judge_max_completion_tokens", 1024),
            completion_token_field="max_completion_tokens",
            extra_body=extras | {"max_prompt_tokens": _get(rank, "judge_max_prompt_tokens", 10240)}))
    return specs


def preflight_training(config, *, dry_run=False):
    preflight_external_models(required_external_models(config), dry_run=dry_run)
