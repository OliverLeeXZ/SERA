"""Best-of-N CLIs reuse Evaluation's partition, resume guards and aggregation."""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "Evaluation"))
from evaluation_common import cli as evaluation_cli, create_plan, default_protocol
from .textworld_bestofn.rubric_selector import RubricSelectionConfig
from .textworld_bestofn.kimi_oracle import ExternalJudgeConfig


def protocol(settings=None, selector=None, judge=None):
    values = default_protocol("TextWorld") | (settings or {})
    selector = selector or RubricSelectionConfig()
    values["best_of_n"] = asdict(selector)
    if selector.selection_mode == "oracle":
        if judge is None:
            raise ValueError("External judge selection requires endpoint configuration")
        judge.validate()
        values["external_judge"] = asdict(judge)
    return values


def cli(command):
    if command != "create_plan":
        return evaluation_cli("TextWorld", command)
    parser = argparse.ArgumentParser(description="Plan a recursive TextWorld Best-of-N evaluation.")
    parser.add_argument("--model", required=True, help="Exact policy served-model ID")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    defaults = default_protocol("TextWorld")
    for key, value in defaults.items():
        parser.add_argument("--" + key.replace("_", "-"),
                            **({"action": argparse.BooleanOptionalAction} if isinstance(value, bool) else {"type": type(value)}), default=value)
    for key, value in asdict(RubricSelectionConfig()).items():
        options = {"type": type(value), "default": value}
        if key == "selection_mode":
            options["choices"] = ["rubric", "oracle", "first", "random"]
        parser.add_argument("--" + key.replace("_", "-"), **options)
    parser.add_argument("--judge-endpoint", default=os.getenv("KIMI_JUDGE_ENDPOINT", os.getenv("KIMI_BASE_URL", "")))
    parser.add_argument("--judge-model", default=os.getenv("KIMI_JUDGE_MODEL", os.getenv("KIMI_MODEL", "kimi-k2.6")))
    parser.add_argument("--judge-api-key-env", default="KIMI_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--judge-timeout", type=float, default=1800.0)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--judge-max-prompt-tokens", type=int, default=10240)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=1024)
    args = parser.parse_args()
    selector = RubricSelectionConfig(**{key: getattr(args, key) for key in asdict(RubricSelectionConfig())})
    judge = ExternalJudgeConfig(endpoint=args.judge_endpoint, model=args.judge_model, api_key_env=args.judge_api_key_env,
                                temperature=args.judge_temperature, timeout=args.judge_timeout, retries=args.judge_retries,
                                max_prompt_tokens=args.judge_max_prompt_tokens,
                                max_completion_tokens=args.judge_max_completion_tokens) if selector.selection_mode == "oracle" else None
    values = protocol({key: getattr(args, key) for key in defaults}, selector, judge)
    print(json.dumps(create_plan("TextWorld", args.model, args.output_root, args.num_shards, values), indent=2))
