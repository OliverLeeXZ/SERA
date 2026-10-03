"""Merge one environment config with experiment-script overrides, then launch AReaL.

--dry-run needs only OmegaConf, not CUDA, AReaL, an API key or a model download.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

from sera_training.method_names import paper_method_name
from sera_training.stage_kernel import StageSchedule

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from sera_training.external_models import preflight_training


def validate(config):
    environment = config.environment
    if environment not in {"textcraft", "textworld"}:
        raise ValueError("Unsupported environment")
    if config.execution_reward not in {"rao", "rubric"}:
        raise ValueError("execution_reward must be rao or rubric")
    schedule = StageSchedule.parse(config.stage_schedule, config.train_dataset.batch_size, config.stage_cycles)
    names = {x.stage for x in schedule.schedule}
    if "execution" not in names:
        raise ValueError("An execution stage is required")
    if len(names) > 1 and config.rollout.max_head_offpolicyness != 0:
        raise ValueError("Multi-stage schedules require max_head_offpolicyness=0")
    if not config.depth_aware or config.step_credit_enabled or config.reward_mode != "rao":
        raise ValueError("These launchers require depth-aware RAO root rewards")
    if config.workflow_config.group_size < 2:
        raise ValueError("Leave-one-out training needs at least two root rollouts")
    params = config.workflow_config.rollout_config.inference_params
    if params.max_prompt_tokens + params.max_completion_tokens > config.sglang.context_length:
        raise ValueError("Prompt and completion budgets exceed the serving context")
    for key in ("rubric_subagent_reward", "rubric_generation_ranking"):
        if key in config and config[key].scoring_temperature != 0.0:
            raise ValueError("Source experiments use deterministic rubric scoring")
    if config.execution_reward == "rubric":
        # Validate routing without importing GPU/dependency-heavy runtime clients.
        generator = config.rubric_subagent_reward.provider
        scorer = config.get("scorer_provider", "policy")
        scope = config.get("rubric_scope", "per_subtask")
        if (generator, scorer) not in {("policy", "policy"), ("kimi", "policy"), ("kimi", "kimi")}:
            raise ValueError("Unsupported rubric generator/scorer pair")
        if scope not in {"per_subtask", "global"}:
            raise ValueError("Unknown rubric scope")
        if scope == "global" and (environment != "textcraft" or (generator, scorer) != ("policy", "policy")):
            raise ValueError("Global rubric uses the TextCraft template and policy scoring")
    rank = config.get("rubric_generation_ranking", {})
    objective = rank.get("objective", "ranking")
    if objective not in {"ranking", "order_discrimination"}:
        raise ValueError("Unknown rubric-generation objective")
    if objective == "order_discrimination":
        if environment != "textcraft" or "rubric_generation" not in names:
            raise ValueError("Order/discrimination requires TextCraft rubric-generation training")
        order = rank.get("order_reward_weight", 0.5)
        disc = rank.get("discrimination_reward_weight", 0.5)
        if order < 0 or disc < 0 or abs(order + disc - 1.0) > 1e-6 or rank.get("discrimination_scale", 2.0) < 0:
            raise ValueError("Invalid order/discrimination weights or scale")
    # Prevent accidental publication of credentials in resolved YAML/logs.
    serialized = OmegaConf.to_yaml(config, resolve=True)
    if any(marker in serialized for marker in ("hf_", "sk-")):
        raise ValueError("Do not put API tokens in configs; use environment variables")
    for key in ("trainer_env_vars", "inference_server_env_vars"):
        env = str(config.launcher.get(key, ""))
        if any(word in env.upper() for word in ("API_KEY=", "HUGGINGFACE_HUB_TOKEN=", "HF_TOKEN=")):
            raise ValueError("Do not serialize credentials in launcher env vars")
    return schedule


def resolve(args):
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs" / f"{args.environment}.yaml"),
        OmegaConf.create({"experiment_name": f"sera-{args.environment}-{args.experiment}",
                          "trial_name": args.run_name}),
        OmegaConf.from_dotlist(args.set),
    )
    output = args.output_root or str(ROOT / "outputs" / args.environment / args.experiment / args.run_name)
    config.cluster.fileroot = str(Path(output).expanduser().resolve())
    if args.model:
        config.actor.path = args.model
    return config, validate(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["textcraft", "textworld"], required=True)
    parser.add_argument("--experiment", required=True,
                        help="Paper method ID matching the script filename; legacy IDs are accepted for resume")
    parser.add_argument("--run-name", default="run1", help="Keep this and the output root fixed to resume")
    parser.add_argument("--model", help="Hugging Face model ID or model path visible on every node")
    parser.add_argument("--output-root", help="Shared filesystem directory visible at the same path on every node")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Last override wins")
    parser.add_argument("--launcher", choices=["ray", "local"], default="ray")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config, schedule = resolve(args)
    # Fail before importing GPU engines or allocating/submitting any workers.
    preflight_training(config, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
        return 0
    # Schema checking happens in the GPU runtime before allocation/submission.
    from sera_training.bootstrap import bootstrap
    bootstrap()
    from sera_training.trainer import TrainingConfig, normalize
    from areal.api.cli_args import to_structured_cfg
    typed = OmegaConf.to_object(to_structured_cfg(config, TrainingConfig))
    normalize(typed)
    destination = Path(config.cluster.fileroot)
    destination.mkdir(parents=True, exist_ok=True)
    # Keep each invocation's configuration without overwriting a previous launch.
    fd, name = tempfile.mkstemp(prefix="resolved-", suffix=".yaml", dir=destination)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(config, resolve=True))
    command = [sys.executable, "-m", f"areal.launcher.{args.launcher}",
               str(ROOT / "train.py"), "--config", name]
    print(f"Starting {paper_method_name(args.experiment)} "
          f"({args.environment}/{args.experiment}); policy-version schedule: {config.stage_schedule}")
    return subprocess.call(command, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
