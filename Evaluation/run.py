"""One-command evaluation: serve a model, evaluate the assigned shard, aggregate."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal

from evaluation_common import (ROOT, aggregate_results, atomic_json, create_plan, default_protocol,
                               evaluate_shard, exclusive_lock, read_json, use_backend)
from local_serving import local_services, models_ready


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("model_path", nargs="?", help="Local HF-format checkpoint directory or Hugging Face model ID")
    result.add_argument("--model-path", dest="model_path_option")
    result.add_argument("--backend", choices=["TextCraft", "TextWorld"], required=True)
    result.add_argument("--model-name", default="sera-policy", help="Served ID; required to match an existing API's model")
    result.add_argument("--base-url", help="Reuse an existing /v1 API; do not start or stop any server")
    result.add_argument("--output-root", type=Path)
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--gpus", default=os.getenv("GPU_IDS", "auto"))
    result.add_argument("--tensor-parallel-size", type=int, default=1)
    result.add_argument("--port-base", type=int, default=8000)
    result.add_argument("--startup-timeout", type=float, default=1800)
    result.add_argument("--trust-remote-code", action="store_true")
    result.add_argument("--server-arg", action="append", default=[], help="Extra vLLM argument token; use --server-arg=--flag")
    result.add_argument("--concurrency", type=int, default=64)
    result.add_argument("--task-retries", type=int, default=1)
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--no-enable-reasoning", action="store_true")
    result.add_argument("--best-of-n", action="store_true", help="Use TestTimeScale's recursive TextWorld selector")
    result.add_argument("--branching-factor", type=int, default=2)
    result.add_argument("--selection-mode", choices=["rubric", "oracle", "first", "random"], default="rubric")
    result.add_argument("--candidate-temperature", type=float, default=1.0)
    result.add_argument("--rubric-temperature", type=float, default=0.0)
    result.add_argument("--scoring-temperature", type=float, default=0.0)
    result.add_argument("--max-forked-environments", type=int, default=32)
    result.add_argument("--dry-run", action="store_true", help="Print settings only; no servers, data writes or GPU imports")
    return result


def execute(args):
    model_path = args.model_path_option or args.model_path
    if args.model_path_option and args.model_path and args.model_path_option != args.model_path:
        raise ValueError("Specify model path only once")
    if not model_path and not args.base_url:
        raise ValueError("Supply a model path/ID, MODEL_PATH, or an existing --base-url")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard count/index")
    if args.num_shards > 1 and args.output_root is None:
        raise ValueError("Multi-machine evaluation requires the same explicit --output-root on every machine")
    if args.concurrency < 1 or args.task_retries < 0 or args.startup_timeout <= 0:
        raise ValueError("Invalid concurrency, retries or startup timeout")
    if args.best_of_n and args.backend != "TextWorld":
        raise ValueError("Best-of-N is only supported for TextWorld")
    if model_path and Path(model_path).exists():
        model_path = str(Path(model_path).resolve())
    values = default_protocol(args.backend)
    values["temperature"] = args.temperature
    if args.backend == "TextWorld":
        values["enable_reasoning"] = not args.no_enable_reasoning
    if args.best_of_n:
        from TestTimeScale.entrypoints import protocol
        from TestTimeScale.textworld_bestofn.rubric_selector import RubricSelectionConfig
        from TestTimeScale.textworld_bestofn.kimi_oracle import ExternalJudgeConfig
        selector = RubricSelectionConfig(branching_factor=args.branching_factor, selection_mode=args.selection_mode,
                                         candidate_temperature=args.candidate_temperature, rubric_temperature=args.rubric_temperature,
                                         scoring_temperature=args.scoring_temperature, max_forked_environments=args.max_forked_environments)
        judge = ExternalJudgeConfig(endpoint=os.getenv("KIMI_JUDGE_ENDPOINT", os.getenv("KIMI_BASE_URL", "")),
                                    model=os.getenv("KIMI_JUDGE_MODEL", os.getenv("KIMI_MODEL", "kimi-k2.6"))) if args.selection_mode == "oracle" else None
        values = protocol(values, selector, judge)
        if judge and not args.dry_run and not os.getenv(judge.api_key_env):
            raise ValueError(f"External judge requires {judge.api_key_env}")
    tag = "bestofn" if args.best_of_n else "n1"
    output_base = ROOT.parent / "TestTimeScale" if args.best_of_n else ROOT
    output = (args.output_root or output_base / "outputs" / args.backend.lower() / f"{tag}-{datetime.now():%Y%m%d-%H%M%S-%f}").resolve()
    model = f"openai/{args.model_name}" if args.backend == "TextCraft" else args.model_name
    identity = dict(model_path=model_path, model=model, backend=args.backend, protocol=values)
    if args.dry_run:
        print(json.dumps(identity | dict(output_root=str(output), num_shards=args.num_shards, shard_index=args.shard_index,
                                        serving="existing-api" if args.base_url else "local-vllm", gpus=args.gpus,
                                        tensor_parallel_size=args.tensor_parallel_size), indent=2))
        return 0
    # This extra identity guards local checkpoint paths as well as served IDs.
    with exclusive_lock(output / ".launch-identity.lock"):
        path = output / "launch_identity.json"
        if path.exists() and read_json(path) != identity:
            raise ValueError("Checkpoint or protocol differs from this run; use a new output root")
        if not path.exists():
            atomic_json(path, identity)
    create_plan(args.backend, model, output, args.num_shards, values)
    # One owner per shard, including its local GPU services. Other shards use
    # separate locks while sharing the same immutable plan and output root.
    with exclusive_lock(output / f".launcher-shard-{args.shard_index:03d}.lock"):
        service = nullcontext(args.base_url) if args.base_url else local_services(
            model_path=model_path, model_name=args.model_name, output=output / "services" / f"shard-{args.shard_index:03d}",
            gpus=args.gpus, tensor_parallel_size=args.tensor_parallel_size, context_length=values["context_length"],
            startup_timeout=args.startup_timeout, port_base=args.port_base,
            trust_remote_code=args.trust_remote_code, server_args=args.server_arg)
        with service as base_url:
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
            if args.base_url and not models_ready(base_url, args.model_name, api_key):
                raise RuntimeError("Existing API is unavailable or does not serve --model-name")
            options = argparse.Namespace(run_root=output, shard_index=args.shard_index, base_url=base_url,
                                         api_key=api_key, concurrency=args.concurrency, task_retries=args.task_retries)
            if args.backend == "TextCraft":
                use_backend(args.backend)
                from textcraft_fulltest.runtime_cleanup import run_with_litellm_cleanup
                asyncio.run(run_with_litellm_cleanup(lambda: evaluate_shard(args.backend, options)))
            else:
                asyncio.run(evaluate_shard(args.backend, options))
    report = aggregate_results(output, args.backend)
    print(json.dumps(dict(output_root=str(output), complete=report["complete"], overall=report["overall"],
                          by_difficulty=report["by_difficulty"]), indent=2))
    return 0 if args.num_shards > 1 or report["complete"] else 2


def main():
    args = parser().parse_args()
    def terminated(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminated)
    try:
        return execute(args)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
