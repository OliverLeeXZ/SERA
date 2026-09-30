"""Scheduler-independent planning, shard execution and durable-result aggregation."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from Dataset.paths import evaluation_manifest
from Runtime.bootstrap import bootstrap as bootstrap_runtime

bootstrap_runtime()
DIFFICULTIES = ("easy", "medium", "hard", "extreme")
MANIFESTS = {
    "TextCraft": "textcraft_synth_val_all_632.json",
    "TextWorld": "textworld_multidish_test_1400_v9.json",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {path}; do not run the same shard twice") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def use_backend(backend: str) -> None:
    if backend not in MANIFESTS:
        raise ValueError(f"Unknown evaluation backend: {backend}")
    sys.path.insert(0, str(ROOT / backend))


def default_protocol(backend: str, *, single_agent: bool = False) -> dict:
    protocol = dict(temperature=0.0, context_length=10240,
                    max_prompt_tokens=9728, max_completion_tokens=512,
                    max_steps=20, subagent_max_steps=20, max_depth=3)
    if backend == "TextWorld":
        protocol.update(context_length=13312, max_prompt_tokens=10240,
                        max_completion_tokens=3072, enable_subagents=True,
                        enable_reasoning=True, shared_environment_max_steps=100)
    if single_agent:
        protocol.update(max_steps=200, max_depth=0)
        if backend == "TextWorld":
            protocol.update(enable_subagents=False, preserve_history=True)
    return protocol


def create_plan(backend: str, model: str, output_root: Path, num_shards: int,
                protocol: dict | None = None, manifest_path: Path | None = None) -> dict:
    use_backend(backend)
    output = output_root.resolve()
    source = (manifest_path or evaluation_manifest(backend)).resolve()
    protocol = protocol or default_protocol(backend)
    if not model.strip():
        raise ValueError("model must not be empty")
    if protocol["max_prompt_tokens"] + protocol["max_completion_tokens"] > protocol["context_length"]:
        raise ValueError("prompt and completion caps exceed context length")
    if any(protocol[key] <= 0 for key in ("context_length", "max_prompt_tokens", "max_completion_tokens",
                                          "max_steps", "subagent_max_steps")) or protocol["max_depth"] < 0:
        raise ValueError("Evaluation budgets must be positive")
    if protocol["temperature"] < 0:
        raise ValueError("temperature must be non-negative")
    if backend == "TextCraft":
        from parallel_fulltest_textcraft.sharding import load_full_manifest, build_shard_manifests
        full = load_full_manifest(source)
        shards = [shard.to_dict() for shard in build_shard_manifests(full, num_shards)]
    else:
        from Runtime.environments.textworld.manifest import load_manifest, build_shards
        full = load_manifest(source)
        if len(full.tasks) != 1400 or any(task.game != "cookingworld_multidish" for task in full.tasks):
            raise ValueError("Expected the fixed 1400-task TextWorld-Sync V9 test manifest")
        shards = [dict(dataset=full.dataset, split=full.split, selection="parallel-shard",
                       parent_manifest_sha256=full.source_sha256, shard_index=index,
                       shard_count=num_shards, tasks=[task.to_dict() for task in tasks])
                  for index, tasks in enumerate(build_shards(full, num_shards))]
    records = []
    for index, shard in enumerate(shards):
        tag = f"shard-{index:03d}-of-{num_shards:03d}"
        counts = Counter(task["difficulty"] for task in shard["tasks"])
        records.append(dict(shard_index=index, tag=tag,
                            manifest_path=f"manifests/{tag}.json", output_dir=f"shards/{tag}",
                            task_count=len(shard["tasks"]), difficulty_counts=dict(counts)))
    plan = dict(schema_version=1, backend=backend, model=model, num_shards=num_shards,
                manifest_file=source.name, manifest_sha256=digest(source),
                dataset=full.dataset, expected_tasks=len(full.tasks), protocol=protocol, shards=records)
    with exclusive_lock(output / ".plan.lock"):
        plan_file = output / "parallel_plan.json"
        if plan_file.exists():
            existing = read_json(plan_file)
            if {k: v for k, v in existing.items() if k != "shards"} != {k: v for k, v in plan.items() if k != "shards"}:
                raise ValueError("Existing run has a different model, protocol, dataset or shard count; use a new output root")
            for record, shard in zip(existing["shards"], shards, strict=True):
                if read_json(output / record["manifest_path"]) != shard:
                    raise ValueError("Existing shard assignment was modified; refusing to overwrite it")
            return existing
        for record, shard in zip(records, shards, strict=True):
            path = output / record["manifest_path"]
            atomic_json(path, shard)
            record["manifest_sha256"] = digest(path)
        atomic_json(plan_file, plan)
    return plan


def load_plan(root: Path, backend: str) -> dict:
    plan = read_json(root / "parallel_plan.json")
    if plan["backend"] != backend:
        raise ValueError(f"This is a {plan['backend']} plan, not a {backend} plan")
    ids = []
    for record in plan["shards"]:
        path = root / record["manifest_path"]
        if digest(path) != record["manifest_sha256"]:
            raise ValueError(f"Shard manifest was modified: {path}")
        tasks = read_json(path)["tasks"]
        if len(tasks) != record["task_count"]:
            raise ValueError("Shard task count mismatch")
        ids.extend(task["task_id"] for task in tasks)
    if len(ids) != plan["expected_tasks"] or len(set(ids)) != len(ids):
        raise ValueError("Shard manifests are not a complete disjoint partition")
    return plan


def shard_identity(plan: dict, record: dict) -> dict:
    return dict(backend=plan["backend"], model=plan["model"], protocol=plan["protocol"],
                manifest_sha256=record["manifest_sha256"])


def load_world_shard(path: Path):
    from Runtime.environments.textworld.manifest import TaskManifest, TaskSpec
    payload = read_json(path)
    return TaskManifest(payload["dataset"], "test", "parallel-shard", "parallel-shard",
                        tuple(TaskSpec(**task) for task in payload["tasks"]), {},
                        payload["parent_manifest_sha256"])


async def evaluate_shard(backend: str, args: argparse.Namespace) -> dict:
    use_backend(backend)
    root = args.run_root.resolve()
    plan = load_plan(root, backend)
    if not 0 <= args.shard_index < plan["num_shards"]:
        raise ValueError("shard-index must be in [0, num-shards)")
    record = plan["shards"][args.shard_index]
    output = root / record["output_dir"]
    with exclusive_lock(output / ".worker.lock"):
        identity_path = output / "run_identity.json"
        identity = shard_identity(plan, record)
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Cannot resume results from a different model or evaluation protocol")
        if not identity_path.exists() and (output / "rollouts").exists():
            raise ValueError("Existing rollouts have no run identity; use a clean output root")
        atomic_json(identity_path, identity)
        options = dict(plan["protocol"], concurrency=args.concurrency, task_retries=args.task_retries)
        if backend == "TextCraft":
            from parallel_fulltest_textcraft.sharding import load_shard_manifest
            from textcraft_fulltest.evaluator import EvaluationSettings, evaluate_checkpoint
            return await evaluate_checkpoint(manifest=load_shard_manifest(root / record["manifest_path"]),
                                             output_dir=output, model=plan["model"], base_url=args.base_url,
                                             api_key=args.api_key, settings=EvaluationSettings(**options), resume=True)
        if "best_of_n" in options:
            from TestTimeScale.textworld_bestofn.evaluator import EvaluationSettings, evaluate_manifest
            from TestTimeScale.textworld_bestofn.rubric_selector import RubricSelectionConfig
            from TestTimeScale.textworld_bestofn.kimi_oracle import ExternalJudgeConfig
            selector = RubricSelectionConfig(**options.pop("best_of_n"))
            judge_payload = options.pop("external_judge", None)
            judge = ExternalJudgeConfig(**judge_payload) if judge_payload else None
            return await evaluate_manifest(manifest=load_world_shard(root / record["manifest_path"]),
                                           output_dir=output, model=plan["model"], base_url=args.base_url,
                                           api_key=args.api_key, settings=EvaluationSettings(**options),
                                           selector=selector, judge_config=judge, resume=True,
                                           resume_retain_errors=getattr(args, "resume_retain_errors", False))
        from textworld_parallel.evaluator import EvaluationSettings, evaluate_manifest
        return await evaluate_manifest(manifest=load_world_shard(root / record["manifest_path"]),
                                       output_dir=output, model=plan["model"], base_url=args.base_url,
                                       api_key=args.api_key, settings=EvaluationSettings(**options), resume=True)


def textcraft_success(collection: dict) -> bool:
    """Match the original workflow's root-only terminal success calculation."""
    trajectories = collection.get("trajectories", {})
    if not trajectories:
        raise ValueError("Missing root trajectory")
    root = next(iter(trajectories.values()))
    steps = root.get("steps", [])
    if steps:
        final = steps[-1].get("misc", {}).get("reward_misc", {})
        for key in ("reward/success", "success"):
            if key in final:
                return float(final[key]) >= 1.0
    components = {}
    for step in steps:
        for key, value in step.get("misc", {}).get("reward_misc", {}).items():
            if key.startswith("reward/"):
                components[key] = components.get(key, 0.0) + float(value)
    reward = components.get("reward/success", 0.0) if components else root.get("reward", 0.0)
    return float(reward) >= 1.0


def bucket(scheduled: int) -> dict:
    return dict(scheduled=scheduled, completed=0, successful=0, failed=0, errored=0, pending=scheduled)


def aggregate_results(root: Path, backend: str) -> dict:
    root = root.resolve()
    with exclusive_lock(root / ".aggregate.lock"):
        plan = load_plan(root, backend)
        by_difficulty = {difficulty: bucket(sum(record["difficulty_counts"].get(difficulty, 0)
                                               for record in plan["shards"])) for difficulty in DIFFICULTIES}
        overall = bucket(plan["expected_tasks"])
        shard_progress = []
        warnings = []
        for record in plan["shards"]:
            output = root / record["output_dir"]
            identity_path = output / "run_identity.json"
            if identity_path.exists() and read_json(identity_path) != shard_identity(plan, record):
                raise ValueError(f"Result identity mismatch: {output}")
            completed = successful = errored = 0
            manifest = read_json(root / record["manifest_path"])
            tasks = manifest["tasks"]
            for task in tasks:
                task_id = task["task_id"]
                path = (output / "rollouts" / task_id / "rollout_0" / "metadata.json" if backend == "TextCraft"
                        else output / "rollouts" / f"{task_id}.json")
                if not path.exists():
                    continue
                if not identity_path.exists():
                    raise ValueError(f"Rollouts have no run identity: {output}")
                try:
                    result = read_json(path)
                    if result["task_id"] != task_id:
                        raise ValueError("Task ID mismatch")
                    error = result.get("error") is not None
                    if backend == "TextWorld":
                        if result.get("manifest_sha256") != manifest["parent_manifest_sha256"]:
                            raise ValueError("Rollout dataset digest mismatch")
                    if not error and result.get("status") != "completed":
                        continue
                    if backend == "TextCraft" and not error:
                        collection = path.parent / "trajectory_collection.json"
                        success = textcraft_success(read_json(collection))
                    else:
                        success = bool(result.get("success")) and not error
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as error:
                    warnings.append(f"Pending/incomplete artifact {path}: {type(error).__name__}")
                    continue
                completed += 1
                successful += int(success)
                errored += int(error)
                for stats in (overall, by_difficulty[task["difficulty"]]):
                    stats["completed"] += 1
                    stats["successful"] += int(success)
                    stats["errored"] += int(error)
                    stats["failed"] += int(not success)
                    stats["pending"] -= 1
            shard_progress.append(dict(shard_index=record["shard_index"], scheduled=record["task_count"],
                                       completed=completed, successful=successful, errored=errored))
        for stats in (overall, *by_difficulty.values()):
            stats["success_rate"] = stats["successful"] / stats["completed"] if stats["completed"] else None
            stats["benchmark_success_rate"] = stats["successful"] / stats["scheduled"] if stats["scheduled"] else None
        report = dict(backend=backend, model=plan["model"], dataset=plan["dataset"], protocol=plan["protocol"],
                      complete=overall["pending"] == 0, overall=overall, by_difficulty=by_difficulty,
                      shards=shard_progress, warnings=warnings,
                      rate_definition="Successful / completed; errors count as failures. Rates are fractions, not percentages.")
        atomic_json(root / "aggregate_progress.json", report)
        if report["complete"]:
            atomic_json(root / "final_report.json", report)
        else:
            # A retry may be replacing an errored record; never leave a stale final report.
            (root / "final_report.json").unlink(missing_ok=True)
        return report


def cli(backend: str, command: str) -> None:
    parser = argparse.ArgumentParser(description=f"{backend}: {command.replace('_', ' ')}")
    if command == "create_plan":
        parser.add_argument("--model", required=True, help="Served model ID (TextCraft: openai/<ID>; TextWorld: <ID>)")
        parser.add_argument("--output-root", type=Path, required=True)
        parser.add_argument("--num-shards", type=int, default=1)
        parser.add_argument("--single-agent", action="store_true", help="Use the 155/156 single-Agent protocol")
        preliminary, _ = parser.parse_known_args()
        defaults = default_protocol(backend, single_agent=preliminary.single_agent)
        if backend == "TextWorld" and preliminary.single_agent:
            parser.add_argument("--tokenizer-path", help="Local served-model tokenizer for exact prompt counting")
        for key, value in defaults.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
            else:
                parser.add_argument(flag, type=type(value), default=value)
        args = parser.parse_args()
        protocol = {key: getattr(args, key) for key in defaults}
        if backend == "TextWorld" and args.single_agent:
            protocol["tokenizer_path"] = args.tokenizer_path
        result = create_plan(backend, args.model, args.output_root, args.num_shards, protocol)
        print(json.dumps(result, indent=2))
    elif command == "evaluate_shard":
        parser.add_argument("--run-root", type=Path, required=True)
        parser.add_argument("--shard-index", type=int, required=True)
        parser.add_argument("--base-url", required=True, help="OpenAI-compatible URL ending in /v1")
        parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
        parser.add_argument("--concurrency", type=int, default=64)
        parser.add_argument("--task-retries", type=int, default=1)
        args = parser.parse_args()
        if args.concurrency < 1 or args.task_retries < 0:
            parser.error("concurrency must be positive and task-retries must be non-negative")
        if backend == "TextCraft":
            use_backend(backend)
            from textcraft_fulltest.runtime_cleanup import run_with_litellm_cleanup
            result = asyncio.run(run_with_litellm_cleanup(lambda: evaluate_shard(backend, args)))
        else:
            result = asyncio.run(evaluate_shard(backend, args))
        print(json.dumps(result, indent=2))
    elif command == "aggregate_results":
        parser.add_argument("--run-root", type=Path, required=True)
        parser.add_argument("--require-complete", action="store_true")
        args = parser.parse_args()
        result = aggregate_results(args.run_root, backend)
        print(json.dumps(result, indent=2))
        if args.require_complete and not result["complete"]:
            raise SystemExit(2)
    else:
        raise ValueError(f"Unknown command: {command}")
