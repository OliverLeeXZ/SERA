#!/usr/bin/env python3
"""Download the public SERA paper checkpoints into Evaluation/ckpt."""

import argparse
import json
from pathlib import Path


CHECKPOINTS = {
    "textcraft": "TextCraft-step250",
    "textworld": "TextWorld-step400",
}
REPOSITORY = Path(__file__).resolve().parents[2]


def verify_checkpoint(directory: Path) -> None:
    """Fail early if a checkpoint is incomplete or its weights are missing."""
    for name in ("config.json", "model.safetensors.index.json"):
        if not (directory / name).is_file():
            raise RuntimeError(f"Incomplete checkpoint: missing {directory / name}")
    with (directory / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        index = json.load(handle)
    shards = set(index.get("weight_map", {}).values())
    if not shards or any(Path(shard).name != shard for shard in shards):
        raise RuntimeError(f"Invalid weight index: {directory}")
    for shard in shards:
        path = directory / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Incomplete checkpoint: missing or empty {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=["both", *CHECKPOINTS], default="both")
    parser.add_argument("--repo-id", default="Litux12138/SERA", help="Hugging Face model repository")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "Evaluation" / "ckpt")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Show destination without downloading")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    selected = list(CHECKPOINTS) if args.checkpoint == "both" else [args.checkpoint]
    destination = args.output_dir.expanduser().resolve()
    for name in selected:
        print(f"{args.repo_id}@{args.revision}:{CHECKPOINTS[name]}/ -> {destination / CHECKPOINTS[name]}", flush=True)
    if args.dry_run:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install -r requirements.txt") from exc

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=destination,
        allow_patterns=[f"{CHECKPOINTS[name]}/*" for name in selected],
        max_workers=args.max_workers,
        token=False,
    )
    for name in selected:
        directory = destination / CHECKPOINTS[name]
        verify_checkpoint(directory)
        print(f"Ready: {directory}", flush=True)


if __name__ == "__main__":
    main()
