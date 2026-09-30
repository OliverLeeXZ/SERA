from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIFFICULTIES = ("easy", "medium", "hard", "extreme")
VALIDATION_SIZE = 632


@dataclass(frozen=True)
class ManifestTask:
    task_id: str
    difficulty: str
    val_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "val_index": self.val_index,
        }


@dataclass(frozen=True)
class FullManifest:
    dataset: str
    split: str
    selection: str
    population_size: int
    tasks: tuple[ManifestTask, ...]
    sha256: str


@dataclass(frozen=True)
class ShardManifest:
    dataset: str
    split: str
    selection: str
    population_size: int
    parent_manifest_sha256: str
    shard_index: int
    shard_count: int
    tasks: tuple[ManifestTask, ...]

    @property
    def difficulty_counts(self) -> dict[str, int]:
        counts = Counter(task.difficulty for task in self.tasks)
        return {difficulty: counts[difficulty] for difficulty in DIFFICULTIES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "selection": self.selection,
            "population_size": self.population_size,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "task_count": len(self.tasks),
            "difficulty_counts": self.difficulty_counts,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _parse_tasks(payload: dict[str, Any]) -> tuple[ManifestTask, ...]:
    return tuple(
        ManifestTask(
            task_id=str(task["task_id"]),
            difficulty=str(task["difficulty"]),
            val_index=int(task["val_index"]),
        )
        for task in payload["tasks"]
    )


def load_full_manifest(path: str | Path) -> FullManifest:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    tasks = _parse_tasks(payload)
    manifest = FullManifest(
        dataset=str(payload["dataset"]),
        split=str(payload["split"]),
        selection=str(payload["selection"]),
        population_size=int(payload["population_size"]),
        tasks=tasks,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    if (
        manifest.dataset != "TextCraft-Synth"
        or manifest.split != "val"
        or manifest.selection != "all"
    ):
        raise ValueError("source manifest must be the TextCraft-Synth full val set")
    if manifest.population_size != VALIDATION_SIZE or len(tasks) != VALIDATION_SIZE:
        raise ValueError(f"source manifest must contain {VALIDATION_SIZE} tasks")
    expected_ids = [f"textcraft_synth.val.{index}" for index in range(VALIDATION_SIZE)]
    if [task.task_id for task in tasks] != expected_ids:
        raise ValueError("source manifest task IDs/order differ from canonical val order")
    for index, task in enumerate(tasks):
        if task.val_index != index or task.difficulty not in DIFFICULTIES:
            raise ValueError(f"invalid canonical task entry: {task}")
    return manifest


def build_shard_manifests(
    full_manifest: FullManifest,
    shard_count: int,
) -> tuple[ShardManifest, ...]:
    if not 1 <= shard_count <= len(full_manifest.tasks):
        raise ValueError(
            f"shard_count must be in [1, {len(full_manifest.tasks)}]"
        )

    buckets: list[list[ManifestTask]] = [[] for _ in range(shard_count)]
    offset = 0
    for difficulty in DIFFICULTIES:
        tasks = [
            task for task in full_manifest.tasks if task.difficulty == difficulty
        ]
        for position, task in enumerate(tasks):
            buckets[(offset + position) % shard_count].append(task)
        offset = (offset + len(tasks)) % shard_count

    manifests = tuple(
        ShardManifest(
            dataset=full_manifest.dataset,
            split=full_manifest.split,
            selection="parallel-shard",
            population_size=full_manifest.population_size,
            parent_manifest_sha256=full_manifest.sha256,
            shard_index=index,
            shard_count=shard_count,
            tasks=tuple(sorted(tasks, key=lambda task: task.val_index)),
        )
        for index, tasks in enumerate(buckets)
    )
    validate_partition(full_manifest, manifests)
    return manifests


def validate_partition(
    full_manifest: FullManifest,
    shards: tuple[ShardManifest, ...] | list[ShardManifest],
) -> None:
    if not shards:
        raise ValueError("partition contains no shards")
    expected_count = len(shards)
    if {shard.shard_count for shard in shards} != {expected_count}:
        raise ValueError("shard_count metadata does not match partition size")
    if {shard.shard_index for shard in shards} != set(range(expected_count)):
        raise ValueError("shard indexes must be contiguous and zero-based")
    task_ids = [task.task_id for shard in shards for task in shard.tasks]
    expected_ids = [task.task_id for task in full_manifest.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("partition contains duplicate task IDs")
    if set(task_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(task_ids))
        extra = sorted(set(task_ids) - set(expected_ids))
        raise ValueError(f"partition mismatch: missing={missing} extra={extra}")


def write_shard_manifest(path: str | Path, manifest: ShardManifest) -> None:
    _atomic_json(Path(path), manifest.to_dict())


def load_shard_manifest(path: str | Path) -> ShardManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = ShardManifest(
        dataset=str(payload["dataset"]),
        split=str(payload["split"]),
        selection=str(payload["selection"]),
        population_size=int(payload["population_size"]),
        parent_manifest_sha256=str(payload["parent_manifest_sha256"]),
        shard_index=int(payload["shard_index"]),
        shard_count=int(payload["shard_count"]),
        tasks=_parse_tasks(payload),
    )
    if (
        manifest.dataset != "TextCraft-Synth"
        or manifest.split != "val"
        or manifest.selection != "parallel-shard"
    ):
        raise ValueError("invalid parallel TextCraft shard manifest")
    if manifest.population_size != VALIDATION_SIZE:
        raise ValueError(f"population_size must be {VALIDATION_SIZE}")
    if not 0 <= manifest.shard_index < manifest.shard_count:
        raise ValueError("shard_index is outside shard_count")
    if not manifest.tasks:
        raise ValueError("shard manifest must contain at least one task")
    if len({task.task_id for task in manifest.tasks}) != len(manifest.tasks):
        raise ValueError("shard manifest contains duplicate tasks")
    if any(task.difficulty not in DIFFICULTIES for task in manifest.tasks):
        raise ValueError("shard manifest contains an unknown difficulty")
    return manifest
