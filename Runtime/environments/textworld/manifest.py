from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GAMES = ("cookingworld", "twc")
# The original TextWorldExpress games remain the default source selection. The
# composite game is deliberately opt-in through the fixed V6 test manifest.
SUPPORTED_GAMES = (*GAMES, "cookingworld_multidish")
SUPPORTED_DATASETS = {
    "TextWorldExpress",
    "TextWorldExpress-MultiDish",
    "TextWorldExpress-MultiDish-V6",
    "TextWorldExpress-MultiDish-V6.1",
    "TextWorldExpress-MultiDish-V7",
    "TextWorldExpress-MultiDish-V7.1",
    "TextWorldExpress-MultiDish-V7.1-Train",
    "TextWorldExpress-MultiDish-V7.2",
    "TextWorldExpress-MultiDish-V7.3",
    "TextWorldExpress-MultiDish-V9-Train",
    "TextWorldExpress-MultiDish-V9-Validation",
    "TextWorldExpress-MultiDish-V9-Test",
}
DIFFICULTIES = ("easy", "medium", "hard", "extreme")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    game: str
    fold: str
    difficulty: str
    seed: int
    game_params: str
    task_description: str
    generation_properties: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "game": self.game,
            "fold": self.fold,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "game_params": self.game_params,
            "task_description": self.task_description,
            "generation_properties": self.generation_properties,
        }


@dataclass(frozen=True)
class TaskManifest:
    dataset: str
    split: str
    game_selection: str
    difficulty_selection: str
    tasks: tuple[TaskSpec, ...]
    source_files: dict[str, str]
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "split": self.split,
            "game_selection": self.game_selection,
            "difficulty_selection": self.difficulty_selection,
            "task_count": len(self.tasks),
            "source_files": self.source_files,
            "source_sha256": self.source_sha256,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _task_from_row(row: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_id=str(row["task_id"]),
        game=str(row["game"]),
        fold=str(row["fold"]),
        difficulty=str(row["difficulty"]),
        seed=int(row["seed"]),
        game_params=str(row.get("game_params", "")),
        task_description=str(row["task_description"]),
        generation_properties=dict(row.get("generation_properties", {})),
    )








def validate_manifest(manifest: TaskManifest) -> None:
    if manifest.dataset not in SUPPORTED_DATASETS:
        raise ValueError("manifest must describe a supported TextWorldExpress dataset")
    if manifest.dataset.startswith("TextWorldExpress-MultiDish-V9-"):
        expected_split = manifest.dataset.rsplit("-", 1)[-1].lower()
        if manifest.split != expected_split:
            raise ValueError("V9 manifest split does not match its dataset name")
    elif manifest.split == "train":
        if manifest.dataset != "TextWorldExpress-MultiDish-V7.1-Train":
            raise ValueError("only the fixed V7.1 train manifest may use the train split")
    elif manifest.split != "test":
        raise ValueError("manifest must describe a supported TextWorldExpress test or train split")
    ids = [task.task_id for task in manifest.tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest task IDs must be unique")
    if any(task.game not in SUPPORTED_GAMES for task in manifest.tasks):
        raise ValueError("manifest contains an unknown game")
    if any(task.difficulty not in DIFFICULTIES for task in manifest.tasks):
        raise ValueError("manifest contains an unknown difficulty")
    if any(task.fold != manifest.split for task in manifest.tasks):
        raise ValueError("manifest task fold does not match the manifest split")


def write_manifest(path: str | Path, manifest: TaskManifest) -> None:
    _atomic_json(Path(path), manifest.to_dict())


def load_manifest(path: str | Path) -> TaskManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = tuple(_task_from_row(row) for row in payload["tasks"])
    manifest = TaskManifest(
        dataset=str(payload["dataset"]),
        split=str(payload["split"]),
        game_selection=str(payload["game_selection"]),
        difficulty_selection=str(payload.get("difficulty_selection", "all")),
        tasks=tasks,
        source_files=dict(payload.get("source_files", {})),
        source_sha256=str(payload.get("source_sha256", "")),
    )
    validate_manifest(manifest)
    expected_count = int(payload.get("task_count", len(tasks)))
    if expected_count != len(tasks):
        raise ValueError("manifest task_count does not match tasks")
    return manifest


def build_shards(manifest: TaskManifest, shard_count: int) -> tuple[tuple[TaskSpec, ...], ...]:
    if not 1 <= shard_count <= len(manifest.tasks):
        raise ValueError("shard_count must be between 1 and the task count")
    buckets: list[list[TaskSpec]] = [[] for _ in range(shard_count)]
    # Stratify by game and difficulty so every machine receives a comparable mix.
    groups = []
    manifest_games = tuple(dict.fromkeys(task.game for task in manifest.tasks))
    for game in manifest_games:
        for difficulty in DIFFICULTIES:
            groups.append(
                [
                    task
                    for task in manifest.tasks
                    if task.game == game and task.difficulty == difficulty
                ]
            )
    offset = 0
    for group in groups:
        for position, task in enumerate(group):
            buckets[(offset + position) % shard_count].append(task)
        offset = (offset + len(group)) % shard_count
    shards = tuple(tuple(sorted(bucket, key=lambda item: item.task_id)) for bucket in buckets)
    flat = [task.task_id for shard in shards for task in shard]
    if len(flat) != len(set(flat)) or set(flat) != set(task.task_id for task in manifest.tasks):
        raise ValueError("invalid task partition")
    return shards
