from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from platoon.envs.base import Task
from Dataset.paths import data_path, evaluation_manifest


FOLDS = ("train", "dev", "test")


def _sync_path(fold: str) -> Path:
    filename = "train.jsonl" if fold == "train" else "dev.jsonl"
    legacy = os.environ.get("TEXTWORLD_SYNC_DATA_DIR")
    if legacy:
        return Path(legacy).expanduser().resolve() / filename
    return data_path("training" if fold == "train" else "validation", "textworld", filename)


@lru_cache(maxsize=3)
def _rows(fold: str) -> tuple[dict, ...]:
    if fold not in FOLDS:
        raise ValueError(f"Unknown TextWorld fold {fold!r}; choose from {FOLDS}")
    rows: list[dict] = []
    if fold in {"train", "dev"}:
        path = _sync_path(fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    else:
        path = evaluation_manifest("TextWorld")
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["tasks"])
    return tuple(rows)


def get_textworld_task_ids(fold: str) -> list[str]:
    return [str(row["task_id"]) for row in _rows(fold)]


@lru_cache(maxsize=4096)
def _row_by_id(task_id: str) -> dict:
    for fold in FOLDS:
        for row in _rows(fold):
            if str(row["task_id"]) == task_id:
                return row
    raise KeyError(f"Unknown TextWorld task: {task_id}")


def get_textworld_task(task_id: str, max_steps: int = 20) -> Task:
    row = _row_by_id(task_id)
    generation_properties = dict(row.get("generation_properties", {}))
    if row.get("game") == "cookingworld_multidish":
        # The composite runtime consumes the task specification through the
        # same metadata channel used by the 66th project's evaluator.
        generation_properties.update(
            {
                "dishes": row.get("dishes", generation_properties.get("dishes", [])),
                "parallelism": row.get("parallelism", generation_properties.get("parallelism", {})),
                "task_family": row.get("task_family", generation_properties.get("task_family", "multi_dish_fork_join")),
            }
        )
    game_params = row.get("game_params", "")
    if isinstance(game_params, dict):
        game_params = json.dumps(game_params, ensure_ascii=False, sort_keys=True)
    return Task(
        id=str(row["task_id"]),
        goal=str(row["task_description"]),
        max_steps=max_steps,
        misc={
            "textworld_game": str(row["game"]),
            # Legacy manifests use ``fold``; TextWorld-Sync uses ``split``.
            "textworld_fold": str(row.get("fold", row.get("split", "train"))),
            "textworld_difficulty": str(row["difficulty"]),
            "textworld_seed": int(row["seed"]),
            "textworld_game_params": str(game_params),
            "generation_properties": generation_properties,
            "root_task_id": str(row["task_id"]),
        },
        fork_strategy="subtask",
    )
