from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from Runtime.environments.textworld.manifest import DIFFICULTIES, TaskManifest


def _bucket(scheduled: int) -> dict[str, int | float]:
    return {
        "scheduled": scheduled,
        "completed": 0,
        "valid": 0,
        "successful": 0,
        "failed": 0,
        "errored": 0,
        "accuracy": 0.0,
    }


def _add(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    bucket["completed"] += 1
    if record.get("error"):
        bucket["errored"] += 1
    else:
        bucket["valid"] += 1
        if bool(record.get("success")):
            bucket["successful"] += 1
        else:
            bucket["failed"] += 1


def _finish(bucket: dict[str, Any]) -> None:
    bucket["accuracy"] = (
        bucket["successful"] / bucket["valid"] if bucket["valid"] else 0.0
    )


def summarize_records(records: Iterable[dict[str, Any]], manifest: TaskManifest) -> dict[str, Any]:
    expected = {task.task_id: task for task in manifest.tasks}
    overall = _bucket(len(manifest.tasks))
    games = tuple(dict.fromkeys(task.game for task in manifest.tasks))
    by_game = {game: _bucket(sum(task.game == game for task in manifest.tasks)) for game in games}
    by_difficulty = {
        difficulty: _bucket(sum(task.difficulty == difficulty for task in manifest.tasks))
        for difficulty in DIFFICULTIES
    }
    by_game_difficulty = {
        f"{game}/{difficulty}": _bucket(
            sum(task.game == game and task.difficulty == difficulty for task in manifest.tasks)
        )
        for game in games
        for difficulty in DIFFICULTIES
    }
    seen: set[str] = set()
    for record in records:
        task_id = str(record.get("task_id"))
        if task_id not in expected:
            continue
        if task_id in seen:
            continue
        seen.add(task_id)
        task = expected[task_id]
        _add(overall, record)
        _add(by_game[task.game], record)
        _add(by_difficulty[task.difficulty], record)
        _add(by_game_difficulty[f"{task.game}/{task.difficulty}"], record)
    for bucket in [overall, *by_game.values(), *by_difficulty.values(), *by_game_difficulty.values()]:
        _finish(bucket)
    return {
        "overall": overall,
        "by_game": by_game,
        "by_difficulty": by_difficulty,
        "by_game_difficulty": by_game_difficulty,
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
