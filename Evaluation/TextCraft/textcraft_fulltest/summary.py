from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from parallel_fulltest_textcraft.sharding import DIFFICULTIES, ShardManifest


def _empty_bucket(scheduled: int) -> dict[str, int | float]:
    return {
        "scheduled": scheduled,
        "completed": 0,
        "valid": 0,
        "successful": 0,
        "failed": 0,
        "errored": 0,
        "accuracy": 0.0,
    }


def summarize_records(
    records: Iterable[Any],
    manifest: ShardManifest,
) -> dict[str, Any]:
    difficulty_by_task = {task.task_id: task.difficulty for task in manifest.tasks}
    by_difficulty = {
        difficulty: _empty_bucket(
            sum(task.difficulty == difficulty for task in manifest.tasks)
        )
        for difficulty in DIFFICULTIES
    }
    overall = _empty_bucket(len(manifest.tasks))

    for record in records:
        task_id = str(record.task_id)
        if task_id not in difficulty_by_task:
            raise ValueError(f"record task is absent from manifest: {task_id}")
        buckets = (overall, by_difficulty[difficulty_by_task[task_id]])
        for bucket in buckets:
            bucket["completed"] += 1
            if record.error is not None:
                bucket["errored"] += 1
                continue
            bucket["valid"] += 1
            if bool(record.success):
                bucket["successful"] += 1
            else:
                bucket["failed"] += 1

    for bucket in (overall, *by_difficulty.values()):
        valid = int(bucket["valid"])
        bucket["accuracy"] = (
            int(bucket["successful"]) / valid if valid else 0.0
        )

    return {"overall": overall, "by_difficulty": by_difficulty}


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
