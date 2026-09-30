#!/usr/bin/env python3
"""Generate V9 TextWorld data directly into Dataset/{training,validation,eval}."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from cooking_tasks import load_json
from parameter_sampling import allocate_counts, build_v7_task

HERE = Path(__file__).resolve().parent
SPLITS = {"training": ("train", "TextWorldExpress-MultiDish-V9-Train", "textworld_multidish_train_1500_v9.json"),
          "validation": ("dev", "TextWorldExpress-MultiDish-V9-Validation", "textworld_multidish_validation_200_v9.json"),
          "eval": ("test", "TextWorldExpress-MultiDish-V9-Test", "textworld_multidish_test_1400_v9.json")}


def build_records(config, db, split):
    """Preserve original task IDs, ordering, family allocation and seed offsets."""
    records = []
    fold = SPLITS[split][0]
    if split == "eval":
        matrix = config["test_matrix"]
        difficulties = matrix["difficulties"]
        families = matrix["families"]
        counts = {difficulty: int(matrix["count_per_family"]) * len(families) for difficulty in difficulties}
        spec = matrix
    else:
        matrix = config["train_dev_matrix"]
        spec = matrix["splits"][fold]
        counts = spec["difficulty_counts"]
        families = matrix["families"]
    if not families or len(families) != len(set(families)):
        raise ValueError("families must be nonempty and unique")
    for difficulty_index, (difficulty, count) in enumerate(counts.items()):
        if difficulty not in config["difficulty_ranges"] or int(count) < 1:
            raise ValueError(f"Invalid difficulty/count: {difficulty}={count}")
        allocations = allocate_counts(int(count), families)
        for family_index, family in enumerate(families):
            seed_start = int(spec["seed_start"]) + difficulty_index * int(matrix["difficulty_seed_stride"]) + family_index * int(matrix["family_seed_stride"])
            for offset in range(allocations[family]):
                records.append(build_v7_task(config=config, db=db, seed=seed_start+offset, offset=offset,
                                             split_name=f"{fold}_{difficulty}_{family}", source_fold=spec["source_fold"],
                                             difficulty_name=difficulty, task_family=family, record_split=fold,
                                             task_id_split=f"{fold}_{family}"))
    ids = [row["task_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate generated task IDs")
    return records


def project_row(row, split):
    return dict(task_id=row["task_id"], game="cookingworld_multidish", fold=split,
                difficulty=row["difficulty"], seed=row["seed"],
                game_params=json.dumps(row.get("game_params", {}), sort_keys=True, separators=(",", ":")),
                task_description=row["task_description"], generation_properties={
                    "base_game": row.get("base_game", "cookingworld"), "task_family": row.get("task_family", ""),
                    "task_view": row.get("task_view", "structured"), "task_descriptions": row.get("task_descriptions", {}),
                    "dishes": row.get("dishes", []), "parallelism": row.get("parallelism", {}), "generation": row.get("generation", {})})


def encode_split(rows, split, source_db_digest, config_digest):
    fold, dataset, manifest_name = SPLITS[split]
    raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in rows).encode()
    raw_digest = hashlib.sha256(raw).hexdigest()
    manifest = dict(schema_version=1, dataset=dataset, split="validation" if fold=="dev" else fold,
                    game_selection="cookingworld_multidish", task_count=len(rows),
                    difficulty_counts=dict(Counter(row["difficulty"] for row in rows)),
                    task_family_counts=dict(Counter(row["task_family"] for row in rows)),
                    source_files={f"textworld_{fold}.jsonl": raw_digest}, source_sha256=raw_digest,
                    source_db_sha256=source_db_digest, generation_config_sha256=config_digest,
                    tasks=[project_row(row,"validation" if fold=="dev" else fold) for row in rows])
    return {f"textworld_{fold}.jsonl": raw, manifest_name: (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+"\n").encode()}


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def generation_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".textworld-generation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another TextWorld generator is writing this dataset root") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def generate(config_path, dataset_root, splits, overwrite=False, source_db=None):
    config_path, dataset_root = Path(config_path).resolve(), Path(dataset_root).resolve()
    config = load_json(config_path)
    db_path = Path(source_db or config["source_db"]).expanduser()
    if not db_path.is_absolute():db_path = config_path.parent / db_path
    db = load_json(db_path)
    db_digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if len(set(splits)) != len(splits) or any(split not in SPLITS for split in splits):
        raise ValueError("Choose unique splits from training, validation, eval")
    # Generate and validate everything before writing any task files.
    outputs, report = {}, {}
    for split in splits:
        rows = build_records(config, db, split)
        for name, raw in encode_split(rows, split, db_digest, config_digest).items():
            outputs[dataset_root / split / name] = raw
        report[split] = dict(tasks=len(rows), difficulties=dict(Counter(row["difficulty"] for row in rows)))
    with generation_lock(dataset_root):
        existing = [path for path in outputs if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"Data already exists; use --overwrite explicitly or another --dataset-root: {existing[0]}")
        for path, raw in outputs.items():atomic_write(path, raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "textworld_v9.json")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("SERA_DATASET_ROOT", HERE.parent)))
    parser.add_argument("--source-db", type=Path, help="Optional override of the bundled CookingWorld database")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace selected TextWorld files; never touches TextCraft")
    args = parser.parse_args()
    print(json.dumps(generate(args.config,args.dataset_root,args.splits,args.overwrite,args.source_db),indent=2))


if __name__ == "__main__":main()
