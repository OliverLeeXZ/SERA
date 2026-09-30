"""Canonical dataset locations; SERA_DATASET_ROOT can select another data root."""
import os
from pathlib import Path


def dataset_root() -> Path:
    return Path(os.environ.get("SERA_DATASET_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()


def data_path(split: str, environment: str, filename: str) -> Path:
    """Resolve a flat split file, adding its environment prefix if needed."""
    if split not in {"training", "validation", "eval"} or environment not in {"textcraft", "textworld"}:
        raise ValueError("Unknown dataset split/environment")
    if not filename or Path(filename).name != filename:
        raise ValueError("Dataset filename must be a basename, not a path")
    if filename.startswith(("textcraft_", "textworld_")) and not filename.startswith(f"{environment}_"):
        raise ValueError("Dataset filename prefix does not match environment")
    prefixed = filename if filename.startswith(f"{environment}_") else f"{environment}_{filename}"
    return dataset_root() / split / prefixed


def evaluation_manifest(backend: str) -> Path:
    if backend == "TextCraft":
        return data_path("eval", "textcraft", "textcraft_synth_val_all_632.json")
    if backend == "TextWorld":
        return data_path("eval", "textworld", "textworld_multidish_test_1400_v9.json")
    raise ValueError(f"Unknown backend: {backend}")
