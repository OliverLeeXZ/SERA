"""Task-construction helpers extracted from the source TextWorld-Sync generator."""
from __future__ import annotations
import copy
import json
import random
from pathlib import Path
from typing import Any
from cooking_tasks import build_task, load_json, sha256_file

def sample_value(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, dict):
        if 'choices' in spec:
            return copy.deepcopy(rng.choice(list(spec['choices'])))
        if 'min' in spec and 'max' in spec:
            lower, upper = (spec['min'], spec['max'])
            if isinstance(lower, int) and isinstance(upper, int):
                return rng.randint(lower, upper)
            return rng.uniform(float(lower), float(upper))
        raise ValueError(f'unsupported range specification: {spec}')
    if isinstance(spec, list):
        return copy.deepcopy(rng.choice(spec))
    return copy.deepcopy(spec)

def sample_profile(config: dict[str, Any], difficulty_name: str, rng: random.Random) -> dict[str, Any]:
    bands = config['difficulty_ranges'][difficulty_name]
    profile = {key: sample_value(value, rng) for key, value in bands.items()}
    if profile['max_prepared_ingredients_per_dish'] < profile['min_prepared_ingredients_per_dish']:
        raise ValueError(f'invalid sampled preparation bounds: {profile}')
    return profile

def family_spec_for_task(config: dict[str, Any], task_family: str, profile: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(config['task_families'][task_family])
    dish_count = int(profile['dish_count'])
    if task_family == 'critical_path':
        spec['complexity_by_dish'] = ['low'] * max(0, dish_count - 2) + ['medium', 'high']
    elif task_family == 'load_balanced':
        spec['complexity_by_dish'] = ['low'] * max(0, dish_count - 1) + ['high']
    elif task_family == 'dependency_gate':
        spec['gated_dishes'] = [f'dish_{index}' for index in range(2, dish_count + 1)]
    return spec

def _build_v7_task_once(*, config: dict[str, Any], db: dict[str, Any], seed: int, offset: int, split_name: str, source_fold: str, difficulty_name: str, task_family: str, record_split: str, task_id_split: str, sampling_seed: int | None=None) -> dict[str, Any]:
    rng = random.Random(seed if sampling_seed is None else sampling_seed)
    profile = sample_profile(config, difficulty_name, rng)
    family_spec = family_spec_for_task(config, task_family, profile)
    task_config = copy.deepcopy(config)
    cooldown = sample_value(config['runtime_ranges'][difficulty_name]['tool_cooldown_steps'], rng)
    shared_steps = sample_value(config['runtime_ranges'][difficulty_name]['shared_environment_max_steps'], rng)
    task_config['runtime_by_difficulty'] = {difficulty_name: {'shared_environment_max_steps': shared_steps, 'mechanisms': {'tool_cooldown_steps': cooldown}}}
    task = build_task(seed=seed, offset=offset, split_name=split_name, source_fold=source_fold, difficulty_name=difficulty_name, difficulty=profile, task_family=task_family, family_spec=family_spec, config=task_config, db=db, record_split=record_split, task_id_split=task_id_split)
    task['generation']['v7_parameter_band'] = copy.deepcopy(config['difficulty_ranges'][difficulty_name])
    task['generation']['v7_sampled_parameters'] = copy.deepcopy(profile)
    task['generation']['v7_runtime_band'] = copy.deepcopy(config['runtime_ranges'][difficulty_name])
    task['generation']['v7_sampled_runtime'] = {'shared_environment_max_steps': shared_steps, 'tool_cooldown_steps': cooldown}
    task['generation']['v7_seed'] = seed
    return task

def build_v7_task(*, config: dict[str, Any], db: dict[str, Any], seed: int, offset: int, split_name: str, source_fold: str, difficulty_name: str, task_family: str, record_split: str, task_id_split: str) -> dict[str, Any]:
    """Build one task, resampling only generation parameters on invalid draws.

    The official source split does not expose identical recipe availability for
    every seed. Resampling the parameter draw keeps the task seed stable while
    avoiding an invalid range combination.
    """
    last_error: Exception | None = None
    for attempt in range(128):
        try:
            return _build_v7_task_once(config=config, db=db, seed=seed, offset=offset, split_name=split_name, source_fold=source_fold, difficulty_name=difficulty_name, task_family=task_family, record_split=record_split, task_id_split=task_id_split, sampling_seed=seed + attempt * 1000003)
        except (RuntimeError, ValueError) as error:
            last_error = error
    raise RuntimeError(f'could not sample a valid V7 task for seed={seed}: {last_error}')

def allocate_counts(total: int, families: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(families))
    return {family: base + int(index < remainder) for index, family in enumerate(families)}
