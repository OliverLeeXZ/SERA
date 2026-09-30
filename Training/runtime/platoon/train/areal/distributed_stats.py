"""Distributed-safe export for AReaL's process-local statistics trackers."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from areal.utils import stats_tracker
from areal.utils.stats_tracker import ReduceType


def _local_schemas() -> dict[str, dict[str, tuple[str, str | None]]]:
    schemas: dict[str, dict[str, tuple[str, str | None]]] = {}
    for tracker_name, tracker in stats_tracker.TRACKERS.items():
        keys = (
            set(tracker.stats)
            | set(tracker.reduce_types)
            | set(tracker.denominators)
        )
        schemas[tracker_name] = {
            key: (
                tracker.reduce_types.get(key, ReduceType.SCALAR).name,
                tracker.denominators.get(key),
            )
            for key in keys
        }
    return schemas


def _merge_schemas(
    schemas_by_rank: list[dict[str, dict[str, tuple[str, str | None]]]],
) -> dict[str, dict[str, tuple[ReduceType, str | None]]]:
    merged: dict[str, dict[str, tuple[ReduceType, str | None]]] = {}
    for rank, rank_schemas in enumerate(schemas_by_rank):
        for tracker_name, schema in rank_schemas.items():
            merged_tracker = merged.setdefault(tracker_name, {})
            for key, (reduce_type_name, denominator) in schema.items():
                candidate = (ReduceType[reduce_type_name], denominator)
                current = merged_tracker.get(key)
                if current is not None and current != candidate:
                    raise RuntimeError(
                        "Inconsistent distributed statistic schema for "
                        f"{tracker_name!r}/{key!r}: rank {rank} has {candidate}, "
                        f"but another rank has {current}."
                    )
                merged_tracker[key] = candidate

    for tracker_name, schema in merged.items():
        for _, denominator in list(schema.values()):
            if denominator is not None:
                expected = (ReduceType.SUM, None)
                current = schema.get(denominator)
                if current is not None and current != expected:
                    raise RuntimeError(
                        "Invalid denominator schema for "
                        f"{tracker_name!r}/{denominator!r}: {current}."
                    )
                schema[denominator] = expected
    return merged


def _pad_tracker(
    tracker: Any,
    schema: dict[str, tuple[ReduceType, str | None]],
    device: torch.device,
) -> None:
    denominator_keys = {
        denominator
        for _, denominator in schema.values()
        if denominator is not None
    }

    for key, (reduce_type, denominator) in schema.items():
        tracker.reduce_types[key] = reduce_type
        if denominator is not None:
            tracker.denominators[key] = denominator

    for key, (reduce_type, denominator) in schema.items():
        if tracker.stats.get(key):
            continue

        if reduce_type == ReduceType.SCALAR:
            # The scalar reducer already treats an empty local list as count zero.
            tracker.stats[key]
            continue

        if reduce_type == ReduceType.SUM:
            dtype = torch.bool if key in denominator_keys else torch.float32
            tracker.stats[key].append(torch.zeros(1, dtype=dtype, device=device))
            continue

        if denominator is None:
            raise RuntimeError(
                f"Statistic {key!r} uses {reduce_type} without a denominator."
            )
        denominator_values = tracker.stats.get(denominator)
        if denominator_values:
            template = denominator_values[0]
            tracker.stats[key].append(
                torch.zeros_like(template, dtype=torch.float32, device=device)
            )
        else:
            tracker.stats[denominator].append(
                torch.zeros(1, dtype=torch.bool, device=device)
            )
            tracker.stats[key].append(
                torch.zeros(1, dtype=torch.float32, device=device)
            )


def export_all_distributed_safe(
    *,
    reduce_group: dist.ProcessGroup | None,
    device: torch.device,
    reset: bool = True,
) -> dict[str, float]:
    """Export metrics after making their collective schemas identical on all ranks.

    AReaL synchronizes metric names before reduction, but its tracker metadata is
    process-local. If one rank did not record a rollout metric, it otherwise
    defaults that metric to ``SCALAR`` while peers may reduce it as
    ``AVG_MIN_MAX``. The resulting different collective order deadlocks NCCL.
    Missing metrics are padded with zero-contribution values here.
    """

    if reduce_group is None:
        return stats_tracker.export_all(reset=reset)

    world_size = dist.get_world_size(reduce_group)
    schemas_by_rank: list[
        dict[str, dict[str, tuple[str, str | None]]] | None
    ] = [None] * world_size
    dist.all_gather_object(
        schemas_by_rank,
        _local_schemas(),
        group=reduce_group,
    )
    merged = _merge_schemas(
        [schema for schema in schemas_by_rank if schema is not None]
    )

    for tracker_name, schema in merged.items():
        _pad_tracker(stats_tracker.get(tracker_name), schema, device)

    return stats_tracker.export_all(reduce_group=reduce_group, reset=reset)
