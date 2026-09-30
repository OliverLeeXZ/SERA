from __future__ import annotations

from types import MethodType
from typing import Any, Callable

import torch
import torch.distributed as dist

from platoon.utils.train import post_process_and_redistribute_tensor_container


def _get_local_batch_size(batch: dict[str, Any]) -> int:
    """Infer a trajectory count without importing the full AReaL trainer."""
    for key in ("attention_mask", "input_ids"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return int(value.shape[0])
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError("Unable to infer local batch size from batch contents")


def _index_local_batch(
    batch: dict[str, Any], keep: torch.Tensor
) -> dict[str, Any]:
    """Apply one datum mask to tensors and per-datum Python lists."""
    local_batch_size = _get_local_batch_size(batch)
    keep_list = keep.cpu().tolist()
    filtered: dict[str, Any] = {}
    for key, value in batch.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 1
            and value.shape[0] == local_batch_size
        ):
            filtered[key] = value[keep.to(value.device)]
        elif isinstance(value, list) and len(value) == local_batch_size:
            filtered[key] = [
                item
                for item, should_keep in zip(value, keep_list)
                if should_keep
            ]
        else:
            filtered[key] = value
    return filtered


def _with_trainable_datums(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if "rewards" in normalized and "trainable_datums" not in normalized:
        normalized["trainable_datums"] = torch.ones_like(
            normalized["rewards"], dtype=torch.bool
        )
    return normalized


def canonicalize_trajectory_dicts(
    tensor_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give task results one deterministic schema before local concatenation."""
    if not tensor_dicts:
        return tensor_dicts

    normalized = [_with_trainable_datums(item) for item in tensor_dicts]
    common_keys = set(normalized[0])
    for item in normalized[1:]:
        common_keys.intersection_update(item)
    ordered_keys = sorted(common_keys)
    return [{key: item[key] for key in ordered_keys} for item in normalized]


def _collective_protocol(value: Any) -> tuple[Any, ...]:
    """Describe the collective calls made by all_gather_tensor_container."""
    if torch.is_tensor(value):
        return ("tensor", str(value.dtype))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (key, _collective_protocol(value[key]))
                for key in sorted(value)
            ),
        )
    if isinstance(value, list):
        return (
            "list",
            len(value),
            tuple(_collective_protocol(item) for item in value),
        )
    return ("object",)


def compatible_rank_keys(
    protocols: list[dict[str, tuple[Any, ...]]],
) -> list[str]:
    """Return keys whose recursive collective protocol agrees on every rank."""
    if not protocols:
        return []
    common_keys = set(protocols[0])
    for protocol in protocols[1:]:
        common_keys.intersection_update(protocol)
    return sorted(
        key
        for key in common_keys
        if all(protocol[key] == protocols[0][key] for protocol in protocols[1:])
    )


def harmonize_rank_batch_schema(
    data: dict[str, Any],
    *,
    group,
) -> dict[str, Any]:
    """Make recursive all-gather traversal identical across data-parallel ranks."""
    normalized = _with_trainable_datums(data)
    local_protocol = {
        key: _collective_protocol(value) for key, value in normalized.items()
    }
    world_size = dist.get_world_size(group)
    protocols: list[dict[str, tuple[Any, ...]] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(protocols, local_protocol, group=group)
    gathered_protocols = [protocol for protocol in protocols if protocol is not None]
    compatible_keys = compatible_rank_keys(gathered_protocols)
    if not compatible_keys:
        raise RuntimeError("No rank-compatible rollout batch keys remain")

    if dist.get_rank(group=group) == 0:
        all_keys = set().union(*(protocol.keys() for protocol in gathered_protocols))
        dropped = sorted(all_keys.difference(compatible_keys))
        if dropped:
            print(
                "[Project26] Dropping rank-incompatible optional batch keys: "
                + ", ".join(dropped),
                flush=True,
            )
    return {key: normalized[key] for key in compatible_keys}


def install_schema_safe_trajectory_concat() -> None:
    """Normalize completed task results before WorkflowExecutor concatenates them."""
    import areal.core.workflow_executor as workflow_executor

    original = workflow_executor.concat_padded_tensors
    if getattr(original, "_project26_schema_safe", False):
        return

    def schema_safe_concat(tensor_dicts, *args, **kwargs):
        return original(
            canonicalize_trajectory_dicts(tensor_dicts), *args, **kwargs
        )

    schema_safe_concat._project26_schema_safe = True
    workflow_executor.concat_padded_tensors = schema_safe_concat
    print("[Project26] Installed schema-safe task-result concatenation.", flush=True)


def install_schema_safe_distributed_rollout() -> None:
    """Keep recursive tensor-container traversal identical across ranks.

    Do not add a presence collective around ``_broadcast_and_redistribute_batch``.
    AReaL already owns the collective protocol in that method, and inserting one
    there can race with a rank that is still finishing rollout generation.  The
    old presence all-reduce was the source of the ``1086324737/4`` diagnostic:
    ranks had entered different collective sequences and the scalar was combined
    with unrelated payload data.
    """
    import areal.core.dist_rollout as dist_rollout

    original_redistribute = dist_rollout.redistribute
    if getattr(original_redistribute, "_project26_schema_safe", False):
        return

    def schema_safe_redistribute(data, granularity=1, group=None):
        data = harmonize_rank_batch_schema(data, group=group)
        return original_redistribute(data, granularity=granularity, group=group)

    schema_safe_redistribute._project26_schema_safe = True
    dist_rollout.redistribute = schema_safe_redistribute
    print("[Project26] Installed schema-safe distributed rollout.", flush=True)


def _all_reduce_int(value: int, *, op: dist.ReduceOp, device: torch.device, group) -> int:
    tensor = torch.tensor([value], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=op, group=group)
    return int(tensor.item())


def synchronize_training_batch(
    batch: dict[str, Any] | None,
    *,
    device: torch.device,
    group,
    shuffle: bool,
    ensure_divisible_by: int,
) -> dict[str, Any] | None:
    """Make filtering and final rank sharding one collective operation.

    Recursive-agent trajectories vary in length and optional fields. The
    upstream token-balanced redistribution may therefore leave ranks with
    different sequence counts.  FSDP requires every rank to execute the same
    number of forward/backward collectives, so this function always rebuilds the
    accepted global batch and gives every rank an equal, non-empty sequence
    shard before PPO starts.
    """
    world_size = dist.get_world_size(group)
    required_keys = {"attention_mask", "input_ids", "rewards"}
    metadata: dict[str, Any] = {
        "present": batch is not None,
        "batch_size": 0,
        "trainable": 0,
        "missing_keys": [],
        "error": None,
    }
    trainable_mask: torch.Tensor | None = None

    if batch is not None:
        try:
            local_batch_size = _get_local_batch_size(batch)
            missing_keys = sorted(required_keys.difference(batch))
            trainable_value = batch.get("trainable_datums")
            if trainable_value is None:
                trainable_mask = torch.ones(
                    local_batch_size, dtype=torch.bool, device=device
                )
            else:
                trainable_mask = trainable_value.to(device=device).bool()
                if trainable_mask.numel() != local_batch_size:
                    raise ValueError(
                        "trainable_datums length does not match the local batch "
                        f"size: {trainable_mask.numel()} != {local_batch_size}"
                    )
            metadata.update(
                batch_size=local_batch_size,
                trainable=int(trainable_mask.sum().item()),
                missing_keys=missing_keys,
            )
        except Exception as exc:  # converted into one rank-consistent skip
            metadata["error"] = f"{type(exc).__name__}: {exc}"

    rank_metadata: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(rank_metadata, metadata, group=group)
    gathered = [item or {} for item in rank_metadata]

    invalid = [
        (rank, item)
        for rank, item in enumerate(gathered)
        if not item.get("present")
        or item.get("error")
        or item.get("missing_keys")
        or int(item.get("batch_size", 0)) <= 0
    ]
    if invalid:
        if dist.get_rank(group=group) == 0:
            print(
                "[Project26] Skipping update before PPO because one or more "
                f"rank batches are invalid: {invalid}",
                flush=True,
            )
        return None

    assert batch is not None
    assert trainable_mask is not None
    local_batch_size = int(metadata["batch_size"])
    local_trainable = int(metadata["trainable"])
    global_trainable = sum(int(item["trainable"]) for item in gathered)
    global_batch_size = sum(int(item["batch_size"]) for item in gathered)

    if global_trainable < world_size:
        if dist.get_rank(group=group) == 0:
            print(
                "[Project26] Skipping update with insufficient trainable data: "
                f"{global_trainable} datums for {world_size} ranks.",
                flush=True,
            )
        return None

    batch.pop("trainable_datums", None)
    if local_trainable != local_batch_size:
        batch = _index_local_batch(batch, trainable_mask)

    # Always redistribute.  This is deliberate: even a fully trainable batch
    # can have unequal per-rank sequence counts after token-balanced rollout
    # redistribution, which makes FSDP ranks execute different microbatch loops.
    batch = post_process_and_redistribute_tensor_container(
        batch,
        shuffle=shuffle,
        ensure_divisible_by=ensure_divisible_by,
        group=group,
    )

    final_local_size = _get_local_batch_size(batch)
    final_sizes: list[int | None] = [None] * world_size
    dist.all_gather_object(final_sizes, final_local_size, group=group)
    if any(size is None or int(size) <= 0 for size in final_sizes):
        if dist.get_rank(group=group) == 0:
            print(
                "[Project26] Skipping update because equal redistribution "
                f"produced an empty rank: {final_sizes}",
                flush=True,
            )
        return None
    if len({int(size) for size in final_sizes if size is not None}) != 1:
        if dist.get_rank(group=group) == 0:
            print(
                "[Project26] Skipping update because final rank batch sizes "
                f"are unequal: {final_sizes}",
                flush=True,
            )
        return None

    if dist.get_rank(group=group) == 0:
        print(
            "[Project26] Prepared equal rank batches for PPO: "
            f"{global_trainable}/{global_batch_size} trainable datums, "
            f"per_rank={final_sizes[0]}.",
            flush=True,
        )
    return batch


def install_sync_safe_prepare_batch(trainer) -> None:
    """Install the Project 26 fix on one trainer instance only."""
    actor = trainer.actor
    original: Callable[..., dict[str, Any] | None] = actor.prepare_batch

    if dist.get_rank(group=actor.data_parallel_group) == 0:
        print(
            "[Project26] Installed rank-consistent trainable-datum filtering.",
            flush=True,
        )

    def sync_safe_prepare_batch(self, *args, **kwargs):
        batch = original(*args, **kwargs)
        return synchronize_training_batch(
            batch,
            device=self.device,
            group=self.data_parallel_group,
            shuffle=trainer.config.rollout.shuffle_cross_task,
            ensure_divisible_by=trainer.config.rollout.ensure_batch_divisible_by,
        )

    actor.prepare_batch = MethodType(sync_safe_prepare_batch, actor)
