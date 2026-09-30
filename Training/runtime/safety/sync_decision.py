from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyncDecision:
    action: str
    global_trainable: int
    global_batch_size: int


def decide_sync_action(
    *,
    present_by_rank: list[bool],
    trainable_by_rank: list[int],
    batch_size_by_rank: list[int],
) -> SyncDecision:
    """Describe the collective action required by a set of rank-local batches."""
    if not present_by_rank or not all(present_by_rank):
        return SyncDecision("skip_missing_batch", 0, sum(batch_size_by_rank))

    global_trainable = sum(trainable_by_rank)
    global_batch_size = sum(batch_size_by_rank)
    if global_trainable < len(present_by_rank):
        return SyncDecision(
            "skip_insufficient_trainable", global_trainable, global_batch_size
        )
    if global_trainable != global_batch_size:
        return SyncDecision("filter_and_redistribute", global_trainable, global_batch_size)
    return SyncDecision("passthrough", global_trainable, global_batch_size)

