"""Return sampling capacity when an accepted task batch is discarded before PPO."""


def release_skipped_rollout_capacity(trainer, consumed_task_groups: int) -> int:
    """Reclassify consumed groups as rejected without advancing the policy version.

    AReaL's strict staleness quota is based on accepted task groups, not token
    datums. Leaving a discarded batch accepted exhausts the current version's
    quota and makes the following prepare_batch wait forever. Only data-parallel
    heads consumed tasks; LoRA consumes tasks exclusively on global rank zero.
    """
    actor = trainer.actor
    if not actor.is_data_parallel_head():
        return 0
    if getattr(trainer, "use_lora", False):
        import torch.distributed as dist
        if dist.get_rank() != 0:
            return 0
    if not isinstance(consumed_task_groups, int) or consumed_task_groups <= 0:
        raise ValueError("Discarded task-group count must be a positive integer")
    engine = getattr(trainer.rollout, "_engine", trainer.rollout)
    executor = getattr(engine, "workflow_executor", None)
    if executor is None:
        raise RuntimeError("Cannot release skipped batch capacity: rollout executor is unavailable")
    manager = executor.staleness_manager
    with manager.lock:
        if manager.rollout_stat.accepted < consumed_task_groups:
            raise RuntimeError("Discarded batch exceeds accepted rollout count")
        manager.rollout_stat.accepted -= consumed_task_groups
        manager.rollout_stat.rejected += consumed_task_groups
    return consumed_task_groups
