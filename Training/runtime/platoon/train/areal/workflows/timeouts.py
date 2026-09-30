"""Bounded cancellation helpers for in-process rollout workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


T = TypeVar("T")
_DETACHED_TASKS: set[asyncio.Task[object]] = set()


class RolloutHardTimeout(TimeoutError):
    """Raised after a rollout exceeds its end-to-end wall-clock budget."""


def _consume_detached_task(task: asyncio.Task[object]) -> None:
    _DETACHED_TASKS.discard(task)
    try:
        task.result()
    except BaseException:
        pass


def _detach(task: asyncio.Task[object]) -> None:
    _DETACHED_TASKS.add(task)
    task.add_done_callback(_consume_detached_task)


async def cancel_tasks(
    tasks: list[asyncio.Task[object]],
    *,
    grace_seconds: float,
) -> None:
    """Cancel tasks without allowing a stuck cancellation to block training."""

    active = [task for task in tasks if not task.done()]
    for task in active:
        task.cancel()
    if not active:
        return

    done, pending = await asyncio.wait(active, timeout=grace_seconds)
    for task in done:
        try:
            task.result()
        except BaseException:
            pass
    for task in pending:
        _detach(task)


async def await_with_hard_timeout(
    awaitable: Awaitable[T],
    *,
    timeout_seconds: float,
    cancellation_grace_seconds: float = 30.0,
    label: str,
) -> T:
    """Await work with a deadline that is not extended by slow cancellation."""

    task = asyncio.create_task(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    except asyncio.CancelledError:
        task.cancel()
        _detach(task)
        raise

    if task in done:
        return task.result()

    task.cancel()
    done, _ = await asyncio.wait(
        {task}, timeout=cancellation_grace_seconds
    )
    if task in done:
        try:
            task.result()
        except BaseException:
            pass
    else:
        _detach(task)
    raise RolloutHardTimeout(
        f"{label} exceeded hard timeout={timeout_seconds:.0f}s"
    )
