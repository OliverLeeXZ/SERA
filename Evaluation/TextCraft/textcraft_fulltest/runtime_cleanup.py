from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar


ResultT = TypeVar("ResultT")
AsyncOperation = Callable[[], Awaitable[ResultT]]
AsyncCleanup = Callable[[], Awaitable[None]]


def _discard_pending_logs(worker) -> int:
    """Close only queued, unstarted SDK telemetry; durable rollouts are untouched."""
    queue = getattr(worker, "_queue", None)
    if queue is None:
        return 0
    discarded = 0
    while True:
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            return discarded
        coroutine = task.get("coroutine") if isinstance(task, dict) else None
        if inspect.iscoroutine(coroutine):
            coroutine.close()
        queue.task_done()
        discarded += 1


async def stop_litellm_logging_worker(worker, *, flush_timeout_seconds: float = 5.0) -> int:
    # Flush best-effort callbacks while their original loop is still alive.
    # Leaving queued callbacks for SDK atexit can run an unbounded coroutine
    # on a fresh loop after the application's final report has been printed.
    try:
        await asyncio.wait_for(worker.flush(), timeout=flush_timeout_seconds)
    except TimeoutError:
        pass
    discarded = _discard_pending_logs(worker)
    try:
        await worker.stop()
    finally:
        discarded += _discard_pending_logs(worker)
    return discarded


async def cleanup_litellm_runtime(*, logging_flush_timeout_seconds: float = 5.0) -> None:
    from litellm import close_litellm_async_clients
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    try:
        discarded = await stop_litellm_logging_worker(
            GLOBAL_LOGGING_WORKER, flush_timeout_seconds=logging_flush_timeout_seconds)
        print(f"[cleanup] LiteLLM logging worker stopped; discarded={discarded}", flush=True)
    finally:
        await close_litellm_async_clients()


async def run_with_litellm_cleanup(
    operation: AsyncOperation[ResultT],
    cleanup: AsyncCleanup | None = None,
    *,
    cleanup_timeout_seconds: float = 30.0,
) -> ResultT:
    if cleanup is None:
        cleanup = cleanup_litellm_runtime

    try:
        return await operation()
    finally:
        try:
            await asyncio.wait_for(cleanup(), timeout=cleanup_timeout_seconds)
            print("[cleanup] LiteLLM async clients closed", flush=True)
        except TimeoutError:
            print(
                "[cleanup] LiteLLM async client cleanup timed out; "
                "continuing process shutdown",
                flush=True,
            )
