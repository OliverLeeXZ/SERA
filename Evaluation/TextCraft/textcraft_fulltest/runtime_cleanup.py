from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


ResultT = TypeVar("ResultT")
AsyncOperation = Callable[[], Awaitable[ResultT]]
AsyncCleanup = Callable[[], Awaitable[None]]


async def run_with_litellm_cleanup(
    operation: AsyncOperation[ResultT],
    cleanup: AsyncCleanup | None = None,
    *,
    cleanup_timeout_seconds: float = 30.0,
) -> ResultT:
    if cleanup is None:
        from litellm import close_litellm_async_clients

        cleanup = close_litellm_async_clients

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
