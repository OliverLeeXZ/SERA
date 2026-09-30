import asyncio
from typing import Any, Awaitable, Callable


async def async_timeout_call(
    function: Callable[..., Awaitable[Any]],
    timeout_seconds: int | None = None,
    *args: tuple,
    **kwargs: Any,
) -> Any:
    """
    Async version of timeout_call that uses asyncio.wait_for for timeout handling.

    Args:
        function: The async function to call
        timeout_seconds: Timeout in seconds, or None for no timeout
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        Exception: If the function times out
    """
    if timeout_seconds is None:
        return await function(*args, **kwargs)

    timeout_seconds = int(timeout_seconds)
    timeout_message = f"Function {function.__name__} execution timed out after {timeout_seconds} seconds."

    try:
        result = await asyncio.wait_for(function(*args, **kwargs), timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError as exception:
        raise Exception(timeout_message) from exception
