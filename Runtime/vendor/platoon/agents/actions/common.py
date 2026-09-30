from platoon.episode.context import finish_message

# TODO: We should probably have some sort of registry with an "action" annotation
# that can be used to register actions. This should also allow setting hooks for
# resetting contextvars at the beginning of a run_agent loop.


def finish(message: str = "") -> str:
    """End the agent trajectory and provide a message to the user.
    E.g., You can use the message to provide the user an answer to the task.

    Args:
        message: The message to provide to the user.

    Returns:
        The message provided to the user.
    """
    finish_message.set(message)
    return message
