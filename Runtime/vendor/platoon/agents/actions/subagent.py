import asyncio
from typing import Any, cast

from platoon.agents.base import ForkableAgent
from platoon.envs.base import ForkableEnv
from platoon.episode.context import (
    budget_tracker,
    current_agent,
    current_env,
    current_trajectory,
    episode_step_timeout,
    subagent_launcher_override,
)
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import BudgetExceededError
from platoon.utils.span_profile import profile_span


def resolve_trajectory_result(finish_message: Any, error_message: Any) -> Any:
    """Prefer a completed result while preserving meaningful falsy values."""
    if finish_message is not None:
        return finish_message
    if error_message is not None:
        return error_message
    return ""


async def launch_subagent(goal: str, max_steps: int = 15, task_misc: dict | None = None, verbose: bool = True) -> Any:
    override = subagent_launcher_override.get()
    if override is not None:
        return await override(
            goal=goal,
            max_steps=max_steps,
            task_misc=task_misc,
            verbose=verbose,
        )
    return await launch_subagent_default(goal, max_steps, task_misc, verbose)


async def launch_subagent_default(
    goal: str,
    max_steps: int = 15,
    task_misc: dict | None = None,
    verbose: bool = True,
) -> Any:
    """Launch a subagent to solve a task.

    Args:
        goal: The goal of the subagent.
        max_steps: The maximum number of steps the subagent can take.

    Returns:
        Returns the result of the subagent's execution.
    """
    # Cast is safe here: launch_subagent only works in contexts with forkable agents/envs
    agent = cast(ForkableAgent, current_agent.get())
    env = cast(ForkableEnv, current_env.get())
    task = env.task
    parent_traj = current_trajectory.get()
    async with profile_span(
        "launch_subagent",
        metadata={
            "goal_len": len(goal),
            "max_steps": max_steps,
            "parent_task_id": getattr(task, "id", None),
            "parent_trajectory_id": parent_traj.id,
        },
    ):
        subtask = task.fork(goal, max_steps, task_misc=task_misc)
        forked_agent = await agent.fork(subtask)
        forked_env = await env.fork(subtask)

        try:
            budget_tracker.get().reserve_budget(max_steps + 1, raise_on_failure=True)
        except (BudgetExceededError, ValueError) as e:
            guidance = getattr(e, "guidance", "")
            msg = f"Not enough budget to launch subagent for goal {goal}. {e}"
            if guidance:
                msg += " " + guidance
            return msg

        try:
            traj = await asyncio.create_task(
                run_episode(
                    forked_agent,
                    forked_env,
                    timeout=episode_step_timeout.get(),
                )
            )
        finally:
            budget_tracker.get().release_budget(max_steps + 1)

        used_recursive = int(budget_tracker.get().used_budget_for(traj.id))
        remaining_total = int(budget_tracker.get().remaining_budget())

        budget_message = (
            f"\n\nBudget used by subagent: {used_recursive}/{max_steps} steps. "
            f"Total remaining budget for the current task is {remaining_total} steps.\n"
        )

        result = resolve_trajectory_result(traj.finish_message, traj.error_message)
        if verbose:
            return str(result) + budget_message
        return result
