from __future__ import annotations

import json
from typing import Any


RUBRIC_SYSTEM_PROMPT = """You are a Parent Agent writing an evaluation rubric before a delegated SubAgent runs.

You will see the Parent Agent's task and its trajectory up to the exact decision
where it calls a SubAgent. You will also see the delegated SubAgent task.

Write a rubric that can later score a completed SubAgent trajectory from 0 to 1.
The rubric should reflect the Parent Agent's intent at delegation time: whether
the SubAgent completes the delegated task, returns useful information or results
to the parent, avoids fabricated results, and avoids unnecessary or harmful work.

Do not judge any future rollout here. Do not assume hidden labels. Return JSON
only and do not include markdown.
"""

SCORING_SYSTEM_PROMPT = """You score one completed SubAgent trajectory using the supplied frozen
rubric. Use only evidence present in the trajectory and final environment state.
If the trajectory does not complete the delegated task, the final score must be
exactly 0.0. Return JSON only and do not include markdown."""

TEACHER_SYSTEM_PROMPT = """You are a strict evaluator for recursive agent trajectories.

You will receive one SubAgent-call node containing 8 independent candidate
SubAgent trajectories. Each trajectory starts from the delegated subtask and
ends when the SubAgent returns to its parent.

First decide whether each trajectory succeeds at the delegated SubAgent task.
Then assign a score with this hard gate:
- If the trajectory fails, the score MUST be exactly 0.0.
- If the trajectory succeeds, the score MUST be in (0.0, 1.0].

For successful trajectories, use the score to express quality:
- 1.0: fully completes the task, returns a useful result to the parent, and is
  direct and efficient.
- 0.7-0.9: completes the task, with minor inefficiency or small extra work.
- 0.4-0.6: completes the task, but with substantial inefficiency, confusion, or
  unnecessary detours.
- 0.1-0.3: barely succeeds or returns a result that is correct but fragile.

A trajectory fails if it loops, crashes, fabricates the result, stops before the
task is done, returns an incomplete result, or does not address the delegated
task. Failed trajectories always receive score 0.0, even if they make partial
progress.

Judge only the trajectory content. Do not use any hidden labels. Return JSON
only and do not include markdown.
"""


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _truncate_text(text: str, max_chars: int = 12000) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 4
    keep_tail = max_chars - keep_head
    return (
        text[:keep_head]
        + "\n\n[... middle of parent trajectory truncated ...]\n\n"
        + text[-keep_tail:]
    )


def _render_messages(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return _stable_json(messages)
    rendered: list[str] = []
    for index, message in enumerate(messages):
        if isinstance(message, dict):
            role = message.get("role", f"message_{index}")
            content = message.get("content", "")
            rendered.extend([f"[Parent Message {index}: {role}]", str(content)])
        else:
            rendered.extend([f"[Parent Message {index}]", str(message)])
    return "\n".join(rendered)


def _render_step_like(step: dict[str, Any], *, include_output: bool = True) -> str:
    lines: list[str] = []
    thought = step.get("thought")
    if thought:
        lines.extend(["Thought:", str(thought)])
    code = step.get("code")
    if code:
        lines.extend(["Action Code:", str(code)])
    if include_output:
        output = step.get("output")
        if output is not None:
            lines.extend(["Observation:", str(output)])
        error = step.get("error")
        if error:
            lines.extend(["Error:", str(error)])
        if "reward" in step:
            lines.extend(["Step Reward:", str(step.get("reward"))])
    return "\n".join(lines)


def _render_trajectory_transcript(
    *,
    trajectory: Any,
    branch_index: int,
    initial_environment_state: Any,
    final_environment_state: Any,
    reward: Any = None,
    finish_message: Any = None,
    error_message: Any = None,
) -> str:
    if isinstance(trajectory, dict):
        task_goal = trajectory.get("task_goal")
        if task_goal is None and isinstance(trajectory.get("task"), dict):
            task_goal = trajectory["task"].get("goal")
        steps = trajectory.get("steps") or []
        finish_message = trajectory.get("finish_message", finish_message)
        error_message = trajectory.get("error_message", error_message)
        reward = trajectory.get("reward", reward)
    else:
        task_goal = None
        steps = []

    lines = [
        "[SubAgent Task]",
        str(task_goal or ""),
        "",
        "[Initial State]",
        _stable_json(initial_environment_state),
    ]
    for index, step in enumerate(steps):
        lines.extend(["", f"[Step {index}]"])
        if isinstance(step, dict):
            lines.append(_render_step_like(step))
        else:
            lines.append(str(step))
    lines.extend(["", "[Return To Parent]", str(finish_message or "")])
    if error_message:
        lines.extend(["Error Message:", str(error_message)])
    lines.extend(
        [
            "Final Reward:",
            str(reward or 0.0),
            "Final State:",
            _stable_json(final_environment_state),
        ]
    )
    del branch_index
    return "\n".join(lines)


def build_rubric_messages(
    *,
    parent_prefix: Any,
    parent_action: str,
    child_goal: str,
    environment_state: Any,
    available_tools: Any,
    execution_budget: Any,
    min_criteria: int,
    parent_task: str | None = None,
    parent_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    parent_context = "\n".join(
        [
            "[Parent Task]",
            str(parent_task or ""),
            "",
            "[Parent Metadata Visible To Rubric Generator]",
            _stable_json(parent_metadata or {}),
            "",
            "[Parent Trajectory Before Delegation]",
            _render_messages(parent_prefix),
            "",
            "[Parent Step: Delegation Decision]",
            "The following action is the parent decision that triggers the SubAgent call. Its observation/output is hidden because it happens after the SubAgent returns.",
            str(parent_action),
            "",
            "[Observable Pre-call Environment]",
            _stable_json(environment_state),
            "",
            "[Delegated SubAgent Task]",
            str(child_goal),
            "",
            "[Child Tools]",
            _stable_json(available_tools),
            "",
            "[Child Execution Budget]",
            _stable_json(execution_budget),
        ]
    )
    minimum = max(4, min_criteria)
    user_prompt = "\n".join(
        [
            "Generate a scoring rubric for the delegated SubAgent task.",
            "",
            _truncate_text(parent_context),
            "",
            "Return exactly this JSON schema:",
            "{",
            '  "rubric_items": [',
            '    {"name": "criterion name", "weight": 0.0, "description": "what to check", "score_0": "what earns 0", "score_full": "what earns full credit"}',
            "  ],",
            '  "success_gate": "conditions that must hold for a nonzero score",',
            '  "failure_conditions": ["condition that forces score 0"],',
            '  "scoring_procedure": "brief instructions for assigning a final 0-1 score"',
            "}",
            "",
            "Constraints:",
            f"- Include {minimum} to 8 rubric_items.",
            "- Weights should be non-negative and should approximately sum to 1.",
            "- The success_gate must be strict: if the SubAgent does not complete the delegated task, the later judge should assign 0.",
            "- Make the rubric specific to the delegated task and parent context, not a generic template.",
        ]
    )
    return [
        {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_scoring_messages(
    *,
    rubric: dict[str, Any],
    child_goal: str,
    trajectory: Any,
    final_environment_state: Any,
) -> list[dict[str, str]]:
    payload = _stable_json(
        {
            "frozen_rubric": rubric,
            "delegated_task": child_goal,
            "trajectory": trajectory,
            "final_environment_state": final_environment_state,
        }
    )
    user_prompt = "\n".join(
        [
            "Score this completed SubAgent trajectory using the frozen rubric.",
            "",
            payload,
            "",
            "Return exactly this JSON schema:",
            "{",
            '  "success": false,',
            '  "criterion_scores": [{"name": "criterion name", "score": 0.0, "reason": "short evidence"}],',
            '  "final_score": 0.0,',
            '  "reason": "short overall reason"',
            "}",
            "",
            "Hard rule: if success is false, final_score must be exactly 0.0; if success is true, final_score must be greater than 0.0 and at most 1.0.",
        ]
    )
    return [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_teacher_messages(
    *,
    child_goal: str,
    initial_environment_state: Any,
    trajectories: list[Any],
    final_environment_states: list[Any],
    parent_task: str | None = None,
    branch_metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    metadata = branch_metadata or [{} for _ in trajectories]
    lines = [
        "Evaluate the 8 candidate SubAgent trajectories below.",
        "",
        "[Parent Task]",
        str(parent_task or ""),
        "",
        "[Delegated SubAgent Task]",
        str(child_goal or ""),
        "",
        "Return exactly this JSON schema:",
        '{"scores":[{"branch_index":0,"success":false,"score":0.0,"reason":"short reason"}, ...]}',
        "The scores array must contain exactly 8 entries, one for each branch_index 0..7.",
        "Hard rule: if success is false, score must be exactly 0.0; if success is true, score must be greater than 0.0 and at most 1.0.",
    ]
    for index, (trajectory, final_state, branch) in enumerate(
        zip(trajectories, final_environment_states, metadata, strict=True)
    ):
        branch_index = int(branch.get("branch_index", index))
        lines.extend(
            [
                "",
                f"===== BRANCH {branch_index} =====",
                _render_trajectory_transcript(
                    trajectory=trajectory,
                    branch_index=branch_index,
                    initial_environment_state=branch.get(
                        "initial_state", initial_environment_state
                    ),
                    final_environment_state=final_state,
                    reward=branch.get("reward"),
                    finish_message=branch.get("finish_message"),
                    error_message=branch.get("error_message"),
                ),
            ]
        )
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
