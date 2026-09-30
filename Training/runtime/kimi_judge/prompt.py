from __future__ import annotations

import json
import re
from typing import Any


def _task_goal(task: Any) -> str:
    if isinstance(task, dict):
        return str(task.get("goal", task.get("task_description", "")))
    return str(getattr(task, "goal", ""))


def _parent_goal(task: Any) -> str:
    if isinstance(task, dict):
        parents = task.get("parent_tasks") or []
        if parents:
            return _task_goal(parents[-1])
    parents = getattr(task, "parent_tasks", None) or []
    return _task_goal(parents[-1]) if parents else ""


def _trajectory_text(trajectory: dict[str, Any]) -> str:
    rows: list[str] = []
    for index, step in enumerate(trajectory.get("steps") or [], start=1):
        step = step if isinstance(step, dict) else {}
        misc = step.get("misc") or {}
        textworld = misc.get("textworld") or {}
        infos = textworld.get("infos", {}) if isinstance(textworld, dict) else {}
        rows.append(
            "\n".join(
                [
                    f"### Step {index}",
                    f"Action: {step.get('action', step.get('code', ''))}",
                    f"Observation/Output: {step.get('output', step.get('observation', ''))}",
                    f"Error: {step.get('error') or '(none)'}",
                    f"Environment info: {json.dumps(infos, ensure_ascii=False)}",
                ]
            )
        )
    return "\n\n".join(rows) or "(The SubAgent produced no environment steps.)"


def build_judge_messages(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    task = trajectory.get("task", {})
    root_goal = _task_goal(task)
    parent_goal = _parent_goal(task) or root_goal
    system = (
        "You are a strict binary evaluator for a recursively delegated TextWorld Agent.\n"
        "Judge only whether this SubAgent completed its assigned goal and produced a "
        "real, useful contribution for its immediate Parent. Do not require it to "
        "complete the entire Root Task. Use the trajectory and environment outputs, "
        "not plausibility or invented claims. If the goal is not completed, the "
        "result is unusable, or the evidence is insufficient, mark it unsuccessful.\n"
        "Return exactly one JSON object and no other text:\n"
        '{"success": true or false, "reason": "brief evidence-based reason"}'
    )
    user = "\n\n".join(
        [
            f"# Root Task\n{root_goal}",
            f"# Immediate Parent Goal\n{parent_goal}",
            f"# Assigned SubAgent Goal\n{root_goal}",
            f"# SubAgent Trajectory\n{_trajectory_text(trajectory)}",
            "# Final Decision\nReturn strict JSON with a boolean success field.",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_binary_success(text: str) -> tuple[bool, str]:
    candidates = [str(text or "").strip()]
    candidates.extend(
        re.findall(r"```(?:json)?\s*(.*?)```", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    )
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload, _ = decoder.raw_decode(candidate.lstrip())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        success = payload.get("success")
        reason = payload.get("reason", "")
        if isinstance(success, bool) and isinstance(reason, str):
            return success, reason
    raise ValueError("KIMI Judge output did not contain a strict binary JSON result")
