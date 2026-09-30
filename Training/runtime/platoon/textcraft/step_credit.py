"""Rule-based step credit for TextCraft-Synth trajectories.

The scorer compares each craft/delegation action against the current task's
gold crafting plan. It deliberately avoids history-dependent bookkeeping so the
reward is deterministic and cheap to compute inside rollout processing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepCreditConfig:
    enabled: bool = False
    lambda_: float = 0.0
    shortage: float = 1.0
    excess: float = 1.0
    unnecessary: float = 1.0
    invalid: float = 1.0
    unreachable: float = 1.0


@dataclass(frozen=True)
class TextCraftOperation:
    kind: str
    targets: dict[str, int]


@dataclass(frozen=True)
class StepPenalty:
    total: float
    reason: str


def _literal_dict(node: ast.AST) -> dict[str, int] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, dict):
        return None
    targets: dict[str, int] = {}
    for key, count in value.items():
        if isinstance(key, str) and isinstance(count, int) and count > 0:
            targets[key] = count
        else:
            return None
    return targets


def _literal_target(node: ast.AST) -> dict[str, int] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    item, count = value
    if not isinstance(item, str) or not isinstance(count, int) or count <= 0:
        return None
    return {item: count}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _parse_textcraft_operations_with_invalid_count(code: str | None) -> tuple[list[TextCraftOperation], int]:
    """Extract scored operations and count malformed scored calls."""
    if not code:
        return [], 0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        invalid_count = int("craft(" in code) + int("launch_subagent(" in code)
        return [], invalid_count

    operations: list[TextCraftOperation] = []
    invalid_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "craft":
            ingredients = _literal_dict(node.args[0]) if len(node.args) >= 1 else None
            targets = _literal_target(node.args[1]) if len(node.args) >= 2 else None
            if ingredients is not None and targets:
                operations.append(TextCraftOperation(kind="craft", targets=targets))
            else:
                invalid_count += 1
        elif name == "launch_subagent":
            targets = _literal_dict(node.args[0]) if node.args else None
            if targets:
                operations.append(TextCraftOperation(kind="launch_subagent", targets=targets))
            else:
                invalid_count += 1
    return operations, invalid_count


def parse_textcraft_operations(code: str | None) -> list[TextCraftOperation]:
    """Extract valid craft and launch_subagent operations from a CodeAct code cell."""
    operations, _ = _parse_textcraft_operations_with_invalid_count(code)
    return operations


def _required_from_target_items(misc: dict[str, Any]) -> dict[str, int]:
    required: dict[str, int] = {}
    for item, count in (misc.get("target_items") or {}).items():
        if isinstance(item, str) and isinstance(count, int):
            required[item] = count
    return required


def _required_from_gold_steps(gold_steps: list[dict[str, Any]]) -> dict[str, int]:
    required: dict[str, int] = {}
    for step in gold_steps:
        if step.get("action") != "craft":
            continue
        target = step.get("target")
        if not isinstance(target, (list, tuple)) or not target:
            continue
        item = target[0]
        if not isinstance(item, str):
            continue
        result_count = step.get("result_count")
        if not isinstance(result_count, int):
            if len(target) >= 2 and isinstance(target[1], int):
                result_count = target[1]
            else:
                continue
        required[item] = required.get(item, 0) + result_count
    return required


def build_required_crafts(trajectory: dict[str, Any]) -> dict[str, int]:
    """Build item -> minimal produced count from this trajectory's local task."""
    task = trajectory.get("task") or {}
    misc = task.get("misc") or {}

    if "local_gold_trajectory" in misc:
        local_gold = misc.get("local_gold_trajectory") or []
        required = _required_from_gold_steps(local_gold)
        return required

    if _local_reference_unavailable(trajectory):
        return {}

    # Old rollout logs may contain a child task with parent gold_trajectory still
    # attached. In that case, prefer the child task target over stale parent gold.
    if trajectory.get("parent_info"):
        return _required_from_target_items(misc)

    required = _required_from_gold_steps(misc.get("gold_trajectory") or [])
    if required:
        return required

    return _required_from_target_items(misc)


def _local_reference_unavailable(trajectory: dict[str, Any]) -> bool:
    task = trajectory.get("task") or {}
    misc = task.get("misc") or {}
    return bool(trajectory.get("parent_info") and misc.get("local_gold_unavailable"))


def _has_unreachable_audit(step: dict[str, Any]) -> bool:
    reward_misc = (step.get("misc") or {}).get("reward_misc") or {}
    audit = reward_misc.get("step_credit/audit") or []
    if not isinstance(audit, list):
        return False
    return any(isinstance(record, dict) and record.get("caused_unreachable") for record in audit)


def _has_direct_craft_error(step: dict[str, Any]) -> bool:
    text = " ".join(str(step.get(key) or "") for key in ("output", "error"))
    if not text:
        return False
    lowered = text.lower()
    return (
        "error" in lowered
        or "failed" in lowered
        or "wrong amount" in lowered
        or "not divisible" in lowered
    )


def _trajectory_succeeded(trajectory: dict[str, Any]) -> bool:
    if float(trajectory.get("reward", 0.0)) >= 1.0:
        return True
    for step in trajectory.get("steps", []):
        reward_misc = (step.get("misc") or {}).get("reward_misc") or {}
        if float(reward_misc.get("reward/success", 0.0)) >= 1.0:
            return True
    return False


def _penalty_for_operations(
    operations: list[TextCraftOperation],
    required: dict[str, int],
    config: StepCreditConfig,
) -> StepPenalty:
    if not operations:
        return StepPenalty(0.0, "no_craft_action")

    max_penalty = 0.0
    reasons: list[str] = []
    for operation in operations:
        for item, count in operation.targets.items():
            required_count = required.get(item)
            if required_count is None:
                max_penalty = max(max_penalty, config.unnecessary)
                reasons.append("unnecessary")
            elif count < required_count:
                max_penalty = max(max_penalty, config.shortage)
                reasons.append("shortage")
            elif count > required_count:
                max_penalty = max(max_penalty, config.excess)
                reasons.append("excess")
            else:
                reasons.append("exact")

    if not reasons:
        return StepPenalty(0.0, "no_craft_action")
    if max_penalty == 0.0:
        return StepPenalty(0.0, "exact")
    return StepPenalty(max_penalty, "+".join(dict.fromkeys(reasons)))


def _with_max_penalty(penalty: StepPenalty, value: float, reason: str) -> StepPenalty:
    if value <= 0:
        return penalty
    reasons = [] if penalty.reason == "no_craft_action" else penalty.reason.split("+")
    if reason not in reasons:
        reasons.append(reason)
    return StepPenalty(max(penalty.total, value), "+".join(reasons) if reasons else reason)


def compute_step_penalties(
    trajectory: dict[str, Any],
    config: StepCreditConfig,
) -> list[StepPenalty]:
    """Compute one penalty value per trajectory step."""
    required = build_required_crafts(trajectory)
    reference_unavailable = _local_reference_unavailable(trajectory)
    task_succeeded = _trajectory_succeeded(trajectory)
    penalties: list[StepPenalty] = []
    for step in trajectory.get("steps", []):
        operations, invalid_count = _parse_textcraft_operations_with_invalid_count(step.get("code"))
        if reference_unavailable and operations:
            penalty = StepPenalty(0.0, "no_reference")
        else:
            penalty = _penalty_for_operations(operations, required, config)
        if invalid_count:
            penalty = _with_max_penalty(penalty, config.invalid, "invalid")
        if any(operation.kind == "craft" for operation in operations) and _has_direct_craft_error(step):
            penalty = _with_max_penalty(penalty, config.invalid, "invalid")
        # A trajectory that ultimately succeeds was not made irreversibly
        # unreachable by an earlier step, even if the online solver said so.
        if operations and not task_succeeded and _has_unreachable_audit(step):
            reason = penalty.reason if penalty.reason != "exact" else "exact"
            penalty = StepPenalty(penalty.total + config.unreachable, f"{reason}+unreachable")
        penalties.append(penalty)
    return penalties
