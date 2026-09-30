from __future__ import annotations

import json
import math
import re
from statistics import mean
from typing import Any, Sequence


def extract_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    )
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output does not contain a JSON object")


def parse_policy_score(text: str) -> tuple[float, dict[str, Any]]:
    payload = extract_json_object(text)
    value = payload.get("final_score", payload.get("score"))
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Policy score must be finite and in [0, 1]")
    success = payload.get("success")
    if success is False and score != 0.0:
        raise ValueError("Failed policy-scored trajectory must have score 0")
    if success is True and score <= 0.0:
        raise ValueError("Successful policy-scored trajectory must have positive score")
    return score, payload


def parse_rubric(text: str, min_criteria: int) -> dict[str, Any]:
    payload = extract_json_object(text)
    rubric_items = payload.get("rubric_items")
    if isinstance(rubric_items, list):
        if len(rubric_items) < min_criteria:
            raise ValueError(f"Rubric needs at least {min_criteria} rubric_items")
        normalized_items = []
        weights = []
        for item in rubric_items:
            if not isinstance(item, dict):
                raise ValueError("Every rubric item must be an object")
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            if not name or not description:
                raise ValueError("Rubric item name and description are required")
            weight = float(item.get("weight", 0.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Rubric item weights must be finite and non-negative")
            weights.append(weight)
            normalized_items.append(
                {
                    "name": name,
                    "weight": weight,
                    "description": description,
                    "score_0": str(item.get("score_0", "")),
                    "score_full": str(item.get("score_full", "")),
                }
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError("Rubric item weights must have positive total weight")
        # Rubric_test allows approximate sums. Normalize here so downstream scoring
        # always receives a stable frozen rubric.
        for item in normalized_items:
            item["weight"] = float(item["weight"]) / total
        return {
            "rubric_items": normalized_items,
            "success_gate": str(payload.get("success_gate", "")),
            "failure_conditions": (
                [str(item) for item in payload.get("failure_conditions")]
                if isinstance(payload.get("failure_conditions"), list)
                else [str(payload.get("failure_conditions", ""))]
            ),
            "scoring_procedure": str(payload.get("scoring_procedure", "")),
        }

    criteria = payload.get("criteria")
    if not isinstance(criteria, list) or len(criteria) < min_criteria:
        raise ValueError(f"Rubric needs at least {min_criteria} criteria")
    weights: list[float] = []
    ids: set[str] = set()
    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise ValueError("Every rubric criterion must be an object")
        criterion_id = str(criterion.get("id", "")).strip()
        if not criterion_id or criterion_id in ids:
            raise ValueError("Rubric criterion ids must be non-empty and unique")
        ids.add(criterion_id)
        if not str(criterion.get("description", "")).strip():
            raise ValueError("Rubric criterion description is required")
        if not str(criterion.get("required_evidence", "")).strip():
            raise ValueError("Rubric required_evidence is required")
        anchors = criterion.get("anchors")
        if not isinstance(anchors, dict) or not all(
            str(key) in anchors for key in ("0", "0.5", "1")
        ):
            raise ValueError("Rubric criteria require 0/0.5/1 anchors")
        weight = float(criterion.get("weight"))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Rubric weights must be finite and non-negative")
        weights.append(weight)
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("Rubric weights must sum to 1")
    return payload


def parse_teacher_scores(text: str, expected: int) -> tuple[list[float], dict[str, Any]]:
    payload = extract_json_object(text)
    values = payload.get("scores")
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"Teacher must return exactly {expected} scores")
    if values and isinstance(values[0], dict):
        normalized = []
        for item in values:
            if not isinstance(item, dict):
                raise ValueError("Teacher score entries must all be objects")
            branch_index = int(item["branch_index"])
            success = bool(item["success"])
            score = float(item["score"])
            if branch_index < 0 or branch_index >= expected:
                raise ValueError(f"Invalid teacher branch_index={branch_index}")
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Teacher scores must be finite and in [0, 1]")
            if not success and score != 0.0:
                raise ValueError("Failed teacher-scored trajectory must have score 0")
            if success and score <= 0.0:
                raise ValueError("Successful teacher-scored trajectory must have positive score")
            normalized.append(
                {
                    "branch_index": branch_index,
                    "success": success,
                    "score": score,
                    "reason": str(item.get("reason", "")),
                }
            )
        normalized.sort(key=lambda item: item["branch_index"])
        if [item["branch_index"] for item in normalized] != list(range(expected)):
            raise ValueError("Teacher scores must cover branch_index values exactly once")
        payload = {**payload, "scores": normalized}
        scores = [float(item["score"]) for item in normalized]
        return scores, payload

    scores = [float(value) for value in values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("Teacher scores must be finite and in [0, 1]")
    return scores, payload


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = mean(left_ranks)
    right_mean = mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_var = sum((value - left_mean) ** 2 for value in left_ranks)
    right_var = sum((value - right_mean) ** 2 for value in right_ranks)
    if left_var == 0.0 or right_var == 0.0:
        return None
    return numerator / math.sqrt(left_var * right_var)


def agreement_score(
    policy_scores: Sequence[float],
    teacher_scores: Sequence[float],
    alpha: float,
) -> dict[str, float]:
    if len(policy_scores) != len(teacher_scores) or len(policy_scores) < 2:
        raise ValueError("Agreement vectors must have equal length >= 2")
    rho = spearman_correlation(policy_scores, teacher_scores)
    rank_score = 0.5 if rho is None else (rho + 1.0) / 2.0
    mae = mean(
        abs(policy - teacher)
        for policy, teacher in zip(policy_scores, teacher_scores, strict=True)
    )
    calibration = 1.0 - mae
    combined = alpha * rank_score + (1.0 - alpha) * calibration
    return {
        "spearman": 0.0 if rho is None else rho,
        "rank_score": rank_score,
        "mae": mae,
        "calibration_score": calibration,
        "agreement_score": combined,
    }


def leave_one_out_agreement_credits(
    policy_scores: Sequence[float],
    teacher_scores: Sequence[float],
    alpha: float,
) -> tuple[list[float], list[float], dict[str, float]]:
    if len(policy_scores) < 3:
        raise ValueError("Leave-one-out agreement credit needs at least 3 branches")
    full = agreement_score(policy_scores, teacher_scores, alpha)
    raw: list[float] = []
    for index in range(len(policy_scores)):
        policy_loo = list(policy_scores[:index]) + list(policy_scores[index + 1 :])
        teacher_loo = list(teacher_scores[:index]) + list(teacher_scores[index + 1 :])
        without = agreement_score(policy_loo, teacher_loo, alpha)
        raw.append(full["agreement_score"] - without["agreement_score"])
    baseline = mean(raw)
    centered = [value - baseline for value in raw]
    return raw, centered, full
