from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any



ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
DELEGATE_PATTERN = re.compile(r"<delegate>\s*(.*?)\s*</delegate>", re.IGNORECASE | re.DOTALL)
FINISH_PATTERN = re.compile(r"<finish>\s*(.*?)\s*</finish>", re.IGNORECASE | re.DOTALL)
INVENTORY_ACTIONS = frozenset({"inventory", "view inventory", "check inventory"})


@dataclass(frozen=True)
class Turn:
    raw_response: str
    action: str
    parse_ok: bool
    observation: str
    infos: dict[str, Any]


@dataclass(frozen=True)
class Delegation:
    goal: str
    max_steps: int


@dataclass(frozen=True)
class Decision:
    kind: str
    raw_response: str
    action: str = ""
    delegations: tuple[Delegation, ...] = ()
    finish_message: str = ""
    parse_ok: bool = False
    error: str | None = None


def extract_action(raw_response: str) -> tuple[str, bool]:
    text = str(raw_response or "").strip()
    match = ACTION_PATTERN.search(text)
    if match:
        return match.group(1).strip(), True
    # Keep malformed outputs observable instead of silently converting them into a
    # different action. The environment will judge whether the fallback is valid.
    fenced = re.search(r"```(?:text|bash|shell)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip().splitlines()[-1].strip(), False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidate = re.sub(r"^(?:action|command)\s*:\s*", "", lines[-1], flags=re.IGNORECASE)
        return candidate.strip(" `\"'"), False
    return "", False


def is_inventory_action(action: str) -> bool:
    """Return whether an environment action requests the shared inventory."""
    normalized = re.sub(r"\s+", " ", str(action or "").strip().lower())
    return normalized in INVENTORY_ACTIONS


def _parse_delegation_payload(payload: str) -> tuple[Delegation, ...]:
    """Parse one or many JSON delegation requests from a tagged payload."""
    parsed = json.loads(payload)
    if isinstance(parsed, dict) and "delegations" in parsed:
        candidates = parsed["delegations"]
    else:
        candidates = [parsed]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("delegate payload must contain an object or non-empty delegations list")

    delegations: list[Delegation] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("every delegation must be a JSON object")
        goal = str(candidate.get("goal", "") or "").strip()
        if not goal:
            raise ValueError("every delegation must provide a non-empty detailed goal")
        try:
            max_steps = int(candidate.get("max_steps", 20))
        except (TypeError, ValueError) as error:
            raise ValueError("delegation max_steps must be an integer") from error
        if max_steps <= 0:
            raise ValueError("delegation max_steps must be positive")
        delegations.append(
            Delegation(
                goal=goal,
                max_steps=max_steps,
            )
        )
    return tuple(delegations)


def extract_decision(raw_response: str) -> Decision:
    """Parse the mutually exclusive TextWorld action/delegation/return protocol."""
    text = str(raw_response or "").strip()
    action_matches = ACTION_PATTERN.findall(text)
    delegate_matches = DELEGATE_PATTERN.findall(text)
    finish_matches = FINISH_PATTERN.findall(text)
    kinds = sum(bool(matches) for matches in (action_matches, delegate_matches, finish_matches))
    if kinds != 1:
        return Decision(
            kind="invalid",
            raw_response=text,
            error="response must contain exactly one action, delegate, or finish block",
        )
    try:
        if action_matches:
            action = action_matches[0].strip()
            if not action:
                raise ValueError("action block is empty")
            return Decision(kind="action", raw_response=text, action=action, parse_ok=True)
        if delegate_matches:
            return Decision(
                kind="delegate",
                raw_response=text,
                delegations=_parse_delegation_payload(delegate_matches[0]),
                parse_ok=True,
            )
        message = finish_matches[0].strip()
        if not message:
            raise ValueError("finish block is empty")
        return Decision(kind="finish", raw_response=text, finish_message=message, parse_ok=True)
    except (ValueError, json.JSONDecodeError) as error:
        return Decision(kind="invalid", raw_response=text, error=str(error))


def _estimate_tokens(messages: list[dict[str, str]]) -> int:
    # TextCraft's model-side tokenizer is not available in every evaluator process.
    # This conservative estimate is only used to decide which old turns to trim.
    return sum(4 + max(1, len(message["content"]) // 4) for message in messages) + 2
