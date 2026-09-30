from __future__ import annotations

import json
import re
from typing import Any


def _field_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _contains_json_object(text: str) -> bool:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return True
    return False


def _looks_like_structured_rubric_or_score(text: str) -> bool:
    """Recognize likely rubric/score payloads without disabling thinking."""

    return _contains_json_object(text) and bool(
        re.search(
            r'"(?:rubric_items|final_score|scores|score|success)"\s*:',
            text,
        )
    )


def message_text(message: Any) -> str:
    """Choose a structured response from compatible message fields.

    Gateways differ on whether a thinking model places its final answer in
    ``content`` or ``reasoning_content``. This helper does not disable
    thinking; it only prefers the field that contains a plausible rubric or
    score payload and leaves filtering to the caller.
    """

    candidates = [
        _field_text(getattr(message, field, None))
        for field in ("content", "reasoning_content", "reasoning")
    ]
    candidates = [candidate for candidate in candidates if candidate.strip()]
    for candidate in candidates:
        if _looks_like_structured_rubric_or_score(candidate):
            return candidate
    for candidate in candidates:
        if _contains_json_object(candidate):
            return candidate
    return candidates[0] if candidates else ""
