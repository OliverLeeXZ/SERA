from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rubric_subagent_reward.client import (
    KimiRubricClient,
    ParsedGeneration,
    PolicyRubricClient,
)
from rubric_subagent_reward.config import RubricSubagentRewardConfig


def _prepare_global_rubric(
    template_path: str | Path,
) -> tuple[Path, dict[str, Any], str, str]:
    path = Path(template_path).expanduser().resolve()
    rubric = _load_template(path)
    text = json.dumps(rubric, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return path, rubric, text, f"global-rubric:{digest}"


def _load_template(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Global rubric template does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Global rubric template is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Global rubric template must be a JSON object")

    success_gate = payload.get("success_gate")
    if not isinstance(success_gate, str) or not success_gate.strip():
        raise ValueError("Global rubric success_gate is required")
    failure_conditions = payload.get("failure_conditions")
    if not isinstance(failure_conditions, list) or not failure_conditions:
        raise ValueError("Global rubric failure_conditions must be non-empty")

    items = payload.get("rubric_items")
    if not isinstance(items, list) or not items:
        raise ValueError("Global rubric rubric_items must be non-empty")
    names: set[str] = set()
    weights: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every global rubric item must be an object")
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            raise ValueError("Global rubric item names must be unique and non-empty")
        names.add(name)
        for field_name in ("description", "score_0", "score_full"):
            if not str(item.get(field_name, "")).strip():
                raise ValueError(
                    f"Global rubric item {name!r} requires {field_name}"
                )
        weight = float(item.get("weight", -1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Global rubric weights must be finite and non-negative")
        weights.append(weight)
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Global rubric weights must sum to 1")

    anchors = payload.get("score_anchors")
    if not isinstance(anchors, dict) or not all(
        key in anchors for key in ("0.0", "0.25", "0.5", "0.75", "1.0")
    ):
        raise ValueError("Global rubric requires fixed 0/0.25/0.5/0.75/1 anchors")
    procedure = payload.get("scoring_procedure")
    if not isinstance(procedure, list) or not procedure:
        raise ValueError("Global rubric scoring_procedure must be non-empty")
    return payload


class _GlobalRubricMixin:
    template_path: Path
    global_rubric: dict[str, Any]
    _rubric_text: str
    _rubric_request_id: str

    def _set_global_rubric(self, template_path: str | Path) -> None:
        (
            self.template_path,
            self.global_rubric,
            self._rubric_text,
            self._rubric_request_id,
        ) = _prepare_global_rubric(template_path)

    async def generate_rubric(
        self, messages: list[dict[str, str]]
    ) -> ParsedGeneration:
        del messages
        return ParsedGeneration(
            text=self._rubric_text,
            parsed=copy.deepcopy(self.global_rubric),
            request_id=self._rubric_request_id,
            cache_hit=True,
        )


class GlobalRubricKimiClient(_GlobalRubricMixin, KimiRubricClient):
    """Use one local rubric for every call and KIMI only for scoring."""

    def __init__(
        self,
        config: RubricSubagentRewardConfig,
        template_path: str | Path,
    ) -> None:
        super().__init__(config)
        self._set_global_rubric(template_path)


class GlobalRubricPolicyClient(_GlobalRubricMixin, PolicyRubricClient):
    """Use one local rubric for every call and the policy only for scoring."""

    def __init__(
        self,
        config: RubricSubagentRewardConfig,
        template_path: str | Path,
        **policy_kwargs: Any,
    ) -> None:
        super().__init__(config, **policy_kwargs)
        self._set_global_rubric(template_path)
