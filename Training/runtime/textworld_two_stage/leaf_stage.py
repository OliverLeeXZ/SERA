from __future__ import annotations

import re
from typing import Any

from three_stage_train.data import completion_text
from three_stage_train.token_spans import output_token_mask_for_char_spans
from reward_first_launch.stage import RewardFirstLaunchDelegationStage


TEXTWORLD_DELEGATE_BLOCK = re.compile(
    r"<delegate\b.*?</delegate>", re.IGNORECASE | re.DOTALL
)


def find_textworld_delegate_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return the output spans occupied by TextWorld delegate actions."""

    return tuple(
        (match.start(), match.end())
        for match in TEXTWORLD_DELEGATE_BLOCK.finditer(str(text or ""))
    )


class TextWorldRewardFirstLaunchDelegationStage(
    RewardFirstLaunchDelegationStage
):
    """Project 68 adapter for TextWorld's XML-like delegate protocol.

    TextWorld stores a delegated action in ``action_misc.raw_response`` and
    intentionally leaves ``step["code"]`` empty because it is not a Python
    tool call. The generic launch stage therefore found the child trajectories
    but produced an all-zero token mask. Align the actual delegate block in
    the model completion so only the launch action is trained.
    """

    def _delegation_mask(
        self,
        completion: Any,
        step: dict[str, Any],
    ) -> tuple[list[int], bool, bool]:
        response = completion.model_response
        text = completion_text(completion)
        spans = find_textworld_delegate_spans(text)
        action_misc = ((step.get("misc") or {}).get("action_misc") or {})
        valid = bool(action_misc.get("parse_ok", True))
        if not spans:
            return [0] * len(response.output_tokens), valid, False
        mask = output_token_mask_for_char_spans(
            response.tokenizer,
            list(response.output_tokens),
            spans,
        )
        return mask, valid, bool(any(mask))
