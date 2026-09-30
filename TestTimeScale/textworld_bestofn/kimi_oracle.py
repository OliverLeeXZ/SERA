from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalJudgeConfig:
    endpoint: str = ""
    model: str = "kimi-k2.6"
    api_key_env: str = "KIMI_API_KEY"
    timeout: float = 1800.0
    retries: int = 2
    max_prompt_tokens: int = 10240
    max_completion_tokens: int = 1024
    temperature: float = 1.0

    def validate(self):
        if not self.endpoint.startswith(("http://", "https://")) or not self.model.strip():
            raise ValueError("External judge requires an HTTP(S) endpoint and model")
        if self.retries < 0 or self.temperature < 0 or min(self.timeout, self.max_prompt_tokens, self.max_completion_tokens) <= 0:
            raise ValueError("Invalid judge timeout, retries, temperature or token budgets")


@dataclass(frozen=True)
class KimiJudgeResult:
    success: bool | None
    reason: str
    raw_output: str
    error: str | None = None
    response_id: str | None = None


def _parse_binary_success(text: str) -> tuple[bool, str]:
    # Some OpenAI-compatible judges (including GLM-5.3) put their analysis in
    # content and append the requested JSON after a closing </think> tag.
    final_text = str(text or '').rsplit('</think>', 1)[-1].strip()
    candidates = [final_text]
    candidates.extend(
        re.findall(
            r'```(?:json)?\s*(.*?)```',
            final_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    decoder = json.JSONDecoder()
    # Accept a trailing JSON object after prose, but never an example JSON
    # embedded earlier in the analysis.
    candidates.extend(final_text[match.start():] for match in reversed(list(re.finditer(r'\{', final_text))))
    for candidate in candidates:
        try:
            trimmed = candidate.lstrip()
            payload, end = decoder.raw_decode(trimmed)
        except json.JSONDecodeError:
            continue
        if trimmed[end:].strip():
            continue
        if not isinstance(payload, dict):
            continue
        success = payload.get('success')
        reason = payload.get('reason', '')
        if isinstance(success, bool) and isinstance(reason, str):
            return success, reason
    raise ValueError('KIMI Judge output did not contain strict binary JSON')


def _trajectory_text(trajectory: dict[str, Any]) -> str:
    rows: list[str] = []
    for index, step in enumerate(trajectory.get('steps_detail') or [], start=1):
        rows.append(
            '\n'.join(
                [
                    f'### Step {index}',
                    f"Action: {step.get('action', '')}",
                    f"Observation/Output: {step.get('observation', '')}",
                    f"Error: {step.get('error') or step.get('parse_error') or '(none)'}",
                    f"Environment info: {json.dumps(step.get('valid_actions', []), ensure_ascii=False)}",
                ]
            )
        )
    return '\n\n'.join(rows) or '(The SubAgent produced no environment steps.)'


def build_kimi_judge_messages(
    *,
    child_goal: str,
    trajectory: dict[str, Any],
    final_environment_state: dict[str, Any],
) -> list[dict[str, str]]:
    """Use the binary TextWorld Judge semantics from Projects 85/172."""
    root_goal = str(trajectory.get('root_goal') or child_goal)
    system = (
        'You are a strict binary evaluator for a recursively delegated TextWorld Agent.\n'
        'Judge only whether this SubAgent completed its assigned goal and produced a '
        'real, useful contribution for its immediate Parent. Do not require it to '
        'complete the entire Root Task. Use the trajectory and environment outputs, '
        'not plausibility or invented claims. If the goal is not completed, the '
        'result is unusable, or the evidence is insufficient, mark it unsuccessful.\n'
        'Return exactly one JSON object and no other text:\n'
        '{"success": true or false, "reason": "brief evidence-based reason"}'
    )
    user = '\n\n'.join(
        [
            f'# Root Task\n{root_goal}',
            f'# Assigned SubAgent Goal\n{child_goal}',
            f'# SubAgent Trajectory\n{_trajectory_text(trajectory)}',
            '# Final Environment State\n'
            + json.dumps(final_environment_state, ensure_ascii=False, default=str),
            '# Final Decision\nReturn strict JSON with a boolean success field.',
        ]
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


class KimiTextWorldJudge:
    """Synchronous OpenAI-compatible KIMI binary Judge for selector calls."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str = 'kimi-k2.6',
        timeout: float = 1800.0,
        retries: int = 2,
        max_prompt_tokens: int = 10240,
        max_completion_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> None:
        if not endpoint:
            raise ValueError('KIMI Judge endpoint is required')
        if not api_key:
            raise ValueError('KIMI_API_KEY is required for oracle selection')
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_completion_tokens = int(max_completion_tokens)
        self.temperature = float(temperature)

    def judge(
        self,
        *,
        child_goal: str,
        trajectory: dict[str, Any],
        final_environment_state: dict[str, Any],
    ) -> KimiJudgeResult:
        messages = build_kimi_judge_messages(
            child_goal=child_goal,
            trajectory=trajectory,
            final_environment_state=final_environment_state,
        )
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': self.max_completion_tokens,
            'reasoning_effort': 'none',
            'chat_template_kwargs': {'enable_thinking': False},
            'extra_body': {'max_prompt_tokens': self.max_prompt_tokens},
        }
        request = urllib.request.Request(
            f'{self.endpoint}/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST',
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode('utf-8'))
                text = str(body['choices'][0]['message'].get('content') or '')
                success, reason = _parse_binary_success(text)
                return KimiJudgeResult(
                    success=success,
                    reason=reason,
                    raw_output=text,
                    response_id=body.get('id'),
                )
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(8.0, 2.0**attempt))
        return KimiJudgeResult(
            success=None,
            reason='',
            raw_output='',
            error=f'{type(last_error).__name__}: {last_error}' if last_error else 'KIMI request failed',
        )
