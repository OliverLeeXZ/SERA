from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .prompt import build_judge_messages, parse_binary_success
from Runtime.clients.external_model import external_request_error


@dataclass(frozen=True)
class KimiJudgeResult:
    success: bool | None = None
    reason: str = ""
    text: str = ""
    error: str | None = None
    completion_id: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.success is not None


class KimiJudgeClient:
    """OpenAI-compatible external KIMI client with thinking disabled."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str = "kimi-k2.6",
        max_concurrency: int = 16,
        max_prompt_tokens: int = 10240,
        max_completion_tokens: int = 1024,
        temperature: float = 1.0,
        timeout_seconds: float = 1800.0,
        max_retries: int = 2,
    ) -> None:
        if not endpoint:
            raise ValueError("KIMI judge endpoint is required")
        if not api_key:
            raise ValueError("KIMI_API_KEY is required for KIMI judge")
        from openai import AsyncOpenAI

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_completion_tokens = int(max_completion_tokens)
        self.temperature = float(temperature)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.endpoint,
            timeout=self.timeout_seconds,
        )

    async def judge_batch(self, trajectories: list[dict[str, Any]]) -> list[KimiJudgeResult]:
        async def judge_one(trajectory: dict[str, Any]) -> KimiJudgeResult:
            messages = build_judge_messages(trajectory)
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    async with self._semaphore:
                        response = await asyncio.wait_for(
                            self._client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=self.temperature,
                                max_completion_tokens=self.max_completion_tokens,
                                extra_body={
                                    "max_prompt_tokens": self.max_prompt_tokens,
                                    "reasoning_effort": "none",
                                    "chat_template_kwargs": {"enable_thinking": False},
                                },
                            ),
                            timeout=self.timeout_seconds,
                        )
                    text = response.choices[0].message.content or ""
                    success, reason = parse_binary_success(text)
                    return KimiJudgeResult(
                        success=success,
                        reason=reason,
                        text=text,
                        completion_id=response.id,
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2.0 ** attempt, 8.0))
            return KimiJudgeResult(
                error=external_request_error(last_error)
            )

        return await asyncio.gather(*(judge_one(item) for item in trajectories))

    async def close(self) -> None:
        await self._client.close()
