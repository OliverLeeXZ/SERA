from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from platoon.train.areal.proxy import ArealProxySession
from .message import message_text


@dataclass
class PolicyGeneration:
    messages: list[dict[str, str]]
    text: str = ""
    completion_id: str | None = None
    completion_entry: Any | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.completion_entry is not None


class ArealPolicyClient:
    """Issues independent chat contexts through the active AReaL rollout proxy."""

    def __init__(
        self,
        *,
        proxy_server: Any,
        model_name: str,
        max_concurrency: int,
        max_prompt_tokens: int | None = None,
        default_completion_tokens: int = 512,
        request_extra_body: dict[str, Any] | None = None,
        request_timeout_seconds: float = 300.0,
    ) -> None:
        self.proxy_server = proxy_server
        self.model_name = model_name
        self.proxy_url = f"{proxy_server.public_addr}/v1"
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.max_prompt_tokens = max_prompt_tokens
        self.default_completion_tokens = default_completion_tokens
        self.request_extra_body = dict(request_extra_body or {})
        thinking_override = os.getenv("PLATOON_QWEN_ENABLE_THINKING", "").strip().lower()
        if thinking_override:
            chat_template_kwargs = dict(
                self.request_extra_body.get("chat_template_kwargs") or {}
            )
            chat_template_kwargs["enable_thinking"] = thinking_override not in {
                "0",
                "false",
                "no",
                "off",
            }
            self.request_extra_body["chat_template_kwargs"] = chat_template_kwargs
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.request_timeout_seconds = request_timeout_seconds

    async def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_completion_tokens: int,
    ) -> list[PolicyGeneration]:
        if not messages_batch:
            return []
        async with ArealProxySession(base_url=self.proxy_url) as session:
            client = AsyncOpenAI(
                api_key="None",
                base_url=session.session_base_url,
                timeout=self.request_timeout_seconds,
            )

            async def generate(messages: list[dict[str, str]]) -> PolicyGeneration:
                record = PolicyGeneration(messages=deepcopy(messages))
                try:
                    extra_body = dict(self.request_extra_body)
                    if self.max_prompt_tokens is not None:
                        extra_body["max_prompt_tokens"] = max(
                            1,
                            self.max_prompt_tokens
                            + self.default_completion_tokens
                            - max_completion_tokens,
                        )
                    if not extra_body:
                        extra_body = None
                    async with self._semaphore:
                        response = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                temperature=temperature,
                                max_completion_tokens=max_completion_tokens,
                                extra_body=extra_body,
                            ),
                            timeout=self.request_timeout_seconds,
                        )
                    record.completion_id = response.id
                    record.text = message_text(response.choices[0].message)
                except Exception as exc:
                    record.error = f"{type(exc).__name__}: {exc}"
                return record

            try:
                records = await asyncio.gather(
                    *(generate(messages) for messages in messages_batch)
                )
                cache = self.proxy_server.session_cache[session.session_id].completions
                for record in records:
                    if record.completion_id is not None:
                        record.completion_entry = cache.get(record.completion_id)
                        if record.completion_entry is None and record.error is None:
                            record.error = (
                                "Completion was not found in the AReaL session cache"
                            )
                return records
            finally:
                # A timeout/cancellation used to bypass this close and leave one
                # HTTP session behind per policy-rubric request.
                await client.close()
