from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning_content: str
    usage: dict[str, Any]
    model: str | None
    response_id: str | None


class VLLMChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 1800.0,
        retries: int = 2,
        enable_reasoning: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.enable_reasoning = enable_reasoning

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_completion_tokens: int,
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_completion_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_reasoning},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = body["choices"][0]
                message = choice.get("message", {})
                content = message.get("content") or ""
                reasoning_content = message.get("reasoning_content") or ""
                return Completion(
                    content=str(content),
                    reasoning_content=str(reasoning_content),
                    usage=dict(body.get("usage") or {}),
                    model=body.get("model"),
                    response_id=body.get("id"),
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(8.0, 2.0**attempt))
        raise RuntimeError(f"chat completion failed after {self.retries + 1} attempts: {last_error}")
