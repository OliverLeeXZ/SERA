from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .config import TeacherConfig
from Runtime.rubric.scoring import parse_teacher_scores


class TeacherClient:
    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        if not config.endpoint:
            raise ValueError("Teacher endpoint is required for Stage 3")
        if not config.api_key:
            raise ValueError(
                f"Teacher API key is missing from environment variable {config.api_key_env}"
            )
        self.client = AsyncOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )
        self.cache_dir = Path(config.cache_dir)

    def _cache_path(self, messages: list[dict[str, str]]) -> Path:
        digest = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    async def score(
        self,
        messages: list[dict[str, str]],
        expected_scores: int,
    ) -> tuple[list[float], dict[str, Any], str]:
        cache_path = self._cache_path(messages)
        if self.config.cache and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            scores, payload = parse_teacher_scores(
                json.dumps(payload, ensure_ascii=False), expected_scores
            )
            if len(scores) == expected_scores:
                return scores, payload, cache_path.stem

        last_error: BaseException | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=0.0,
                    max_completion_tokens=2048,
                    extra_body=self.config.extra_body,
                )
                text = response.choices[0].message.content or ""
                scores, payload = parse_teacher_scores(text, expected_scores)
                if self.config.cache:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    temp = cache_path.with_suffix(".tmp")
                    temp.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    temp.replace(cache_path)
                return scores, payload, cache_path.stem
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"Teacher failed after {self.config.max_retries} attempts: {last_error}"
        )

    async def close(self) -> None:
        await self.client.close()
