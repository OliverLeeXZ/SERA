from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

from Runtime.rubric.scoring import parse_policy_score, parse_rubric
from Runtime.clients.external_model import external_request_error

from .config import RubricSubagentRewardConfig


@dataclass
class ParsedGeneration:
    text: str = ""
    parsed: Any = None
    request_id: str | None = None
    cache_hit: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.parsed is not None


class KimiRubricClient:
    """KIMI client using the exact 3Stage rubric and score parsers."""

    def __init__(self, config: RubricSubagentRewardConfig) -> None:
        config.validate()
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )
        self.cache_dir = Path(config.cache_dir)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    def _cache_path(
        self,
        operation: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
    ) -> Path:
        payload = {
            "operation": operation,
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "extra_body": self.config.extra_body,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / operation / f"{digest}.json"

    async def generate_rubric(
        self, messages: list[dict[str, str]]
    ) -> ParsedGeneration:
        return await self._generate(
            operation="rubric",
            messages=messages,
            temperature=self.config.rubric_temperature,
            max_completion_tokens=self.config.max_rubric_tokens,
            parser=lambda text: parse_rubric(
                text, self.config.min_rubric_criteria
            ),
        )

    async def score_trajectory(
        self, messages: list[dict[str, str]]
    ) -> ParsedGeneration:
        return await self._generate(
            operation="score",
            messages=messages,
            temperature=self.config.scoring_temperature,
            max_completion_tokens=self.config.max_scoring_tokens,
            parser=parse_policy_score,
        )

    async def _generate(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
        parser: Callable[[str], Any],
    ) -> ParsedGeneration:
        cache_path = self._cache_path(
            operation, messages, temperature, max_completion_tokens
        )
        if self.config.cache and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                parsed = parser(str(payload["text"]))
                return ParsedGeneration(
                    text=str(payload["text"]),
                    parsed=parsed,
                    request_id=str(payload.get("request_id") or cache_path.stem),
                    cache_hit=True,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)

        last_error: BaseException | None = None
        for attempt in range(self.config.max_retries):
            try:
                async with self._semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                        extra_body=self.config.extra_body,
                    )
                text = response.choices[0].message.content or ""
                parsed = parser(text)
                record = ParsedGeneration(
                    text=text,
                    parsed=parsed,
                    request_id=response.id,
                )
                if self.config.cache:
                    self._write_cache(cache_path, record)
                return record
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.config.max_retries:
                    await asyncio.sleep(min(2**attempt, 30))
        return ParsedGeneration(
            error=(
                f"{operation} failed after {self.config.max_retries} attempts: "
                + external_request_error(last_error)
            )
        )

    @staticmethod
    def _write_cache(path: Path, record: ParsedGeneration) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "text": record.text,
                    "request_id": record.request_id,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    async def close(self) -> None:
        await self.client.close()


class PolicyRubricClient:
    """Use the active AReaL policy snapshot as rubric writer and scorer."""

    def __init__(
        self,
        config: RubricSubagentRewardConfig,
        *,
        policy_client: Any | None = None,
        proxy_server: Any | None = None,
        model_name: str | None = None,
        max_prompt_tokens: int | None = None,
        default_completion_tokens: int = 512,
    ) -> None:
        config.validate()
        if config.provider != "policy":
            raise ValueError("PolicyRubricClient requires provider=policy")
        if policy_client is None:
            if proxy_server is None or not model_name:
                raise ValueError(
                    "proxy_server and model_name are required for policy rewards"
                )
            from three_stage_train.policy_client import ArealPolicyClient

            policy_client = ArealPolicyClient(
                proxy_server=proxy_server,
                model_name=model_name,
                max_concurrency=config.max_concurrency,
                max_prompt_tokens=max_prompt_tokens,
                default_completion_tokens=default_completion_tokens,
                request_timeout_seconds=config.timeout_seconds,
            )
        self.config = config
        self.policy_client = policy_client

    async def generate_rubric(
        self, messages: list[dict[str, str]]
    ) -> ParsedGeneration:
        return await self._generate(
            messages,
            temperature=self.config.rubric_temperature,
            max_completion_tokens=self.config.max_rubric_tokens,
            parser=lambda text: parse_rubric(
                text, self.config.min_rubric_criteria
            ),
        )

    async def score_trajectory(
        self, messages: list[dict[str, str]]
    ) -> ParsedGeneration:
        return await self._generate(
            messages,
            temperature=self.config.scoring_temperature,
            max_completion_tokens=self.config.max_scoring_tokens,
            parser=parse_policy_score,
        )

    async def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_completion_tokens: int,
        parser: Callable[[str], Any],
    ) -> ParsedGeneration:
        records = await self.policy_client.generate_batch(
            [messages],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        record = records[0]
        if not record.valid:
            return ParsedGeneration(error=record.error or "policy request failed")
        try:
            parsed = parser(record.text)
        except Exception as exc:
            return ParsedGeneration(
                text=record.text,
                request_id=record.completion_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ParsedGeneration(
            text=record.text,
            parsed=parsed,
            request_id=record.completion_id,
        )

    async def close(self) -> None:
        return None
