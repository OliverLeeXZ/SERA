from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

active_policy_judge: contextvars.ContextVar["TextWorldPolicyJudge | None"] = (
    contextvars.ContextVar("textworld_rao_policy_judge", default=None)
)


def set_active_policy_judge(
    judge: "TextWorldPolicyJudge | None",
) -> contextvars.Token:
    return active_policy_judge.set(judge)


def _json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            # Qwen/SGLang may append a generation terminator such as
            # ``<|im_end|>`` after an otherwise valid JSON response. Decode
            # the first complete JSON value and ignore only the trailing
            # generation marker/text.
            payload, _ = decoder.raw_decode(candidate.lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Policy Judge output did not contain a JSON object")


def parse_binary_success(text: str) -> tuple[int, str]:
    payload = _json_object(text)
    success = payload.get("success")
    if not isinstance(success, bool):
        raise ValueError("Policy Judge success must be a JSON boolean")
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("Policy Judge reason must be a string")
    return int(success), reason


def _task_goal(task: Any) -> str:
    if isinstance(task, dict):
        return str(task.get("goal", task.get("task_description", "")))
    return str(getattr(task, "goal", ""))


def _parent_goal(task: Any) -> str:
    if isinstance(task, dict):
        parents = task.get("parent_tasks") or []
        if parents:
            return _task_goal(parents[-1])
    parents = getattr(task, "parent_tasks", None) or []
    return _task_goal(parents[-1]) if parents else ""


def _trajectory_text(trajectory: dict[str, Any]) -> str:
    rows: list[str] = []
    for index, step in enumerate(trajectory.get("steps") or [], start=1):
        misc = step.get("misc", {})
        textworld = misc.get("textworld", {})
        rows.append(
            "\n".join(
                [
                    f"### Step {index}",
                    f"Action: {step.get('action', step.get('code', ''))}",
                    f"Observation/Output: {step.get('output', step.get('observation', ''))}",
                    f"Error: {step.get('error') or '(none)'}",
                    f"Environment info: {json.dumps(textworld.get('infos', {}), ensure_ascii=False)}",
                ]
            )
        )
    return "\n\n".join(rows) or "(The SubAgent produced no environment steps.)"


def build_judge_messages(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    task = trajectory.get("task", {})
    root_goal = _task_goal(task)
    parent_goal = _parent_goal(task) or root_goal
    child_goal = root_goal
    system = (
        "You are a strict binary evaluator for a recursively delegated TextWorld Agent.\n"
        "Judge only whether this SubAgent completed its assigned goal and produced a "
        "real, useful contribution for its immediate Parent. Do not require it to "
        "complete the entire Root Task. Use the trajectory and environment outputs, "
        "not plausibility or invented claims. If the goal is not completed, the "
        "result is unusable, or the evidence is insufficient, mark it unsuccessful.\n"
        "Return exactly one JSON object and no other text:\n"
        '{"success": true or false, "reason": "brief evidence-based reason"}'
    )
    user = "\n\n".join(
        [
            f"# Root Task\n{root_goal}",
            f"# Immediate Parent Goal\n{parent_goal}",
            f"# Assigned SubAgent Goal\n{child_goal}",
            f"# SubAgent Trajectory\n{_trajectory_text(trajectory)}",
            "# Final Decision\nReturn strict JSON with a boolean success field.",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _direct_children(collection: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    children: dict[str, list[dict[str, Any]]] = {}
    for trajectory in collection.get("trajectories", {}).values():
        parent_info = trajectory.get("parent_info") or {}
        parent_id = parent_info.get("id")
        if parent_id:
            children.setdefault(str(parent_id), []).append(trajectory)
    return children


def mask_failed_judge_trajectories(
    collection: dict[str, Any],
) -> dict[str, Any]:
    """Keep the tree while removing failed Judge nodes from PPO token data.

    The AReaL data extractor does not interpret ``skip_gradient_update`` by
    itself. Keeping the node with an empty ``steps`` list preserves parent
    links and depth labels for descendants while making only this node produce
    no optimizer sequence.
    """

    trajectories = collection.get("trajectories", {})
    masked: dict[str, Any] = {}
    changed = False
    for trajectory_id, trajectory in trajectories.items():
        if trajectory.get("misc", {}).get("skip_gradient_update"):
            replacement = dict(trajectory)
            replacement["steps"] = []
            masked[trajectory_id] = replacement
            changed = True
        else:
            masked[trajectory_id] = trajectory
    if not changed:
        return collection
    return {**collection, "trajectories": masked}


@dataclass
class _JudgeRecord:
    text: str = ""
    completion_id: str | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and bool(self.text.strip())


class _ActivePolicyJudgeClient:
    """Independent active-policy client with thinking disabled for JSON output."""

    def __init__(
        self,
        *,
        proxy_server: Any,
        model_name: str,
        max_concurrency: int,
        max_prompt_tokens: int,
        request_timeout_seconds: float,
    ) -> None:
        from platoon.train.areal.proxy import ArealProxySession

        self.proxy_server = proxy_server
        self.model_name = model_name
        self.max_prompt_tokens = max_prompt_tokens
        self.request_timeout_seconds = request_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._session_type = ArealProxySession

    async def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_completion_tokens: int,
    ) -> list[_JudgeRecord]:
        if not messages_batch:
            return []
        from openai import AsyncOpenAI

        async with self._session_type(
            base_url=f"{self.proxy_server.public_addr}/v1"
        ) as session:
            client = AsyncOpenAI(
                api_key="None",
                base_url=session.session_base_url,
                timeout=self.request_timeout_seconds,
            )

            async def generate(messages: list[dict[str, str]]) -> _JudgeRecord:
                try:
                    async with self._semaphore:
                        response = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                temperature=temperature,
                                max_completion_tokens=max_completion_tokens,
                                extra_body={
                                    "max_prompt_tokens": self.max_prompt_tokens,
                                    "chat_template_kwargs": {
                                        "enable_thinking": False,
                                    },
                                },
                            ),
                            timeout=self.request_timeout_seconds,
                        )
                    return _JudgeRecord(
                        text=response.choices[0].message.content or "",
                        completion_id=response.id,
                    )
                except Exception as exc:
                    return _JudgeRecord(error=f"{type(exc).__name__}: {exc}")

            try:
                return await asyncio.gather(
                    *(generate(messages) for messages in messages_batch)
                )
            finally:
                await client.close()


class TextWorldPolicyJudge:
    """Use the active Policy snapshot as a strict 0/1 SubAgent Judge."""

    def __init__(
        self,
        *,
        proxy_server: Any,
        model_name: str,
        max_concurrency: int = 32,
        max_prompt_tokens: int = 10240,
        max_completion_tokens: int = 1024,
        temperature: float = 1.0,
        request_timeout_seconds: float = 1800.0,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.client = _ActivePolicyJudgeClient(
            proxy_server=proxy_server,
            model_name=model_name,
            max_concurrency=max_concurrency,
            max_prompt_tokens=max_prompt_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.request_timeout_seconds = request_timeout_seconds

    async def judge_collection(
        self,
        collection: dict[str, Any],
        *,
        task_id: str,
        collection_id: str,
    ) -> dict[str, float]:
        trajectories = collection.get("trajectories", {})
        children_by_parent = _direct_children(collection)
        candidates = [
            trajectory
            for trajectory in trajectories.values()
            if trajectory.get("parent_info")
        ]
        messages = [build_judge_messages(trajectory) for trajectory in candidates]
        records = await self.client.generate_batch(
            messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )
        metrics = {
            "policy_judge/candidates": float(len(candidates)),
            "policy_judge/valid": 0.0,
            "policy_judge/failed": 0.0,
        }
        artifact_records: list[dict[str, Any]] = []
        for trajectory, record in zip(candidates, records, strict=True):
            success = 0
            reason = ""
            error: str | None = None
            if not record.valid:
                error = record.error or "Policy Judge request failed"
            else:
                try:
                    success, reason = parse_binary_success(record.text)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            misc = trajectory.setdefault("misc", {})
            misc["policy_judge_success"] = success
            misc["policy_judge"] = {
                "success": success,
                "reason": reason,
                "error": error,
                "request_id": record.completion_id,
            }
            if error is not None:
                # Only this SubAgent is removed by get_train_data_for_trajectory_collection.
                misc["skip_gradient_update"] = {
                    "reason": "policy_judge_failure",
                    "error": error,
                }
                metrics["policy_judge/failed"] += 1.0
            else:
                metrics["policy_judge/valid"] += 1.0
            artifact_records.append(
                {
                    "task_id": task_id,
                    "collection_id": collection_id,
                    "trajectory_id": trajectory.get("id"),
                    "parent_trajectory_id": (trajectory.get("parent_info") or {}).get("id"),
                    "success": success,
                    "reason": reason,
                    "error": error,
                    "judge_text": record.text,
                    "messages": build_judge_messages(trajectory),
                }
            )

        for trajectory in trajectories.values():
            direct = children_by_parent.get(str(trajectory.get("id")), [])
            misc = trajectory.setdefault("misc", {})
            misc["rao_direct_subagent_launched"] = len(direct)
            misc["rao_direct_subagent_succeeded"] = float(
                sum(float(child.get("misc", {}).get("policy_judge_success", 0)) for child in direct)
            )

        if self.artifact_dir and artifact_records:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            path = self.artifact_dir / f"{task_id}__{collection_id}.json"
            path.write_text(
                json.dumps(
                    {"metrics": metrics, "records": artifact_records},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return metrics

    async def close(self) -> None:
        return None


class _KimiClient:
    """OpenAI-compatible external KIMI client with thinking disabled."""

    def __init__(
        self,
        *,
        model_name: str,
        endpoint: str,
        api_key: str,
        max_concurrency: int,
        max_prompt_tokens: int,
        request_timeout_seconds: float,
    ) -> None:
        from openai import AsyncOpenAI

        self.model_name = model_name
        self.max_prompt_tokens = max_prompt_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            timeout=request_timeout_seconds,
        )

    async def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_completion_tokens: int,
    ) -> list[_JudgeRecord]:
        async def generate(messages: list[dict[str, str]]) -> _JudgeRecord:
            try:
                async with self._semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                        extra_body={
                            "max_prompt_tokens": self.max_prompt_tokens,
                            "reasoning_effort": "none",
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                return _JudgeRecord(
                    text=response.choices[0].message.content or "",
                    completion_id=response.id,
                )
            except Exception as exc:
                return _JudgeRecord(error=f"{type(exc).__name__}: {exc}")

        return await asyncio.gather(
            *(generate(messages) for messages in messages_batch)
        )

    async def close(self) -> None:
        await self.client.close()


class TextWorldKimiJudge(TextWorldPolicyJudge):
    """Strict binary SubAgent Judge backed by the external KIMI endpoint."""

    def __init__(
        self,
        *,
        model_name: str = "kimi-k2.6",
        endpoint: str = "",
        api_key: str | None = None,
        max_concurrency: int = 16,
        max_prompt_tokens: int = 10240,
        max_completion_tokens: int = 1024,
        temperature: float = 1.0,
        request_timeout_seconds: float = 1800.0,
        artifact_dir: str | Path | None = None,
    ) -> None:
        key = api_key or os.getenv("KIMI_API_KEY")
        if not key:
            raise ValueError("KIMI_API_KEY is required for TextWorldKimiJudge")
        self.client = _KimiClient(
            model_name=model_name,
            endpoint=endpoint,
            api_key=key,
            max_concurrency=max_concurrency,
            max_prompt_tokens=max_prompt_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.request_timeout_seconds = request_timeout_seconds

    async def close(self) -> None:
        await self.client.close()
