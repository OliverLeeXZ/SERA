from __future__ import annotations

import os
import uuid
from typing import Any

from platoon.config_defs import InferenceParams
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActAction
from platoon.episode.context import finish_message
from platoon.utils.llm_client import LiteLLMClient
from textworld_continuous_policy.manifest import TaskSpec
from textworld_continuous_policy.prompts import Decision, TextWorldPromptBuilder, Turn, extract_decision


def _task_spec(task: Task) -> TaskSpec:
    misc = task.misc
    return TaskSpec(
        task_id=str(task.id),
        game=str(misc.get("textworld_game", "unknown")),
        fold=str(misc.get("textworld_fold", "train")),
        difficulty=str(misc.get("textworld_difficulty", "medium")),
        seed=int(misc.get("textworld_seed", 0)),
        game_params=str(misc.get("textworld_game_params", "")),
        task_description=str(task.goal or ""),
        generation_properties=dict(misc.get("generation_properties", {})),
    )


class TextWorldAgent:
    def __init__(
        self,
        *,
        llm_client: LiteLLMClient,
        inference_params: InferenceParams,
        prompt_builder: TextWorldPromptBuilder,
    ) -> None:
        self.llm_client = llm_client
        self.inference_params = inference_params
        self.prompt_builder = prompt_builder

    async def act(self, obs: Any) -> CodeActAction:
        turns: list[Turn] = []
        for step in obs.history:
            turn_misc = step.misc.get("textworld", {})
            turns.append(
                Turn(
                    raw_response=step.raw_response,
                    action=step.action,
                    parse_ok=not bool(step.error),
                    observation=step.observation,
                    infos=dict(turn_misc.get("infos", {})),
                )
            )
        messages, estimated_tokens, truncated = self.prompt_builder.build_messages(
            _task_spec(obs.task),
            obs.initial_observation,
            dict(obs.initial_infos),
            turns,
            agent_id=str(getattr(obs.task, "id", "agent")),
            depth=self._depth(obs.task),
        )
        request_id = str(uuid.uuid4())
        enable_thinking = os.getenv("TEXTWORLD_ENABLE_THINKING", "true").lower() in {
            "1", "true", "yes", "on"
        }
        extra_body = {
            "chat_template_kwargs": {
                "platoon_max_prompt_tokens": self.inference_params.max_prompt_tokens,
                "enable_thinking": enable_thinking,
            }
        }
        # AReaL's patched OpenAI adapter interprets `max_tokens` as the total
        # prompt-plus-completion budget when `max_completion_tokens` is also
        # present. Passing only 512 here makes every prompt longer than 512
        # tokens fail before the local model is called.
        request_max_tokens = self.inference_params.max_completion_tokens
        if self.inference_params.max_prompt_tokens is not None:
            request_max_tokens = (
                self.inference_params.max_prompt_tokens
                + self.inference_params.max_completion_tokens
            )
        response = await self.llm_client.async_chat_completion(
            messages,
            temperature=self.inference_params.temperature or 0.0,
            max_tokens=request_max_tokens,
            max_completion_tokens=self.inference_params.max_completion_tokens,
            extra_body=extra_body,
            timeout=1800,
        )
        raw_response = response.choices[0].message.content or ""
        decision: Decision = extract_decision(raw_response)
        action = CodeActAction(
            action=raw_response,
            parsed_code=decision.action,
            parsed_thought=None,
            misc={
                "raw_response": raw_response,
                "decision": decision.kind,
                "parse_ok": decision.parse_ok,
                "parse_error": decision.error,
                "delegations": [
                    {"goal": item.goal, "max_steps": item.max_steps}
                    for item in decision.delegations
                ],
                "finish_message": decision.finish_message,
                "estimated_prompt_tokens": estimated_tokens,
                "prompt_truncated": truncated,
                "request_id": request_id,
                "usage": response.usage.to_dict() if response.usage else {},
                "model": response.model,
                "completion_id": response.id,
            },
        )
        action._request_messages = [dict(message) for message in messages]
        if decision.kind == "finish" and decision.finish_message:
            finish_message.set(decision.finish_message)
        return action

    @staticmethod
    def _depth(task: Task) -> int:
        parents = getattr(task, "parent_tasks", None)
        return len(parents) if isinstance(parents, list) else 0

    async def reset(self) -> None:
        return None

    async def fork(self, task: Task) -> "TextWorldAgent":
        return type(self)(
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            prompt_builder=self.prompt_builder,
        )

    async def close(self) -> None:
        await self.llm_client.aclose()
