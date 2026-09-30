from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentRuntime:
    agent: Any
    env: Any


class ThreeStageEnvironmentAdapter(ABC):
    name: str

    @abstractmethod
    def canonical_subtask_key(self, trajectory: dict[str, Any]) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def serialize_agent_trajectory(
        self, trajectory: dict[str, Any], mode: str = "judge"
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def final_state_from_trajectory(
        self, trajectory: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def fork_isolated(
        self, parent_agent: Any, parent_env: Any, subtask: Any
    ) -> AgentRuntime:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, env: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def commit(self, parent_env: Any, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_success(self, trajectory: Any) -> bool:
        raise NotImplementedError
