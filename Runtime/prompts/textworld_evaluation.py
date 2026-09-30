from __future__ import annotations

from typing import Any, Callable
from Runtime.environments.textworld.manifest import TaskSpec
from Runtime.agents.textworld_protocol import (
    ACTION_PATTERN, DELEGATE_PATTERN, FINISH_PATTERN, INVENTORY_ACTIONS,
    Turn, Delegation, Decision, extract_action, is_inventory_action,
    _parse_delegation_payload, extract_decision, _estimate_tokens,
)


class TextWorldPromptBuilder:
    """TextWorld equivalent of TextCraft's sequence-extension prompt builder."""

    def __init__(
        self,
        max_prompt_tokens: int = 10240,
        *,
        max_depth: int = 3,
        max_subagent_steps: int = 20,
        allow_subagents: bool = True,
        preserve_history: bool = False,
        token_counter: Callable[[list[dict[str, str]]], int] | None = None,
    ):
        self.max_prompt_tokens = max_prompt_tokens
        self.max_depth = max_depth
        self.max_subagent_steps = max_subagent_steps
        self.allow_subagents = allow_subagents
        self.preserve_history = preserve_history
        self.token_counter = token_counter

    def _count_tokens(self, messages: list[dict[str, str]]) -> int:
        return self.token_counter(messages) if self.token_counter is not None else _estimate_tokens(messages)

    def build_messages(
        self,
        task: TaskSpec,
        initial_observation: str,
        initial_infos: dict[str, Any],
        history: list[Turn],
        *,
        agent_id: str = "root",
        depth: int = 0,
        agent_max_steps: int = 20,
    ) -> tuple[list[dict[str, str]], int, bool]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._system_prompt(task, agent_id, depth),
            },
            {
                "role": "user",
                "content": self._initial_user(
                    task,
                    initial_observation,
                    initial_infos,
                    agent_max_steps=agent_max_steps,
                ),
            },
        ]
        for index, turn in enumerate(history):
            messages.append({"role": "assistant", "content": turn.raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": self._observation_user(
                        index,
                        turn,
                        agent_max_steps=agent_max_steps,
                    ),
                }
            )

        if self.preserve_history:
            count = self._count_tokens(messages)
            return messages, count, count > self.max_prompt_tokens

        truncated = False
        while len(messages) > 2 and self._count_tokens(messages) > self.max_prompt_tokens:
            # Preserve the fixed system/task prefix and the latest environment state.
            del messages[2:4]
            truncated = True
        return messages, self._count_tokens(messages), truncated

    def _system_prompt(self, task: TaskSpec, agent_id: str, depth: int) -> str:
        can_delegate = self.allow_subagents and depth < self.max_depth
        delegation = self._delegation_strategy(depth) if can_delegate else ""
        collaboration = (
            "Before selecting the next environment action, briefly consider whether the current goal contains multiple subgoals that can be solved independently or in parallel.\n"
            "If decomposition appears useful, identify the concrete subgoals and decide whether delegating one or more of them to identical Agents would improve progress under the remaining step budget. A delegated goal should be specific, self-contained, and useful to the parent task.\n"
            "BUDGET-AWARE COLLABORATION:\n"
            "The current Agent's remaining step budget counts only actions taken by the current Agent. A delegated SubAgent executes with its own independent rollout budget. SubAgent actions do not consume the current Agent's remaining step budget.\n"
            "Delegation can reduce the parent's sequential workload when independent subgoals are executed in parallel.\n"
            if self.allow_subagents else ""
        )
        agent_context = f"Your current delegation depth is {depth} (root is depth 0).\n" if self.allow_subagents else ""
        recursive_context = (
            "If you are a SubAgent, the User Prompt contains the goal assigned by your Parent. "
            "All Agents use the same goal-based recursive interface; a SubAgent goal may be more "
            "detailed than its Parent's goal so that the required work is unambiguous.\n"
            if self.allow_subagents else ""
        )
        response_instruction = (
            "Return exactly one block from the Action Space in the User Prompt and nothing else.\n"
            if self.allow_subagents else
            "Return exactly one block from the Action Space in the User Prompt and nothing else. The next response must contain your next action.\n"
        )
        return (
            "You are an agent in a TextWorldExpress interactive task.\n"
            f"You are Agent {agent_id}.\n"
            + agent_context
            + "Your goal is to complete the task by issuing textual environment commands.\n"
            + "You have access to the current observation and Valid Actions list.\n"
            + collaboration
            + "The shared inventory is not shown automatically. Use the inventory action in the "
            "Action Space whenever you need to inspect its current contents.\n"
            "Use only an action that is present in the current Valid Actions list when possible.\n"
            "Do not invent state, task completion, or hidden objects.\n"
            "Each Agent has its own action/observation history and logical location, while all "
            "Agents interact with the same underlying world and its resources.\n"
            + recursive_context
            + "\n<TIPS>\n"
            "INTERACTION STRATEGY:\n"
            "- Inspect the current observation and Valid Actions before acting.\n"
            "- Track prerequisites: move, open, take, and preparation actions may need to happen "
            "before a later action is possible.\n"
            "- Always verify the world state before claiming that an object or route is unavailable.\n"
            "\n"
            + delegation
            + "</TIPS>\n\n"
            + response_instruction
            + "The task is evaluated by the environment, not by a language-model judge."
        )

    def _delegation_strategy(self, depth: int) -> str:
        return (
            "DELEGATION STRATEGY:\n"
            "- It is **highly recommended** to delegate a coherent independent subtask when "
            "another Agent can make useful progress without waiting for your next action.\n"
            "- Break complex tasks into INDEPENDENT subtasks that can be solved separately.\n"
            "- For tasks that are sufficiently complex, it is recommended to recursively delegate; "
            "i.e., SubAgents can further delegate to other SubAgents.\n"
            "- Delegate one group of related objectives at a time, not everything at once.\n"
            "- Use the current task state, observed locations, prerequisites, and remaining action "
            "budget to estimate whether a delegated subtask is feasible.\n"
            "- Independent subtasks can be delegated in parallel if they do not depend on each other; "
            "dependent subtasks should be handled sequentially.\n"
            "- Reserve budget for yourself to perform final assembly and verification after subtasks complete.\n"
            "- Delegated Agents interact with the same world and shared resources; their state changes "
            "and results become available to you, but their action/observation histories and logical "
            "locations remain separate.\n"
            "- A subtask that needs the result of another subtask is not independent. Do not launch "
            "duplicate subtasks that compete for the same unique object or irreversible transition.\n"
            "\n"
            "SUBAGENT ACTION:\n"
            "A delegation is accepted only when it contains one specific, sufficiently detailed goal. "
            "The goal should state what the new SubAgent needs to accomplish in the environment; "
            "do not write a vague label such as 'help with the task'. Do not explain a prescribed "
            "format or hidden execution plan in the goal: describe the work clearly and let the "
            "SubAgent decide how to execute it.\n"
            'To launch one SubAgent, output exactly: <delegate>{"goal":"detailed task description"}</delegate>\n'
            "To launch several independent SubAgents in parallel, output one block with a list:\n"
            '<delegate>{"delegations":[{"goal":"inspect the pantry and report useful objects and routes"},'
            '{"goal":"inspect the fridge and report available ingredients"}]}</delegate>\n'
            "If you cannot write a clear, concrete goal, perform the work yourself instead of delegating.\n"
            f"The maximum delegation depth is {self.max_depth}; the current depth is {depth}.\n"
        )

    @staticmethod
    def _action_space(can_delegate: bool) -> str:
        if can_delegate:
            return (
                "AVAILABLE ACTIONS:\n"
                "1. <action>your environment command</action>\n"
                "   Execute one command from the current Valid Actions list.\n"
                "2. <action>inventory</action>\n"
                "   Inspect the current shared inventory without advancing the world state.\n"
                "3. <delegate>{\"goal\":\"detailed task description\"}</delegate>\n"
                "   The goal must make the requested work unambiguous; max_steps is optional.\n"
                "   Delegate one coherent subtask, or use the delegations list for independent parallel subtasks.\n"
                "4. <finish>brief result for the Parent</finish>\n"
                "   Use this only when you are a SubAgent and have completed or investigated the delegated task.\n"
                "Return exactly one block and nothing else."
            )
        return (
            "AVAILABLE ACTIONS:\n"
            "1. <action>your environment command</action>\n"
            "   Execute one command from the current Valid Actions list.\n"
            "2. <action>inventory</action>\n"
            "   Inspect the current shared inventory without advancing the world state.\n"
            "Return exactly one action block and nothing else."
        )

    def _initial_user(
        self,
        task: TaskSpec,
        observation: str,
        infos: dict[str, Any],
        *,
        agent_max_steps: int,
    ) -> str:
        return (
            "# Task\n\n"
            f"{task.task_description}\n\n"
            f"Budget: You have a total budget of {agent_max_steps} steps to complete this task.\n\n"
            "# Initial Observation\n\n"
            f"{observation}\n\n"
            "# Valid Actions\n\n"
            f"{self._valid_actions(infos)}\n\n"
            "# Action Space\n\n"
            f"{self._action_space(self.allow_subagents)}\n\n"
            + (
                "# Delegation\n\n"
                "If another Agent can independently handle part of this task, use the detailed "
                "goal-based <delegate> JSON protocol from the System Prompt.\n\n"
                if self.allow_subagents
                else ""
            )
            + ("Now provide the first action or delegation." if self.allow_subagents else "Now provide the first action.")
        )

    def _observation_user(
        self,
        index: int,
        turn: Turn,
        *,
        agent_max_steps: int,
    ) -> str:
        infos = turn.infos
        steps_used = min(index + 1, agent_max_steps)
        remaining_steps = max(agent_max_steps - steps_used, 0)
        return (
            f"[Step {index} Observation]\n"
            f"{turn.observation or '(No observation)'}\n\n"
            f"Budget used by current Agent: {steps_used}/{agent_max_steps} steps.\n"
            f"Current Agent remaining budget: {remaining_steps} steps.\n\n"
            f"Valid Actions: {self._valid_actions(infos)}\n"
            f"Score: {infos.get('score', 0.0)}\n"
            f"Task Success: {bool(infos.get('tasksuccess', False))}\n"
            f"Task Failure: {bool(infos.get('taskfailure', False))}\n"
            f"Done: {bool(infos.get('done', False))}\n\n"
            + (
                "A SubAgent result, if present, is included in the observation above. "
                "Use it to decide the next action or delegation.\n"
                if self.allow_subagents
                else ""
            )
            + ("Provide your next action or delegation." if self.allow_subagents else "Provide your next action.")
        )

    @staticmethod
    def _valid_actions(infos: dict[str, Any]) -> str:
        actions = infos.get("validActions", [])
        if not isinstance(actions, list):
            return str(actions)
        return "\n".join(f"- {action}" for action in actions) or "(none)"

    @staticmethod
    def format_inventory(infos: dict[str, Any]) -> str:
        inventory = infos.get("inventory", "(unavailable)")
        return str(inventory).strip() or "(empty)"
