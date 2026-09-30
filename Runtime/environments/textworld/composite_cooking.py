from __future__ import annotations

"""A deterministic, shared-world adapter for composite TextWorld-Sync tasks.

The official JVM CookingWorld has one recipe per game instance.  The fixed
benchmark instead describes several recipes in one task, so this adapter keeps
the same evaluator-facing interface while implementing the composite state in
Python.  Agent locations and histories are private; ingredients, prepared
items, meals, gates, and the global step budget are shared.
"""

import asyncio
import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from .composite_scorer import score_task


_ROOMS = ("kitchen", "pantry", "garden", "supermarket", "hallway", "dining room", "garage", "study", "basement", "attic", "courtyard")
_DIRECTIONS = {
    "north": "pantry",
    "south": "dining room",
    "east": "garden",
    "west": "supermarket",
}
_COOKING_ACTION = {
    "fried": "stove",
    "roasted": "oven",
    "grilled": "grill",
    "baked": "oven",
    "boiled": "stove",
}
_CUTTING_ACTIONS = {"chopped": "chop", "sliced": "slice", "diced": "dice", "minced": "mince"}


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _source_room(source: str) -> str:
    value = _normalise(source).split(".", 1)[0]
    aliases = {"kitchen": "kitchen", "pantry": "pantry", "garden": "garden", "supermarket": "supermarket"}
    return aliases.get(value, "kitchen")


def _prep_set(values: Any) -> set[str]:
    if isinstance(values, str):
        return {part.strip().lower() for part in values.split(",") if part.strip()}
    if isinstance(values, (list, tuple, set, frozenset)):
        return {str(part).strip().lower() for part in values if str(part).strip()}
    return set()


@dataclass
class _Event:
    sequence: int
    agent_id: str
    action: str
    accepted: bool
    reward: float
    done: bool
    observation: str
    before_location: str
    after_location: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "agent_id": self.agent_id,
            "action": self.action,
            "accepted": self.accepted,
            "reward": self.reward,
            "done": self.done,
            "observation": self.observation,
            "before_location": self.before_location,
            "after_location": self.after_location,
            "error": self.error,
        }


class CompositeAgentView:
    def __init__(self, coordinator: "CompositeCookingWorldCoordinator", agent_id: str):
        self.coordinator = coordinator
        self.agent_id = agent_id

    async def observe_async(self) -> tuple[str, dict[str, Any]]:
        return await self.coordinator.observe_agent(self.agent_id)

    async def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        return await self.coordinator.step(self.agent_id, action)


class CompositeCookingWorldCoordinator:
    """One shared composite CookingWorld with independent agent locations."""

    def __init__(self, task: Any, *, env_step_limit: int = 100):
        self.task = task
        self.env_step_limit = int(env_step_limit)
        if self.env_step_limit <= 0:
            raise ValueError("env_step_limit must be positive")
        self._lock = asyncio.Lock()
        self._events: list[_Event] = []
        self._agent_locations: dict[str, str] = {}
        self._last: tuple[str, dict[str, Any]] | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        properties = dict(getattr(self.task, "generation_properties", {}) or {})
        params = json.loads(getattr(self.task, "game_params", "{}") or "{}")
        self._properties = properties
        self._params = params
        self._dishes = copy.deepcopy(properties.get("dishes", []))
        self._family = str(properties.get("task_family", "multi_dish_fork_join"))
        self._rooms = tuple(_ROOMS[: max(4, min(len(_ROOMS), int(params.get("numLocations", 6))))])
        self._inventory_capacity = int(params.get("limitInventorySize", 0) or 0)
        parallelism = dict(properties.get("parallelism", {}) or {})
        mechanisms = dict(parallelism.get("mechanisms", {}) or {})
        self._tool_cooldown_steps = int(mechanisms.get("tool_cooldown_steps", 0) or 0)
        if self._tool_cooldown_steps < 0:
            raise ValueError("tool_cooldown_steps must be non-negative")
        self._tool_cooldowns: dict[str, int] = {}
        capacity = parallelism.get("resource_capacity") or {}
        if isinstance(capacity, dict) and capacity.get("inventory"):
            self._inventory_capacity = int(capacity["inventory"])
        self._gate_open = not bool(params.get("includeDoors", 0))
        dependencies = parallelism.get("dependencies") or []
        self._gated_dishes = {
            str(edge.get("to"))
            for edge in dependencies
            if isinstance(edge, dict) and str(edge.get("from")) == "gate_1"
        }
        self._inventory: list[str] = []
        self._prepared_on_counter: set[str] = set()
        self._meal_prepared: set[str] = set()
        self._meal_eaten: set[str] = set()
        self._items: dict[str, dict[str, Any]] = {}
        for dish in self._dishes:
            dish_id = str(dish["dish_id"])
            for recipe in dish.get("recipe", []):
                name = str(recipe["name"])
                expected = _prep_set(recipe.get("preparation", [])) or {"raw", "uncut"}
                source_candidates = list(recipe.get("candidate_source_locations", []))
                initial_actual = {value for value in ("raw", "uncut") if value in expected}
                self._items[name] = {
                    "name": name,
                    "dish_id": dish_id,
                    "location": _source_room(source_candidates[0] if source_candidates else "kitchen"),
                    "expected": expected,
                    "actual": initial_actual,
                    "collected": False,
                    "deleted": False,
                }
        self._distractors = [f"distractor_{index + 1}" for index in range(int(params.get("numDistractorItems", 0) or 0))]
        self._steps = 0
        self._done = False
        self._failure = False
        self._agent_locations.clear()
        self._events.clear()
        self._last = None

    def reset(self, **_: Any) -> tuple[str, dict[str, Any]]:
        self._reset_state()
        self._agent_locations["root"] = "kitchen"
        return self.observe("root")

    def fork_shared(self, agent_id: str) -> CompositeAgentView:
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        self._agent_locations.setdefault(agent_id, self._agent_locations.get("root", "kitchen"))
        return CompositeAgentView(self, agent_id)

    async def observe_agent(self, agent_id: str) -> tuple[str, dict[str, Any]]:
        async with self._lock:
            return self.observe(agent_id)

    def observe(self, agent_id: str = "root") -> tuple[str, dict[str, Any]]:
        self._agent_locations.setdefault(agent_id, "kitchen")
        infos = self._infos(agent_id)
        observation = self._observation(agent_id, infos)
        self._last = (observation, copy.deepcopy(infos))
        return observation, copy.deepcopy(infos)

    async def step(self, agent_id: str, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        async with self._lock:
            return self._step_locked(agent_id, action)

    def _step_locked(self, agent_id: str, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self._agent_locations.setdefault(agent_id, "kitchen")
        action = str(action or "").strip()
        before_location = self._agent_locations[agent_id]
        valid_actions = set(self._valid_actions(agent_id))
        accepted = action in valid_actions or _normalise(action) in {"look around", "inventory", "view inventory", "check inventory"}
        error = None
        reward = 0.0
        if not accepted:
            error = "Action is not valid in the current composite world state."
        else:
            self._apply(agent_id, action)
        self._steps += 1
        report = self._score_report()
        if report["task_success"]:
            self._done = True
            reward = 1.0
        if report["task_failure"]:
            self._failure = True
            self._done = True
        if self._steps >= self.env_step_limit:
            self._done = True
        self._advance_tool_cooldowns()
        observation, infos = self.observe(agent_id)
        self._events.append(_Event(len(self._events) + 1, agent_id, action, accepted, reward, self._done, observation, before_location, self._agent_locations[agent_id], error))
        return observation, reward, self._done, infos

    def _apply(self, agent_id: str, action: str) -> None:
        normalized = _normalise(action)
        location = self._agent_locations[agent_id]
        if normalized in {"look around", "inventory", "view inventory", "check inventory", "read cookbook"}:
            return
        if normalized == "open door":
            self._gate_open = True
            return
        if normalized.startswith("go to "):
            destination = normalized[6:].strip()
            if destination in self._rooms:
                self._agent_locations[agent_id] = destination
            return
        if normalized.startswith("move "):
            destination = _DIRECTIONS.get(normalized[5:].strip())
            if destination in self._rooms:
                self._agent_locations[agent_id] = destination
            return
        if normalized.startswith("take "):
            name = action[5:].strip()
            item = self._items.get(name)
            if item is None:
                item = self._items.get(_normalise(name))
            if item is not None:
                item["collected"] = True
                if name not in self._inventory:
                    self._inventory.append(item["name"])
                if item["expected"] <= {"raw", "uncut"}:
                    self._mark_ready(item["name"])
            return
        if normalized.startswith("place ") and " on counter" in normalized:
            name = action[6:].lower().split(" on counter", 1)[0].strip()
            self._mark_ready(name)
            return
        if normalized.startswith("cook "):
            match = re.match(r"cook (.+?) (?:in|on) (.+)$", normalized)
            if match:
                self._consume_tool(match.group(2))
                self._set_preparation(match.group(1), match.group(2))
            return
        for operation in ("chop", "slice", "dice", "mince"):
            prefix = operation + " "
            if normalized.startswith(prefix):
                self._consume_tool("knife")
                self._set_preparation(normalized[len(prefix):], operation)
                return
        if normalized.startswith("prepare "):
            dish_id = normalized[8:].strip()
            if self._dish_ready(dish_id):
                self._meal_prepared.add(dish_id)
            return
        if normalized.startswith("eat "):
            dish_id = normalized[4:].strip()
            if dish_id in self._meal_prepared:
                self._meal_eaten.add(dish_id)

    def _set_preparation(self, name: str, tool: str) -> None:
        item = self._items.get(name)
        if item is None:
            return
        for prep, expected_tool in _COOKING_ACTION.items():
            if expected_tool == tool and prep in item["expected"]:
                if "uncut" not in item["expected"]:
                    item["actual"].discard("uncut")
                item["actual"].add(prep)
        for prep, operation in _CUTTING_ACTIONS.items():
            if operation == tool and prep in item["expected"]:
                item["actual"].discard("uncut")
                item["actual"].add(prep)
        if item["actual"] == item["expected"]:
            self._mark_ready(item["name"])

    def _tool_available(self, tool: str) -> bool:
        return self._tool_cooldowns.get(tool, 0) <= 0

    def _consume_tool(self, tool: str) -> None:
        if self._tool_cooldown_steps <= 0:
            return
        self._tool_cooldowns[tool] = self._tool_cooldown_steps + 1

    def _advance_tool_cooldowns(self) -> None:
        if not self._tool_cooldowns:
            return
        self._tool_cooldowns = {
            tool: remaining - 1
            for tool, remaining in self._tool_cooldowns.items()
            if remaining > 1
        }

    def _mark_ready(self, name: str) -> None:
        item = self._items.get(name)
        if item is None:
            return
        item["collected"] = True
        if name not in self._prepared_on_counter:
            self._prepared_on_counter.add(name)
        self._inventory = [value for value in self._inventory if value != name]

    def _dish_ready(self, dish_id: str) -> bool:
        dish = next((item for item in self._dishes if str(item["dish_id"]) == dish_id), None)
        if dish is None:
            return False
        return all(str(recipe["name"]) in self._prepared_on_counter for recipe in dish.get("recipe", []))

    def _valid_actions(self, agent_id: str) -> list[str]:
        location = self._agent_locations.get(agent_id, "kitchen")
        actions = ["look around", "inventory", "read cookbook"]
        actions.extend(f"go to {room}" for room in self._rooms if room != location)
        actions.extend(f"move {direction}" for direction, destination in _DIRECTIONS.items() if destination in self._rooms and destination != location)
        if not self._gate_open:
            actions.append("open door")
        if location == "kitchen":
            actions.append("take knife")
        for item in self._items.values():
            if item["deleted"] or item["collected"] or item["name"] in self._prepared_on_counter:
                continue
            if item["location"] != location or not self._item_accessible(item):
                continue
            if self._inventory_capacity and len(self._inventory) >= self._inventory_capacity:
                continue
            actions.append(f"take {item['name']}")
        for name in list(self._inventory):
            item = self._items[name]
            expected = item["expected"]
            actual = item["actual"]
            if "chopped" in expected and "chopped" not in actual:
                if self._tool_available("knife"):
                    actions.append(f"chop {name}")
            if "sliced" in expected and "sliced" not in actual:
                if self._tool_available("knife"):
                    actions.append(f"slice {name}")
            if "diced" in expected and "diced" not in actual:
                if self._tool_available("knife"):
                    actions.append(f"dice {name}")
            if "minced" in expected and "minced" not in actual:
                if self._tool_available("knife"):
                    actions.append(f"mince {name}")
            for prep, tool in _COOKING_ACTION.items():
                if prep in expected and prep not in actual:
                    if self._tool_available(tool):
                        connector = "on" if tool == "grill" else "in"
                        actions.append(f"cook {name} {connector} {tool}")
        actions.extend(f"prepare {dish['dish_id']}" for dish in self._dishes if self._dish_ready(str(dish["dish_id"])) and str(dish["dish_id"]) not in self._meal_prepared)
        actions.extend(f"eat {dish['dish_id']}" for dish in self._dishes if str(dish["dish_id"]) in self._meal_prepared and str(dish["dish_id"]) not in self._meal_eaten)
        return actions

    def _item_accessible(self, item: dict[str, Any]) -> bool:
        return self._gate_open or item["dish_id"] not in self._gated_dishes

    def _score_report(self) -> dict[str, Any]:
        state = self.getObjectTree()
        return score_task(self._dishes, state)

    def _infos(self, agent_id: str) -> dict[str, Any]:
        report = self._score_report()
        return {
            "validActions": self._valid_actions(agent_id),
            "inventory": ", ".join(self._inventory) if self._inventory else "(empty)",
            "score": report["score"],
            "tasksuccess": report["task_success"],
            "taskfailure": self._failure or report["task_failure"],
            "done": self._done,
            "steps": self._steps,
            "remaining_shared_steps": max(self.env_step_limit - self._steps, 0),
            "agent_location": self._agent_locations.get(agent_id, "kitchen"),
            "completed_dishes": report["completed_dishes"],
            "total_dishes": report["total_dishes"],
        }

    def _observation(self, agent_id: str, infos: dict[str, Any]) -> str:
        location = self._agent_locations.get(agent_id, "kitchen")
        visible = [item["name"] for item in self._items.values() if item["location"] == location and not item["collected"] and self._item_accessible(item)]
        progress = []
        for dish in self._dishes:
            dish_id = str(dish["dish_id"])
            ready = sum(str(recipe["name"]) in self._prepared_on_counter for recipe in dish.get("recipe", []))
            progress.append(f"{dish_id}: {ready}/{len(dish.get('recipe', []))} ingredients prepared, meal_prepared={dish_id in self._meal_prepared}, meal_eaten={dish_id in self._meal_eaten}")
        gate = "open" if self._gate_open else "closed"
        cooldowns = ", ".join(
            f"{tool}:{remaining}"
            for tool, remaining in sorted(self._tool_cooldowns.items())
        ) or "none"
        return (
            f"You are in the {location}.\n"
            f"Visible objects: {', '.join(visible) if visible else '(none)'}\n"
            f"Shared kitchen gate: {gate}.\n"
            f"Shared tool cooldowns: {cooldowns}.\n"
            f"Shared prepared counter: {', '.join(sorted(self._prepared_on_counter)) if self._prepared_on_counter else '(empty)'}\n"
            "Dish progress:\n- " + "\n- ".join(progress) + "\n"
            f"Shared steps remaining: {infos['remaining_shared_steps']}\n"
            "Valid Actions are listed below."
        )

    def getObjectTree(self) -> dict[str, Any]:
        dishes: dict[str, Any] = {}
        for dish in self._dishes:
            dish_id = str(dish["dish_id"])
            ingredients = {}
            for recipe in dish.get("recipe", []):
                name = str(recipe["name"])
                item = self._items[name]
                ingredients[name] = {"collected": item["collected"], "deleted": item["deleted"], "preparation": sorted(item["actual"]), "ready": name in self._prepared_on_counter}
            dishes[dish_id] = {"ingredients": ingredients, "meal_prepared": dish_id in self._meal_prepared, "meal_eaten": dish_id in self._meal_eaten}
        return {
            "dishes": dishes,
            "inventory": list(self._inventory),
            "prepared_on_counter": sorted(self._prepared_on_counter),
            "agent_locations": dict(self._agent_locations),
            "steps": self._steps,
            "gate_open": self._gate_open,
            "tool_cooldowns": dict(self._tool_cooldowns),
        }

    def get_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    def get_generation_properties(self) -> dict[str, Any]:
        return copy.deepcopy(self._properties)

    def get_task_description(self) -> str:
        return str(getattr(self.task, "task_description", ""))

    def close(self) -> None:
        return None
