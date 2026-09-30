from __future__ import annotations

from typing import Any


def _prep(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(part) for part in value)
    return frozenset()


def score_task(task_dishes: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    state_dishes = state.get("dishes", {})
    dish_results: list[dict[str, Any]] = []
    invalid = False
    for dish in task_dishes:
        dish_id = str(dish["dish_id"])
        current = state_dishes.get(dish_id, {})
        current_ingredients = current.get("ingredients", {})
        ingredient_results = []
        dish_invalid = False
        for required in dish.get("recipe", []):
            name = str(required["name"])
            actual = current_ingredients.get(name, {})
            collected = bool(actual.get("collected", False))
            deleted = bool(actual.get("deleted", False))
            expected = _prep(required.get("preparation", []))
            observed = _prep(actual.get("preparation", []))
            matches = observed == expected
            # An in-progress ingredient is allowed to be collected but not yet
            # fully prepared. Only an actually wrong irreversible preparation
            # (or deletion) makes the branch invalid.
            wrong_preparation = bool((observed - {"raw", "uncut"}) - expected)
            dish_invalid = dish_invalid or deleted or wrong_preparation
            ingredient_results.append({"name": name, "collected": collected, "deleted": deleted, "preparation_matches": matches, "complete": collected and not deleted and matches})
        complete = all(item["complete"] for item in ingredient_results)
        meal_prepared = bool(current.get("meal_prepared", False))
        meal_eaten = bool(current.get("meal_eaten", False))
        success = complete and meal_prepared and meal_eaten
        invalid = invalid or dish_invalid
        dish_results.append({"dish_id": dish_id, "ingredients_complete": complete, "meal_prepared": meal_prepared, "meal_eaten": meal_eaten, "task_success": success, "invalid_or_deleted": dish_invalid, "completed_ingredients": sum(item["complete"] for item in ingredient_results), "total_ingredients": len(ingredient_results), "ingredients": ingredient_results})
    completed = sum(bool(item["task_success"]) for item in dish_results)
    return {"task_success": bool(dish_results) and completed == len(dish_results), "task_failure": invalid, "score": 1.0 if dish_results and completed == len(dish_results) else 0.0, "completed_dishes": completed, "total_dishes": len(dish_results), "dish_results": dish_results}
