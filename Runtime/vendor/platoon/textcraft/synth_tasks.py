"""
Task generation for TextCraft-Synth synthetic crafting tasks.

Creates tasks with difficulty tagging based on crafting depth:
- easy: depth 2-3 (simple multi-step crafting)
- medium: depth 4-6 (moderate complexity)
- hard: depth 7-9 (deep hierarchies)
- extreme: depth 10+ (very deep, cross-domain crafting)
"""

import argparse
import json
import os
from Dataset.paths import data_path
import pathlib
import random
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from platoon.envs.base import Task

from .synth_recipe_generator import SynthRecipeDatabase, set_naming_mode


class Difficulty(Enum):
    """Difficulty levels based on crafting depth."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


# Difficulty level definitions:
# (
#   min_depth,
#   max_depth,
#   max_targets,
#   max_count_per_target
# )
# Single-target configuration to fit within 200-step budget
DIFFICULTY_CONFIG = {
    Difficulty.EASY: (2, 3, 1, 3),
    Difficulty.MEDIUM: (4, 6, 1, 3),
    Difficulty.HARD: (7, 9, 1, 2),
    Difficulty.EXTREME: (10, 12, 1, 2),
}


def get_difficulty_for_depth(depth: int) -> Difficulty:
    """Get difficulty level for a given crafting depth."""
    if depth <= 3:
        return Difficulty.EASY
    elif depth <= 6:
        return Difficulty.MEDIUM
    elif depth <= 9:
        return Difficulty.HARD
    else:
        return Difficulty.EXTREME


def extract_base_materials_synth(
    recipe_db: SynthRecipeDatabase,
    target_items: Dict[str, int],
    initial_inventory: Dict[str, int],
    _cache: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, int]:
    """
    Extract all base materials needed to craft target items.
    Returns a dictionary mapping base item names to required counts.

    Uses memoization for efficiency with shared cache.
    """
    needed_base_items: Dict[str, int] = {}

    # Cache: item -> {base_item: count_per_one}
    if _cache is None:
        _cache = {}

    def get_base_materials_for_one(item: str, visited: Optional[Set[str]] = None) -> Dict[str, int]:
        """Get base materials needed to craft ONE of an item."""
        if item in _cache:
            return _cache[item]

        if visited is None:
            visited = set()

        if item in visited:
            return {}
        visited.add(item)

        # Base item: needs itself
        if recipe_db.is_base_item(item):
            result = {item: 1}
            _cache[item] = result
            return result

        # Craftable: collect from ingredients
        if recipe_db.can_craft(item):
            recipes = recipe_db.get_recipes_for_item(item)
            if recipes:
                recipe = recipes[0]
                result: Dict[str, int] = {}

                for ing_item, ing_count in recipe.ingredients.items():
                    ing_bases = get_base_materials_for_one(ing_item, visited.copy())
                    for base, base_count in ing_bases.items():
                        result[base] = result.get(base, 0) + base_count * ing_count

                # Divide by result_count to get per-one requirements
                # (kept as fractions implicitly via ceiling when used)
                _cache[item] = result
                return result

        return {}

    for item, count in target_items.items():
        base_per_one = get_base_materials_for_one(item)
        for base, base_count in base_per_one.items():
            total_needed = base_count * count
            current_have = initial_inventory.get(base, 0)
            needed = total_needed - current_have
            if needed > 0:
                needed_base_items[base] = needed_base_items.get(base, 0) + needed

    return needed_base_items


def solve_crafting_task_synth(
    recipe_db: SynthRecipeDatabase,
    target_items: Dict[str, int],
    initial_inventory: Dict[str, int],
    max_recursion_depth: int = 20,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, int]]]:
    """
    Solve a synthetic crafting task by finding a valid sequence of crafting steps.

    Recursively crafts only the deficit between each required quantity and the
    current simulated inventory. This makes the plan inventory-aware at every
    dependency level instead of expanding the full recipe tree first.

    Returns:
        Tuple of (trajectory, required_base_materials) or None if no solution found.
    """
    inventory = initial_inventory.copy()
    trajectory: List[Dict[str, Any]] = []

    def ensure_inventory(
        item: str,
        required_count: int,
        active_items: Set[str],
        depth: int,
    ) -> bool:
        """Ensure that inventory contains required_count of item."""
        if inventory.get(item, 0) >= required_count:
            return True
        if depth >= max_recursion_depth or item in active_items:
            return False

        recipes = recipe_db.get_recipes_for_item(item)
        if not recipes:
            return False

        for recipe in recipes:
            inventory_snapshot = inventory.copy()
            trajectory_length = len(trajectory)
            deficit = required_count - inventory.get(item, 0)
            crafts_needed = (deficit + recipe.result_count - 1) // recipe.result_count
            ingredient_needs = {
                ing_item: crafts_needed * ing_count
                for ing_item, ing_count in recipe.ingredients.items()
            }

            # A higher-depth ingredient may consume a lower-depth ingredient
            # that the same recipe also needs. Preparing deeper dependencies
            # first leaves the direct lower-depth requirement available last.
            ordered_ingredients = sorted(
                ingredient_needs.items(),
                key=lambda pair: recipe_db.get_crafting_depth(pair[0]),
                reverse=True,
            )
            next_active = active_items | {item}
            can_prepare = all(
                ensure_inventory(ing_item, count, next_active, depth + 1)
                for ing_item, count in ordered_ingredients
            )
            ingredients_available = all(
                inventory.get(ing_item, 0) >= count
                for ing_item, count in ingredient_needs.items()
            )
            if not can_prepare or not ingredients_available:
                inventory.clear()
                inventory.update(inventory_snapshot)
                del trajectory[trajectory_length:]
                continue

            for ing_item, consumed in ingredient_needs.items():
                inventory[ing_item] -= consumed
                if inventory[ing_item] <= 0:
                    del inventory[ing_item]

            crafted_count = crafts_needed * recipe.result_count
            inventory[item] = inventory.get(item, 0) + crafted_count
            trajectory.append(
                {
                    "action": "craft",
                    "target": (item, crafts_needed),
                    "ingredients": ingredient_needs,
                    "result_count": crafted_count,
                }
            )
            return True

        return False

    # Higher-depth final targets are completed first. If another requested
    # target is one of their ingredients, it is replenished afterwards.
    ordered_targets = sorted(
        target_items.items(),
        key=lambda pair: recipe_db.get_crafting_depth(pair[0]),
        reverse=True,
    )
    for item, count in ordered_targets:
        if not ensure_inventory(item, count, set(), 0):
            return None

    if any(inventory.get(item, 0) < count for item, count in target_items.items()):
        return None

    required_base = extract_base_materials_synth(recipe_db, target_items, initial_inventory)
    return trajectory, required_base


def create_synth_task(
    recipe_db: SynthRecipeDatabase,
    task_id: str,
    difficulty: Difficulty,
    rng: random.Random,
) -> Optional[Task]:
    """Create a single synthetic task with the given difficulty."""

    config = DIFFICULTY_CONFIG[difficulty]
    min_depth, max_depth, max_targets, max_count = config

    # Get items in the depth range
    available_items = recipe_db.get_items_by_depth_range(min_depth, max_depth)

    if not available_items:
        return None

    # Select target items
    num_targets = rng.randint(1, max_targets)
    target_items_list = rng.sample(available_items, min(num_targets, len(available_items)))

    # Create target dictionary with random counts
    target_items: Dict[str, int] = {}
    for item in target_items_list:
        target_items[item] = rng.randint(1, max_count)

    # Extract required base materials
    required_base = extract_base_materials_synth(recipe_db, target_items, {})

    if not required_base:
        return None

    # Generate initial inventory with required materials + buffer
    initial_inventory: Dict[str, int] = {}

    for item, count in required_base.items():
        # Add 0-50% extra buffer
        buffer = rng.randint(0, max(1, count // 2))
        initial_inventory[item] = count + buffer

    # Add some extra random base items for distraction
    all_base = list(recipe_db.base_items)
    num_extra = rng.randint(1, 3)
    for _ in range(num_extra):
        extra_item = rng.choice(all_base)
        if extra_item not in initial_inventory:
            initial_inventory[extra_item] = rng.randint(1, 5)

    # Verify task is solvable
    result = solve_crafting_task_synth(recipe_db, target_items, initial_inventory)
    if result is None:
        return None

    gold_trajectory, _ = result

    # Calculate actual max depth for this task
    max_item_depth = max(recipe_db.get_crafting_depth(item) for item in target_items.keys())

    # Create task
    target_str = ", ".join([f"{count}x {item}" for item, count in target_items.items()])
    goal = f"Craft the following items: {target_str}"

    # Calculate max_steps based on trajectory length and depth
    # Fixed budget for all tasks - agents should batch actions to stay within budget
    max_steps = 75

    return Task(
        goal=goal,
        max_steps=max_steps,
        misc={
            "target_items": target_items,
            "initial_inventory": initial_inventory,
            "gold_trajectory": gold_trajectory,
            "difficulty": difficulty.value,
            "max_depth": max_item_depth,
            "num_craft_steps": len(gold_trajectory),
        },
        id=task_id,
    )


def create_synth_datasets(
    seed: int = 42,
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
    recipes_dir: Optional[pathlib.Path] = None,
    difficulty_distribution: Optional[Dict[Difficulty, float]] = None,
    verbose: bool = True,
    semantic_names: bool = False,
    val_item_fraction: float = 0.2,
    items_per_domain_tier: int = 25,
) -> Tuple[List[Task], List[Task]]:
    """
    Create train and validation datasets for TextCraft-Synth tasks.

    Ensures NO overlap between train and val by splitting craftable items:
    - A fraction of items are designated "val-only"
    - Train tasks only use train items as targets
    - Val tasks only use val items as targets
    - This guarantees the agent never sees val target items during training

    Args:
        seed: Random seed
        num_samples_train: Number of training samples
        num_samples_val: Number of validation samples
        recipes_dir: Path to synthetic recipes directory (will generate if doesn't exist)
        difficulty_distribution: Dict mapping Difficulty -> proportion (should sum to 1.0)
            Default is equal distribution across difficulties.
        verbose: Print progress updates
        semantic_names: If True, use human-readable names (iron_refined).
                       If False (default), use generic names (m0_i1) to avoid LLM prior conflicts.
        val_item_fraction: Fraction of items to reserve for validation (default 0.2 = 20%)
        items_per_domain_tier: Number of items per domain per tier (default 25 for ~1500 items)

    Returns:
        Tuple of (train_tasks, val_tasks)
    """
    rng = random.Random(seed)

    # Set naming mode
    set_naming_mode(semantic=semantic_names)

    # Default: equal distribution
    if difficulty_distribution is None:
        difficulty_distribution = {
            Difficulty.EASY: 0.25,
            Difficulty.MEDIUM: 0.35,
            Difficulty.HARD: 0.25,
            Difficulty.EXTREME: 0.15,
        }

    # Generate or load recipes
    if recipes_dir is None:
        recipes_dir = pathlib.Path(__file__).parent / "synth_recipes"

    if verbose:
        print(f"Using {'semantic' if semantic_names else 'generic'} naming mode")
        print("Loading/generating synthetic recipe database...")

    recipe_db = SynthRecipeDatabase()
    recipe_db.generate_all_recipes(seed=seed, items_per_domain_tier=items_per_domain_tier)

    if verbose:
        print(f"Recipe database has {len(recipe_db.all_items)} items, max depth {max(recipe_db.item_depths.values())}")

    # === SPLIT ITEMS INTO TRAIN AND VAL SETS ===
    # This ensures NO overlap: val tasks use items the model never saw during training
    train_items_by_tier: Dict[int, List[str]] = {i: [] for i in range(13)}
    val_items_by_tier: Dict[int, List[str]] = {i: [] for i in range(13)}

    for tier in range(1, 13):  # Skip tier 0 (base items - not craftable targets)
        tier_items = recipe_db.items_by_tier.get(tier, [])
        if not tier_items:
            continue

        # Shuffle deterministically
        shuffled = list(tier_items)
        rng.shuffle(shuffled)

        # Split: first (1-val_item_fraction) go to train, rest to val
        split_idx = int(len(shuffled) * (1 - val_item_fraction))
        train_items_by_tier[tier] = shuffled[:split_idx]
        val_items_by_tier[tier] = shuffled[split_idx:]

    if verbose:
        train_total = sum(len(v) for v in train_items_by_tier.values())
        val_total = sum(len(v) for v in val_items_by_tier.values())
        print(f"Item split: {train_total} train items, {val_total} val items ({val_item_fraction * 100:.0f}% held out)")

    train_tasks: List[Task] = []
    val_tasks: List[Task] = []

    # Track seen tasks to prevent duplicates within each split
    seen_train_tasks: Set[str] = set()
    seen_val_tasks: Set[str] = set()

    # Helper to create task with specific item pool
    def create_task_from_pool(
        items_by_tier: Dict[int, List[str]],
        task_id: str,
        difficulty: Difficulty,
        seen_tasks: Set[str],
    ) -> Optional[Task]:
        config = DIFFICULTY_CONFIG[difficulty]
        min_depth, max_depth, max_targets, max_count = config

        # Get items in the depth range from this pool
        available_items = []
        for depth in range(min_depth, max_depth + 1):
            available_items.extend(items_by_tier.get(depth, []))

        if not available_items:
            return None

        # Select target items
        num_targets = rng.randint(1, max_targets)
        target_items_list = rng.sample(available_items, min(num_targets, len(available_items)))

        # Create target dictionary with random counts
        target_items: Dict[str, int] = {}
        for item in target_items_list:
            target_items[item] = rng.randint(1, max_count)

        # Check for duplicate (by item names AND counts)
        # This allows "craft 1x A" and "craft 2x A" as separate tasks
        task_key = str(sorted(target_items.items()))
        if task_key in seen_tasks:
            return None

        # Extract required base materials
        required_base = extract_base_materials_synth(recipe_db, target_items, {})
        if not required_base:
            return None

        # Generate initial inventory
        initial_inventory: Dict[str, int] = {}
        for item, count in required_base.items():
            buffer = rng.randint(0, max(1, count // 2))
            initial_inventory[item] = count + buffer

        # Add distractor items
        all_base = list(recipe_db.base_items)
        num_extra = rng.randint(1, 3)
        for _ in range(num_extra):
            extra_item = rng.choice(all_base)
            if extra_item not in initial_inventory:
                initial_inventory[extra_item] = rng.randint(1, 5)

        # Verify solvable
        result = solve_crafting_task_synth(recipe_db, target_items, initial_inventory)
        if result is None:
            return None

        gold_trajectory, _ = result
        max_item_depth = max(recipe_db.get_crafting_depth(item) for item in target_items.keys())

        # Create task
        target_str = ", ".join([f"{count}x {item}" for item, count in target_items.items()])
        goal = f"Craft the following items: {target_str}"

        # Fixed budget for all tasks - agents should batch actions to stay within budget
        max_steps = 75

        seen_tasks.add(task_key)

        return Task(
            goal=goal,
            max_steps=max_steps,
            misc={
                "target_items": target_items,
                "initial_inventory": initial_inventory,
                "gold_trajectory": gold_trajectory,
                "difficulty": difficulty.value,
                "max_depth": max_item_depth,
                "num_craft_steps": len(gold_trajectory),
            },
            id=task_id,
        )

    # Calculate target counts per difficulty for each split
    train_difficulty_counts = {diff: int(num_samples_train * prop) for diff, prop in difficulty_distribution.items()}
    val_difficulty_counts = {diff: int(num_samples_val * prop) for diff, prop in difficulty_distribution.items()}

    # Adjust for rounding
    train_sum = sum(train_difficulty_counts.values())
    if train_sum < num_samples_train:
        train_difficulty_counts[Difficulty.MEDIUM] += num_samples_train - train_sum

    val_sum = sum(val_difficulty_counts.values())
    if val_sum < num_samples_val:
        val_difficulty_counts[Difficulty.MEDIUM] += num_samples_val - val_sum

    # Generate TRAIN tasks
    if verbose:
        print("\n=== Generating TRAIN tasks ===")

    for difficulty, target_count in train_difficulty_counts.items():
        if verbose:
            print(f"Generating {target_count} {difficulty.value} train tasks...")

        generated = 0
        attempts = 0
        max_attempts = target_count * 30

        while generated < target_count and attempts < max_attempts:
            attempts += 1
            task_id = f"textcraft_synth.train.{len(train_tasks)}"

            task = create_task_from_pool(train_items_by_tier, task_id, difficulty, seen_train_tasks)

            if task is not None:
                train_tasks.append(task)
                generated += 1

                if verbose and generated % 500 == 0:
                    print(f"  Generated {generated}/{target_count} {difficulty.value} tasks")

        if verbose:
            print(f"  Completed {generated}/{target_count} {difficulty.value} tasks")

    # Generate VAL tasks
    if verbose:
        print("\n=== Generating VAL tasks ===")

    for difficulty, target_count in val_difficulty_counts.items():
        if verbose:
            print(f"Generating {target_count} {difficulty.value} val tasks...")

        generated = 0
        attempts = 0
        max_attempts = target_count * 30

        while generated < target_count and attempts < max_attempts:
            attempts += 1
            task_id = f"textcraft_synth.val.{len(val_tasks)}"

            task = create_task_from_pool(val_items_by_tier, task_id, difficulty, seen_val_tasks)

            if task is not None:
                val_tasks.append(task)
                generated += 1

                if verbose and generated % 100 == 0:
                    print(f"  Generated {generated}/{target_count} {difficulty.value} tasks")

        if verbose:
            print(f"  Completed {generated}/{target_count} {difficulty.value} tasks")

    # Final check
    if len(train_tasks) < num_samples_train or len(val_tasks) < num_samples_val:
        print(
            f"Warning: Generated {len(train_tasks)}/{num_samples_train} train and "
            f"{len(val_tasks)}/{num_samples_val} val tasks"
        )

    # Verify no overlap
    train_target_items = set()
    for task in train_tasks:
        train_target_items.update(task.misc["target_items"].keys())

    val_target_items = set()
    for task in val_tasks:
        val_target_items.update(task.misc["target_items"].keys())

    overlap = train_target_items & val_target_items
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping items between train and val!")
    elif verbose:
        print(f"\n✓ Verified: 0 overlapping targets train({len(train_target_items)}) / val({len(val_target_items)})")

    # Shuffle tasks (with deterministic seed for reproducibility)
    shuffle_rng = random.Random(seed + 1)  # Different seed to avoid correlation with generation
    shuffle_rng.shuffle(train_tasks)
    shuffle_rng.shuffle(val_tasks)

    # Re-assign IDs after shuffling to maintain sequential ordering
    for i, task in enumerate(train_tasks):
        task.id = f"textcraft_synth.train.{i}"
    for i, task in enumerate(val_tasks):
        task.id = f"textcraft_synth.val.{i}"

    if verbose:
        print("✓ Tasks shuffled (deterministically)")

    return train_tasks, val_tasks


# Data loading functions
SYNTH_TRAIN_DATA: Optional[List[str]] = None
SYNTH_VAL_DATA: Optional[List[str]] = None
SYNTH_TASKS: Dict[str, Task] = {}


def get_synth_task_ids(
    split: Literal["train", "val"],
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
) -> List[str]:
    """Get task IDs for a split."""
    if split == "train":
        return [f"textcraft_synth.train.{i}" for i in range(num_samples_train)]
    elif split == "val":
        return [f"textcraft_synth.val.{i}" for i in range(num_samples_val)]
    else:
        raise ValueError(f"Invalid split: {split}")


def get_synth_task_ids_by_difficulty(
    split: Literal["train", "val"],
    difficulty: Difficulty,
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
) -> List[str]:
    """Get task IDs for a split filtered by difficulty."""
    all_ids = get_synth_task_ids(split, num_samples_train, num_samples_val)
    filtered = []

    for task_id in all_ids:
        try:
            task = get_synth_task(task_id)
            if task.misc.get("difficulty") == difficulty.value:
                filtered.append(task_id)
        except (FileNotFoundError, IndexError):
            break

    return filtered


def load_synth_task_from_disk(id: str) -> Task:
    """Load a synthetic task from disk."""
    parent = pathlib.Path(__file__).parent

    if id.startswith("textcraft_synth.train."):
        global SYNTH_TRAIN_DATA
        if SYNTH_TRAIN_DATA is None:
            file = pathlib.Path(os.environ.get("TEXTCRAFT_SYNTH_TRAIN_PATH", data_path("training", "textcraft", "textcraft_synth_train.jsonl")))
            if file.exists():
                SYNTH_TRAIN_DATA = file.read_text().splitlines()
            else:
                raise FileNotFoundError(f"Training data file not found: {file}. Run task generation first.")
        idx = int(id.split(".")[2])
        if idx >= len(SYNTH_TRAIN_DATA):
            raise IndexError(f"Task index {idx} out of range for training data")
        return Task.from_dict(json.loads(SYNTH_TRAIN_DATA[idx]))

    elif id.startswith("textcraft_synth.val."):
        global SYNTH_VAL_DATA
        if SYNTH_VAL_DATA is None:
            file = pathlib.Path(os.environ.get("TEXTCRAFT_SYNTH_VAL_PATH", data_path("eval", "textcraft", "textcraft_synth_val.jsonl")))
            if file.exists():
                SYNTH_VAL_DATA = file.read_text().splitlines()
            else:
                raise FileNotFoundError(f"Validation data file not found: {file}. Run task generation first.")
        idx = int(id.split(".")[2])
        if idx >= len(SYNTH_VAL_DATA):
            raise IndexError(f"Task index {idx} out of range for validation data")
        return Task.from_dict(json.loads(SYNTH_VAL_DATA[idx]))

    else:
        raise ValueError(f"Invalid task id: {id}")


def get_synth_task(id: str) -> Task:
    """Get a synthetic task by ID (with caching)."""
    global SYNTH_TASKS
    if id in SYNTH_TASKS:
        return SYNTH_TASKS[id]
    task = load_synth_task_from_disk(id)
    SYNTH_TASKS[id] = task
    return task


def print_dataset_statistics(train_tasks: List[Task], val_tasks: List[Task]):
    """Print statistics about the generated dataset."""
    print("\n" + "=" * 60)
    print("TextCraft-Synth Dataset Statistics")
    print("=" * 60)

    for name, tasks in [("Train", train_tasks), ("Val", val_tasks)]:
        print(f"\n{name} set: {len(tasks)} tasks")

        # Count by difficulty
        difficulty_counts = {}
        depth_sum = 0
        steps_sum = 0

        for task in tasks:
            diff = task.misc.get("difficulty", "unknown")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            depth_sum += task.misc.get("max_depth", 0)
            steps_sum += task.misc.get("num_craft_steps", 0)

        print("  By difficulty:")
        for diff, count in sorted(difficulty_counts.items()):
            print(f"    {diff}: {count} ({100 * count / len(tasks):.1f}%)")

        if tasks:
            print(f"  Average max depth: {depth_sum / len(tasks):.2f}")
            print(f"  Average craft steps: {steps_sum / len(tasks):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TextCraft-Synth dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=1000,
        help="Number of validation/test samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "extreme", "all"],
        default="all",
        help="Generate tasks of specific difficulty or all difficulties",
    )
    parser.add_argument(
        "--semantic-names",
        action="store_true",
        help="Use semantic names (iron_refined) instead of generic (m0_i1). "
        "Generic names (default) are recommended for benchmarking to avoid LLM prior conflicts.",
    )
    args = parser.parse_args()

    # Adjust distribution if specific difficulty requested
    difficulty_distribution = None
    if args.difficulty != "all":
        diff = Difficulty(args.difficulty)
        difficulty_distribution = {d: 0.0 for d in Difficulty}
        difficulty_distribution[diff] = 1.0

    train_data, val_data = create_synth_datasets(
        seed=args.seed,
        num_samples_train=args.num_samples,
        num_samples_val=args.eval_size,
        difficulty_distribution=difficulty_distribution,
        semantic_names=args.semantic_names,
    )

    print_dataset_statistics(train_data, val_data)

    # Save training data
    train_file = data_path("training", "textcraft", "textcraft_synth_train.jsonl")
    train_file.parent.mkdir(parents=True, exist_ok=True)
    with open(train_file, "w") as f:
        for task in train_data:
            # Convert tuple to list for JSON serialization in gold_trajectory
            task_dict = asdict(task)
            if task_dict.get("misc", {}).get("gold_trajectory"):
                for step in task_dict["misc"]["gold_trajectory"]:
                    if isinstance(step.get("target"), tuple):
                        step["target"] = list(step["target"])
            json.dump(task_dict, f)
            f.write("\n")

    # Save validation data
    val_file = data_path("validation", "textcraft", "textcraft_synth_val.jsonl")
    val_file.parent.mkdir(parents=True, exist_ok=True)
    with open(val_file, "w") as f:
        for task in val_data:
            task_dict = asdict(task)
            if task_dict.get("misc", {}).get("gold_trajectory"):
                for step in task_dict["misc"]["gold_trajectory"]:
                    if isinstance(step.get("target"), tuple):
                        step["target"] = list(step["target"])
            json.dump(task_dict, f)
            f.write("\n")

    print(f"\nSaved {len(train_data)} training samples to {train_file}")
    print(f"Saved {len(val_data)} validation samples to {val_file}")
