"""Task-construction helpers extracted from the source TextWorld-Sync generator."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from typing import Any

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def db_fold(source_fold: str) -> str:
    return 'valid' if source_fold == 'dev' else source_fold

def preparation_work(preparation: list[str]) -> int:
    work = 1
    if 'uncut' not in preparation:
        work += 1
    if 'raw' not in preparation:
        work += 1
    return work

def choose_preparation(options: list[list[str]], rng: random.Random, complexity: str | None) -> list[str]:
    if not complexity or complexity == 'medium':
        return list(rng.choice(options))
    scored = sorted(((preparation_work(list(option)), list(option)) for option in options))
    if complexity == 'low':
        return scored[0][1]
    if complexity == 'high':
        return scored[-1][1]
    raise ValueError(f'unknown preparation complexity: {complexity}')

def is_raw_uncut(preparation: list[str]) -> bool:
    return {'raw', 'uncut'} <= set(preparation)

def resolve_food_data(name: str, food_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve a recipe name against compact food records and their aliases."""
    direct = food_data.get(name)
    if direct is not None:
        return direct
    for record in food_data.values():
        aliases = record.get('names', [])
        if name in aliases:
            return record
    return {}

def sample_recipe(*, rng: random.Random, preparations: dict[str, list[list[str]]], food_data: dict[str, dict[str, Any]], ingredient_count: int, used_ingredients: set[str], min_prepared: int, max_prepared: int | None=None, complexity: str | None=None) -> list[dict[str, Any]]:
    candidates = sorted(set(preparations) - used_ingredients)
    if len(candidates) < ingredient_count:
        raise ValueError('not enough disjoint ingredients in the selected split')
    prepared_candidates = [name for name in candidates if any((not is_raw_uncut(list(option)) for option in preparations[name]))]
    if len(prepared_candidates) < min_prepared:
        raise ValueError('not enough ingredients with a non-trivial preparation for the requested recipe')
    if max_prepared is not None and max_prepared < ingredient_count:
        raw_candidates = [name for name in candidates if any((is_raw_uncut(list(option)) for option in preparations[name]))]
        if len(raw_candidates) < ingredient_count - min_prepared:
            raise ValueError('not enough raw-capable ingredients for the requested preparation bound')
    for _ in range(256):
        required_names = rng.sample(prepared_candidates, min_prepared)
        required_name_set = set(required_names)
        remaining_candidates = [name for name in candidates if name not in required_name_set]
        bounded_recipe = max_prepared is not None and max_prepared < ingredient_count
        if bounded_recipe:
            simple_needed = ingredient_count - min_prepared
            raw_capable = [name for name in remaining_candidates if any((is_raw_uncut(list(option)) for option in preparations[name]))]
            if len(raw_capable) < simple_needed:
                continue
            raw_only = [name for name in raw_capable if not any((not is_raw_uncut(list(option)) for option in preparations[name]))]
            preferred = rng.sample(raw_only, min(len(raw_only), simple_needed))
            fallback = [name for name in raw_capable if name not in preferred]
            simple_names = preferred + rng.sample(fallback, simple_needed - len(preferred))
            names = required_names + simple_names
        else:
            names = required_names + rng.sample(remaining_candidates, ingredient_count - min_prepared)
        rng.shuffle(names)
        recipe: list[dict[str, Any]] = []
        if bounded_recipe:
            required_prepared = min_prepared
            prepared_names = required_name_set
        else:
            prepared_names = required_name_set
        for name in names:
            options = preparations[name]
            if name in prepared_names:
                options = [option for option in options if not is_raw_uncut(list(option))]
            elif bounded_recipe:
                options = [option for option in options if is_raw_uncut(list(option))]
            if not options:
                break
            prep = choose_preparation(options, rng, complexity)
            compact = resolve_food_data(name, food_data)
            recipe.append({'name': name, 'preparation': prep, 'requires_preparation': not {'raw', 'uncut'} <= set(prep), 'candidate_source_locations': list(compact.get('locations', [])), 'branch_work': preparation_work(prep)})
        if len(recipe) != ingredient_count:
            continue
        prepared_count = sum((item['requires_preparation'] for item in recipe))
        if prepared_count >= min_prepared and (max_prepared is None or prepared_count <= max_prepared):
            used_ingredients.update(names)
            return recipe
    raise RuntimeError('could not sample a recipe meeting preparation constraints')

def render_structured(dishes: list[dict[str, Any]]) -> str:
    lines = [f'Prepare and eat {len(dishes)} independent dishes. Complete every dish.', 'The dishes are separate objectives and may be prepared independently before the final meal actions.']
    for dish in dishes:
        lines.append(f"Dish {dish['dish_id']}: prepare the following ingredients:")
        for item in dish['recipe']:
            prep = ', '.join(item['preparation'])
            lines.append(f"- {item['name']} ({prep})")
    lines.append('After all required ingredients for a dish are ready, prepare and eat that dish.')
    return '\n'.join(lines)

def render_coarse(dishes: list[dict[str, Any]]) -> str:
    lines = [f'Prepare and eat {len(dishes)} dishes. Check the cookbook and complete every dish.']
    for dish in dishes:
        names = ', '.join((item['name'] for item in dish['recipe']))
        lines.append(f"Dish {dish['dish_id']} uses: {names}.")
    return '\n'.join(lines)

def render_family_context(task_description: str, task_family: str, family_spec: dict[str, Any], profile: dict[str, Any]) -> str:
    """Expose task constraints without prescribing delegation."""
    lines = [task_description]
    if task_family == 'resource_constrained':
        lines.append(f"The shared inventory can hold at most {profile['inventory_capacity']} ingredients at once.")
        lines.append('The kitchen and preparation tools are shared across all dishes.')
    elif task_family == 'dependency_gate':
        lines.append('Some ingredient locations are behind an access gate that must be opened first.')
    elif task_family == 'conflict':
        lines.append('Several dishes require shared kitchen tools; coordinate their use.')
    elif task_family == 'hierarchical':
        lines.append('Each dish contains multiple ingredient-level objectives that can be handled as a group.')
    elif task_family == 'critical_path':
        lines.append('The dishes do not require equal amounts of preparation work.')
    elif task_family == 'load_balanced':
        lines.append('The preparation workloads of the dishes are intentionally different.')
    return '\n'.join(lines)

def build_task(*, seed: int, offset: int, split_name: str, source_fold: str, difficulty_name: str, difficulty: dict[str, Any], task_family: str, family_spec: dict[str, Any], config: dict[str, Any], db: dict[str, Any], record_split: str | None=None, task_id_split: str | None=None) -> dict[str, Any]:
    rng = random.Random(seed)
    prep_split = db['FOOD_PREPARATIONS_SPLITS'][db_fold(source_fold)]
    food_data = db.get('FOODS_COMPACT', {})
    used: set[str] = set()
    dish_count_by_difficulty = dict(family_spec.get('dish_count_by_difficulty', {}))
    dish_count = int(dish_count_by_difficulty.get(difficulty_name, family_spec.get('dish_count', difficulty['dish_count'])))
    min_prepared_by_dish = list(family_spec.get('min_prepared_by_dish', []))
    max_prepared_by_dish = list(family_spec.get('max_prepared_by_dish', []))
    complexity_by_dish = list(family_spec.get('complexity_by_dish', []))
    dishes: list[dict[str, Any]] = []
    for dish_index in range(dish_count):
        min_prepared = int(min_prepared_by_dish[dish_index]) if dish_index < len(min_prepared_by_dish) else int(difficulty['min_prepared_ingredients_per_dish'])
        max_prepared = int(max_prepared_by_dish[dish_index]) if dish_index < len(max_prepared_by_dish) else int(difficulty['max_prepared_ingredients_per_dish']) if difficulty.get('max_prepared_ingredients_per_dish') is not None else None
        complexity = str(complexity_by_dish[dish_index]) if dish_index < len(complexity_by_dish) else str(difficulty.get('preparation_complexity', 'medium'))
        recipe = sample_recipe(rng=rng, preparations=prep_split, food_data=food_data, ingredient_count=int(difficulty['ingredients_per_dish']), used_ingredients=used, min_prepared=min_prepared, max_prepared=max_prepared, complexity=complexity)
        dishes.append({'dish_id': f'dish_{dish_index + 1}', 'recipe': recipe, 'ingredient_names': [item['name'] for item in recipe], 'estimated_branch_work': sum((item['branch_work'] for item in recipe))})
    branch_work = [dish['estimated_branch_work'] for dish in dishes]
    total_branch_work = sum(branch_work)
    critical_path = max(branch_work) + 2 * len(dishes)
    task_views_by_difficulty = dict(config.get('task_view_by_difficulty', {}))
    task_view = str(task_views_by_difficulty.get(difficulty_name, config['task_view']))
    descriptions = {'structured': render_structured(dishes), 'coarse': render_coarse(dishes)}
    if task_view not in descriptions:
        raise ValueError(f'unknown task_view: {task_view}')
    profile = dict(difficulty)
    profile.update({key: value for key, value in family_spec.items() if key in {'num_locations', 'num_distractor_items', 'include_doors', 'limit_inventory_size', 'inventory_capacity'}})
    inventory_capacity_by_difficulty = dict(family_spec.get('inventory_capacity_by_difficulty', {}))
    if difficulty_name in inventory_capacity_by_difficulty:
        profile['inventory_capacity'] = int(inventory_capacity_by_difficulty[difficulty_name])
    descriptions = {view: render_family_context(text, task_family, family_spec, profile) for view, text in descriptions.items()}
    params = {'numLocations': int(profile['num_locations']), 'numDishes': len(dishes), 'numIngredientsPerDish': int(difficulty['ingredients_per_dish']), 'numIngredientsTotal': sum((len(dish['recipe']) for dish in dishes)), 'numDistractorItems': int(profile['num_distractor_items']), 'includeDoors': int(bool(profile['include_doors'])), 'limitInventorySize': int(bool(profile['limit_inventory_size']))}
    dependencies: list[dict[str, Any]] = []
    if task_family == 'dependency_gate':
        for dish_id in family_spec.get('gated_dishes', []):
            dependencies.append({'from': 'gate_1', 'to': dish_id, 'type': family_spec.get('gate_type', 'door')})
    elif task_family == 'hierarchical':
        for dish in dishes:
            dependencies.append({'from': 'root', 'to': dish['dish_id'], 'type': 'decompose'})
            for item in dish['recipe']:
                dependencies.append({'from': dish['dish_id'], 'to': f"{dish['dish_id']}::{item['name']}", 'type': 'decompose'})
    shared_resources = list(family_spec.get('shared_resources', ['kitchen', 'meal_preparation', 'meal_eating']))
    resource_capacity_by_difficulty = dict(family_spec.get('resource_capacity_by_difficulty', {}))
    resource_capacity = resource_capacity_by_difficulty.get(difficulty_name, family_spec.get('resource_capacity'))
    if resource_capacity is None and task_family == 'resource_constrained':
        resource_capacity = {'inventory': int(profile.get('inventory_capacity', 3))}
    record_split = record_split or split_name
    task_id_split = task_id_split or split_name
    task_id = f'cookingworld_multidish_{task_id_split}_{difficulty_name}_{offset:04d}'
    runtime_by_difficulty = dict(config.get('runtime_by_difficulty', {}))
    runtime_overrides = dict(runtime_by_difficulty.get(difficulty_name, {}) or {})
    mechanisms = dict(runtime_overrides.pop('mechanisms', {}) or {})
    return {'schema_version': 1, 'task_id': task_id, 'game': 'cookingworld_multidish', 'base_game': 'cookingworld', 'task_family': task_family, 'split': record_split, 'source_fold': source_fold, 'difficulty': difficulty_name, 'seed': seed, 'task_view': task_view, 'task_description': descriptions[task_view], 'task_descriptions': descriptions, 'game_params': params, 'dishes': dishes, 'parallelism': {'pattern': family_spec.get('pattern', task_family), 'branches': len(dishes), 'ingredient_disjoint': len(used) == params['numIngredientsTotal'], 'total_branch_work': total_branch_work, 'critical_path_estimate': critical_path, 'estimated_sequential_actions': total_branch_work + 2 * len(dishes) + int(bool(profile['include_doors'])), 'ideal_parallel_speedup': round(total_branch_work / critical_path, 4), 'shared_bottlenecks': shared_resources, 'resource_conflict': task_family == 'conflict', 'resource_capacity': resource_capacity, 'mechanisms': mechanisms, 'dependencies': dependencies, 'hierarchy_depth': family_spec.get('hierarchy_depth', 1), **runtime_overrides}, 'generation': {'task_family': task_family, 'requires_composite_environment': True, 'requires_composite_scorer': True, 'source_db_fold': db_fold(source_fold), 'source_db_seed': seed, 'require_disjoint_ingredients': bool(config['require_disjoint_ingredients'])}}
