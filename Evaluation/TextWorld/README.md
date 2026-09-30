# TextWorld-Sync V9 evaluation

Evaluates the recursive agent on **1,400 fixed test tasks**, with **350 tasks
per difficulty** (Easy / Medium / Hard / Extreme). Task descriptions, recipe
requirements, gates and environment parameters are included in the manifest.
The V9 composite cooking environment and programmatic success scorer are
implemented in Python; **no Java, JVM jar or TextWorldExpress installation is
required**. This is single-trajectory recursive evaluation, not Best-of-N.
The `textworld_*` manifest is read directly from `Dataset/eval/`. Training/dev
files are `Dataset/training/textworld_train.jsonl` and
`Dataset/validation/textworld_dev.jsonl`.
See [Dataset/README.md](../../Dataset/README.md) for the bundled data and the
standalone generator that writes directly to these directories.

## Dataset construction and changes from TextWorldExpress

The starting point is TextWorldExpress's **CookingWorld**, where an agent
collects ingredients, performs the required cutting/cooking operations, prepares
a meal and eats it. TextWorld-Sync retains that task vocabulary and uses the
original `cooking_world.json` database as the source of valid foods, preparation
combinations and candidate source locations. It is an extended benchmark, **not
an unchanged rerun of standard CookingWorld**. The bundled runtime reimplements
the composite world in Python rather than running the original single-avatar
JVM game directly.

The changes are:

- **One recipe → multiple dishes.** Sample several recipes per task, with
  disjoint ingredient names so dish branches can be delegated independently.
  All dishes must be prepared and consumed for root-task success.
- **One avatar → recursive agent-local views of one shared state.** Agents
  have independent positions/histories, but access the same inventory, items,
  gates, preparation progress and global action budget. Actions commit under a
  lock; delegation does not clone the environment.
- **Recipe execution → coordination-sensitive tasks.** Sample one of seven
  structures, which vary branch workloads, shared-resource constraints, access
  prerequisites and opportunities for nested delegation.
- **Single-instance score → a composite verifier.** The program scorer checks
  every required ingredient and dish in the final shared state. Missing work is
  incomplete; deleting an ingredient or applying an incompatible irreversible
  preparation can make the task fail. Root-task scoring is separate from any
  external LLM labels used for arbitrary subtasks during training.

### Seven task families

| Family | Construction / coordination requirement |
| --- | --- |
| `multi_dish_fork_join` | Independent dish branches join at overall completion. |
| `critical_path` | Deliberately unequal preparation complexity creates a long branch. |
| `load_balanced` | Heterogeneous branch work calls for balanced delegation. |
| `resource_constrained` | Shared inventory capacity and tools constrain execution. |
| `dependency_gate` | Selected dishes require opening an access gate first. |
| `conflict` | Branches contend for shared tools, with readiness/cooldown constraints. |
| `hierarchical` | Dish-level work contains ingredient-level subbranches; the agent chooses its delegation tree. |

These are instance structures and metadata, not mandatory agent call trees.
Their effects are implemented by the coordinator's actions and validity checks;
the benchmark does not simulate independent full games for different agents.

### Generation and difficulty bands

Generation uses deterministic seeds and difficulty-specific parameter sampling.
It reads the corresponding CookingWorld food/preparation fold (`train`,
`valid`, or `test`), samples compatible disjoint recipes and a task family, and
records the chosen parameters and recipes. Invalid parameter draws are resampled
deterministically while preserving the task's instance seed. The test seed
schedule starts at 820,000, with separate difficulty/family ranges.

| Difficulty | Dishes | Ingredients/dish | Locations | Distractors | Global action budget | Tool cooldown |
| --- | --- | --- | --- | --- | --- | --- |
| Easy | 2 | 3 | 5–6 | 3–5 | 115–125 | 0 |
| Medium | 2–4 | 3–4 | 7–10 | 6–12 | 95–100 | 0–1 |
| Hard | 4–5 | 4 | 10–11 | 12–18 | 75–90 | 1 |
| Extreme | 5 | 4 | 10–11 | 16–20 | 65–80 | 1 |

These are **generation bands**, not a guarantee that every combination occurs.
Recipe availability and family-specific overrides constrain valid draws.
Higher levels also require more nontrivial cutting/cooking operations and tighter
inventory/access constraints. Easy/Medium/Hard task descriptions are structured;
Extreme uses a coarser description. Inspect each task's `game_params` and
`generation_properties` for its exact recipe, sampled parameters, dependencies
and effective runtime settings.

The fixed test set contains **50 tasks in every family × difficulty cell**:
7 × 4 × 50 = 1,400 tasks. V9 retains the fixed V8 test instances and changes
training/validation quotas: 1,500 training tasks (900 Medium, 150 Hard, 450
Extreme) and 200 validation tasks (50 per level). The fixed test manifest,
training/validation splits, generation tools and source database are bundled
centrally in `Dataset/`. Evaluation reconstructs manifest tasks without new
sampling, so all checkpoints see identical task specifications.

## Setup

Use Linux and Python 3.12 (also compatible with Python 3.10+). The evaluator
requires only the Python standard library. Start an OpenAI-compatible model
server separately, with a context window of at least 13,312 tokens.

From the SERA repository root:

```bash
export OPENAI_API_KEY=EMPTY  # Replace if authentication is required.
```

Use the server's **exact served model ID**, without a LiteLLM `openai/` prefix.
The server should support `chat_template_kwargs.enable_thinking`. For models
without reasoning, create the plan with `--no-enable-reasoning`; if the serving
stack rejects chat-template options entirely, adapt the client for that stack.
Keep `Scripts/`, `Dataset/`, `Runtime/` and the entire `Evaluation/` directory
together, including `evaluation_common.py`. The shared composite environment,
task schema, HTTP client and evaluation prompt variant live in Runtime.

## Single machine

```bash
python Scripts/Eval/textworld/create_plan.py \
  --model qwen3-4b --num-shards 1 \
  --output-root Evaluation/outputs/textworld/run-01

python Scripts/Eval/textworld/evaluate_shard.py \
  --run-root Evaluation/outputs/textworld/run-01 --shard-index 0 \
  --base-url http://localhost:8000/v1 --concurrency 64

python Scripts/Eval/textworld/aggregate_results.py \
  --run-root Evaluation/outputs/textworld/run-01 --require-complete
```

Use a unique run directory for each checkpoint and independent repeat. Served
model IDs are recorded, but model weights are not automatically fingerprinted.

## Multiple machines

Create one plan on shared storage:

```bash
python Scripts/Eval/textworld/create_plan.py \
  --model qwen3-4b --num-shards 4 \
  --output-root /shared/evaluations/textworld/run-01
```

Run one worker on each machine. Set `SHARD_INDEX` to **0, 1, 2, or 3**:

```bash
export SHARD_INDEX=0
python Scripts/Eval/textworld/evaluate_shard.py \
  --run-root /shared/evaluations/textworld/run-01 --shard-index "$SHARD_INDEX" \
  --base-url http://localhost:8000/v1 --concurrency 64
```

Each shard contains 350 tasks, with balanced difficulty allocation and no
duplicates. `--num-shards` accepts 1 through 1,400; GPU counts and job schedulers
are not part of the evaluator. All endpoints must serve identical model weights
and compatible settings. Concurrency is per worker; size it for serving capacity.
Each machine needs the code and shared run directory. Without shared storage,
copy the complete plan/manifests to all machines, then merge their `shards/`
directories into one run root before aggregation.

## Shared environment semantics

Each task has exactly one environment coordinator. Agents have private locations
and histories, but **inventory, ingredients, preparations, meals, gates, and
the global environment-action budget are shared**. Delegation creates another
view of that same world, not a cloned world. Sibling agents may run concurrently;
individual environment actions are serialized by the coordinator lock. Program
success is computed from the final shared world state, not an LLM judge.

## Protocol, resume and reporting

Defaults: temperature 0; reasoning enabled; context 13,312; prompt/completion
caps 10,240/3,072; root/subagent budgets 20/20; maximum recursion depth 3;
subagents enabled. The global environment-action budget defaults to 100, but
the fixed task's `parallelism.shared_environment_max_steps` takes precedence
when present. Changing that planner option therefore does not override an
explicit task budget. These match the original evaluation protocol.

Use `create_plan.py --help` for protocol options, including
`--no-enable-subagents` for a single-agent ablation. Workers accept
`--concurrency` and `--task-retries` (default: 1 retry after an exception).
Re-run the same worker command to resume: completed valid tasks, including
ordinary failures, are retained; missing/corrupt/error records are retried.
A new independent evaluation requires a **new run directory**. Existing plans
cannot silently change model, shard count, dataset or protocol. Temperature 0
does not ensure bitwise-identical outputs across model-serving runs.

Task artifacts are written atomically to
`shards/<shard>/rollouts/<task_id>.json`. They include the recursive agent tree,
environment events, final environment state and terminal success/error status.
Aggregate at any time:

```bash
python Scripts/Eval/textworld/aggregate_results.py \
  --run-root /shared/evaluations/textworld/run-01
```

`aggregate_progress.json` contains completed/successful/failed/errored/pending
counts, per-difficulty rates and per-shard progress. **Errors count as failures**.
`success_rate` is successful/completed; `benchmark_success_rate` is
successful/all scheduled tasks. Rates are fractions, not percentages. Pending
tasks are listed explicitly. `final_report.json` is emitted only when all 1,400
tasks have terminal records; `--require-complete` exits 2 otherwise.

The overall rate is task-weighted; for the complete balanced V9 test set it also
equals the four-difficulty macro average. Per-shard legacy `accuracy` fields
exclude errors, so use the aggregate report for final benchmark numbers.
