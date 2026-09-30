# TextCraft-Synth evaluation

Evaluates a recursive CodeAct agent on all **632 fixed validation tasks**:
147 Easy, 213 Medium, 136 Hard and 136 Extreme. The manifest and matching task
payloads are included; synthetic recipes are regenerated deterministically.
The full-test runner lives in `textcraft_fulltest/`; the old 100-task
subsample module is not included.
Success is determined by the environment's root-task completion check, not by
an external LLM judge. The default is the original recursive, single-trajectory
evaluation; the separate 155-style single-Agent protocol is also supported.

## Setup

Use Linux and **Python 3.12**. From the SERA repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Evaluation/TextCraft/requirements.txt
export OPENAI_API_KEY=EMPTY  # Replace for authenticated servers; never commit credentials.
```

Start an OpenAI-compatible model server separately, with a context window of at
least 10,240 tokens. The evaluator sends Python code to an embedded IPython
executor: **model-generated code is not a security sandbox**. Run it in an
isolated container/account with no sensitive credentials or writable host data.

No installation of the original training repository is needed. The shared
`Runtime/vendor/platoon/` directory contains the execution core and prompts.
Keep `Scripts/`, `Evaluation/`, `Runtime/` and the root `Dataset/` directory together. The evaluator
reads `textcraft_*` manifests and task payloads from `Dataset/eval/`; no data is
stored under the runtime package. See [Dataset/README.md](../../Dataset/README.md).

## Single machine

Use the server's model name with LiteLLM's **`openai/` prefix**. For example, if
the API advertises `qwen3-4b`, use `openai/qwen3-4b`:

```bash
python Scripts/Eval/textcraft/create_plan.py \
  --model openai/qwen3-4b --num-shards 1 \
  --output-root Evaluation/outputs/textcraft/run-01

python Scripts/Eval/textcraft/evaluate_shard.py \
  --run-root Evaluation/outputs/textcraft/run-01 --shard-index 0 \
  --base-url http://localhost:8000/v1 --concurrency 64

python Scripts/Eval/textcraft/aggregate_results.py \
  --run-root Evaluation/outputs/textcraft/run-01 --require-complete
```

The served model ID is recorded, but weights are not fingerprinted. Use a
different run directory for each checkpoint and every independent repeat.

## Single-Agent protocol (source project 155)

The same 632-task manifest, TextCraft environment, model client, shard runner
and aggregation code are reused. Only the agent/prompt and rollout limits
change: use the nonrecursive `TextCraftAgent`/`create_synth_env`, 200 root steps
and depth zero; temperature 0 and the 10,240/9,728/512 context/prompt/completion
limits stay unchanged. No SubAgent action is exposed. With a local checkpoint:

```bash
bash Scripts/Eval/textcraft/run_single_agent.sh /path/to/hf-checkpoint
```

For an existing API or manual multi-machine plan, add `--single-agent` to
`Scripts/Eval/textcraft/create_plan.py`, then use the normal shard and aggregate
commands below. A recursive run directory cannot be reused for this protocol.

## Multiple machines

Create the plan **once**, on shared storage visible to all machines:

```bash
python Scripts/Eval/textcraft/create_plan.py \
  --model openai/qwen3-4b --num-shards 4 \
  --output-root /shared/evaluations/textcraft/run-01
```

Run this command on each machine, setting `SHARD_INDEX` to **0, 1, 2, or 3**:

```bash
export SHARD_INDEX=0
python Scripts/Eval/textcraft/evaluate_shard.py \
  --run-root /shared/evaluations/textcraft/run-01 --shard-index "$SHARD_INDEX" \
  --base-url http://localhost:8000/v1 --concurrency 64
```

The four shards contain 158 tasks each, stratified by difficulty, with no
duplicates or missing IDs. The number of machines/GPUs is not hardcoded: set
`--num-shards` to any integer from 1 to 632. Each worker can use a local server
or an accessible remote endpoint. Set concurrency according to serving capacity.
Every machine must have this code, dependencies and the shared run directory.
If storage is not shared, distribute the entire plan/manifests to each machine
and collect the resulting `shards/` directories into one run root to aggregate.

## Protocol, resume and results

Defaults preserve the original evaluation: temperature 0; context 10,240;
prompt/completion caps 9,728/512; root/subagent budgets 20/20; maximum recursive
depth 3; one trajectory per task. Budget/temperature options belong to
`create_plan.py` so all workers share the same protocol. Workers accept
`--concurrency` and `--task-retries` (default: 1 retry after an exception).

Re-run the same worker command to resume. Finished valid tasks, whether
successful or unsuccessful, are skipped; missing, corrupt or errored rollouts
are attempted again. To repeat all tasks independently, create a **new run
directory**. Changing the model, task partition or protocol in an existing run
is rejected. Temperature 0 is not a guarantee of bitwise reproducibility.

Each task saves `trajectory_collection.json`, `metadata.json`, and agent event
logs under `shards/<shard>/rollouts/<task>/rollout_0/`. A terminal record is
counted only after its artifacts have been written. You can aggregate while
workers are running:

```bash
python Scripts/Eval/textcraft/aggregate_results.py \
  --run-root /shared/evaluations/textcraft/run-01
```

`aggregate_progress.json` reports completed/successful/failed/errored/pending
counts overall and by difficulty. **Errors count as failures** in this aggregate.
`success_rate` is successful/completed, including errors in the denominator;
`benchmark_success_rate` is successful/all scheduled tasks (pending tasks are
not successes). Both are fractions; multiply by 100 for percentages.
`final_report.json` is created only when all 632 task IDs have terminal records;
`--require-complete` exits with status 2 if any are still missing.

The overall rate is **task-weighted**, not the unweighted average of the four
difficulty rates. Per-shard legacy summaries retain their `accuracy` field
(successful/valid, excluding errors); use the aggregate for benchmark reporting.
