# Evaluation

Portable evaluation for recursive and single-Agent policies on two fixed benchmarks:

| Benchmark | Tasks | Difficulty counts (Easy / Medium / Hard / Extreme) | Runtime |
| --- | ---: | --- | --- |
| [TextCraft-Synth](TextCraft/README.md) | 632 | 147 / 213 / 136 / 136 | Python CodeAct + synthetic crafting recipes |
| [TextWorld-Sync V9](TextWorld/README.md) | 1400 | 350 / 350 / 350 / 350 | Pure-Python shared cooking world |

Both evaluators use an OpenAI-compatible chat-completions API. The low-level
workers connect to an existing server. Optional `Scripts/Eval/*/run.sh` and
`run_single_agent.sh` launchers
start and clean up local vLLM replicas automatically; no cluster scheduler is
required, and the scripts never request or submit GPU jobs.
This directory includes evaluation-specific orchestration; common environments,
Agents/trajectories, prompts and inference helpers live in
[Runtime/](../Runtime/README.md). Keep Runtime alongside Evaluation and Dataset.
it does not include training code, checkpoints, API credentials, job-submission
scripts, logs or previously generated results.
Fixed datasets live centrally in [Dataset/](../Dataset/README.md). Default
evaluation plans read `textcraft_*` or `textworld_*` files directly from `Dataset/eval/`.
Set `SERA_DATASET_ROOT` to select another root with the same layout. Training
uses that root's `training/` and `validation/` folders. The TextWorld generator
in `Dataset/generation/` writes these files directly; no copying is required.

## How TextWorld-Sync is constructed

TextWorld-Sync extends the **CookingWorld task design and recipe data in
TextWorldExpress**. Instead of solving one recipe with a single avatar, each
instance contains several dishes in one shared world. The benchmark samples
valid food/preparation combinations from the original CookingWorld database,
assigns disjoint ingredients to dishes, and adds coordination structures such
as unequal branch workloads, dependency gates and shared-resource constraints.

The central change is **shared state with agent-local contexts**: recursive
agents share ingredients, inventory, tools, doors and dish progress, but keep
their own logical positions and conversation histories. Model requests may run
in parallel; state transitions are locked and applied atomically. The final
programmatic check requires every ingredient to be collected and processed
correctly and every dish to be prepared and consumed. Partial subtask progress
does not count as root-task success.

Task generation uses seeded sampling within four difficulty-specific parameter
bands, validates recipe availability, and records the concrete parameters in a
fixed manifest. The test set crosses **seven task families × four difficulty
levels × 50 instances = 1,400 tasks**. V9 retains the preceding fixed test set;
its training/validation change concerns sampling quotas, not a new test set.
This evaluation release reconstructs those fixed tasks using its pure-Python
composite-world implementation; it does not call the original JVM simulator or
regenerate test tasks at evaluation time.

See [TextWorld/README.md](TextWorld/README.md#dataset-construction-and-changes-from-textworldexpress)
for the seven families, difficulty bands and generation details.

Launch scripts live in `../Scripts/Eval/textcraft/` and
`../Scripts/Eval/textworld/`; no script directories remain in Evaluation.
See [Scripts/README.md](../Scripts/README.md) for their locations.
Each benchmark has three low-level entry points: `create_plan.py`, `evaluate_shard.py` and
`aggregate_results.py`. A single shard runs on one machine; multiple shards can
run concurrently on different machines. Use one shared writable run directory
and one worker per shard. Each worker can point to its own model-server URL,
provided all endpoints serve the same model weights and settings.

Plans use relative shard/output paths and can be moved as a unit. Run identities
and manifest hashes guard against mixing models, protocols or task partitions;
file locks prevent accidentally running the same shard twice. On Linux/shared
storage, filesystem `flock` and atomic rename must be supported.

Run-independent tests (no GPU or model server required):

```bash
python -m unittest discover -s test/evaluation -v
```

The local-only test suite is excluded by the root `.gitignore`. Tests that
exercise the TextCraft runtime additionally require its dependencies
and Python 3.12. These are local/mock-server tests, not full model evaluations.

The vendored Platoon runtime retains its
[MIT license notice](../Runtime/licenses/platoon-MIT.txt).

## Evaluate with only a model path

Install the serving dependencies into a Python 3.12 environment with a
compatible CUDA/PyTorch stack:

```bash
pip install -r Evaluation/requirements-serving.txt
bash Scripts/Eval/textcraft/run.sh /path/to/hf-checkpoint
bash Scripts/Eval/textworld/run.sh /path/to/hf-checkpoint
```

Use a Hugging Face-format directory (model weights, config and tokenizer), not
an optimizer/FSDP recovery directory. A Hugging Face model ID also works. You
can instead fill in `MODEL_PATH` at the top of either bash file, or export it,
then run that file without arguments. `PYTHON=/path/to/python` selects Python.
HF authentication, if necessary, is handled by your installed Hugging Face tools.

The launcher uses all visible GPUs, one independent replica per GPU by default,
waits for `/v1/models` readiness, runs evaluation through a local round-robin
proxy, saves each rollout atomically and writes the aggregate report. It stops
only its own server processes on exit. Occupied server ports are an error;
unrelated services are neither reused nor killed. Logs and caches live under
the run's `services/` directory. Results default to a timestamped directory
under `Evaluation/outputs/` (ignored by Git).

```bash
# Inspect settings without GPU access, downloads, network calls or file writes.
bash Scripts/Eval/textworld/run.sh /path/to/hf-checkpoint --dry-run

# Restrict GPUs; use TP for models that cannot fit on one GPU.
bash Scripts/Eval/textworld/run.sh /path/to/hf-checkpoint \
  --gpus 0,1,2,3 --tensor-parallel-size 2 --output-root /shared/eval/run1

# Connect to an existing server instead: --model-name must match its served ID.
bash Scripts/Eval/textworld/run.sh --base-url http://localhost:8000/v1 \
  --model-name qwen3-4b --output-root /shared/eval/run2
```

For four machines, run the same command on each with the same checkpoint,
shared `--output-root` and `--num-shards 4`, setting `--shard-index` to 0, 1,
2 or 3 respectively. Each machine uses its own local GPUs. Repeat the same
command/output root to resume: valid completed rollouts are skipped, while
error rollouts are retried. Model path, served ID, protocol and partition are
pinned to prevent mixing different runs. `aggregate_results.py` can inspect
progress while workers run; a final report is produced only after all expected
task files are present. The one-command launcher also aggregates on exit.
