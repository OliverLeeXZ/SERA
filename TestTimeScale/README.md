# Bottom-up rubric-guided node selection on TextWorld-Sync

Portable test-time scaling on the same 1,400 fixed TextWorld-Sync V9 tasks as
single-trajectory evaluation. Data comes directly from `Dataset/eval/`; the
environment, API client, prompt and rubric helpers are imported from `Runtime/`;
immutable shard/aggregation utilities remain in `Evaluation/`. This module
does not import Training or require AReaL, Ray,
Java or a cluster scheduler. Launchers live in `Scripts/TestTimeScaling/`.

## Selection and environment semantics

At every delegation, clone the **entire** current environment into N independent
candidates. Complete each candidate's recursive subtree before scoring it.
Commit the selected candidate's complete state back into the parent coordinator
in place, including inventory, item/preparation state, gates, per-agent positions
and action budget. Discarded candidates never share mutable state with their
siblings. If one parent response delegates several subtasks, resolve them
**sequentially**: each subsequent subtask starts from the previous selected
branch's committed state, not the pre-delegation snapshot.

- **Policy judge (`selection_mode=rubric`):** generate one rubric per delegation from
  the parent context; the same policy scores all finished candidates using that
  frozen rubric. Choose the highest valid score; ties choose the first branch.
  An invalid rubric or all-invalid scores fall back to the first branch and
  record the reason. No external judge is instantiated or requested.
- **Kimi judge (`selection_mode=oracle`):** an external OpenAI-compatible LLM judges
  each completed candidate against the delegated subtask, returning binary
  success. Choose the first judged-successful candidate, or the first candidate
  with a recorded fallback if none passes. `oracle` is the historical mode name:
  this is **not** access to ground-truth root-task outcomes. It generates no
  policy rubric and makes no policy scorer requests.
- **No selection (`N=1`):** a single rollout without candidate selection, as
  reported in the paper's test-time-scaling table.
- Root success always comes from the environment program, not a judge score.
  `finish` and the selected recursive tree are retained in rollout files, along
  with the candidate groups, scores, outputs and fallback reasons.

Defaults: **N=2**, root temperature 0, candidate temperature 1, rubric generation
temperature 0, policy scoring temperature 0; root/child step limits 20, depth
limit 3; context 13,312, prompt budget 10,240, completion budget 3,072. This is
not the mixed-temperature 0/0.7 protocol: all candidate branches use the same
candidate temperature. Override it explicitly if needed.

The task-wide cumulative fork cap defaults to **32**. It counts every created
candidate coordinator, including discarded branches and nested calls, rather
than just live environments. Following the source evaluator, a delegation
that would exceed the cap creates no candidate: it returns a recorded
"complete this delegation locally" message so the parent can continue solving.
Reaching the cap does not itself mark the root task failed. Step/depth and
environment action budgets still apply.

## One-command launch

Install a compatible CUDA/PyTorch environment and serving dependencies:

```bash
pip install -r Evaluation/requirements-serving.txt
bash Scripts/TestTimeScaling/run_policy.sh /path/to/hf-checkpoint
```

The bash script can instead read `MODEL_PATH` (export it or fill it in at the
top of the script). A Hugging Face model ID also works. `PYTHON` selects the
interpreter. vLLM replicas start on the visible GPUs, one per GPU by default;
the launcher waits for the served model, evaluates through a local proxy,
aggregates, then stops only its own processes. `--gpus 0,1,2,3` restricts GPUs;
`--tensor-parallel-size 2` uses two GPUs per replica. Additional vLLM arguments
can be passed as individual `--server-arg=--flag` / `--server-arg=value` tokens.
Model code execution is opt-in with `--trust-remote-code`.

External judge settings must be available on every worker. Keys are read at
runtime and are not written into plans/configs:

```bash
export KIMI_JUDGE_ENDPOINT="https://your-openai-compatible-endpoint/v1"
export KIMI_JUDGE_MODEL="kimi-k2.6"
export KIMI_API_KEY="your-api-key"
bash Scripts/TestTimeScaling/run_kimi.sh /path/to/hf-checkpoint
```

`KIMI_BASE_URL` / `KIMI_MODEL` are fallback environment names. External judge
defaults are temperature 1, 1,024 completion tokens, 1,800s timeout and two
retries; its requests disable thinking. Another compatible external model can
be selected through the same variables. Policy mode needs none of them.

```bash
# No GPU imports, servers, network calls or writes.
bash Scripts/TestTimeScaling/run_policy.sh /path/to/hf-checkpoint --dry-run

# Reuse an existing policy service; --model-name must match its served ID.
bash Scripts/TestTimeScaling/run_policy.sh \
  --base-url http://localhost:8000/v1 --model-name qwen3-4b \
  --output-root /shared/sera/bestofn/run1
```

## Multi-machine sharding and resume

Run the following on four machines, changing only `--shard-index` to 0, 1, 2
or 3. All machines use the same model weights and a shared writable output root:

```bash
bash Scripts/TestTimeScaling/run_policy.sh /path/to/hf-checkpoint \
  --num-shards 4 --shard-index 0 --output-root /shared/sera/bestofn/run2
```

Seven or any other positive shard count works too. Assignment is deterministic,
non-overlapping and balanced by difficulty. The plan pins the model identity,
manifest and protocol (including N, temperatures, fork cap, selector and judge
settings); conflicting launches fail instead of mixing results. A separate
launcher/worker lock prevents duplicate ownership of a shard.

Repeat the identical command/output root to resume. Valid completed rollouts
are skipped; error records are retried. Outputs default to timestamped paths
under `TestTimeScale/outputs/` and are ignored by Git. Inspect progress with:

```bash
python Scripts/TestTimeScaling/aggregate_results.py \
  --run-root /shared/sera/bestofn/run2
```

Aggregates count error tasks as failures. Partial reports divide by completed
tasks, not pending tasks, and report expected counts separately. Final reports
require all 1,400 distinct task records; the overall score is task-weighted
(equivalent to the four-difficulty average here because each has 350 tasks).

Low-level `create_plan.py`, `evaluate_shard.py` and `aggregate_results.py`
support `--help`. They do not start servers, so they can be used independently
with existing endpoints. `create_plan.py` additionally exposes all root/child
budgets and external judge parameters.

## Provenance and tests

The recursive evaluator, selector and external judge are ported from the
internal TextWorld BestOfN evaluator (project 174). Private endpoints, scheduler
wrappers, retry-job scripts, credentials, logs and checkpoints are excluded.
The default N is 2 instead of the original source's generic default of 4.
Rubric prompt/parsing helpers are shared through Runtime, without importing
Training. The vendored Platoon runtime retains its
[MIT license notice](../Runtime/licenses/platoon-MIT.txt).

```bash
python -m unittest discover -s test/test_time_scaling -v
python -m unittest discover -s test/evaluation -v
```

These local-only tests are ignored by Git and use CPU/mock APIs, not a claim
that a real GPU checkpoint evaluation
has been run.
