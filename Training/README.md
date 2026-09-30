# Training

Portable main and ablation training recipes for TextCraft-Synth and TextWorld-Sync. No scheduler
submission scripts, run outputs, credentials, or checkpoints are included.
The environment/agent, task schemas and rubric helpers come from
[Runtime/](../Runtime/README.md), not Evaluation. Keep `Scripts/`, `Training/`,
`Runtime/` and `Dataset/` together; Evaluation is not required to load training.
Training does not depend on an R3AO checkout.

## Layout

```
Training/
  configs/textcraft.yaml       # shared TextCraft model/optimizer/rollout settings
  configs/textworld.yaml       # shared TextWorld model/optimizer/rollout settings
  sera_training/stage_kernel.py
  sera_training/stage_workflow.py
  sera_training/trainer.py     # environment/reward-specific workflow assembly
  runtime/                    # training-only credit processors and AReaL integration
  launch.py                   # merge config + script overrides; launch or dry-run
  train.py                    # common worker entry point for every recipe
```

Launch recipes live in `../Scripts/Main/textcraft/` and
`../Scripts/Main/textworld/` (six per environment); there is no local scripts
directory. See [Scripts/README.md](../Scripts/README.md) for all launch entries.

There are exactly two shared YAML configs. Reward, stage lengths, seed, and
experiment-specific client/leaf/ranking settings live in each launch script.
All recipes share one trainer; no per-experiment trainer copies or YAMLs are used.
Both training and evaluation read the root [Dataset/](../Dataset/README.md).
No dataset copies remain inside Training or Evaluation. `SERA_DATASET_ROOT`
can select another directory with the same split layout.

## Recipes and source correspondence

Stage lengths are **policy-version steps**, not task counts or number of agents.
The same full-parameter policy is used throughout each schedule.

| Paper method | Script (each environment) | Execution credit | Implementation schedule | TextCraft training source | TextWorld training source |
|---|---|---|---|---|---|
| RAO | `rao.sh` | native binary RAO | execution only | 142 | 85 |
| RAO + D | `rao_leaf.sh` | native binary RAO | execution 12 → delegation 4 | 103 | 86 |
| RAO + RT | `rao_rubric_training.sh` | native binary RAO | execution 16 → rubric generation 2 | 171 | 172 |
| SERA w/o D & RT | `rubric_reward.sh` | policy rubric score | execution only | 143 | 82 |
| SERA w/o D | `rubric_reward_training.sh` | policy rubric score | execution 16 → rubric generation 2 | 87 | 91 |
| SERA | `three_stage.sh` | policy rubric score | execution 16 → delegation 2 → rubric generation 2 | 101 | 102 |

Here **D** is Decomposition reward (leaf coverage), and **RT** is Rubric
Training (training the rubric-generation action). Scoring execution with a
rubric alone is not RT. The TextCraft RAO and SERA w/o D & RT entries above
name their training sources (142 and 143); the paper's evaluated-checkpoint
project IDs are 110 and 3, respectively. Script names and `--experiment`
identifiers are retained for existing runs and resume compatibility.
Four additional TextCraft training recipes live in `../Scripts/Ablation/textcraft/`.
They follow the paper's ablation axes **Generator / Scorer / Training**:
Kimi / Policy / -- (159), Kimi / Kimi / -- (145), Policy (global) / Policy / --
(4), and Policy / Policy / Rank+Disc. (106). See
[Scripts/README.md](../Scripts/README.md#ablation-training-recipes)
for the script-to-experiment mapping and external endpoint setup. No experiment
196 exists in the source snapshot; the Rank+Disc. table entry corresponds to 106.

The root execution trajectory always keeps the executable task-success reward.
For binary subagent credit, TextCraft uses environment/program checks; TextWorld
uses an external OpenAI-compatible binary judge. Rubric execution credit is
generated and scored by the active policy snapshot, not an external judge.
The delegation objective uses root-balanced leaf credit without a subagent
success gate and trains only launch/delegate token spans. Rubric-generation
training uses fork-8 counterfactual continuations, margin 0.2, and the original
ranking objective; only the rubric-generation completion receives this credit.
TextCraft ranking labels are programmatic; TextWorld ranking labels use the
external judge. TextWorld's delegation/leaf stage does not judge subagent
trajectories: RAO execution still uses binary subagent judgments, while rubric
execution uses policy scores and rubric-generation training uses judge labels.

For the 106 ablation, execution uses policy rubric scores for 16 steps, followed
by two rubric-generation steps. For each mixed-label fork group with scores
`s_i`, success set `P`, failure set `N`, and margin `m=0.2`:

```
R_order = mean_{i in P, j in N} clip((s_i - s_j) / m, 0, 1)
R_disc  = clip(2 * population_std(s), 0, 1)
R_group = 0.5 * R_order + 0.5 * R_disc
```

This is the source 106 combined objective, not the unchanged negative-hinge
ranking objective used by the main rubric-training recipes. Groups with only
successes or only failures are skipped. Leave-one-out group advantage trains
one rubric-generation completion per call node; the eight scorer completions
and environment execution traces are not trained in this stage. Forking,
branch-0 commit and stage routing reuse the existing shared implementations.

## Installation and cluster preparation

Use Python 3.12 and a CUDA-capable environment compatible with the pinned AReaL
revision. From the repository root, run `uv pip install -r requirements.txt`.
The AReaL revision is the source runs' declared dependency, not an arbitrary
latest release. Its CUDA/SGLang installation may need your site's toolchain.
All nodes must use the same environment, repository path, and shared output root.

The default allocation is two nodes with eight GPUs each: eight SGLang inference
workers and eight FSDP trainer ranks (`sglang:d8p1t1+d8p1t1`). Start a Ray cluster
on those nodes before invoking a recipe and export `RAY_ADDRESS` on the launcher.
These scripts neither submit rjobs nor start/stop a Ray cluster. Configure your
own network interfaces/transport via AReaL `launcher.*_env_vars` overrides; no
cluster-specific network interface or private API address is hardcoded here.

For TextWorld recipes requiring an external judge, set `JUDGE_API_URL`,
`JUDGE_MODEL` (optional, default `kimi-k2.6`), and `KIMI_API_KEY` in the environment
of **every Ray trainer worker** (e.g. before starting Ray on each node). API keys
are never written to resolved configs. `textworld/rubric_reward.sh` is entirely
policy-backed and does **not** require or request Kimi.

## Launch, inspect, override, resume

Run from any working directory, using the Python environment above:

```bash
# CPU-only inspection: no Ray, GPU, model download, or external judge call.
bash Scripts/Main/textcraft/three_stage.sh --dry-run
bash Scripts/Main/textworld/three_stage.sh --dry-run

# Launch on your existing Ray cluster; paths must be shared across all nodes.
bash Scripts/Main/textcraft/three_stage.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-root /shared/sera/textcraft-three-stage/run1 \
  --run-name run1 --set total_train_steps=250

bash Scripts/Main/textworld/rao.sh \
  --output-root /shared/sera/textworld-rao/run1 \
  --run-name run1 --set total_train_steps=400

# Any recipe setting can be overridden; the final --set wins.
bash Scripts/Main/textcraft/three_stage.sh --dry-run \
  --set 'stage_schedule=execution:16,delegation:2,rubric_generation:2'
```

Examples above are relative to the repository root. From another working
directory, use an absolute script path (or `../Scripts/Main/...` from `Training/`).
`PYTHON=/path/to/python` selects the interpreter used by a shell recipe. The
default model is the public model ID; `--model` can select a local checkpoint.
The shared configs default to 250 TextCraft or 400 TextWorld optimizer steps,
matching the paper's evaluated checkpoints. `--set total_train_steps=...` can
change this limit; ten epochs remain a safety bound. For finite cycles,
`stage_cycles` defines the horizon as cycle length × cycle count instead.

To resume, reuse the same experiment script, `--run-name`, and `--output-root`.
AReaL `recover.mode=auto` restores the latest recovery state. The shared kernel
routes from `engine.get_version()` after restore, not from a local task counter.
Do not change the schedule/reward recipe during resume. Each invocation saves
its resolved config; checkpoints/recovery default to every 50 steps.

## Stage kernel and artifacts

`sera_training.stage_kernel.StageController` is the sole scheduling algorithm.
It consolidates the original three-stage, RAO/leaf two-stage, rubric two-stage,
and alternating-cycle variants. Legacy processor imports of `three_stage_train.schedule`
are a facade to this same kernel. Processor wrappers contain a fixed single-stage
schedule; only the outer `SharedStageWorkflow` switches objectives.

For 16+2+2, zero-based versions 0–15 execute, 16–17 train delegation, and 18–19
train rubric generation; version 20 starts the next cycle. Repeated rollout
requests at one version do not advance the schedule. Multi-stage runs preserve
`max_head_offpolicyness=0`; the two TextCraft execution-only recipes retain 3.
Zero-gradient rubric groups keep the source valid/no-op handling rather than
turning into replacement-task prefetch. Policy version need not equal the count
of nonzero-gradient optimizer updates; `optimizer_steps.jsonl` records updates
and skips explicitly.

`stage_routes/version-*.json` records the outer routing decision.
`rollouts/`, leaf/ranking/rubric artifact directories, checkpoints, recovery
state, stats, and `optimizer_steps.jsonl` are written under your output root.

## Dataset and environment protocol

`Dataset/training/textcraft_synth_train.jsonl` bundles all 2,522 training records and filters to the 852 medium tasks,
matching the source recipes. The trainer constructs a 100-task TextCraft validation
subset from `Dataset/validation/textcraft_synth_val.jsonl`, but the TextCraft
evaluator has no frequency configured, so it never runs periodic validation.
The reported held-out benchmark uses all 632 tasks via `Scripts/Eval/textcraft/`.
TextWorld reads V9 train (1,500) and dev (200) from `Dataset/training/textworld_train.jsonl`
and `Dataset/validation/textworld_dev.jsonl`; see
[Evaluation/README.md](../Evaluation/README.md) for generation details.

TextCraft uses context 10,240, prompt budget 9,600, completion budget 512.
TextWorld uses context 13,312, prompt budget 10,240, completion budget 3,072.
Both use eight root samples per task, batch size eight, depth three, 20 steps per
agent, temperature 1 during training, and CISPO with learning rate 3e-6.
TextWorld training retains its source global simulator budget (100); the held-out
evaluation protocol can override this with task-specific budgets. Training-time
dev sampling retains source behavior: TextCraft temperature 1; TextWorld 0.

The recipes preserve effective launch overrides, not stale YAML defaults:
notably ranking uses a budget of 16 counterfactual environments per root rollout
(including the root), rather than TextWorld's unused YAML default of 96. The
source budget reserves complete fork-8 groups: a group started below the cap can
cross it, after which calls fall back to ordinary single subagents. This preserves
the original soft-cap behavior; it is not a strict simultaneous-live-env limit.
Normal subagents
share the live inventory; ranking counterfactuals clone it and commit only branch 0.
Changing objectives/stage boundaries does not change the environment-sharing rule.

## Validation and attribution

```bash
python -m unittest discover -s test/training -v
python -m unittest discover -s test/evaluation -v
```

These development tests are local-only and ignored by Git. CPU tests cover
every recipe/config, scheduling boundaries/resume, launcher
overrides, and inventory isolation/commit. With the pinned AReaL dependency,
they also validate typed configs and construct all sixteen real workflow graphs
using a mock proxy; the policy-only TextWorld recipe is checked to never
initialize an external judge. Actual distributed GPU training must
be smoke-tested on your cluster. This extraction does not claim a GPU training
run was performed. The vendored Platoon runtime retains its
[MIT license notice](../Runtime/licenses/platoon-MIT.txt);
review dataset/model redistribution rights before publishing the repository.
