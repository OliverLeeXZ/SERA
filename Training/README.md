# Training

Training recipes for TextCraft-Synth and TextWorld-Sync share one policy-version
stage kernel. Environment, agent, task-schema, and rubric utilities come from
[Runtime/](../Runtime/README.md); all datasets are read from
[Dataset/](../Dataset/README.md).

## Layout

```
Training/
  configs/textcraft.yaml       # shared TextCraft model/optimizer/rollout settings
  configs/textworld.yaml       # shared TextWorld model/optimizer/rollout settings
  sera_training/stage_kernel.py
  sera_training/stage_workflow.py
  sera_training/trainer.py     # environment/reward-specific workflow assembly
  runtime/                    # credit processors and AReaL integration
  launch.py                   # merge config + recipe overrides
  train.py                    # common worker entry point
```

Launch recipes live in `Scripts/Main/textcraft/` and `Scripts/Main/textworld/`.
Each environment has one shared YAML config; reward settings, stage lengths,
seed, and recipe-specific overrides live in its launch scripts.
`SERA_DATASET_ROOT` selects an alternative dataset root with the same layout.

## Recipes

Stage lengths count policy-version steps. All stages update the same
full-parameter policy.

| Paper method | Script (each environment) | Execution credit | Schedule |
| --- | --- | --- | --- |
| RAO | `rao.sh` | binary outcome | execution only |
| RAO with Decomposition Reward | `rao_with_decomposition_reward.sh` | binary outcome | execution 12 → delegation 4 |
| RAO with Rubric Training | `rao_with_rubric_training.sh` | binary outcome | execution 16 → rubric generation 2 |
| SERA without Decomposition Reward and Rubric Training | `sera_without_decomposition_reward_and_rubric_training.sh` | policy rubric score | execution only |
| SERA without Decomposition Reward | `sera_without_decomposition_reward.sh` | policy rubric score | execution 16 → rubric generation 2 |
| SERA | `sera.sh` | policy rubric score | execution 16 → delegation 2 → rubric generation 2 |

**SERA** means Self-Evaluating Recursive Agents and **RAO** means Recursive Agent
Optimization. The paper abbreviates Decomposition Reward as D and Rubric
Training as RT. Decomposition reward uses leaf coverage; rubric training
updates rubric-generation actions. Applying a rubric to a trajectory alone
does not train rubric generation.

The root always receives the environment's task-success reward. For binary
subagent execution credit, TextCraft uses program checks and TextWorld uses
an external judge. Rubric-based execution uses scores from the active policy.
The decomposition stage updates delegation-goal tokens using root-balanced
leaf coverage, without judging subagent trajectories or gating coverage on
subtask success.

Rubric-generation training samples eight counterfactual continuations per
subtask and uses a margin-ranking objective with margin 0.2. Only the
rubric-generation completion receives this credit; execution and scorer
completions are excluded from the rubric-generation loss. TextCraft labels
come from its program verifier; TextWorld labels come from the external judge.

### Rubric-design ablations

Four additional TextCraft recipes live in `Scripts/Ablation/textcraft/`,
covering Kimi-generated rubrics with policy or Kimi scoring, a global policy
rubric, and ranking plus discriminativeness training. See
[the ablation recipe table](../Scripts/README.md#ablation-training-recipes)
for launch scripts and endpoint configuration.

The ranking-plus-discriminativeness recipe alternates 16 execution steps with
two rubric-generation steps. Given rubric scores `s_i`, successful
continuations `P`, failed continuations `N`, and margin `m=0.2`:

```
R_order = mean_{i in P, j in N} clip((s_i - s_j) / m, 0, 1)
R_disc  = clip(2 * population_std(s), 0, 1)
R_group = 0.5 * R_order + 0.5 * R_disc
```

Groups containing only successes or only failures are skipped. A
leave-one-out baseline converts group rewards into rubric-generation
advantages. Branch 0 continues the live rollout; the other branches supply
counterfactual training data.

## Installation and cluster preparation

Use Python 3.12 and a CUDA-capable environment compatible with the AReaL,
PyTorch, and SGLang versions pinned in the root requirements file:

```bash
pip install -r requirements.txt
```

All nodes must use the same environment, repository path, and shared output
root. The default allocation is two nodes with eight GPUs each: eight SGLang
inference workers and eight FSDP trainer ranks
(`sglang:d8p1t1+d8p1t1`). Start a Ray cluster on those nodes and export
`RAY_ADDRESS` before launching. Configure network interfaces and transport
through AReaL's `launcher.*_env_vars` overrides.

For TextWorld recipes requiring an external judge, set `JUDGE_API_URL`,
`JUDGE_MODEL` (default `kimi-k2.6`), and `KIMI_API_KEY` in the environment
of every Ray trainer worker before starting Ray. Credentials are read at
runtime and are not written to resolved configs. The TextWorld
`sera_without_decomposition_reward_and_rubric_training.sh` recipe requires
no external judge.

Required external backends are validated and probed before training is
submitted, and again in each trainer process before GPU initialization. The
probe sends a small completion using the configured model and verifies the
binary-judge, rubric, or score schema as appropriate. Missing endpoint/model/key,
authentication failures, unavailable models, and malformed completions stop
startup; transient transport failures have at most one retry with a 20-second
timeout per request. Pure-policy methods send no external requests.
`--dry-run` checks configuration structure without API calls or credentials and
explicitly reports that availability has not been verified. External rubric
failures may drop a trajectory or skip a rollout, but cannot use
`fallback_binary` rewards.

## Launch, inspect, override, resume

Run from the repository root:

```bash
# Inspect settings without launching training.
bash Scripts/Main/textcraft/sera.sh --dry-run
bash Scripts/Main/textworld/sera.sh --dry-run

# Launch on an existing Ray cluster.
bash Scripts/Main/textcraft/sera.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-root /shared/sera/textcraft/run1 --run-name run1

bash Scripts/Main/textworld/sera.sh \
  --output-root /shared/sera/textworld/run1 --run-name run1

# Override settings; the last --set wins.
bash Scripts/Main/textcraft/sera.sh --dry-run \
  --set 'stage_schedule=execution:16,delegation:2,rubric_generation:2'
```

The default training limits are 250 TextCraft or 400 TextWorld optimizer
steps. Override them with `--set total_train_steps=...`; ten epochs provide
an additional stopping bound. For finite schedules, `stage_cycles` sets
the horizon to cycle length × cycle count. `--model` accepts a Hugging Face
model ID or a local checkpoint visible on all nodes; `PYTHON` selects the
interpreter. Absolute script paths work from other working directories.

To resume, repeat the same recipe, run name, output root, and model/config
settings. AReaL's `recover.mode=auto` restores the latest recovery state.
Stage routing resumes from the restored policy version. Checkpoints and
recovery state are saved every 50 steps; each invocation records its resolved
configuration.

## Stage kernel and outputs

`sera_training.stage_kernel.StageController` defines stage scheduling and
`SharedStageWorkflow` routes rollouts to the active objective. For SERA's
16:2:2 cycle, zero-based versions 0–15 train execution, 16–17 train
delegation, and 18–19 train rubric generation. Version 20 begins the next
cycle. Repeated rollout requests at one version do not advance the schedule.

Multi-stage recipes use `max_head_offpolicyness=0`; TextCraft's two
execution-only recipes use 3. Groups without informative rubric rankings
can produce skipped updates. `optimizer_steps.jsonl` records optimizer
updates and skips, while `stage_routes/version-*.json` records routing.
Rollouts, reward artifacts, checkpoints, recovery state, and statistics are
saved beneath the selected output root.

## Dataset and environment settings

TextCraft includes 2,522 training records and trains on the 852 Medium tasks.
The trainer constructs a 100-task validation subset from
`Dataset/validation/textcraft_synth_val.jsonl`; periodic TextCraft
validation is disabled by default. Final benchmark evaluation uses all 632
tasks through `Scripts/Eval/textcraft/`. TextWorld uses 1,500 training
tasks and a separate 200-task development split.

TextCraft uses context/prompt/completion budgets of 10,240/9,600/512;
TextWorld uses 13,312/10,240/3,072. Both use eight root samples per task,
batch size eight, depth three, 20 steps per agent, training temperature 1,
and CISPO with learning rate 3e-6. TextWorld's training simulator budget is
100 actions; evaluation uses task-specific action budgets where provided.
Training-time validation temperatures are 1 for TextCraft and 0 for TextWorld.

Rubric-generation training uses a counterfactual-environment budget of 16,
including the root. Complete groups of eight are reserved: a group started
below the cap can cross it, after which further calls use ordinary
subagents. This is a cumulative soft cap, not a limit on simultaneously live
environments. Ordinary subagents share inventory; counterfactual branches
clone the environment and only branch 0 is committed.

## Attribution

The vendored Platoon runtime includes its
[MIT license notice](../Runtime/licenses/platoon-MIT.txt).
