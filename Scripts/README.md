# Scripts

Launch entry points live here; implementations and shared configs live in
`Training/`, `Evaluation/` and `TestTimeScale/`, backed by the common `Runtime/`.
Data is read from the root `Dataset/` directory.

```
Scripts/
  Download/download.py         # public TextCraft/TextWorld paper checkpoints
  Eval/
    evaluate_textcraft_ckpt.sh # evaluate downloaded TextCraft-step250
    evaluate_textworld_ckpt.sh # evaluate downloaded TextWorld-step400
    textcraft/{run.sh,run_single_agent.sh,create_plan.py,evaluate_shard.py,aggregate_results.py}
    textworld/{run.sh,run_single_agent.sh,create_plan.py,evaluate_shard.py,aggregate_results.py}
  Main/
    textcraft/*.sh             # six main training recipes
    textworld/*.sh             # six main training recipes
  Ablation/
    textcraft/*.sh             # four rubric-design training ablations
  TestTimeScaling/{run_policy.sh,run_kimi.sh,create_plan.py,evaluate_shard.py,aggregate_results.py}
```

## Evaluation

Download the two public paper checkpoints and evaluate them directly:

```bash
python Scripts/Download/download.py
bash Scripts/Eval/evaluate_textcraft_ckpt.sh
bash Scripts/Eval/evaluate_textworld_ckpt.sh
```

The checkpoints are saved under `Evaluation/ckpt/TextCraft-step250` and
`Evaluation/ckpt/TextWorld-step400` (ignored by Git). Download just one with
`--checkpoint textcraft` or `--checkpoint textworld`; use `--dry-run` to inspect
destinations without network access. The evaluation scripts accept the same
options as `Evaluation/run.py`, including `--dry-run`, `--gpus`, `--output-root`,
`--num-shards` and `--shard-index`. For a multi-machine run, download the
checkpoint on each machine, set the same shared `--output-root`, and use a
different `--shard-index` per machine.

The simplest GPU launch needs only your HF-format model directory (or HF model
ID). Install `Evaluation/requirements-serving.txt` first:

```bash
bash Scripts/Eval/textcraft/run.sh /path/to/hf-checkpoint
bash Scripts/Eval/textworld/run.sh /path/to/hf-checkpoint

# Single-agent baselines on the same test sets.
bash Scripts/Eval/textcraft/run_single_agent.sh /path/to/hf-checkpoint
bash Scripts/Eval/textworld/run_single_agent.sh /path/to/hf-checkpoint
```

Alternatively, fill in/export `MODEL_PATH` and run the bash file without
arguments. These launch local vLLM replicas, wait for readiness, evaluate and
aggregate, then stop only the servers they started. `--dry-run` inspects without
launching anything. All visible GPUs are used by default; `--gpus` and
`--tensor-parallel-size` control replicas/TP. Multi-machine launches use a shared
`--output-root`, `--num-shards` and a distinct `--shard-index` per machine.
See [one-command evaluation](../Evaluation/README.md#evaluate-with-only-a-model-path).

For an already-running OpenAI-compatible API, low-level commands remain:

```bash
python Scripts/Eval/textcraft/create_plan.py \
  --model openai/Qwen3-4B-Instruct --num-shards 4 \
  --output-root /shared/sera/evaluation/textcraft/run1
python Scripts/Eval/textcraft/aggregate_results.py \
  --run-root /shared/sera/evaluation/textcraft/run1
```

`evaluate_shard.py` runs each shard against your existing model endpoint.
Use `create_plan.py --single-agent` when manually planning a single-Agent
multi-machine run. See [TextCraft evaluation](../Evaluation/TextCraft/README.md) and
[TextWorld evaluation](../Evaluation/TextWorld/README.md) for worker commands,
multi-machine sharding and resume support. All six entry points support `--help`.

## Test-time scaling

```bash
# N=2, paper's Policy judge: policy-generated rubric + policy scorer; no Kimi requests.
bash Scripts/TestTimeScaling/run_policy.sh /path/to/hf-checkpoint

# N=2, paper's Kimi judge: external binary judge. Set endpoint/model/key first.
bash Scripts/TestTimeScaling/run_kimi.sh /path/to/hf-checkpoint
```

Both support the same model-path, local serving, existing API, multi-machine
sharding and resume options as evaluation. The implementation is in
[TestTimeScale/](../TestTimeScale/README.md); logs/results are not bundled.

## Main training recipes

```bash
bash Scripts/Main/textcraft/sera.sh --dry-run
bash Scripts/Main/textworld/sera.sh --dry-run
```

Each environment contains the same six main training recipes. Filenames and
default `--experiment` identifiers correspond to the paper methods:

| Paper method | Launch file in `textcraft/` or `textworld/` |
|---|---|
| RAO | `rao.sh` |
| RAO with Decomposition Reward | `rao_with_decomposition_reward.sh` |
| RAO with Rubric Training | `rao_with_rubric_training.sh` |
| SERA without Decomposition Reward and Rubric Training | `sera_without_decomposition_reward_and_rubric_training.sh` |
| SERA without Decomposition Reward | `sera_without_decomposition_reward.sh` |
| SERA | `sera.sh` |

In the paper, **SERA** expands to Self-Evaluating Recursive Agents, **RAO** to
Recursive Agent Optimization, **D** to Decomposition Reward, and **RT** to
Rubric Training. Scoring execution with a rubric alone is not rubric training.
To resume, repeat the recipe with the same output root and run name. See
[Training/README.md](../Training/README.md) for dependencies, cluster setup and
launch/resume instructions. `PYTHON=/path/to/python` selects the interpreter.

The commands above assume the repository root as the working directory. Using
absolute script paths works from any directory; paths to the shared runtime
are resolved relative to each script, not your current working directory.

## Ablation training recipes

| Script under `Ablation/textcraft/` | Generator | Scorer | Training |
| --- | --- | --- | --- |
| `kimi_rubric_policy_scorer.sh` | Kimi | Policy | None |
| `kimi_rubric_kimi_scorer.sh` | Kimi | Kimi | None |
| `global_rubric_policy_scorer.sh` | Policy (global) | Policy | None |
| `rubric_training_discriminativeness.sh` | Policy | Policy | Ranking + Discriminativeness |

The main recipe `sera_without_decomposition_reward.sh`
is the paper's Policy / Policy / Ranking ablation row. The first three rows
above do not train rubric generation; Ranking + Discriminativeness trains it
with ranking plus discriminativeness credit.

All four use `Training/launch.py`, the shared TextCraft config and the same
policy-version stage kernel as Main. The global template is bundled at
`Training/assets/textcraft_global_rubric.json`; it is not generated dynamically
and needs no external model. The discriminativeness objective uses success/failure ordering
plus population score dispersion; scorer completions remain evaluation-only.

```bash
# Every recipe supports CPU-only inspection without credentials or network calls.
bash Scripts/Ablation/textcraft/kimi_rubric_policy_scorer.sh --dry-run
bash Scripts/Ablation/textcraft/kimi_rubric_kimi_scorer.sh --dry-run
bash Scripts/Ablation/textcraft/global_rubric_policy_scorer.sh --dry-run
bash Scripts/Ablation/textcraft/rubric_training_discriminativeness.sh --dry-run

# For actual Kimi-backed training, export these on EVERY trainer worker before Ray starts.
export KIMI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export KIMI_MODEL="kimi-k2.6"
export KIMI_API_KEY="your-api-key"
bash Scripts/Ablation/textcraft/kimi_rubric_policy_scorer.sh \
  --output-root /shared/sera/textcraft-kimi-policy/run1 --run-name run1
```

`JUDGE_API_URL` is an endpoint fallback when `KIMI_BASE_URL` is not set.
Endpoint/model can also be selected using the config override mechanism
(the environment model override takes precedence). Credentials are only read
from environment variables, never from script literals or resolved YAML files.
The global and discriminativeness recipes are entirely policy/program-backed:
they require no Kimi endpoint or key. See the Training README for cluster,
model-path, and resume configuration.
