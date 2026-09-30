# SERA

Official implementation of "Self-Evaluating Recursive Agents".

For training and local-GPU evaluation, use Python 3.12 and install the complete
environment from the repository root with `pip install -r requirements.txt`.
The smaller `Evaluation/requirements-serving.txt` is available when only local
evaluation serving is needed.

## Scripts

All evaluation and training launch scripts live in [Scripts/](Scripts/README.md):
`Eval/` for evaluation, `Main/` for the main training recipes, `Ablation/` for
rubric-design training ablations, and `TestTimeScaling/` for recursive Best-of-N.

## Dataset

All training, validation and evaluation data live in
[Dataset/](Dataset/README.md). `Dataset/generation/` includes a standalone
TextWorld generator that writes directly into those three splits.

## Runtime

[Runtime/](Runtime/README.md) is the common environment, Agent/trajectory,
prompt, rubric and ordinary inference core used by Training, Evaluation and
TestTimeScale. Training-only optimization/credit assignment and evaluation/
Best-of-N rollout policies remain in their respective directories.

## Evaluation

See [Evaluation/README.md](Evaluation/README.md) for portable TextCraft-Synth and
TextWorld-Sync evaluation, including recursive and 155/156 single-Agent
protocols, multi-machine sharding and resumable runs.

## Training

See [Training/README.md](Training/README.md) for the twelve TextCraft/TextWorld
main training recipes plus four TextCraft ablations, two shared environment
configs, and one policy-version stage kernel shared by RAO, decomposition
reward, and rubric-training experiments.
The [paper-to-recipe mapping](Training/README.md#recipes-and-source-correspondence)
uses the main-table names RAO, RAO + D, RAO + RT, SERA w/o D & RT,
SERA w/o D, and SERA; existing script names remain stable.

## Test-time scaling

See [TestTimeScale/README.md](TestTimeScale/README.md) for recursive TextWorld
Best-of-N with the paper's Policy judge or Kimi judge selectors. Evaluation
and test-time scaling both provide model-path-only bash launchers under Scripts.

## Local tests

Development tests live under `test/{dataset,evaluation,runtime,test_time_scaling,training}/`
in the local checkout. The root `.gitignore` excludes `test/`, so this suite
is not included in the published repository.
