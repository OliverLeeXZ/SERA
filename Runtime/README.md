# Shared Runtime

Runtime provides the environments, recursive-agent execution, prompts, rubrics,
and inference utilities shared by Training, Evaluation, and TestTimeScale.
It has no dependency on those application modules or on AReaL, Ray, or
PyTorch. TextWorld and rubric utilities use the Python standard library;
TextCraft requires the CodeAct/IPython/LiteLLM dependencies.

## Layout

```
Runtime/
  vendor/platoon/
    agents/                  # recursive CodeAct agents and action/context handling
    episode/                 # episode loop, trajectory tree, per-agent budgets
    envs/                    # CodeAct executor and base environment types
    textcraft/               # crafting environment, recipes, tasks, and agent
    inference/               # inference workflow
    utils/, visualization/   # inference helpers and trajectory event sinks
  environments/textworld/
    composite_cooking.py     # shared cooking world and agent-local views
    composite_scorer.py      # root-task programmatic success check
    manifest.py              # task schema, loader, and sharding utilities
  agents/textworld_protocol.py # action/delegate/finish parsing and types
  prompts/
    textworld_training.py    # training prompt builder
    textworld_evaluation.py  # evaluation and Best-of-N prompt builder
  rubric/
    prompts.py               # rubric-generation, scoring, and teacher prompts
    scoring.py               # structured rubric/score parsing
  clients/openai_chat.py     # HTTP client and completion types
  clients/external_model.py # required-model validation and startup API probes
  bootstrap.py               # initialize the vendored platoon namespace
  requirements-textcraft.txt
  licenses/platoon-MIT.txt
```

TextCraft uses the `platoon` namespace for agent classes, recipes, prompts,
and execution contexts. Training extends it with optimization and
training-specific utilities.

## Using the shared code

Launch scripts initialize import paths automatically. For direct imports from
the repository root:

```python
from Runtime.environments.textworld import CompositeCookingWorldCoordinator, TaskSpec
from Runtime.prompts.textworld_evaluation import TextWorldPromptBuilder
from Runtime.rubric.scoring import parse_rubric, parse_policy_score

from Runtime.bootstrap import bootstrap
bootstrap()
from platoon.textcraft.env import create_synth_depth_aware_env
```

Install TextCraft dependencies with Python 3.12:

```bash
pip install -r Runtime/requirements-textcraft.txt
```

Training and serving requirements include this dependency list. TextWorld and
rubric-only imports do not require the optional TextCraft packages. Task data
is read from the root `Dataset/` directory. The TextCraft executor runs
model-generated Python; use an isolated execution environment.

## Application responsibilities

| Module | Responsibilities |
| --- | --- |
| Training | Optimization, stage routing, reward assignment, trainable token masks, and counterfactual training branches |
| Evaluation | Model serving, shard ownership, rollout orchestration, resume, and result aggregation |
| TestTimeScale | Candidate recursion, fork budgets, selection, branch-state commit, and selection artifacts |

Training and evaluation have separate TextWorld prompt builders for their
respective instructions and budgets. They share the action protocol and
environment implementation.

Ordinary recursive delegation creates views of a shared environment.
Counterfactual and Best-of-N candidates use isolated copies. Application
workflows decide which branch to commit, how to order multiple delegations,
and how to handle fork limits.

## Attribution

The vendored Platoon runtime includes its
[MIT license notice](licenses/platoon-MIT.txt).
