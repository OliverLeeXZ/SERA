# Shared Runtime

The common execution code used by Training, Evaluation and TestTimeScale lives
here. This is a code-location refactor, not a new rollout protocol. Runtime
does not import any of those application directories, AReaL, Ray or PyTorch.
TextWorld and rubric utilities use the Python standard library; TextCraft has
optional CodeAct/IPython/LiteLLM dependencies.

```
Runtime/
  vendor/platoon/
    agents/                  # recursive CodeAct agents and action/context handling
    episode/                 # episode loop, trajectory tree, per-agent budgets
    envs/                    # CodeAct executor and base environment types
    textcraft/               # crafting environment, recipe/task loaders and agent
    inference/               # ordinary inference workflow
    utils/, visualization/   # inference helpers and trajectory event sinks
  environments/textworld/
    composite_cooking.py     # canonical shared cooking world and agent views
    composite_scorer.py      # root-task programmatic success check
    manifest.py              # one task schema/loader, plus existing shard utilities
  agents/textworld_protocol.py # shared action/delegate/finish parsing and types
  prompts/
    textworld_training.py    # preserved historical training prompt variant
    textworld_evaluation.py  # preserved historical evaluation/Best-of-N variant
  rubric/
    prompts.py               # rubric generation/scoring and teacher prompts
    scoring.py               # JSON parsers and existing score-agreement helpers
  clients/openai_chat.py     # unchanged ordinary HTTP client and Completion type
  bootstrap.py               # expose the vendored `platoon` namespace
  requirements-textcraft.txt
  licenses/platoon-MIT.txt
```

TextCraft remains under its upstream `platoon` namespace to preserve relative
imports, prompt/recipe resources and shared context variables. Its actual code
has one home under Runtime; there is no second environment copy in Training or
Evaluation. Training adds only its existing training-specific `platoon.train`,
`platoon.utils` and `platoon.textcraft` extensions. The training bootstrap also
handles the case where an ordinary evaluator imported Platoon earlier in the
same process, without creating duplicate classes or context variables.

## Using the shared code

Existing launch scripts initialize paths automatically. For direct imports from
the repository root:

```python
from Runtime.environments.textworld import CompositeCookingWorldCoordinator, TaskSpec
from Runtime.prompts.textworld_evaluation import TextWorldPromptBuilder
from Runtime.rubric.scoring import parse_rubric, parse_policy_score

# Only needed when directly importing the upstream TextCraft namespace.
from Runtime.bootstrap import bootstrap
bootstrap()
from platoon.textcraft.env import create_synth_depth_aware_env
```

Install TextCraft dependencies with Python 3.12:

```bash
pip install -r Runtime/requirements-textcraft.txt
```

The old `Evaluation/TextCraft/requirements.txt` remains an include-only
compatibility entry point. Training and serving requirements now include the
Runtime dependency list directly. Basic Runtime/TextWorld imports do not require
these optional packages or any training dependencies. Task payloads remain in
the root Dataset directory; Runtime contains no training/evaluation data copies.
The TextCraft executor runs model-generated Python and is **not a security
sandbox**; use an isolated execution environment.

## What remains application-specific

- **Training:** optimizer/FSDP/AReaL integration, proxy completion/session
  tracking, stage routing, reward/credit assignment, trainable token masks,
  environment-to-training adapters, counterfactual branch selection and commit.
- **Evaluation:** worker orchestration, immutable plans, shard ownership, resume,
  error-as-failure aggregation, N=1 rollout policy and model-server lifecycle.
- **TestTimeScale:** N=2 candidate recursion, task-wide fork cap, selector/judge
  policy, complete-branch state commit and detailed selection artifacts.

The two TextWorld prompt builders are deliberately **not** merged: evaluation
retains its explicit budget/decomposition instructions, while training retains
its original wording and defaults. Only their byte-equivalent action protocol
definitions were deduplicated. The two training/evaluation fork/commit adapters
are also not unified. Ordinary delegation can use shared views; counterfactual
candidates use isolated environments. Which branch to commit, whether to
advance sibling delegates sequentially, and how to handle fork limits remain
decisions of the caller, not implicit changes to Runtime.

An old direct `platoon` import from `Evaluation/TextCraft/` still resolves via
its compatibility namespace shim; it contains no execution code. Existing
TextWorld environment/manifest/client/prompt imports under Evaluation,
training rubric/prompt imports, and TestTimeScale's former rubric helper modules
remain thin re-exports for compatibility. Internal consumers import Runtime
directly. This refactor changes no prompt text, action semantics, root success
criterion, temperature, budget, credit formula, selector or statistics definition.

## Tests and licenses

```bash
python -m unittest discover -s test/runtime -v
python -m unittest discover -s test/evaluation -v
python -m unittest discover -s test/training -v
python -m unittest discover -s test/test_time_scaling -v
```

These local-only tests are ignored by Git and use CPU/local mock APIs.
They are not a real GPU training or evaluation run. The vendored Platoon
runtime retains its [MIT license notice](licenses/platoon-MIT.txt).
