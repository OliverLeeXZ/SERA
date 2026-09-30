from pathlib import Path
import os
import sys

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRAINING_ROOT.parent


def bootstrap() -> None:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from Runtime.bootstrap import bootstrap as bootstrap_runtime
    bootstrap_runtime()
    for path in (TRAINING_ROOT, TRAINING_ROOT / "runtime"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # Also support training after ordinary evaluation imported the shared core
    # in this process. Extend the existing namespaces instead of loading a second
    # copy of their classes/contextvars or changing the environment implementation.
    shared_platoon = REPOSITORY_ROOT / "Runtime/vendor/platoon"
    for name, subpath in (("platoon", ""), ("platoon.textcraft", "textcraft"), ("platoon.utils", "utils")):
        module = sys.modules.get(name)
        if module is not None:
            module.__path__ = [str(TRAINING_ROOT / "runtime/platoon" / subpath),
                               str(shared_platoon / subpath)]
    from Dataset.paths import data_path
    os.environ.setdefault("TEXTCRAFT_SYNTH_TRAIN_PATH", str(data_path("training", "textcraft", "textcraft_synth_train.jsonl")))
    os.environ.setdefault("TEXTCRAFT_SYNTH_VAL_PATH", str(data_path("validation", "textcraft", "textcraft_synth_val.jsonl")))
