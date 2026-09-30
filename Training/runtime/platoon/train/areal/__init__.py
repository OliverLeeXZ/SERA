"""Lazy AReaL exports: configuration inspection must not import CUDA engines."""
import importlib

_EXPORTS = {
    "LossFnConfig": "config_defs", "PlatoonArealRLTrainerConfig": "config_defs",
    "RolloutConfig": "config_defs", "WorkflowConfig": "config_defs",
    "PlatoonArealRLTrainer": "rl", "PlatoonPPOActor": "actor", "create_actor": "actor",
    "ArealProxySession": "proxy", "cispo_loss_fn": "loss_functions",
    "get_loss_fn": "loss_functions", "grpo_loss_fn": "loss_functions",
    "list_loss_fns": "loss_functions", "register_loss_fn": "loss_functions",
}
__all__ = list(_EXPORTS)
_patched = False


def __getattr__(name):
    global _patched
    if name not in _EXPORTS:
        raise AttributeError(name)
    if _EXPORTS[name] != "config_defs" and not _patched:
        from .patches import apply_all_patches
        apply_all_patches()
        _patched = True
    value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
