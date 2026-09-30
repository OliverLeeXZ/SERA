from pathlib import Path
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import stats_tracker
from .stage_kernel import StageController


class SharedStageWorkflow(RolloutWorkflow):
    """The sole outer router used by all twelve experiment launchers."""

    def __init__(self, schedule, workflows, stats_scope="train_rollout", audit_dir=None):
        self.controller = StageController(schedule)
        expected = {item.stage for item in schedule.schedule}
        if set(workflows) != expected:
            raise ValueError(f"Expected workflows for {sorted(expected)}, got {sorted(workflows)}")
        self.workflows = workflows
        self.stats_scope = stats_scope
        self.audit_dir = Path(audit_dir) if audit_dir else None

    async def arun_episode(self, engine, data):
        version = int(engine.get_version())
        position = self.controller.position(version)
        stats_tracker.get(self.stats_scope).scalar(**{
            "stage/global_step": float(version), "stage/cycle_index": float(position.cycle_index),
            "stage/stage_index": float(position.schedule_index), "stage/step_in_stage": float(position.step_in_stage),
            **{f"stage/is_{name}": float(position.stage == name) for name in self.workflows},
        })
        if self.audit_dir:
            self.controller.write_state(self.audit_dir / f"version-{version:06d}.json", version)
        result = await self.workflows[position.stage].arun_episode(engine, data)
        # Match the old shared controller's logging-only schema normalization.
        if result is not None and {"rewards", "task_reward"} <= result.keys():
            import torch
            for key in ("reward/success", "reward/subagent_launched", "reward/subagent_succeeded"):
                result.setdefault(key, torch.zeros_like(result["rewards"]))
                result.setdefault(f"root_{key}", torch.zeros_like(result["task_reward"]))
        return result
