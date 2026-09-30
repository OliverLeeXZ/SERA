"""One policy-version stage controller, extracted from the original stage controllers.

Queue submissions, accepted task groups and epochs do not advance this schedule:
the AReaL inference engine's policy version is the sole routing clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class StageScheduleItem:
    stage: str
    train_steps: int
    batch_size: int = 8

    def validate(self) -> None:
        if not self.stage or self.train_steps <= 0 or self.batch_size <= 0:
            raise ValueError("Stage names must be non-empty and lengths/batch sizes positive")


@dataclass
class StageSchedule:
    schedule: list[StageScheduleItem]
    cycles: int = -1

    @property
    def steps_per_cycle(self) -> int:
        return sum(item.train_steps for item in self.schedule)

    @property
    def finite_total_steps(self) -> int | None:
        return None if self.cycles == -1 else self.steps_per_cycle * self.cycles

    def validate(self) -> None:
        if not self.schedule or self.cycles == 0 or self.cycles < -1:
            raise ValueError("A schedule needs stages and cycles must be positive or -1")
        for item in self.schedule:
            item.validate()

    @classmethod
    def parse(cls, text: str, batch_size: int = 8, cycles: int = -1):
        stages = []
        for block in text.split(","):
            name, count = block.strip().split(":")
            if name not in {"execution", "delegation", "rubric_generation"}:
                raise ValueError(f"Unknown stage: {name}")
            stages.append(StageScheduleItem(name, int(count), batch_size))
        result = cls(stages, cycles)
        result.validate()
        if len({item.stage for item in stages}) != len(stages):
            raise ValueError("Repeated stage names are not supported")
        return result


@dataclass(frozen=True)
class StagePosition:
    stage: str
    cycle_index: int
    schedule_index: int
    step_in_stage: int
    stage_steps: int
    batch_size: int
    global_step: int

    @property
    def is_stage_start(self) -> bool:
        return self.step_in_stage == 0

    @property
    def is_stage_end(self) -> bool:
        return self.step_in_stage + 1 == self.stage_steps


class StageController:
    """Also accepts the vendored credit processors' single-stage configurations."""

    def __init__(self, config):
        config.validate()
        self.config = config

    def position(self, global_step: int) -> StagePosition:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        total = self.config.finite_total_steps
        if total is not None and global_step >= total:
            raise StopIteration(f"Policy version {global_step} exceeds the finite schedule ({total})")
        cycle, offset = divmod(global_step, self.config.steps_per_cycle)
        for index, item in enumerate(self.config.schedule):
            if offset < item.train_steps:
                return StagePosition(item.stage, cycle, index, offset, item.train_steps, item.batch_size, global_step)
            offset -= item.train_steps
        raise AssertionError("Validated schedule must cover its cycle")

    def write_state(self, path: str | Path, global_step: int) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = asdict(self.position(global_step)) | dict(cycles=self.config.cycles, steps_per_cycle=self.config.steps_per_cycle)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
            Path(temporary).replace(destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
