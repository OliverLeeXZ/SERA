from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.distributed as dist
from areal.api.io_struct import SaveLoadMeta
from areal.utils.saver import Saver

from .schedule import StageController


class StageBoundarySaver:
    """Adds exact stage-boundary checkpoints without replacing normal Saver policy."""

    def __init__(
        self,
        delegate: Saver,
        controller: StageController,
        manifest_path: str | Path,
    ) -> None:
        self.delegate = delegate
        self.controller = controller
        self.manifest_path = Path(manifest_path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def state_dict(self):
        return self.delegate.state_dict()

    def load_state_dict(self, state_dict):
        return self.delegate.load_state_dict(state_dict)

    def save(
        self,
        engine,
        epoch: int,
        step: int,
        global_step: int,
        name: str = "default",
        tokenizer=None,
        processor=None,
        base_model_path: str | None = None,
    ) -> None:
        self.delegate.save(
            engine,
            epoch,
            step,
            global_step,
            name=name,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        position = self.controller.position(global_step)
        if not position.is_stage_end:
            return

        config = self.delegate.config
        path = Saver.get_model_save_path(
            config.experiment_name,
            config.trial_name,
            config.fileroot,
            epoch,
            step,
            global_step,
            name,
        )
        engine.save(
            SaveLoadMeta(
                path=path,
                weight_format="hf",
                with_optim=False,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
            )
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            self._append_manifest(position, path)

    def _append_manifest(self, position, checkpoint_path: str) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": position.global_step,
            "completed_stage": position.stage,
            "cycle_index": position.cycle_index,
            "schedule_index": position.schedule_index,
            "checkpoint_path": checkpoint_path,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
