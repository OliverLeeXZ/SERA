from __future__ import annotations

from typing import Any

import torch

from three_stage_train.stages.common import RawRollout, StageBatchResult
from three_stage_train.stages.delegation import DelegationStage

from .config import TwoStageRaoLeafConfig


SEQUENCE_KEYS = {
    "input_ids",
    "loss_mask",
    "logprobs",
    "versions",
    "attention_mask",
    "token_rewards",
}


def truncate_after_last_trainable_token(
    datum: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Make the final launch token the terminal rewarded action.

    AReaL places a scalar trajectory reward at the final action position and
    propagates it backwards only through contiguous trainable tokens. A Leaf
    datum otherwise retains non-trainable text after ``launch_subagent``, so
    that suffix blocks the reward before it reaches the launch span.
    """

    loss_mask = datum["loss_mask"]
    if loss_mask.ndim != 2 or loss_mask.shape[0] != 1:
        raise ValueError("Leaf datum loss_mask must have shape [1, sequence]")
    trainable = torch.nonzero(loss_mask[0], as_tuple=False).flatten()
    if trainable.numel() == 0:
        raise ValueError("Leaf datum contains no trainable launch token")

    terminal = int(trainable[-1].item()) + 1
    original_length = int(loss_mask.shape[1])
    result = dict(datum)
    for key in SEQUENCE_KEYS:
        value = result.get(key)
        if value is None:
            continue
        if value.ndim != 2 or value.shape[1] != original_length:
            raise ValueError(f"Leaf datum {key} does not match sequence length")
        result[key] = value[:, :terminal]

    input_tokens = int(result["num_input_tokens"].item())
    result["num_output_tokens"] = torch.tensor(
        [float(max(0, terminal - input_tokens))],
        dtype=result["num_output_tokens"].dtype,
        device=result["num_output_tokens"].device,
    )
    return result


class LaunchTerminalDelegationStage(DelegationStage):
    """Launch-only stage shared by the four two-stage credit ablations."""

    def __init__(
        self,
        config: Any,
        adapter: Any,
        launch_credit_config: TwoStageRaoLeafConfig | None = None,
    ) -> None:
        super().__init__(config, adapter)
        self.launch_credit_config = (
            launch_credit_config or TwoStageRaoLeafConfig()
        )

    def _event_credit(
        self,
        raw: RawRollout,
        tree: Any,
        event: Any,
    ) -> tuple[float, dict[str, Any]]:
        mode = self.launch_credit_config.launch_credit_mode
        if mode == "leaf":
            weight, metadata = _leaf_credit(
                raw.collection,
                tree,
                event,
                success_gate=self.launch_credit_config.subagent_success_gate,
            )
        elif mode == "workload":
            weight, metadata = _workload_credit(
                raw.collection,
                tree,
                event,
                success_gate=self.launch_credit_config.subagent_success_gate,
                cap=self.launch_credit_config.workload_weight_cap,
            )
        else:  # validate() normally catches this before rollout starts.
            raise ValueError(f"Unsupported launch credit mode: {mode}")

        return weight, {
            "credit_mode": mode,
            "subagent_success_gate": (
                self.launch_credit_config.subagent_success_gate
            ),
            **metadata,
        }

    async def process(
        self,
        raw_rollouts: list[RawRollout],
        policy_client: Any,
    ) -> StageBatchResult:
        result = await super().process(raw_rollouts, policy_client)
        result.datums = [
            truncate_after_last_trainable_token(datum)
            for datum in result.datums
        ]
        result.metrics["stage2/launch_terminal_datums"] = float(
            len(result.datums)
        )
        return result


def _trajectory_success(trajectory: dict[str, Any]) -> bool:
    for step in reversed(trajectory.get("steps") or []):
        reward_misc = (step.get("misc") or {}).get("reward_misc") or {}
        if "reward/success" in reward_misc:
            return float(reward_misc["reward/success"]) > 0.0
    return float(trajectory.get("reward", 0.0)) > 0.0


def _owning_root(tree: Any, node_id: str) -> str:
    current = node_id
    while tree.nodes[current].parent_trajectory_id in tree.nodes:
        current = tree.nodes[current].parent_trajectory_id
    return current


def _gold_step_count(trajectory: dict[str, Any]) -> int | None:
    task = trajectory.get("task") or {}
    misc = task.get("misc") or {}
    if misc.get("local_gold_unavailable"):
        return None
    gold = misc.get("local_gold_trajectory")
    if gold is None:
        gold = misc.get("gold_trajectory")
    if not isinstance(gold, list):
        return None
    return sum(
        1
        for step in gold
        if isinstance(step, dict) and step.get("action") == "craft"
    )


def _leaf_credit(
    collection: dict[str, Any],
    tree: Any,
    event: Any,
    *,
    success_gate: bool,
) -> tuple[float, dict[str, Any]]:
    trajectories = collection["trajectories"]
    root_id = _owning_root(tree, event.parent_trajectory_id)
    denominator = len(tree.descendant_leaves(root_id))
    credited_leaves: set[str] = set()
    successful_children = 0
    for child_id in event.child_trajectory_ids:
        child_success = _trajectory_success(trajectories[child_id])
        successful_children += int(child_success)
        if success_gate and not child_success:
            continue
        credited_leaves.update(tree.descendant_leaves(child_id))
    numerator = len(credited_leaves)
    weight = numerator / denominator if denominator else 0.0
    return weight, {
        "credit_numerator": float(numerator),
        "credit_denominator": float(denominator),
        "successful_direct_children": float(successful_children),
        "direct_children": float(len(event.child_trajectory_ids)),
    }


def _workload_credit(
    collection: dict[str, Any],
    tree: Any,
    event: Any,
    *,
    success_gate: bool,
    cap: float,
) -> tuple[float, dict[str, Any]]:
    trajectories = collection["trajectories"]
    root_id = _owning_root(tree, event.parent_trajectory_id)
    denominator = _gold_step_count(trajectories[root_id])
    numerator = 0
    successful_children = 0
    missing_child_gold = 0
    for child_id in event.child_trajectory_ids:
        child = trajectories[child_id]
        child_success = _trajectory_success(child)
        successful_children += int(child_success)
        if success_gate and not child_success:
            continue
        child_steps = _gold_step_count(child)
        if child_steps is None:
            missing_child_gold += 1
            continue
        numerator += child_steps

    raw_weight = (
        numerator / denominator
        if denominator is not None and denominator > 0
        else 0.0
    )
    weight = min(raw_weight, cap)
    return weight, {
        "credit_numerator": float(numerator),
        "credit_denominator": float(denominator or 0),
        "raw_workload_weight": float(raw_weight),
        "workload_weight_capped": float(raw_weight > cap),
        "missing_root_gold": float(denominator is None or denominator <= 0),
        "missing_child_gold": float(missing_child_gold),
        "successful_direct_children": float(successful_children),
        "direct_children": float(len(event.child_trajectory_ids)),
    }
