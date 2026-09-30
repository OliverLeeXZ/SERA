from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from three_stage_train.agent_tree import build_agent_tree, build_delegation_events
from three_stage_train.data import completion_to_datum
from three_stage_train.stages.common import (
    RawRollout,
    StageBatchResult,
    root_reward,
    step_completion_id,
)
from three_stage_train.token_spans import find_delegation_statement_spans
from two_stage_rao_leaf.config import TwoStageRaoLeafConfig
from two_stage_rao_leaf.leaf_credit import (
    LaunchTerminalDelegationStage,
    truncate_after_last_trainable_token,
)


@dataclass
class _LaunchCandidate:
    raw: RawRollout
    tree: Any
    parent_trajectory_id: str
    parent_step: int
    completion: Any
    mask: list[int]
    shaped_reward: float
    credit_weight: float
    root_reward: float
    valid_ast: bool
    aligned: bool
    metadata: dict[str, Any]
    invalid_delegation: bool = False


def flat_loo_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("Flat launch-event LOO requires at least two actions")
    total = sum(rewards)
    return [
        reward - (total - reward) / (len(rewards) - 1)
        for reward in rewards
    ]


def root_balanced_loo_advantages(
    rewards_by_rollout: list[list[float]],
) -> list[list[float]]:
    if len(rewards_by_rollout) < 2:
        raise ValueError("Root-balanced LOO requires at least two rollouts")
    root_means = [
        sum(rewards) / len(rewards) if rewards else 0.0
        for rewards in rewards_by_rollout
    ]
    mean_sum = sum(root_means)
    return [
        [
            reward - (mean_sum - root_means[index]) / (len(root_means) - 1)
            for reward in rewards
        ]
        for index, rewards in enumerate(rewards_by_rollout)
    ]


class RewardFirstLaunchDelegationStage(LaunchTerminalDelegationStage):
    """Compute shaped launch rewards before either LOO baseline."""

    async def process(
        self,
        raw_rollouts: list[RawRollout],
        policy_client: Any,
    ) -> StageBatchResult:
        del policy_client
        if len(raw_rollouts) < 2:
            return StageBatchResult(
                metrics={"stage2/skipped_insufficient_rollouts": 1.0}
            )

        candidates_by_rollout: list[list[_LaunchCandidate]] = []
        ast_failures = 0
        alignment_failures = 0
        invalid_parent_step_events = 0
        for raw in raw_rollouts:
            candidates, metrics = self._collect_candidates(raw)
            candidates_by_rollout.append(candidates)
            ast_failures += int(metrics["ast_failures"])
            alignment_failures += int(metrics["alignment_failures"])
            invalid_parent_step_events += int(
                metrics["invalid_parent_step_events"]
            )

        mode = self.launch_credit_config.launch_advantage_mode
        rewards_by_rollout = [
            [candidate.shaped_reward for candidate in candidates]
            for candidates in candidates_by_rollout
        ]
        if mode == "reward_first_flat":
            flat_candidates = [
                candidate
                for candidates in candidates_by_rollout
                for candidate in candidates
            ]
            if len(flat_candidates) < 2:
                return StageBatchResult(
                    metrics={
                        "stage2/skipped_insufficient_launch_actions": 1.0,
                        "stage2/reward_first_launch_actions": float(
                            len(flat_candidates)
                        ),
                    }
                )
            flat_advantages = flat_loo_advantages(
                [candidate.shaped_reward for candidate in flat_candidates]
            )
            advantages_by_rollout: list[list[float]] = []
            cursor = 0
            for candidates in candidates_by_rollout:
                size = len(candidates)
                advantages_by_rollout.append(
                    flat_advantages[cursor : cursor + size]
                )
                cursor += size
        elif mode == "reward_first_root_balanced":
            advantages_by_rollout = root_balanced_loo_advantages(
                rewards_by_rollout
            )
        else:
            raise ValueError(f"Unsupported reward-first mode: {mode}")

        datums: list[dict] = []
        records: list[dict[str, Any]] = []
        for candidates, advantages in zip(
            candidates_by_rollout,
            advantages_by_rollout,
            strict=True,
        ):
            started_trajectories: set[str] = set()
            rollout_mean = (
                sum(candidate.shaped_reward for candidate in candidates)
                / len(candidates)
                if candidates
                else 0.0
            )
            for candidate, advantage in zip(candidates, advantages, strict=True):
                depth = candidate.tree.nodes[
                    candidate.parent_trajectory_id
                ].depth
                datum = completion_to_datum(
                    candidate.completion,
                    advantage,
                    output_loss_mask=candidate.mask,
                    trajectory_depth=(
                        depth if self._include_depth_metadata() else None
                    ),
                    trajectory_start=(
                        candidate.parent_trajectory_id
                        not in started_trajectories
                    ),
                )
                datums.append(truncate_after_last_trainable_token(datum))
                started_trajectories.add(candidate.parent_trajectory_id)
                records.append(
                    {
                        "rollout_index": candidate.raw.rollout_index,
                        "parent_trajectory_id": candidate.parent_trajectory_id,
                        "parent_step": candidate.parent_step,
                        "root_reward": candidate.root_reward,
                        "credit_weight": candidate.credit_weight,
                        "shaped_launch_reward": candidate.shaped_reward,
                        "rollout_launch_reward_mean": rollout_mean,
                        "action_advantage": advantage,
                        "launch_advantage_mode": mode,
                        "invalid_delegation": candidate.invalid_delegation,
                        "valid_ast": candidate.valid_ast,
                        "token_alignment_succeeded": candidate.aligned,
                        **candidate.metadata,
                    }
                )

        return StageBatchResult(
            datums=datums,
            records=records,
            metrics={
                "stage2/delegation_actions": float(len(records)),
                "stage2/reward_first_launch_actions": float(len(records)),
                "stage2/reward_first_root_rollouts": float(len(raw_rollouts)),
                "stage2/reward_first_flat_group": float(
                    mode == "reward_first_flat"
                ),
                "stage2/reward_first_root_balanced_group": float(
                    mode == "reward_first_root_balanced"
                ),
                "stage2/launch_terminal_datums": float(len(datums)),
                "stage2/ast_failures": float(ast_failures),
                "stage2/token_alignment_failures": float(alignment_failures),
                "stage2/invalid_parent_step_events": float(
                    invalid_parent_step_events
                ),
            },
        )

    def _collect_candidates(
        self,
        raw: RawRollout,
    ) -> tuple[list[_LaunchCandidate], dict[str, int]]:
        collection = raw.collection
        trajectories = collection["trajectories"]
        tree = build_agent_tree(collection)
        reward = root_reward(collection, tree)
        represented_steps: set[tuple[str, int]] = set()
        skip_invalid_trajectories: set[str] = set()
        candidates: list[_LaunchCandidate] = []
        ast_failures = 0
        alignment_failures = 0
        invalid_parent_step_events = 0

        for event in build_delegation_events(collection, tree):
            parent = trajectories[event.parent_trajectory_id]
            parent_steps = parent.get("steps") or []
            if not 0 <= event.parent_step < len(parent_steps):
                invalid_parent_step_events += 1
                skip_invalid_trajectories.add(event.parent_trajectory_id)
                continue
            represented_steps.add(
                (event.parent_trajectory_id, event.parent_step)
            )
            step = parent_steps[event.parent_step]
            completion_id = event.completion_id or step_completion_id(step)
            if completion_id is None or completion_id not in raw.completions:
                continue
            completion = raw.completions[completion_id]
            mask, valid_ast, aligned = self._delegation_mask(completion, step)
            ast_failures += int(not valid_ast)
            alignment_failures += int(not aligned)
            if not any(mask):
                continue
            credit_weight, metadata = self._event_credit(raw, tree, event)
            candidates.append(
                _LaunchCandidate(
                    raw=raw,
                    tree=tree,
                    parent_trajectory_id=event.parent_trajectory_id,
                    parent_step=event.parent_step,
                    completion=completion,
                    mask=mask,
                    shaped_reward=reward * credit_weight,
                    credit_weight=credit_weight,
                    root_reward=reward,
                    valid_ast=valid_ast,
                    aligned=aligned,
                    metadata={
                        "child_trajectory_ids": list(
                            event.child_trajectory_ids
                        ),
                        "descendant_leaf_ids": list(
                            event.descendant_leaf_ids
                        ),
                        "structural_weight": event.structural_weight,
                        **metadata,
                    },
                )
            )

        for trajectory_id, trajectory in trajectories.items():
            if trajectory_id in skip_invalid_trajectories:
                continue
            for step_index, step in enumerate(trajectory.get("steps") or []):
                if (trajectory_id, step_index) in represented_steps:
                    continue
                parsed = find_delegation_statement_spans(step.get("code") or "")
                if not parsed.attempted_delegation:
                    continue
                completion_id = step_completion_id(step)
                if completion_id is None or completion_id not in raw.completions:
                    continue
                completion = raw.completions[completion_id]
                mask, valid_ast, aligned = self._delegation_mask(
                    completion, step
                )
                ast_failures += int(not valid_ast)
                alignment_failures += int(not aligned)
                if not any(mask):
                    continue
                candidates.append(
                    _LaunchCandidate(
                        raw=raw,
                        tree=tree,
                        parent_trajectory_id=trajectory_id,
                        parent_step=step_index,
                        completion=completion,
                        mask=mask,
                        shaped_reward=self.stage_config.invalid_delegation_reward,
                        credit_weight=0.0,
                        root_reward=reward,
                        valid_ast=valid_ast,
                        aligned=aligned,
                        metadata={"credit_mode": "invalid"},
                        invalid_delegation=True,
                    )
                )

        return candidates, {
            "ast_failures": ast_failures,
            "alignment_failures": alignment_failures,
            "invalid_parent_step_events": invalid_parent_step_events,
        }
