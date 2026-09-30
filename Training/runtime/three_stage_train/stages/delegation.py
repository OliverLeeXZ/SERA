from __future__ import annotations

from typing import Any

from ..agent_tree import build_agent_tree, build_delegation_events
from ..config import ThreeStageTrainConfig
from ..data import completion_text, completion_to_datum
from ..token_spans import (
    find_delegation_statement_spans,
    locate_code_spans_in_completion,
    output_token_mask_for_char_spans,
)
from .common import RawRollout, StageBatchResult, root_reward, step_completion_id


class DelegationStage:
    def __init__(self, config: ThreeStageTrainConfig, adapter: Any) -> None:
        self.config = config
        self.stage_config = config.delegation
        self.adapter = adapter

    async def process(self, raw_rollouts: list[RawRollout], policy_client: Any) -> StageBatchResult:
        del policy_client
        if len(raw_rollouts) < 2:
            return StageBatchResult(metrics={"stage2/skipped_insufficient_rollouts": 1.0})

        rewards = [root_reward(raw.collection) for raw in raw_rollouts]
        reward_sum = sum(rewards)
        datums: list[dict] = []
        records: list[dict[str, Any]] = []
        ast_failures = 0
        alignment_failures = 0
        invalid_parent_step_events = 0

        for raw, reward in zip(raw_rollouts, rewards, strict=True):
            root_advantage = reward - (reward_sum - reward) / (len(rewards) - 1)
            tree = build_agent_tree(raw.collection)
            events = build_delegation_events(raw.collection, tree)
            represented_steps: set[tuple[str, int]] = set()
            started_trajectories: set[str] = set()
            skip_invalid_attempt_trajectories: set[str] = set()
            for event in events:
                parent = raw.collection["trajectories"][event.parent_trajectory_id]
                parent_steps = parent.get("steps") or []
                if not 0 <= event.parent_step < len(parent_steps):
                    invalid_parent_step_events += 1
                    skip_invalid_attempt_trajectories.add(event.parent_trajectory_id)
                    continue
                represented_steps.add((event.parent_trajectory_id, event.parent_step))
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
                credit_weight, credit_metadata = self._event_credit(
                    raw,
                    tree,
                    event,
                )
                action_advantage = credit_weight * root_advantage
                datums.append(
                    completion_to_datum(
                        completion,
                        action_advantage,
                        output_loss_mask=mask,
                        trajectory_depth=(
                            tree.nodes[event.parent_trajectory_id].depth
                            if self._include_depth_metadata()
                            else None
                        ),
                        trajectory_start=(
                            event.parent_trajectory_id not in started_trajectories
                        ),
                    )
                )
                started_trajectories.add(event.parent_trajectory_id)
                records.append(
                    {
                        "rollout_index": raw.rollout_index,
                        "parent_trajectory_id": event.parent_trajectory_id,
                        "parent_step": event.parent_step,
                        "child_trajectory_ids": list(event.child_trajectory_ids),
                        "descendant_leaf_ids": list(event.descendant_leaf_ids),
                        "structural_weight": event.structural_weight,
                        "credit_weight": credit_weight,
                        "root_reward": reward,
                        "root_advantage": root_advantage,
                        "action_advantage": action_advantage,
                        "valid_ast": valid_ast,
                        "token_alignment_succeeded": aligned,
                        **credit_metadata,
                    }
                )

            invalid_datums, invalid_records = self._invalid_attempts(
                raw,
                tree,
                represented_steps,
                started_trajectories,
                skip_invalid_attempt_trajectories,
            )
            datums.extend(invalid_datums)
            records.extend(invalid_records)

        return StageBatchResult(
            datums=datums,
            records=records,
            metrics={
                "stage2/delegation_actions": float(len(records)),
                "stage2/ast_failures": float(ast_failures),
                "stage2/token_alignment_failures": float(alignment_failures),
                "stage2/invalid_parent_step_events": float(invalid_parent_step_events),
            },
        )

    def _event_credit(
        self,
        raw: RawRollout,
        tree: Any,
        event: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Return the launch-event credit while preserving legacy behavior.

        Two-stage ablations override this hook. The original three-stage
        delegation objective remains exactly the descendant-leaf fraction.
        """

        del raw, tree
        return event.structural_weight, {"credit_mode": "leaf"}

    def _delegation_mask(
        self,
        completion: Any,
        step: dict[str, Any],
    ) -> tuple[list[int], bool, bool]:
        response = completion.model_response
        code = step.get("code") or ""
        parsed = find_delegation_statement_spans(code)
        text = completion_text(completion)
        spans = locate_code_spans_in_completion(text, code, parsed.spans)
        aligned = bool(spans)
        if not aligned and parsed.attempted_delegation:
            return [0] * len(response.output_tokens), parsed.valid_ast, False
        mask = output_token_mask_for_char_spans(
            response.tokenizer,
            list(response.output_tokens),
            spans,
        )
        return mask, parsed.valid_ast, aligned

    def _invalid_attempts(
        self,
        raw: RawRollout,
        tree: Any,
        represented_steps: set[tuple[str, int]],
        started_trajectories: set[str],
        skip_trajectory_ids: set[str] | None = None,
    ) -> tuple[list[dict], list[dict[str, Any]]]:
        datums: list[dict] = []
        records: list[dict[str, Any]] = []
        skip_trajectory_ids = skip_trajectory_ids or set()
        for trajectory_id, trajectory in raw.collection["trajectories"].items():
            if trajectory_id in skip_trajectory_ids:
                continue
            for step_index, step in enumerate(trajectory.get("steps") or []):
                if (trajectory_id, step_index) in represented_steps:
                    continue
                code = step.get("code") or ""
                parsed = find_delegation_statement_spans(code)
                if not parsed.attempted_delegation:
                    continue
                completion_id = step_completion_id(step)
                if completion_id is None or completion_id not in raw.completions:
                    continue
                completion = raw.completions[completion_id]
                mask, _, _ = self._delegation_mask(completion, step)
                if not any(mask):
                    continue
                reward = self.stage_config.invalid_delegation_reward
                datums.append(
                    completion_to_datum(
                        completion,
                        reward,
                        output_loss_mask=mask,
                        trajectory_depth=(
                            tree.nodes[trajectory_id].depth
                            if self._include_depth_metadata()
                            else None
                        ),
                        trajectory_start=(
                            trajectory_id not in started_trajectories
                        ),
                    )
                )
                started_trajectories.add(trajectory_id)
                records.append(
                    {
                        "rollout_index": raw.rollout_index,
                        "parent_trajectory_id": trajectory_id,
                        "parent_step": step_index,
                        "invalid_delegation": True,
                        "action_advantage": reward,
                    }
                )
        return datums, records

    def _include_depth_metadata(self) -> bool:
        optimization = self.config.optimization
        return (
            optimization.depth_level_weighting
            or optimization.depth_level_discount_gamma is not None
        )
