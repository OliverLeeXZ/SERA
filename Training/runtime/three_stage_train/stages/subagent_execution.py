from __future__ import annotations

from statistics import mean
from typing import Any

from ..config import ThreeStageTrainConfig
from ..data import completion_text, completion_to_datum
from Runtime.rubric.prompts import build_rubric_messages, build_scoring_messages
from Runtime.rubric.scoring import parse_policy_score, parse_rubric
from ..token_spans import (
    find_delegation_statement_spans,
    locate_code_spans_in_completion,
    output_token_mask_for_char_spans,
)
from .common import (
    RawRollout,
    StageBatchResult,
    SubagentCandidate,
    build_subagent_candidates,
    rubric_context,
    step_completion_id,
)


class SubagentExecutionStage:
    def __init__(self, config: ThreeStageTrainConfig, adapter: Any) -> None:
        self.config = config
        self.stage_config = config.subagent_execution
        self.adapter = adapter

    async def process(self, raw_rollouts: list[RawRollout], policy_client: Any) -> StageBatchResult:
        candidates = [
            candidate
            for raw in raw_rollouts
            for candidate in build_subagent_candidates(raw, self.adapter)
        ]
        total_candidates = len(candidates)
        maximum = self.stage_config.max_subagent_trajectories_per_task
        if maximum is not None:
            candidates = candidates[:maximum]
        if len(candidates) < self.stage_config.min_valid_subagent_trajectories_per_task:
            return StageBatchResult(
                metrics={
                    "stage1/subagent_candidates": float(total_candidates),
                    "stage1/skipped_insufficient_trajectories": 1.0,
                }
            )

        min_criteria = self.config.rubric_generation.num_rubric_criteria_min
        for candidate in candidates:
            context = rubric_context(candidate)
            candidate.rubric_messages = build_rubric_messages(
                **context,
                min_criteria=min_criteria,
            )
        rubric_outputs = await policy_client.generate_batch(
            [candidate.rubric_messages for candidate in candidates],
            temperature=self.stage_config.rubric_temperature,
            max_completion_tokens=self.stage_config.max_rubric_tokens,
        )

        valid_rubrics: list[SubagentCandidate] = []
        for candidate, generation in zip(candidates, rubric_outputs, strict=True):
            candidate.rubric_generation = generation
            if not generation.valid:
                continue
            try:
                candidate.rubric = parse_rubric(generation.text, min_criteria)
            except (TypeError, ValueError):
                continue
            candidate.scoring_messages = build_scoring_messages(
                rubric=candidate.rubric,
                child_goal=candidate.child_goal,
                trajectory=self.adapter.serialize_agent_trajectory(
                    candidate.trajectory
                ),
                final_environment_state=self.adapter.final_state_from_trajectory(
                    candidate.trajectory
                ),
            )
            valid_rubrics.append(candidate)

        scoring_outputs = await policy_client.generate_batch(
            [candidate.scoring_messages for candidate in valid_rubrics],
            temperature=self.stage_config.scoring_temperature,
            max_completion_tokens=self.stage_config.max_scoring_tokens,
        )
        scored: list[SubagentCandidate] = []
        for candidate, generation in zip(valid_rubrics, scoring_outputs, strict=True):
            candidate.scoring_generation = generation
            if not generation.valid:
                continue
            try:
                candidate.score, _ = parse_policy_score(generation.text)
            except (TypeError, ValueError):
                continue
            scored.append(candidate)

        minimum = self.stage_config.min_valid_subagent_trajectories_per_task
        if len(scored) < minimum:
            return StageBatchResult(
                metrics={
                    "stage1/subagent_candidates": float(total_candidates),
                    "stage1/valid_rubrics": float(len(valid_rubrics)),
                    "stage1/valid_scores": float(len(scored)),
                    "stage1/skipped_insufficient_valid_scores": 1.0,
                }
            )

        datums: list[dict] = []
        records: list[dict[str, Any]] = []
        scores = [float(candidate.score) for candidate in scored]
        if max(scores) == min(scores):
            return StageBatchResult(
                metrics={
                    "stage1/subagent_candidates": float(total_candidates),
                    "stage1/valid_rubrics": float(len(valid_rubrics)),
                    "stage1/valid_scores": float(len(scored)),
                    "stage1/zero_variance_groups": 1.0,
                    "stage1/mean_score": mean(scores),
                }
            )

        score_sum = sum(scores)
        for candidate in scored:
            advantage = float(candidate.score) - (
                score_sum - float(candidate.score)
            ) / (len(scored) - 1)
            candidate_datums = self._trajectory_datums(candidate, advantage)
            if not candidate_datums:
                continue
            datums.extend(candidate_datums)
            records.append(
                {
                    "trajectory_id": candidate.trajectory_id,
                    "depth": candidate.depth,
                    "subtask_key": candidate.subtask_key,
                    "score": candidate.score,
                    "advantage": advantage,
                    "rubric_completion_id": candidate.rubric_generation.completion_id,
                    "scoring_completion_id": candidate.scoring_generation.completion_id,
                    "rubric_input_messages": candidate.rubric_messages,
                    "rubric_output": candidate.rubric_generation.text,
                    "rubric": candidate.rubric,
                    "scoring_input_messages": candidate.scoring_messages,
                    "scoring_output": candidate.scoring_generation.text,
                }
            )

        return StageBatchResult(
            datums=datums,
            records=records,
            metrics={
                "stage1/subagent_candidates": float(total_candidates),
                "stage1/valid_rubrics": float(len(valid_rubrics)),
                "stage1/valid_scores": float(len(scored)),
                "stage1/zero_variance_groups": 0.0,
                "stage1/mean_score": mean(
                    [float(candidate.score) for candidate in scored]
                ) if scored else 0.0,
            },
        )

    def _trajectory_datums(
        self, candidate: SubagentCandidate, advantage: float
    ) -> list[dict]:
        datums: list[dict] = []
        for step in candidate.trajectory.get("steps") or []:
            completion_id = step_completion_id(step)
            if completion_id is None or completion_id not in candidate.raw.completions:
                continue
            completion = candidate.raw.completions[completion_id]
            response = completion.model_response
            output_mask = [1] * len(response.output_tokens)
            if not self.stage_config.include_delegation_tokens:
                code = step.get("code") or ""
                parsed = find_delegation_statement_spans(code)
                if parsed.attempted_delegation:
                    text = completion_text(completion)
                    spans = locate_code_spans_in_completion(text, code, parsed.spans)
                    if not spans:
                        continue
                    delegation_mask = output_token_mask_for_char_spans(
                        response.tokenizer,
                        list(response.output_tokens),
                        spans,
                    )
                    output_mask = [1 - value for value in delegation_mask]
            if not any(output_mask):
                continue
            datums.append(
                completion_to_datum(
                    completion,
                    advantage,
                    output_loss_mask=output_mask,
                    trajectory_depth=(
                        candidate.depth if self._include_depth_metadata() else None
                    ),
                    trajectory_start=not datums,
                )
            )
        return datums

    def _include_depth_metadata(self) -> bool:
        optimization = self.config.optimization
        return (
            optimization.depth_level_weighting
            or optimization.depth_level_discount_gamma is not None
        )
