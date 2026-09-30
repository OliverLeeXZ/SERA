from __future__ import annotations

from typing import Any, Sequence

import torch


def completion_to_datum(
    completion_entry: Any,
    reward: float,
    *,
    output_loss_mask: Sequence[int] | None = None,
    trajectory_depth: int | None = None,
    trajectory_start: bool = True,
) -> dict[str, torch.Tensor]:
    response = completion_entry.model_response
    input_tokens = list(response.input_tokens)
    output_tokens = list(response.output_tokens)
    if output_loss_mask is None:
        output_loss_mask = [1] * len(output_tokens)
    if len(output_loss_mask) != len(output_tokens):
        raise ValueError("output_loss_mask length must equal completion output length")
    if not any(output_loss_mask):
        raise ValueError("Completion datum contains no trainable output token")

    sequence = input_tokens + output_tokens
    loss_mask = [0] * len(input_tokens) + [int(value) for value in output_loss_mask]
    output_logprobs = list(response.output_logprobs)
    output_versions = list(response.output_versions)
    if len(output_logprobs) != len(output_tokens):
        raise ValueError("Completion output_logprobs length mismatch")
    if len(output_versions) != len(output_tokens):
        raise ValueError("Completion output_versions length mismatch")

    datum: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(sequence, dtype=torch.long).unsqueeze(0),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * len(input_tokens) + output_logprobs,
            dtype=torch.float32,
        ).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * len(input_tokens) + output_versions,
            dtype=torch.long,
        ).unsqueeze(0),
        "attention_mask": torch.ones(
            (1, len(sequence)), dtype=torch.bool
        ),
        "num_input_tokens": torch.tensor([float(len(input_tokens))]),
        "num_output_tokens": torch.tensor([float(len(output_tokens))]),
        "num_steps": torch.tensor([1.0]),
        "rewards": torch.tensor([float(reward)], dtype=torch.float32),
        "token_rewards": torch.full(
            (1, len(sequence)), float(reward), dtype=torch.float32
        ),
    }
    if trajectory_depth is not None:
        datum["traj_depth"] = torch.tensor(
            [float(trajectory_depth)], dtype=torch.float32
        )
        datum["traj_start"] = torch.tensor(
            [float(trajectory_start)], dtype=torch.float32
        )
    return datum


def completion_text(completion_entry: Any) -> str:
    response = completion_entry.model_response
    return response.tokenizer.decode(
        list(response.output_tokens), skip_special_tokens=False
    )
