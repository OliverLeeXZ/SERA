from __future__ import annotations

import torch

from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors
from platoon.utils.areal_data_processing import pad_missing_metric_tensors

from rubric_subagent_reward.workflow import RubricRewardArealWorkflow


class TextWorldRubricRewardArealWorkflow(RubricRewardArealWorkflow):
    """Run the TextWorld rubric group path through AReaL's public entrypoint.

    The shared rubric workflow implements ``_arun_episode_group`` but the
    installed StepWiseArealWorkflow calls ``_arun_episode_single`` from
    ``arun_episode``.  This project-local entrypoint keeps group scoring and
    the normal AReaL advantage/statistics handling together.
    """

    async def arun_episode(self, engine, data):
        results = await self._arun_episode_group(engine, data)
        results = [result for result in results if result is not None]
        if not results:
            print(f"[TextWorldRubricWorkflow] No results for task {data['task_id']}")
            return None

        # Optional reward diagnostics are emitted only when the corresponding
        # event occurs.  Normalize the per-rollout schemas before concatenating;
        # otherwise a single missing metric can reject the whole workflow task
        # and make AReaL keep sampling without reaching an optimizer update.
        pad_missing_metric_tensors(results)
        train_data = concat_padded_tensors(results)
        mean_unprocessed_reward = torch.mean(train_data["rewards"])

        if self.config.leave_one_out_baseline and len(results) > 1:
            task_rewards = train_data["task_reward"]
            count = len(task_rewards)
            total_reward = task_rewards.sum()
            loo_baselines = (total_reward - task_rewards) / (count - 1)
            datum_counts = torch.tensor([item["rewards"].shape[0] for item in results])
            per_datum_baselines = torch.repeat_interleave(loo_baselines, datum_counts)
            train_data["rewards"] = train_data["rewards"] - per_datum_baselines
        else:
            train_data["rewards"] = train_data["rewards"] - torch.mean(
                train_data["task_reward"]
            )

        tracker = stats_tracker.get(self.stats_scope)
        task_reward_mask = torch.ones_like(
            train_data["task_reward"], dtype=torch.bool
        ).to(self.device)
        output_token_mask = torch.ones_like(
            train_data["num_output_tokens"], dtype=torch.bool
        ).to(self.device)
        input_token_mask = torch.ones_like(
            train_data["num_input_tokens"], dtype=torch.bool
        ).to(self.device)
        num_steps_mask = torch.ones_like(
            train_data["num_steps"], dtype=torch.bool
        ).to(self.device)

        num_steps = train_data["num_steps"].to(self.device)
        num_input_tokens = train_data["num_input_tokens"].to(self.device)
        num_output_tokens = train_data["num_output_tokens"].to(self.device)
        safe_num_steps = torch.clamp(num_steps, min=1.0)
        avg_input_tokens_per_step = num_input_tokens / safe_num_steps
        avg_output_tokens_per_step = num_output_tokens / safe_num_steps

        tracker.denominator(
            task_reward_mask=task_reward_mask,
            num_output_tokens_mask=output_token_mask,
            num_input_tokens_mask=input_token_mask,
            num_steps_mask=num_steps_mask,
            avg_input_tokens_per_step_mask=num_steps_mask,
            avg_output_tokens_per_step_mask=num_steps_mask,
        )
        tracker.stat(
            task_reward=train_data["task_reward"].to(self.device),
            denominator="task_reward_mask",
        )
        tracker.stat(
            num_output_tokens=num_output_tokens,
            denominator="num_output_tokens_mask",
        )
        tracker.stat(
            num_input_tokens=num_input_tokens,
            denominator="num_input_tokens_mask",
        )
        tracker.stat(num_steps=num_steps, denominator="num_steps_mask")
        tracker.stat(
            avg_input_tokens_per_step=avg_input_tokens_per_step,
            denominator="avg_input_tokens_per_step_mask",
        )
        tracker.stat(
            avg_output_tokens_per_step=avg_output_tokens_per_step,
            denominator="avg_output_tokens_per_step_mask",
        )

        task_rewards = train_data["task_reward"].to(self.device)
        task_reward_at_k_mask = torch.ones(1, dtype=torch.bool).to(self.device)
        tracker.denominator(task_reward_at_k_mask=task_reward_at_k_mask)
        tracker.stat(
            task_reward_at_k_mean=torch.mean(task_rewards).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )
        tracker.stat(
            task_reward_at_k_max=torch.max(task_rewards).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )
        tracker.stat(
            task_reward_at_k_min=torch.min(task_rewards).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )

        for key, value in train_data.items():
            if key.startswith("root_"):
                # Root token-level fields do not have the same length as the
                # task-level reward vector.  Give each field a matching
                # denominator so stats tracking cannot abort a rollout group.
                root_mask = torch.ones_like(value, dtype=torch.bool).to(self.device)
                tracker.denominator(**{f"{key}_mask": root_mask})
                tracker.stat(
                    **{key: value.to(self.device)},
                    denominator=f"{key}_mask",
                )
                tracker.stat(
                    **{f"{key}_at_k_mean": torch.mean(value).unsqueeze(0).to(self.device)},
                    denominator="task_reward_at_k_mask",
                )
                tracker.stat(
                    **{f"{key}_at_k_max": torch.max(value).unsqueeze(0).to(self.device)},
                    denominator="task_reward_at_k_mask",
                )
                tracker.stat(
                    **{f"{key}_at_k_min": torch.min(value).unsqueeze(0).to(self.device)},
                    denominator="task_reward_at_k_mask",
                )
            elif key.startswith("reward/"):
                reward_mask = torch.ones_like(value, dtype=torch.bool).to(self.device)
                tracker.denominator(**{f"{key}_mask": reward_mask})
                tracker.stat(
                    **{key: value.to(self.device)},
                    denominator=f"{key}_mask",
                )

        if not self.config.filter_zero_variance_groups:
            train_data["trainable_datums"] = torch.ones_like(
                train_data["rewards"], dtype=torch.bool
            )

        if train_data["rewards"].max() == train_data["rewards"].min() and len(results) > 1:
            tracker.scalar(zero_variance_reward_group=1.0)
            print(
                f"[TextWorldRubricWorkflow] All rewards same for task "
                f"{data['task_id']}: {mean_unprocessed_reward.item():.2f}"
            )
            if self.config.filter_zero_variance_groups:
                return None
            # Keep the group in the fixed-size training batch.  A zero-
            # variance group has zero centered advantage, so this update is
            # mathematically a zero-gradient optimizer step; marking every
            # datum non-trainable would make the trainer drop the whole batch
            # and prevent global_step from advancing forever.
            train_data["trainable_datums"] = torch.ones_like(
                train_data["rewards"], dtype=torch.bool
            )

        return train_data
