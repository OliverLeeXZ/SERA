"""Source 106 reward, reusing the existing fork/score/LOO/datum pipeline."""
from statistics import mean

from rubric_generation_ranking.stage import RubricGenerationRankingStage
from .reward import calculate_rubric_reward


class RubricGenerationDiscriminationStage(RubricGenerationRankingStage):
    """Change only the group reward; train rubric generation, never scoring."""

    def _ranking_result(self, scores, successes):
        config = self.ranking_config
        return calculate_rubric_reward(
            scores, successes, config.margin,
            order_weight=config.order_reward_weight,
            discrimination_weight=config.discrimination_reward_weight,
            discrimination_scale=config.discrimination_scale,
        )

    def _ranking_record(self, ranking):
        return dict(order_reward=ranking.order_reward,
                    discrimination_reward=ranking.discrimination_reward,
                    discrimination_score_std=ranking.score_std)

    def _ranking_metrics(self, ranked_groups):
        rewards = [ranking for _, ranking in ranked_groups]
        return {f"rubric_generation_ranking/{metric}": mean(getattr(row, field) for row in rewards) if rewards else 0.0
                for metric, field in (("order_reward", "order_reward"),
                                      ("discrimination_reward", "discrimination_reward"),
                                      ("discrimination_score_std", "score_std"),
                                      ("combined_reward", "group_reward"))}
