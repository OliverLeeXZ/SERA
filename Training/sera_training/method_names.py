"""Expanded paper method names and legacy recipe identifiers for resume."""

PAPER_METHOD_NAMES = {
    "rao": "RAO",
    "rao_with_decomposition_reward": "RAO with Decomposition Reward",
    "rao_with_rubric_training": "RAO with Rubric Training",
    "sera_without_decomposition_reward_and_rubric_training": "SERA without Decomposition Reward and Rubric Training",
    "sera_without_decomposition_reward": "SERA without Decomposition Reward",
    "sera": "SERA",
    "kimi_rubric_policy_scorer": "Rubric ablation: Kimi Generator / Policy Scorer / No Rubric Training",
    "kimi_rubric_kimi_scorer": "Rubric ablation: Kimi Generator / Kimi Scorer / No Rubric Training",
    "global_rubric_policy_scorer": "Rubric ablation: Global Policy Rubric / Policy Scorer / No Rubric Training",
    "rubric_training_discriminativeness": "Rubric ablation: Policy Generator / Policy Scorer / Ranking and Discriminativeness",
}

LEGACY_RECIPE_IDS = {
    "rao": "rao",
    "rao_leaf": "rao_with_decomposition_reward",
    "rao_rubric_training": "rao_with_rubric_training",
    "rubric_reward": "sera_without_decomposition_reward_and_rubric_training",
    "rubric_reward_training": "sera_without_decomposition_reward",
    "three_stage": "sera",
}


def paper_method_name(experiment: str) -> str:
    """Expand the method name, leaving old run/checkpoint identities intact."""
    identifier = LEGACY_RECIPE_IDS.get(experiment, experiment)
    return PAPER_METHOD_NAMES.get(identifier, experiment)
