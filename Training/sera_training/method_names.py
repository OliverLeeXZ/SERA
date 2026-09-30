"""Paper-facing names for stable training recipe identifiers.

Keep the identifier in run names and checkpoint paths for resume compatibility.
The names below follow the main and rubric-ablation tables in the SERA paper.
"""

PAPER_METHOD_NAMES = {
    "rao": "RAO",
    "rao_leaf": "RAO + D",
    "rao_rubric_training": "RAO + RT",
    "rubric_reward": "SERA w/o D & RT",
    "rubric_reward_training": "SERA w/o D",
    "three_stage": "SERA",
    "kimi_rubric_policy_scorer": "Rubric ablation: Kimi / Policy / --",
    "kimi_rubric_kimi_scorer": "Rubric ablation: Kimi / Kimi / --",
    "global_rubric_policy_scorer": "Rubric ablation: Policy (global) / Policy / --",
    "rubric_training_discriminativeness": "Rubric ablation: Policy / Policy / Rank+Disc.",
}


def paper_method_name(experiment: str) -> str:
    """Return the public label without changing the experiment identifier."""
    return PAPER_METHOD_NAMES.get(experiment, experiment)
