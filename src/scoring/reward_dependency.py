
"""Calculate transparent reward-dependency proxy scores."""

import numpy as np
import pandas as pd


def calculate_reward_dependency(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a provisional 0-100 reward-dependency score.

    Scores use discount sensitivity for every customer. For customers
    offered a reward, observed redemption contributes additional evidence.

    This is a heuristic, not a causal estimate of reward effectiveness.
    """

    required = {
        "customer_id",
        "discount_sensitivity",
        "reward_offered",
        "reward_redeemed",
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = data.copy()

    sensitivity = result["discount_sensitivity"].astype(float)

    if not sensitivity.between(0, 1).all():
        raise ValueError("discount_sensitivity must be between 0 and 1.")

    offered = result["reward_offered"].astype(bool)
    redeemed = result["reward_redeemed"].astype(bool)

    if (redeemed & ~offered).any():
        raise ValueError("A customer cannot redeem a reward that was not offered.")

    # Discount sensitivity provides a common baseline for every customer.
    # Redemption adds evidence only where reward exposure was observed.
    observed_score = (
        0.70 * sensitivity + 0.30 * redeemed.astype(float)
    ) * 100

    proxy_score = sensitivity * 100

    result["reward_dependency_score"] = np.where(
        offered,
        observed_score,
        proxy_score,
    ).round(1)

    result["score_evidence"] = np.where(
        offered,
        "reward_exposure_observed",
        "discount_sensitivity_proxy",
    )

    result["dependency_band"] = pd.cut(
        result["reward_dependency_score"],
        bins=[-0.01, 33.33, 66.67, 100],
        labels=["Low", "Moderate", "High"],
    ).astype(str)

    return result
