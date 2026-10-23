
"""Generate and validate synthetic customer reward data.

The dataset simulates randomized reward offers and customer purchasing
behavior for a reward-dependency and uplift-modeling research project.

IMPORTANT:
- All records are synthetic.
- Potential outcomes and probability columns are simulation ground truth.
- These columns must not be used as production model features.
- Synthetic results do not establish real-world causal effectiveness.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 10_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "reward_customer_data.csv"
)


def sigmoid(values):
    """Convert logits into probabilities in the interval [0, 1]."""
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def generate_customer_data(
    n_customers: int = N_CUSTOMERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate synthetic customer, treatment, and potential-outcome data."""
    if n_customers <= 0:
        raise ValueError("n_customers must be greater than zero.")

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # 1. Generate latent behavioral profiles
    # ---------------------------------------------------------------
    segments = np.array([
        "organic_loyalist",
        "reward_responsive",
        "incentive_dependent",
        "disengaged",
    ])

    segment_probabilities = [0.30, 0.30, 0.20, 0.20]

    simulation_segment = rng.choice(
        segments,
        size=n_customers,
        p=segment_probabilities,
    )

    profiles = {
        "organic_loyalist": {
            "purchase_probability": 0.72,
            "reward_effect": 0.04,
            "discount_sensitivity": 0.20,
            "engagement_score": 0.82,
            "retention_probability": 0.82,
        },
        "reward_responsive": {
            "purchase_probability": 0.40,
            "reward_effect": 0.25,
            "discount_sensitivity": 0.70,
            "engagement_score": 0.65,
            "retention_probability": 0.63,
        },
        "incentive_dependent": {
            "purchase_probability": 0.18,
            "reward_effect": 0.38,
            "discount_sensitivity": 0.92,
            "engagement_score": 0.48,
            "retention_probability": 0.42,
        },
        "disengaged": {
            "purchase_probability": 0.10,
            "reward_effect": 0.03,
            "discount_sensitivity": 0.35,
            "engagement_score": 0.18,
            "retention_probability": 0.20,
        },
    }

    profile_data = pd.DataFrame(
        [profiles[segment] for segment in simulation_segment]
    )

    baseline_purchase_profile = profile_data[
        "purchase_probability"
    ].to_numpy(dtype=float)

    reward_effect_strength = profile_data[
        "reward_effect"
    ].to_numpy(dtype=float)

    discount_sensitivity = np.clip(
        rng.normal(
            loc=profile_data["discount_sensitivity"].to_numpy(),
            scale=0.12,
        ),
        0.01,
        0.99,
    )

    engagement_score = np.clip(
        rng.normal(
            loc=profile_data["engagement_score"].to_numpy(),
            scale=0.12,
        ),
        0.01,
        0.99,
    )

    retention_profile = profile_data[
        "retention_probability"
    ].to_numpy(dtype=float)

    # ---------------------------------------------------------------
    # 2. Generate customer characteristics
    # ---------------------------------------------------------------
    tenure_months = np.clip(
        rng.gamma(shape=2.5, scale=12.0, size=n_customers),
        1,
        120,
    ).round().astype(int)

    prior_orders = rng.poisson(
        lam=np.maximum(0.5, tenure_months / 5.0)
    )

    avg_order_value_eur = np.clip(
        rng.normal(loc=48.0, scale=15.0, size=n_customers),
        8.0,
        200.0,
    ).round(2)

    # ---------------------------------------------------------------
    # 3. Randomized reward assignment
    # ---------------------------------------------------------------
    reward_offered = rng.binomial(
        n=1,
        p=0.50,
        size=n_customers,
    )

    possible_reward_amounts = np.array([3.0, 5.0, 10.0, 15.0])

    reward_amount_eur = np.where(
        reward_offered == 1,
        rng.choice(
            possible_reward_amounts,
            size=n_customers,
        ),
        0.0,
    )

    # ---------------------------------------------------------------
    # 4. Purchase probability without a reward
    # ---------------------------------------------------------------
    baseline_logit = (
        -1.25
        + 2.1 * baseline_purchase_profile
        + 0.025 * np.log1p(prior_orders)
        + 0.45 * engagement_score
    )

    baseline_probability = sigmoid(baseline_logit)

    # ---------------------------------------------------------------
    # 5. Purchase probability with the assigned reward
    # ---------------------------------------------------------------
    amount_factor = np.minimum(
        reward_amount_eur / 10.0,
        1.5,
    )

    offered_logit = (
        baseline_logit
        + 2.0 * reward_effect_strength * amount_factor
    )

    offered_probability = sigmoid(offered_logit)

    # ---------------------------------------------------------------
    # 6. Standardized €10 potential probability (NEW)
    # ---------------------------------------------------------------
    # This is the ground-truth probability for every customer if given
    # exactly €10, independent of the reward actually assigned.
    #
    # It is simulation-only information for oracle benchmarking and
    # validation. Do not include it in production model features.
    probability_with_reward_10 = sigmoid(
        baseline_logit + 2.0 * reward_effect_strength
    )

    # ---------------------------------------------------------------
    # 7. Generate binary potential outcomes
    # ---------------------------------------------------------------
    purchase_without_reward = rng.binomial(
        n=1,
        p=baseline_probability,
    )

    purchase_with_assigned_reward = rng.binomial(
        n=1,
        p=offered_probability,
    )

    # Binary potential outcome under a standardized €10 offer.
    potential_purchase_with_reward_10 = rng.binomial(
        n=1,
        p=probability_with_reward_10,
    )

    # Observed purchase follows the randomized assignment:
    # untreated customers use the no-reward potential outcome;
    # treated customers use the outcome for their assigned reward.
    purchase = np.where(
        reward_offered == 1,
        purchase_with_assigned_reward,
        purchase_without_reward,
    ).astype(int)

    # ---------------------------------------------------------------
    # 8. Reward redemption and reward costs
    # ---------------------------------------------------------------
    redemption_draw = rng.random(n_customers)

    reward_redeemed = (
        (reward_offered == 1)
        & (purchase == 1)
        & (redemption_draw < discount_sensitivity)
    ).astype(int)

    reward_cost_eur = (
        reward_redeemed * reward_amount_eur
    ).round(2)

    # ---------------------------------------------------------------
    # 9. Simulate 90-day retention
    # ---------------------------------------------------------------
    retention_logit = (
        -0.75
        + 2.2 * retention_profile
        + 0.65 * engagement_score
        + 0.45 * purchase
    )

    retention_probability = sigmoid(retention_logit)

    retained_90d = rng.binomial(
        n=1,
        p=retention_probability,
    )

    # ---------------------------------------------------------------
    # 10. Assemble dataset
    # ---------------------------------------------------------------
    data = pd.DataFrame({
        "customer_id": [
            f"CUST_{i:06d}"
            for i in range(1, n_customers + 1)
        ],
        "tenure_months": tenure_months,
        "prior_orders": prior_orders,
        "avg_order_value_eur": avg_order_value_eur,
        "engagement_score": engagement_score.round(4),
        "discount_sensitivity": discount_sensitivity.round(4),
        "reward_offered": reward_offered,
        "reward_amount_eur": reward_amount_eur,
        "reward_redeemed": reward_redeemed,
        "purchase": purchase,
        "retained_90d": retained_90d,
        "reward_cost_eur": reward_cost_eur,
        "simulation_segment": simulation_segment,

        # Binary potential outcomes retained for compatibility.
        "potential_purchase_no_reward": purchase_without_reward,
        "potential_purchase_with_reward": (
            potential_purchase_with_reward_10
        ),

        # NEW: underlying expected probabilities for validation.
        "purchase_probability_no_reward": (
            baseline_probability.round(6)
        ),
        "purchase_probability_with_reward_10": (
            probability_with_reward_10.round(6)
        ),
    })

    validate_customer_data(
        data=data,
        purchase_with_assigned_reward=purchase_with_assigned_reward,
    )

    return data


def validate_customer_data(
    data: pd.DataFrame,
    purchase_with_assigned_reward: np.ndarray,
) -> None:
    """Validate schema, probability ranges, and logical consistency."""
    required_columns = [
        "customer_id",
        "tenure_months",
        "prior_orders",
        "avg_order_value_eur",
        "engagement_score",
        "discount_sensitivity",
        "reward_offered",
        "reward_amount_eur",
        "reward_redeemed",
        "purchase",
        "retained_90d",
        "reward_cost_eur",
        "simulation_segment",
        "potential_purchase_no_reward",
        "potential_purchase_with_reward",
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]

    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if data["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if data.isna().any().any():
        raise ValueError("Missing values detected.")

    binary_columns = [
        "reward_offered",
        "reward_redeemed",
        "purchase",
        "retained_90d",
        "potential_purchase_no_reward",
        "potential_purchase_with_reward",
    ]

    for column in binary_columns:
        if not data[column].isin([0, 1]).all():
            raise ValueError(
                f"{column} contains values outside 0 and 1."
            )

    # NEW: Validate the ground-truth probability columns.
    probability_columns = [
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]

    for column in probability_columns:
        if not data[column].between(0, 1).all():
            raise ValueError(
                f"{column} contains invalid probabilities."
            )

    # Customers not offered rewards must have zero offer amount.
    no_offer = data["reward_offered"].eq(0)
    if not data.loc[no_offer, "reward_amount_eur"].eq(0).all():
        raise ValueError(
            "Customers without an offer have non-zero reward amounts."
        )

    # Customers offered rewards must have a valid offer amount.
    valid_reward_amounts = {3.0, 5.0, 10.0, 15.0}
    offered_amounts = set(
        data.loc[
            data["reward_offered"].eq(1),
            "reward_amount_eur",
        ].unique()
    )

    if not offered_amounts.issubset(valid_reward_amounts):
        raise ValueError(
            f"Unexpected reward amounts: {offered_amounts}"
        )

    # The observed purchase outcome must agree with assignment.
    treated = data["reward_offered"].eq(1)
    untreated = ~treated

    if not np.array_equal(
        data.loc[untreated, "purchase"].to_numpy(),
        data.loc[untreated, "potential_purchase_no_reward"].to_numpy(),
    ):
        raise ValueError(
            "Untreated observed purchases disagree with their "
            "no-reward potential outcomes."
        )

    if not np.array_equal(
        data.loc[treated, "purchase"].to_numpy(),
        np.asarray(purchase_with_assigned_reward)[treated.to_numpy()],
    ):
        raise ValueError(
            "Treated observed purchases disagree with outcomes under "
            "their assigned reward."
        )

    # Redemption requires a reward offer and a purchase.
    invalid_redemption = (
        data["reward_redeemed"].eq(1)
        & (
            data["reward_offered"].eq(0)
            | data["purchase"].eq(0)
        )
    )

    if invalid_redemption.any():
        raise ValueError(
            "Reward redemption without an offer and purchase detected."
        )

    expected_reward_cost = (
        data["reward_redeemed"] * data["reward_amount_eur"]
    ).round(2)

    if not np.allclose(
        data["reward_cost_eur"],
        expected_reward_cost,
    ):
        raise ValueError("Reward costs are inconsistent with redemption.")


def main():
    """Generate, validate, and save the synthetic dataset."""
    data = generate_customer_data()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 65)
    print("REWARD DEPENDENCY INTELLIGENCE SYSTEM")
    print("Synthetic data generation and validation complete")
    print("=" * 65)

    print(f"Customers generated: {len(data):,}")
    print(f"Columns generated: {len(data.columns)}")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(f"Duplicate customer IDs: {int(data['customer_id'].duplicated().sum())}")
    print(f"Purchase rate: {data['purchase'].mean():.1%}")
    print(f"90-day retention rate: {data['retained_90d'].mean():.1%}")
    print(f"Reward redemption rate: {data['reward_redeemed'].mean():.1%}")
    print(f"Reward offer rate: {data['reward_offered'].mean():.1%}")
    print(f"Total reward cost: €{data['reward_cost_eur'].sum():,.2f}")

    print("\nSimulation segment distribution:")
    print(data["simulation_segment"].value_counts())

    print("\nGround-truth purchase probabilities:")
    print(
        data[
            [
                "purchase_probability_no_reward",
                "purchase_probability_with_reward_10",
            ]
        ].describe().round(4)
    )

    print(f"\nSaved validated dataset to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
