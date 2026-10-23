
"""Generate and validate synthetic customer reward data."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 10_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "reward_customer_data.csv"


def sigmoid(values):
    """Convert logits into probabilities between zero and one."""
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def generate_customer_data(n_customers=N_CUSTOMERS, seed=SEED):
    """Generate synthetic customers, randomized rewards, and outcomes."""
    if n_customers <= 0:
        raise ValueError("n_customers must be greater than zero.")

    rng = np.random.default_rng(seed)

    segments = np.array([
        "organic_loyalist",
        "reward_responsive",
        "incentive_dependent",
        "disengaged",
    ])

    simulation_segment = rng.choice(
        segments,
        size=n_customers,
        p=[0.30, 0.30, 0.20, 0.20],
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
            profile_data["discount_sensitivity"].to_numpy(),
            0.12,
        ),
        0.01,
        0.99,
    )

    engagement_score = np.clip(
        rng.normal(
            profile_data["engagement_score"].to_numpy(),
            0.12,
        ),
        0.01,
        0.99,
    )

    retention_profile = profile_data[
        "retention_probability"
    ].to_numpy(dtype=float)

    # Customer characteristics
    tenure_months = np.clip(
        rng.gamma(2.5, 12.0, size=n_customers),
        1,
        120,
    ).round().astype(int)

    prior_orders = rng.poisson(
        np.maximum(0.5, tenure_months / 5.0)
    )

    avg_order_value_eur = np.clip(
        rng.normal(48.0, 15.0, size=n_customers),
        8.0,
        200.0,
    ).round(2)

    # Randomized reward assignment
    reward_offered = rng.binomial(1, 0.50, size=n_customers)

    reward_amount_eur = np.where(
        reward_offered == 1,
        rng.choice([3.0, 5.0, 10.0, 15.0], size=n_customers),
        0.0,
    )

    # Purchase probability without a reward
    baseline_logit = (
        -1.25
        + 2.1 * baseline_purchase_profile
        + 0.025 * np.log1p(prior_orders)
        + 0.45 * engagement_score
    )

    baseline_probability = sigmoid(baseline_logit)

    # Purchase probability under the customer's assigned reward
    amount_factor = np.minimum(reward_amount_eur / 10.0, 1.5)

    offered_logit = (
        baseline_logit
        + 2.0 * reward_effect_strength * amount_factor
    )

    offered_probability = sigmoid(offered_logit)

    # Ground-truth probability under a standardized €10 reward.
    # Simulation-only information; not a production model feature.
    probability_with_reward_10 = sigmoid(
        baseline_logit + 2.0 * reward_effect_strength
    )

    # Generate potential outcomes
    purchase_without_reward = rng.binomial(
        1, baseline_probability
    )

    purchase_with_assigned_reward = rng.binomial(
        1, offered_probability
    )

    potential_purchase_with_reward_10 = rng.binomial(
        1, probability_with_reward_10
    )

    # Observed outcome follows randomized assignment.
    purchase = np.where(
        reward_offered == 1,
        purchase_with_assigned_reward,
        purchase_without_reward,
    ).astype(int)

    # Redemption and costs
    reward_redeemed = (
        (reward_offered == 1)
        & (purchase == 1)
        & (rng.random(n_customers) < discount_sensitivity)
    ).astype(int)

    reward_cost_eur = (
        reward_redeemed * reward_amount_eur
    ).round(2)

    # 90-day retention
    retention_logit = (
        -0.75
        + 2.2 * retention_profile
        + 0.65 * engagement_score
        + 0.45 * purchase
    )

    retention_probability = sigmoid(retention_logit)
    retained_90d = rng.binomial(1, retention_probability)

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
        "potential_purchase_no_reward": purchase_without_reward,
        "potential_purchase_with_reward": (
            potential_purchase_with_reward_10
        ),
        "purchase_probability_no_reward": baseline_probability.round(6),
        "purchase_probability_with_reward_10": (
            probability_with_reward_10.round(6)
        ),
    })

    validate_customer_data(
        data,
        purchase_with_assigned_reward,
    )

    return data


def validate_customer_data(data, purchase_with_assigned_reward):
    """Validate schema, probabilities, treatment consistency, and costs."""
    required_columns = {
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
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Dataset is empty.")

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

    # Validate both simulation ground-truth probability columns.
    probability_columns = [
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]

    for column in probability_columns:
        if not data[column].between(0, 1).all():
            raise ValueError(
                f"{column} contains invalid probabilities."
            )

    if not data["engagement_score"].between(0, 1).all():
        raise ValueError("Invalid engagement scores.")

    if not data["discount_sensitivity"].between(0, 1).all():
        raise ValueError("Invalid discount sensitivity values.")

    if (data["avg_order_value_eur"] <= 0).any():
        raise ValueError("Average order values must be positive.")

    # Reward amounts must agree with treatment assignment.
    no_offer = data["reward_offered"].eq(0)
    offered = data["reward_offered"].eq(1)

    if not data.loc[no_offer, "reward_amount_eur"].eq(0).all():
        raise ValueError(
            "Customers without a reward offer have non-zero reward amounts."
        )

    valid_amounts = {3.0, 5.0, 10.0, 15.0}
    actual_amounts = set(
        data.loc[offered, "reward_amount_eur"].unique()
    )

    if not actual_amounts.issubset(valid_amounts):
        raise ValueError(
            f"Unexpected reward amounts: {actual_amounts}"
        )

    # Untreated observed purchases must match the no-reward outcome.
    if not np.array_equal(
        data.loc[no_offer, "purchase"].to_numpy(),
        data.loc[
            no_offer, "potential_purchase_no_reward"
        ].to_numpy(),
    ):
        raise ValueError(
            "Untreated purchases are inconsistent with assignment."
        )

    # Treated observed purchases must match their assigned-reward outcome.
    assigned_outcome = np.asarray(purchase_with_assigned_reward)

    if assigned_outcome.shape != (len(data),):
        raise ValueError("Assigned-reward outcome has an invalid shape.")

    if not np.array_equal(
        data.loc[offered, "purchase"].to_numpy(),
        assigned_outcome[offered.to_numpy()],
    ):
        raise ValueError(
            "Treated purchases are inconsistent with assignment."
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

    expected_cost = (
        data["reward_redeemed"] * data["reward_amount_eur"]
    ).round(2)

    if not np.allclose(
        data["reward_cost_eur"],
        expected_cost,
    ):
        raise ValueError(
            "Reward costs do not match redeemed reward amounts."
        )

    # Sanity check: the standardized €10 probability should be at least
    # the no-reward probability for this generator's non-negative effects.
    if (
        data["purchase_probability_with_reward_10"]
        < data["purchase_probability_no_reward"] - 1e-6
    ).any():
        raise ValueError(
            "Unexpected negative treatment effect in the generator."
        )


def main():
    """Generate, validate, and save the customer dataset."""
    data = generate_customer_data()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(OUTPUT_FILE, index=False)

    print("=" * 65)
    print("REWARD DEPENDENCY INTELLIGENCE SYSTEM")
    print("Synthetic data generation and validation complete")
    print("=" * 65)

    print(f"Customers generated: {len(data):,}")
    print(f"Columns generated: {len(data.columns)}")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(
        "Duplicate customer IDs: "
        f"{int(data['customer_id'].duplicated().sum())}"
    )
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
