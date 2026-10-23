
"""Evaluate reward policies with a held-out synthetic test set.

Policies:
- No rewards
- Blanket €10 rewards
- Oracle targeting using hidden simulation labels
- Practical targeting using a learned uplift model

All results are simulated estimates, not real-world causal evidence.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "reward_customer_data.csv"
OUTPUT_FILE = (
    PROJECT_ROOT / "reports" / "metrics" / "reward_policy_comparison.csv"
)

SEED = 42
REWARD_AMOUNT_EUR = 10.0
PROFIT_MARGIN = 0.30
TARGETING_BUDGET_SHARE = 0.20

FEATURES = [
    "tenure_months",
    "prior_orders",
    "avg_order_value_eur",
    "engagement_score",
    "discount_sensitivity",
]


def load_data():
    """Load the dataset and validate required columns."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}. "
            "Run the data generator first."
        )

    data = pd.read_csv(INPUT_FILE)

    required = {
        "customer_id",
        "avg_order_value_eur",
        "discount_sensitivity",
        "reward_amount_eur",
        "reward_offered",
        "purchase",
        "simulation_segment",
        "potential_purchase_no_reward",
        "potential_purchase_with_reward",
        *FEATURES,
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["customer_id"].duplicated().any():
        raise ValueError("Customer IDs must be unique.")

    if data[FEATURES].isna().any().any():
        raise ValueError("Model features contain missing values.")

    return data


def split_data(data):
    """Create reproducible training and test partitions."""
    train_ids, test_ids = train_test_split(
        data.index,
        test_size=0.20,
        random_state=SEED,
        stratify=data["reward_offered"],
    )

    return data.loc[train_ids].copy(), data.loc[test_ids].copy()


def train_uplift_models(train):
    """Train separate purchase models for control and €10 treatment.

    The control model uses customers assigned no reward.
    The treatment model uses customers assigned exactly €10.
    Other reward amounts are excluded from this standardized comparison.
    """
    control = train.loc[train["reward_offered"].eq(0)]
    treatment_10 = train.loc[
        train["reward_offered"].eq(1)
        & np.isclose(train["reward_amount_eur"], REWARD_AMOUNT_EUR)
    ]

    if len(control) < 100 or len(treatment_10) < 100:
        raise ValueError(
            "Not enough control or €10 treatment observations to train "
            "the uplift models. Check the generated reward amounts."
        )

    if control["purchase"].nunique() < 2 or treatment_10["purchase"].nunique() < 2:
        raise ValueError("Each model training group must contain both outcomes.")

    common_settings = {
        "n_estimators": 250,
        "min_samples_leaf": 30,
        "max_features": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
    }

    no_reward_model = RandomForestClassifier(**common_settings)
    reward_model = RandomForestClassifier(**common_settings)

    no_reward_model.fit(control[FEATURES], control["purchase"])
    reward_model.fit(treatment_10[FEATURES], treatment_10["purchase"])

    return no_reward_model, reward_model


def predict_uplift(test, no_reward_model, reward_model):
    """Estimate purchase-probability change from a standardized €10 offer."""
    p_no_reward = no_reward_model.predict_proba(test[FEATURES])[:, 1]
    p_reward = reward_model.predict_proba(test[FEATURES])[:, 1]

    return p_reward - p_no_reward


def create_policy_assignments(test, estimated_uplift):
    """Create policy assignments using test features and model predictions."""
    no_rewards = pd.Series(0, index=test.index, dtype=int)
    blanket = pd.Series(1, index=test.index, dtype=int)

    # Oracle benchmark: hidden segment labels are available only because
    # this is a synthetic experiment. Never use this as production logic.
    oracle = test["simulation_segment"].eq("incentive_dependent").astype(int)

    # Practical targeting: choose the top 20% of test customers by
    # predicted incremental purchase probability.
    n_target = max(
        1,
        math.ceil(len(test) * TARGETING_BUDGET_SHARE),
    )
    top_positions = np.argsort(estimated_uplift)[-n_target:]
    practical = pd.Series(0, index=test.index, dtype=int)
    practical.iloc[top_positions] = 1

    return {
        "No rewards": no_rewards,
        "Blanket rewards": blanket,
        "Oracle targeting (benchmark)": oracle,
        "Practical targeting (uplift model)": practical,
    }


def evaluate_policy(test, assignment, policy_name):
    """Evaluate policy using synthetic potential outcomes on held-out data."""
    assignment = assignment.reindex(test.index).astype(int)

    y0 = test["potential_purchase_no_reward"].astype(int)
    y10 = test["potential_purchase_with_reward"].astype(int)

    # Simulated policy outcome: use the relevant potential outcome
    # for each customer's assigned policy.
    purchases = assignment * y10 + (1 - assignment) * y0

    # Expected redemption is conditional on purchase. The generator uses
    # discount_sensitivity as the redemption probability.
    sensitivity = test["discount_sensitivity"].clip(0, 1)
    expected_redemptions = assignment * purchases * sensitivity
    expected_reward_cost = expected_redemptions * REWARD_AMOUNT_EUR

    revenue = purchases * test["avg_order_value_eur"]
    gross_profit = revenue * PROFIT_MARGIN
    net_profit = gross_profit - expected_reward_cost

    return {
        "policy": policy_name,
        "test_customers": len(test),
        "customers_targeted": int(assignment.sum()),
        "targeting_rate_pct": round(assignment.mean() * 100, 2),
        "simulated_purchases": int(purchases.sum()),
        "simulated_purchase_rate_pct": round(purchases.mean() * 100, 2),
        "estimated_revenue_eur": round(revenue.sum(), 2),
        "estimated_gross_profit_eur": round(gross_profit.sum(), 2),
        "estimated_reward_cost_eur": round(expected_reward_cost.sum(), 2),
        "estimated_net_profit_eur": round(net_profit.sum(), 2),
    }


def main():
    data = load_data()
    train, test = split_data(data)

    no_reward_model, reward_model = train_uplift_models(train)
    uplift = predict_uplift(test, no_reward_model, reward_model)

    assignments = create_policy_assignments(test, uplift)

    results = [
        evaluate_policy(test, assignment, name)
        for name, assignment in assignments.items()
    ]
    comparison = pd.DataFrame(results)

    baseline = comparison.loc[
        comparison["policy"] == "No rewards"
    ].iloc[0]

    comparison["incremental_purchases_vs_no_rewards"] = (
        comparison["simulated_purchases"]
        - baseline["simulated_purchases"]
    )
    comparison["incremental_net_profit_vs_no_rewards_eur"] = (
        comparison["estimated_net_profit_eur"]
        - baseline["estimated_net_profit_eur"]
    )

    comparison["assumed_gross_margin_pct"] = PROFIT_MARGIN * 100
    comparison["standardized_reward_amount_eur"] = REWARD_AMOUNT_EUR
    comparison["evaluation_basis"] = (
        "Held-out synthetic potential outcomes; estimated redemption costs"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_FILE, index=False)

    print("=" * 76)
    print("REWARD POLICY EVALUATION — HELD-OUT TEST SET")
    print("=" * 76)
    print(f"Training customers: {len(train):,}")
    print(f"Test customers: {len(test):,}")
    print(f"Standardized reward: €{REWARD_AMOUNT_EUR:.2f}")
    print(f"Assumed gross margin: {PROFIT_MARGIN:.0%}")
    print(f"Practical targeting budget: {TARGETING_BUDGET_SHARE:.0%}")
    print()
    print(
        comparison[
            [
                "policy",
                "customers_targeted",
                "simulated_purchase_rate_pct",
                "simulated_purchases",
                "estimated_reward_cost_eur",
                "estimated_net_profit_eur",
                "incremental_purchases_vs_no_rewards",
                "incremental_net_profit_vs_no_rewards_eur",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved comparison to: {OUTPUT_FILE}")
    print()
    print(
        "Caveat: the oracle uses hidden synthetic labels. Practical targeting "
        "uses a learned model, but its policy performance is still evaluated "
        "with synthetic potential outcomes rather than real observed outcomes."
    )


if __name__ == "__main__":
    main()
