
"""Rigorous evaluation of reward targeting policies.

Research design:
- Restrict comparison to no reward versus an assigned €10 reward.
- Split data into model training, policy validation, and final test sets.
- Train separate outcome models for control and €10 treatment.
- Select the targeting budget using validation-set IPW policy value.
- Evaluate the selected policy once on the held-out test set.
- Report bootstrap confidence intervals for estimated net profit.

This is a synthetic randomized experiment. Results are not evidence of
real-world effectiveness.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "reward_customer_data.csv"
OUTPUT_DIR = ROOT / "reports" / "metrics"

POLICY_OUTPUT = OUTPUT_DIR / "policy_budget_sensitivity.csv"
FINAL_OUTPUT = OUTPUT_DIR / "final_policy_evaluation.csv"

SEED = 42
REWARD_EUR = 10.0
GROSS_MARGIN = 0.30
TREATMENT_PROPENSITY = 0.20

BUDGET_SHARES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BOOTSTRAP_REPLICATES = 1000

FEATURES = [
    "tenure_months",
    "prior_orders",
    "avg_order_value_eur",
    "engagement_score",
    "discount_sensitivity",
]


def load_eligible_data():
    """Load control and standardized €10 treatment observations."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)

    required = {
        "customer_id",
        "reward_offered",
        "reward_amount_eur",
        "purchase",
        "reward_redeemed",
        "reward_cost_eur",
        "avg_order_value_eur",
        "potential_purchase_no_reward",
        "potential_purchase_with_reward",
        "simulation_segment",
        *FEATURES,
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    # Control group: no reward.
    # Treatment group: assigned exactly €10.
    control_mask = data["reward_offered"].eq(0)
    treatment_mask = (
        data["reward_offered"].eq(1)
        & np.isclose(data["reward_amount_eur"], REWARD_EUR)
    )

    eligible = data.loc[control_mask | treatment_mask].copy()
    eligible["treatment"] = treatment_mask.loc[eligible.index].astype(int)

    if eligible[FEATURES].isna().any().any():
        raise ValueError("Missing model features detected.")

    observed_propensity = eligible["treatment"].mean()

    if not 0 < observed_propensity < 1:
        raise ValueError("Both treatment and control groups are required.")

    print(f"Eligible observations: {len(eligible):,}")
    print(f"Control observations: {(eligible['treatment'] == 0).sum():,}")
    print(f"€10 treatment observations: {(eligible['treatment'] == 1).sum():,}")
    print(f"Observed treatment share among eligible: {observed_propensity:.3f}")
    print(f"Design treatment propensity assumed: {TREATMENT_PROPENSITY:.3f}")

    # In this generator, reward offer is randomized and the amount is
    # uniformly selected from four values. Conditional on being eligible
    # for this comparison, the theoretical €10 treatment probability is 0.2.
    if abs(observed_propensity - TREATMENT_PROPENSITY) > 0.04:
        raise ValueError(
            "Observed treatment share differs materially from the expected "
            "randomized design. Audit the generator before proceeding."
        )

    return eligible


def split_data(data):
    """Create independent train, validation, and final test partitions."""
    train, temporary = train_test_split(
        data,
        test_size=0.40,
        random_state=SEED,
        stratify=data["treatment"],
    )

    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=SEED + 1,
        stratify=temporary["treatment"],
    )

    return train.copy(), validation.copy(), test.copy()


def train_outcome_models(train):
    """Estimate P(purchase | X, control) and P(purchase | X, €10)."""
    control = train.loc[train["treatment"].eq(0)]
    treated = train.loc[train["treatment"].eq(1)]

    settings = {
        "n_estimators": 250,
        "min_samples_leaf": 30,
        "max_features": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
    }

    control_model = RandomForestClassifier(**settings)
    treatment_model = RandomForestClassifier(**settings)

    control_model.fit(control[FEATURES], control["purchase"])
    treatment_model.fit(treated[FEATURES], treated["purchase"])

    return control_model, treatment_model


def estimate_uplift(data, control_model, treatment_model):
    """Predict the individual change in purchase probability."""
    p0 = control_model.predict_proba(data[FEATURES])[:, 1]
    p1 = treatment_model.predict_proba(data[FEATURES])[:, 1]

    return p0, p1, p1 - p0


def assign_top_share(uplift, share):
    """Select the top-ranked customers for a given budget share."""
    n = len(uplift)
    n_target = int(np.ceil(n * share))

    assignment = np.zeros(n, dtype=int)

    if n_target > 0:
        selected = np.argsort(-np.asarray(uplift))[:n_target]
        assignment[selected] = 1

    return assignment


def observed_customer_profit(data):
    """Construct observed profit outcomes under each randomized arm."""
    revenue_margin = (
        data["purchase"].astype(float)
        * data["avg_order_value_eur"].astype(float)
        * GROSS_MARGIN
    )

    actual_reward_cost = data["reward_cost_eur"].astype(float)

    # In this restricted dataset, treated customers were assigned €10;
    # control customers have zero reward cost.
    return revenue_margin - actual_reward_cost


def ipw_policy_contributions(data, assignment):
    """Calculate row-level IPW contributions for a deterministic policy.

    If the policy selects control, only randomized controls contribute.
    If it selects treatment, only randomized €10-treated customers
    contribute. The known design propensities are used for weighting.
    """
    assignment = np.asarray(assignment, dtype=int)
    treatment = data["treatment"].to_numpy(dtype=int)

    observed_profit = observed_customer_profit(data).to_numpy()

    p1 = TREATMENT_PROPENSITY
    p0 = 1.0 - p1

    matches_policy = treatment == assignment
    propensity = np.where(assignment == 1, p1, p0)

    return np.where(
        matches_policy,
        observed_profit / propensity,
        0.0,
    )


def policy_value(data, assignment):
    """Estimate policy value per eligible customer using IPW."""
    contributions = ipw_policy_contributions(data, assignment)
    return float(np.mean(contributions))


def bootstrap_policy_value(data, assignment, rng):
    """Bootstrap the IPW value estimate and return a 95% percentile CI."""
    assignment = np.asarray(assignment, dtype=int)
    contributions = ipw_policy_contributions(data, assignment)

    n = len(contributions)
    estimates = np.empty(BOOTSTRAP_REPLICATES)

    for b in range(BOOTSTRAP_REPLICATES):
        sample_idx = rng.integers(0, n, size=n)
        estimates[b] = contributions[sample_idx].mean()

    lower, upper = np.quantile(estimates, [0.025, 0.975])

    return float(contributions.mean()), float(lower), float(upper)


def choose_budget(validation, validation_uplift):
    """Select budget using validation-set IPW value only."""
    records = []

    no_reward_assignment = np.zeros(len(validation), dtype=int)
    baseline_value = policy_value(validation, no_reward_assignment)

    for share in BUDGET_SHARES:
        assignment = assign_top_share(validation_uplift, share)
        value = policy_value(validation, assignment)

        records.append({
            "budget_share_pct": share * 100,
            "customers_targeted": int(assignment.sum()),
            "estimated_net_profit_per_customer_eur": value,
            "incremental_net_profit_per_customer_vs_control_eur": (
                value - baseline_value
            ),
        })

    results = pd.DataFrame(records)

    # Include zero reward spending as a valid candidate.
    zero_row = pd.DataFrame([{
        "budget_share_pct": 0.0,
        "customers_targeted": 0,
        "estimated_net_profit_per_customer_eur": baseline_value,
        "incremental_net_profit_per_customer_vs_control_eur": 0.0,
    }])

    results = pd.concat([zero_row, results], ignore_index=True)

    # Deterministic tie-break: choose the lower budget when values tie.
    results = results.sort_values(
        ["estimated_net_profit_per_customer_eur", "budget_share_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)

    selected_share = float(results.iloc[0]["budget_share_pct"]) / 100.0

    return selected_share, results


def evaluate_final_test(test, test_uplift, selected_share):
    """Evaluate frozen policies on the untouched test set."""
    rng = np.random.default_rng(SEED + 100)
    policies = {
        "No rewards": np.zeros(len(test), dtype=int),
        "Blanket €10 rewards": np.ones(len(test), dtype=int),
        "Practical uplift targeting": assign_top_share(
            test_uplift, selected_share
        ),
    }

    # Oracle benchmark is shown separately because it uses hidden
    # simulation labels unavailable in a production system.
    policies["Oracle benchmark (hidden labels)"] = (
        test["simulation_segment"]
        .eq("incentive_dependent")
        .astype(int)
        .to_numpy()
    )

    records = []

    for name, assignment in policies.items():
        value, lower, upper = bootstrap_policy_value(test, assignment, rng)

        records.append({
            "policy": name,
            "test_customers": len(test),
            "customers_targeted": int(np.sum(assignment)),
            "targeting_rate_pct": round(np.mean(assignment) * 100, 2),
            "estimated_net_profit_per_customer_eur": round(value, 4),
            "ci_95_lower_eur_per_customer": round(lower, 4),
            "ci_95_upper_eur_per_customer": round(upper, 4),
        })

    results = pd.DataFrame(records)

    baseline_value = results.loc[
        results["policy"] == "No rewards",
        "estimated_net_profit_per_customer_eur",
    ].iloc[0]

    results["incremental_profit_per_customer_vs_control_eur"] = (
        results["estimated_net_profit_per_customer_eur"] - baseline_value
    )

    # Approximate total impact if the eligible test population were scaled
    # to 10,000 customers. This is a linear extrapolation, not a new estimate.
    results["scaled_incremental_profit_per_10000_customers_eur"] = (
        results["incremental_profit_per_customer_vs_control_eur"] * 10000
    )

    results["evaluation_method"] = "IPW on randomized held-out observations"
    results["gross_margin_assumption_pct"] = GROSS_MARGIN * 100

    return results


def main():
    data = load_eligible_data()
    train, validation, test = split_data(data)

    print("\nPARTITION SIZES")
    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Final test: {len(test):,}")

    control_model, treatment_model = train_outcome_models(train)

    _, _, validation_uplift = estimate_uplift(
        validation, control_model, treatment_model
    )
    selected_share, budget_results = choose_budget(
        validation, validation_uplift
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget_results.to_csv(POLICY_OUTPUT, index=False)

    print("\nVALIDATION: BUDGET SENSITIVITY")
    print(budget_results.to_string(index=False))
    print(f"\nSelected budget share: {selected_share:.0%}")

    _, _, test_uplift = estimate_uplift(test, control_model, treatment_model)

    final_results = evaluate_final_test(
        test, test_uplift, selected_share
    )
    final_results.to_csv(FINAL_OUTPUT, index=False)

    print("\nFINAL HELD-OUT TEST EVALUATION")
    print(final_results.to_string(index=False))

    print("\nSaved files:")
    print(f"Budget sensitivity: {POLICY_OUTPUT}")
    print(f"Final evaluation:   {FINAL_OUTPUT}")

    print(
        "\nInterpretation caveats:\n"
        "- IPW estimates are noisy, especially for smaller treatment groups.\n"
        "- Confidence intervals describe sampling uncertainty, not all model "
        "or simulation uncertainty.\n"
        "- The oracle uses hidden labels and is not deployable.\n"
        "- The uplift model is evaluated on synthetic randomized data.\n"
        "- Budget selection is performed on validation data; the final test "
        "set is reserved for final evaluation."
    )


if __name__ == "__main__":
    main()
