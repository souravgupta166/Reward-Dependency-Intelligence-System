
"""Advanced evaluation of reward targeting policies.

Research objectives:
1. Estimate heterogeneous purchase treatment effects.
2. Select a practical reward budget using validation data.
3. Compare practical targeting with a simulation oracle.
4. Estimate paired bootstrap confidence intervals for policy value.
5. Compare predicted uplift with known simulated ground truth.

All results are synthetic. Ground-truth columns are reserved for
simulation validation and must not be used as production features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "reward_customer_data.csv"
OUTPUT_DIR = ROOT / "reports" / "metrics"

BUDGET_FILE = OUTPUT_DIR / "advanced_policy_budget_sensitivity.csv"
POLICY_FILE = OUTPUT_DIR / "advanced_policy_comparison.csv"
DIAGNOSTICS_FILE = OUTPUT_DIR / "uplift_ranking_diagnostics.csv"

SEED = 20261023
REWARD_EUR = 10.0
GROSS_MARGIN = 0.30
TREATMENT_PROPENSITY = 0.20
BUDGET_SHARES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BOOTSTRAP_REPLICATES = 2000

# Observable features only: no latent segments, potential outcomes,
# or simulation ground-truth probabilities.
FEATURES = [
    "tenure_months",
    "prior_orders",
    "avg_order_value_eur",
    "engagement_score",
    "discount_sensitivity",
]


def load_data():
    """Load randomized control and standardized €10 treatment records."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)

    required = {
        "customer_id",
        "reward_offered",
        "reward_amount_eur",
        "purchase",
        "reward_cost_eur",
        "avg_order_value_eur",
        "discount_sensitivity",
        "potential_purchase_no_reward",
        "potential_purchase_with_reward",
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
        "simulation_segment",
        *FEATURES,
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if data[FEATURES].isna().any().any():
        raise ValueError("Model features contain missing values.")

    # Control = no reward; treatment = exactly €10.
    control_mask = data["reward_offered"].eq(0)
    treatment_mask = (
        data["reward_offered"].eq(1)
        & np.isclose(data["reward_amount_eur"], REWARD_EUR)
    )

    eligible = data.loc[control_mask | treatment_mask].copy()
    eligible["treatment"] = treatment_mask.loc[eligible.index].astype(int)

    observed_propensity = eligible["treatment"].mean()

    if abs(observed_propensity - TREATMENT_PROPENSITY) > 0.04:
        raise ValueError(
            f"Unexpected €10 treatment share: {observed_propensity:.3f}. "
            "Check randomized assignment and reward amounts."
        )

    print(f"Eligible customers: {len(eligible):,}")
    print(f"Control customers: {(eligible['treatment'] == 0).sum():,}")
    print(f"€10 treatment customers: {(eligible['treatment'] == 1).sum():,}")
    print(f"Observed treatment share: {observed_propensity:.3f}")

    return eligible


def split_data(data):
    """Create fixed train, validation, and test partitions."""
    train, remaining = train_test_split(
        data,
        test_size=0.40,
        random_state=SEED,
        stratify=data["treatment"],
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=SEED + 1,
        stratify=remaining["treatment"],
    )

    return train.copy(), validation.copy(), test.copy()


def train_outcome_models(train):
    """Estimate purchase probability under control and €10 treatment."""
    control = train.loc[train["treatment"].eq(0)]
    treated = train.loc[train["treatment"].eq(1)]

    settings = {
        "n_estimators": 300,
        "min_samples_leaf": 30,
        "max_features": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
    }

    model_0 = RandomForestClassifier(**settings)
    model_1 = RandomForestClassifier(**settings)

    model_0.fit(control[FEATURES], control["purchase"])
    model_1.fit(treated[FEATURES], treated["purchase"])

    return model_0, model_1


def predict_policy_scores(data, model_0, model_1):
    """Predict uplift and incremental economic benefit per customer."""
    p0 = model_0.predict_proba(data[FEATURES])[:, 1]
    p1 = model_1.predict_proba(data[FEATURES])[:, 1]

    uplift = p1 - p0

    # Expected incremental contribution from the extra purchases,
    # less expected voucher cost if the customer purchases and redeems.
    aov = data["avg_order_value_eur"].to_numpy(dtype=float)
    sensitivity = data["discount_sensitivity"].to_numpy(dtype=float)

    predicted_net_benefit = (
        uplift * aov * GROSS_MARGIN
        - p1 * sensitivity * REWARD_EUR
    )

    return p0, p1, uplift, predicted_net_benefit


def assign_top_share(scores, share):
    """Target the top-ranked share of customers."""
    scores = np.asarray(scores, dtype=float)
    assignment = np.zeros(len(scores), dtype=int)

    n_target = int(np.ceil(len(scores) * share))
    if n_target > 0:
        order = np.argsort(-scores, kind="stable")
        assignment[order[:n_target]] = 1

    return assignment


def observed_customer_profit(data):
    """Calculate observed contribution after actual reward costs."""
    gross_profit = (
        data["purchase"].astype(float)
        * data["avg_order_value_eur"].astype(float)
        * GROSS_MARGIN
    )

    return gross_profit - data["reward_cost_eur"].astype(float)


def ipw_contributions(data, assignment):
    """Inverse-propensity-weighted contributions for a fixed policy."""
    assignment = np.asarray(assignment, dtype=int)
    treatment = data["treatment"].to_numpy(dtype=int)
    profit = observed_customer_profit(data).to_numpy()

    propensity = np.where(
        assignment == 1,
        TREATMENT_PROPENSITY,
        1.0 - TREATMENT_PROPENSITY,
    )

    follows_policy = treatment == assignment

    return np.where(
        follows_policy,
        profit / propensity,
        0.0,
    )


def policy_value(data, assignment):
    """Estimate expected net profit per eligible customer."""
    return float(ipw_contributions(data, assignment).mean())


def select_practical_budget(validation, predicted_benefit):
    """Select a targeting share by validation-set IPW policy value."""
    rows = []

    candidate_shares = [0.0] + BUDGET_SHARES

    for share in candidate_shares:
        assignment = assign_top_share(predicted_benefit, share)

        rows.append({
            "budget_share_pct": share * 100,
            "customers_targeted": int(assignment.sum()),
            "ipw_net_profit_per_customer_eur": policy_value(
                validation, assignment
            ),
        })

    results = pd.DataFrame(rows)

    baseline_value = results.loc[
        results["budget_share_pct"].eq(0),
        "ipw_net_profit_per_customer_eur",
    ].iloc[0]

    results["incremental_profit_vs_no_reward_eur"] = (
        results["ipw_net_profit_per_customer_eur"] - baseline_value
    )

    # Conservative tie-break: choose the smaller budget.
    results = results.sort_values(
        ["ipw_net_profit_per_customer_eur", "budget_share_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)

    selected_share = float(results.iloc[0]["budget_share_pct"]) / 100.0

    return selected_share, results


def true_incremental_net_benefit(data):
    """Expected incremental net benefit from simulation ground truth.

    This is oracle-only information and must not enter production features.
    """
    p0 = data["purchase_probability_no_reward"].to_numpy(dtype=float)
    p1 = data["purchase_probability_with_reward_10"].to_numpy(dtype=float)

    aov = data["avg_order_value_eur"].to_numpy(dtype=float)
    sensitivity = data["discount_sensitivity"].to_numpy(dtype=float)

    incremental_margin = (p1 - p0) * aov * GROSS_MARGIN
    expected_reward_cost = p1 * sensitivity * REWARD_EUR

    return incremental_margin - expected_reward_cost


def select_oracle_budget(validation):
    """Find the best oracle budget using validation ground truth only."""
    benefit = true_incremental_net_benefit(validation)
    rows = []

    for share in [0.0] + BUDGET_SHARES:
        assignment = assign_top_share(benefit, share)

        rows.append({
            "budget_share_pct": share * 100,
            "customers_targeted": int(assignment.sum()),
            "total_expected_incremental_net_benefit_eur": float(
                benefit[assignment == 1].sum()
            ),
        })

    results = pd.DataFrame(rows).sort_values(
        ["total_expected_incremental_net_benefit_eur", "budget_share_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)

    selected_share = float(results.iloc[0]["budget_share_pct"]) / 100.0

    return selected_share, results


def paired_bootstrap_difference(data, assignment, rng):
    """95% bootstrap CI for policy profit minus no-reward profit."""
    assignment = np.asarray(assignment, dtype=int)
    control_assignment = np.zeros(len(data), dtype=int)

    policy_contrib = ipw_contributions(data, assignment)
    control_contrib = ipw_contributions(data, control_assignment)

    paired_differences = policy_contrib - control_contrib

    estimate = float(paired_differences.mean())
    n = len(paired_differences)

    bootstrap_means = np.empty(BOOTSTRAP_REPLICATES)

    for b in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[b] = paired_differences[indices].mean()

    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    return estimate, float(lower), float(upper)


def evaluate_test_policies(
    test,
    predicted_benefit,
    practical_share,
    oracle_share,
):
    """Evaluate frozen policies on the held-out test set."""
    rng = np.random.default_rng(SEED + 100)

    oracle_benefit = true_incremental_net_benefit(test)

    policies = {
        "No rewards": np.zeros(len(test), dtype=int),
        "Blanket €10 rewards": np.ones(len(test), dtype=int),
        "Practical targeting": assign_top_share(
            predicted_benefit, practical_share
        ),
        "Simulation oracle": assign_top_share(
            oracle_benefit, oracle_share
        ),
    }

    records = []

    for name, assignment in policies.items():
        estimate, lower, upper = paired_bootstrap_difference(
            test, assignment, rng
        )

        records.append({
            "policy": name,
            "test_customers": len(test),
            "customers_targeted": int(assignment.sum()),
            "targeting_rate_pct": round(assignment.mean() * 100, 2),
            "incremental_profit_per_customer_eur": estimate,
            "paired_ci_lower_eur_per_customer": lower,
            "paired_ci_upper_eur_per_customer": upper,
            "scaled_incremental_profit_per_10000_customers_eur": (
                estimate * 10_000
            ),
            "true_expected_incremental_net_benefit_eur": (
                float(oracle_benefit[assignment == 1].sum())
            ),
        })

    return pd.DataFrame(records)


def uplift_ranking_diagnostics(test, predicted_uplift, predicted_benefit):
    """Assess ranking against the simulator's known treatment effects."""
    p0 = test["purchase_probability_no_reward"].to_numpy(dtype=float)
    p1 = test["purchase_probability_with_reward_10"].to_numpy(dtype=float)

    true_purchase_uplift = p1 - p0
    true_economic_benefit = true_incremental_net_benefit(test)

    uplift_corr = np.corrcoef(
        predicted_uplift, true_purchase_uplift
    )[0, 1]

    benefit_corr = np.corrcoef(
        predicted_benefit, true_economic_benefit
    )[0, 1]

    rows = []

    for share in [0.05, 0.10, 0.15, 0.20, 0.30, 1.0]:
        selected = assign_top_share(predicted_benefit, share).astype(bool)

        rows.append({
            "top_share_pct": share * 100,
            "customers_selected": int(selected.sum()),
            "mean_true_purchase_uplift": float(
                true_purchase_uplift[selected].mean()
            ),
            "mean_true_net_benefit_eur": float(
                true_economic_benefit[selected].mean()
            ),
            "total_true_net_benefit_eur": float(
                true_economic_benefit[selected].sum()
            ),
            "predicted_uplift_vs_true_uplift_correlation": float(
                uplift_corr
            ),
            "predicted_net_benefit_vs_true_benefit_correlation": float(
                benefit_corr
            ),
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    train, validation, test = split_data(data)

    print("\nDATA PARTITIONS")
    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Final test: {len(test):,}")

    model_0, model_1 = train_outcome_models(train)

    _, _, validation_uplift, validation_benefit = predict_policy_scores(
        validation, model_0, model_1
    )

    practical_share, practical_budget_results = select_practical_budget(
        validation, validation_benefit
    )

    oracle_share, oracle_budget_results = select_oracle_budget(validation)

    practical_budget_results.to_csv(BUDGET_FILE, index=False)

    print("\nVALIDATION: PRACTICAL POLICY BUDGET")
    print(practical_budget_results.to_string(index=False))
    print(f"\nSelected practical budget: {practical_share:.0%}")

    print("\nVALIDATION: ORACLE BENCHMARK BUDGET")
    print(oracle_budget_results.to_string(index=False))
    print(f"\nSelected oracle benchmark budget: {oracle_share:.0%}")

    test_p0, test_p1, test_uplift, test_benefit = predict_policy_scores(
        test, model_0, model_1
    )

    diagnostics = uplift_ranking_diagnostics(
        test, test_uplift, test_benefit
    )
    diagnostics.to_csv(DIAGNOSTICS_FILE, index=False)

    print("\nUPLIFT AND ECONOMIC RANKING DIAGNOSTICS")
    print(diagnostics.to_string(index=False))

    policy_results = evaluate_test_policies(
        test,
        test_benefit,
        practical_share,
        oracle_share,
    )
    policy_results.to_csv(POLICY_FILE, index=False)

    print("\nFINAL HELD-OUT TEST POLICY COMPARISON")
    print(policy_results.to_string(index=False))

    print("\nSAVED OUTPUTS")
    print(f"Practical budget sensitivity: {BUDGET_FILE}")
    print(f"Policy comparison:             {POLICY_FILE}")
    print(f"Uplift diagnostics:             {DIAGNOSTICS_FILE}")

    print(
        "\nRESEARCH CAVEATS\n"
        "- Policy budgets are selected on validation data only.\n"
        "- The final test set is used for final policy evaluation.\n"
        "- Paired bootstrap intervals quantify sampling uncertainty.\n"
        "- The oracle uses known synthetic probabilities unavailable "
        "in real deployment.\n"
        "- The practical policy uses only observable customer features.\n"
        "- Results depend on the assumed gross margin and redemption model.\n"
        "- These results are synthetic and do not establish real-world "
        "causal effectiveness."
    )


if __name__ == "__main__":
    main()
