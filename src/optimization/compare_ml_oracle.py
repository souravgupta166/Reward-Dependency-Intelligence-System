
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "reward_customer_data.csv"
METRICS_DIR = ROOT / "reports" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261023
REWARD_EUR = 10.0
GROSS_MARGIN = 0.30

# Conditional treatment propensity among customers who received
# either no reward or exactly a €10 reward:
# P(€10 treatment | no reward or €10 treatment) = 0.20.
TREATMENT_PROPENSITY = 0.20

# Include 0% to allow the no-reward policy to win.
BUDGET_SHARES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

BOOTSTRAP_REPS = 2000

FEATURES = [
    "tenure_months",
    "prior_orders",
    "avg_order_value_eur",
    "engagement_score",
    "discount_sensitivity",
]

REQUIRED_COLUMNS = [
    "customer_id",
    "reward_offered",
    "reward_amount_eur",
    "purchase",
    "reward_cost_eur",
    "avg_order_value_eur",
    "discount_sensitivity",
    "purchase_probability_no_reward",
    "purchase_probability_with_reward_10",
    *FEATURES,
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run src/data/generate_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    numeric_columns = list(dict.fromkeys(
        FEATURES
        + [
            "reward_offered",
            "reward_amount_eur",
            "purchase",
            "reward_cost_eur",
            "purchase_probability_no_reward",
            "purchase_probability_with_reward_10",
        ]
    ))

    if df[numeric_columns].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Infinite values found in required columns.")

    # Eligible comparison: no reward versus exactly €10 reward.
    control = df[
        (df["reward_offered"] == 0)
        & (df["reward_amount_eur"] == 0)
    ].copy()

    treatment = df[
        (df["reward_offered"] == 1)
        & np.isclose(df["reward_amount_eur"], REWARD_EUR)
    ].copy()

    eligible = pd.concat([control, treatment], ignore_index=True)

    if eligible.empty or control.empty or treatment.empty:
        raise ValueError(
            "Could not find both control customers and exactly €10 "
            "treatment customers."
        )

    eligible["treatment"] = (
        (eligible["reward_offered"] == 1)
        & np.isclose(eligible["reward_amount_eur"], REWARD_EUR)
    ).astype(int)

    observed_propensity = eligible["treatment"].mean()

    print(f"Eligible customers: {len(eligible):,}")
    print(f"Control customers: {(eligible['treatment'] == 0).sum():,}")
    print(f"€10 treatment customers: {(eligible['treatment'] == 1).sum():,}")
    print(f"Observed treatment share: {observed_propensity:.3f}")

    if not np.isclose(
        observed_propensity,
        TREATMENT_PROPENSITY,
        atol=0.025,
    ):
        raise ValueError(
            "Observed treatment share is unexpectedly far from the "
            f"assumed conditional propensity {TREATMENT_PROPENSITY:.2f}. "
            "Check the assignment design before proceeding."
        )

    if not eligible["purchase"].isin([0, 1]).all():
        raise ValueError("Purchase must be binary (0/1).")

    if (eligible["avg_order_value_eur"] <= 0).any():
        raise ValueError("Order values must be positive.")

    if not eligible["discount_sensitivity"].between(0, 1).all():
        raise ValueError("Discount sensitivity must be between 0 and 1.")

    if (eligible["reward_cost_eur"] < 0).any():
        raise ValueError("Reward costs cannot be negative.")

    return eligible.reset_index(drop=True)


# ============================================================
# DATA PARTITIONING
# ============================================================

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train, validation, and final test sets.

    Treatment stratification preserves the treatment/control balance.
    The test set is not used for training or budget selection.
    """
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["treatment"],
    )

    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=SEED,
        stratify=train_val["treatment"],
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("\nDATA PARTITIONS")
    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Final test: {len(test):,}")

    return train, validation, test


# ============================================================
# TRAIN PRACTICAL ML MODELS
# ============================================================

def train_outcome_models(
    train: pd.DataFrame,
) -> tuple[RandomForestClassifier, RandomForestClassifier]:
    """
    Train separate purchase outcome models for control and treatment.

    Only observable customer features are used.
    Ground-truth probabilities, latent segments, costs, and potential
    outcomes are excluded from the model features.
    """
    control = train[train["treatment"] == 0]
    treatment = train[train["treatment"] == 1]

    if control["purchase"].nunique() < 2:
        raise ValueError("Control training data needs both purchase classes.")

    if treatment["purchase"].nunique() < 2:
        raise ValueError(
            "Treatment training data needs both purchase classes."
        )

    common_params = {
        "n_estimators": 300,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
        "class_weight": None,
        "random_state": SEED,
        "n_jobs": -1,
    }

    model_control = RandomForestClassifier(**common_params)
    model_treatment = RandomForestClassifier(**common_params)

    model_control.fit(
        control[FEATURES],
        control["purchase"],
    )

    model_treatment.fit(
        treatment[FEATURES],
        treatment["purchase"],
    )

    print("\nPRACTICAL MODELS")
    print(f"Control model training rows: {len(control):,}")
    print(f"Treatment model training rows: {len(treatment):,}")
    print(f"Observable features: {', '.join(FEATURES)}")

    return model_control, model_treatment


def predict_economics(
    data: pd.DataFrame,
    model_control: RandomForestClassifier,
    model_treatment: RandomForestClassifier,
) -> pd.DataFrame:
    """
    Estimate practical purchase uplift and expected net benefit.

    Reward cost is estimated using the model's treated purchase
    probability and discount sensitivity as a redemption proxy.
    """
    result = pd.DataFrame(index=data.index)

    p0 = model_control.predict_proba(data[FEATURES])[:, 1]
    p1 = model_treatment.predict_proba(data[FEATURES])[:, 1]

    order_value = data["avg_order_value_eur"].to_numpy(dtype=float)
    sensitivity = data["discount_sensitivity"].to_numpy(dtype=float)

    uplift = p1 - p0

    expected_incremental_gross_profit = (
        uplift * order_value * GROSS_MARGIN
    )

    expected_reward_cost = p1 * sensitivity * REWARD_EUR

    result["predicted_p0"] = p0
    result["predicted_p1"] = p1
    result["predicted_uplift"] = uplift
    result["predicted_net_benefit"] = (
        expected_incremental_gross_profit - expected_reward_cost
    )

    return result


# ============================================================
# GROUND-TRUTH SIMULATION ECONOMICS
# ============================================================

def true_customer_net_benefit(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the simulation's true expected incremental net benefit.

    This is used only for oracle benchmarking and diagnostic reporting.
    It must not be used to train or select the practical ML policy.
    """
    p0 = data["purchase_probability_no_reward"].to_numpy(dtype=float)
    p1 = data["purchase_probability_with_reward_10"].to_numpy(dtype=float)

    order_value = data["avg_order_value_eur"].to_numpy(dtype=float)
    sensitivity = data["discount_sensitivity"].to_numpy(dtype=float)

    return (
        (p1 - p0) * order_value * GROSS_MARGIN
        - p1 * sensitivity * REWARD_EUR
    )


# ============================================================
# POLICY SELECTION
# ============================================================

def select_top_customers(
    scores: np.ndarray,
    share: float,
) -> np.ndarray:
    """Return a binary targeting decision for a given budget share."""
    n = len(scores)

    if share <= 0:
        return np.zeros(n, dtype=int)

    number_targeted = min(n, int(np.ceil(n * share)))

    ranking = np.argsort(-scores)
    selected = np.zeros(n, dtype=int)
    selected[ranking[:number_targeted]] = 1

    return selected


def ipw_incremental_contributions(
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Estimate treatment-versus-control incremental contribution using IPW.

    Per-customer observed contribution:
        T * outcome / propensity
        - (1-T) * outcome / (1-propensity)

    Outcome includes gross profit minus realized voucher cost.
    The estimate relies on randomized assignment and the known
    conditional treatment propensity.
    """
    treatment = data["treatment"].to_numpy(dtype=float)
    purchase = data["purchase"].to_numpy(dtype=float)

    order_value = data["avg_order_value_eur"].to_numpy(dtype=float)
    realized_reward_cost = data["reward_cost_eur"].to_numpy(dtype=float)

    outcome = (
        purchase * order_value * GROSS_MARGIN
        - realized_reward_cost
    )

    e = TREATMENT_PROPENSITY

    return (
        treatment * outcome / e
        - (1 - treatment) * outcome / (1 - e)
    )


def policy_ipw_value(
    data: pd.DataFrame,
    policy: np.ndarray,
) -> float:
    """IPW incremental net value per eligible customer."""
    contributions = ipw_incremental_contributions(data)
    return float(np.mean(policy * contributions))


def select_practical_budget(
    validation: pd.DataFrame,
    validation_scores: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """
    Select the practical budget using validation IPW value only.

    The 0% option is included. Ties are resolved in favor of the
    smallest targeting share.
    """
    scores = validation_scores["predicted_net_benefit"].to_numpy()

    rows = []

    for share in BUDGET_SHARES:
        policy = select_top_customers(scores, share)
        value = policy_ipw_value(validation, policy)

        rows.append({
            "budget_share_pct": share * 100,
            "customers_targeted": int(policy.sum()),
            "validation_ipw_net_benefit_per_customer_eur": value,
            "validation_incremental_benefit_per_10000_eur":
                value * 10000,
        })

    results = pd.DataFrame(rows).sort_values(
        ["validation_ipw_net_benefit_per_customer_eur", "budget_share_pct"],
        ascending=[False, True],
    )

    best = results.iloc[0]
    selected_share = float(best["budget_share_pct"]) / 100

    return selected_share, results


def select_oracle_budget(
    validation: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """
    Select the oracle budget using true simulated benefits on validation.

    This is intentionally not available to the practical ML model.
    """
    benefits = true_customer_net_benefit(validation)
    rows = []

    for share in BUDGET_SHARES:
        policy = select_top_customers(benefits, share)

        rows.append({
            "budget_share_pct": share * 100,
            "customers_targeted": int(policy.sum()),
            "validation_true_expected_benefit_eur":
                float(np.sum(policy * benefits)),
        })

    results = pd.DataFrame(rows).sort_values(
        ["validation_true_expected_benefit_eur", "budget_share_pct"],
        ascending=[False, True],
    )

    best = results.iloc[0]
    selected_share = float(best["budget_share_pct"]) / 100

    return selected_share, results


# ============================================================
# BOOTSTRAP UNCERTAINTY
# ============================================================

def bootstrap_policy_values(
    data: pd.DataFrame,
    policies: dict[str, np.ndarray],
    reps: int = BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Paired nonparametric bootstrap over held-out customers.

    Reports percentile 95% confidence intervals for each policy's
    IPW incremental value versus no rewards.
    """
    rng = np.random.default_rng(SEED)

    n = len(data)
    contributions = ipw_incremental_contributions(data)

    point_values = {
        name: float(np.mean(policy * contributions))
        for name, policy in policies.items()
    }

    bootstrap_values = {
        name: np.empty(reps, dtype=float)
        for name in policies
    }

    for b in range(reps):
        indices = rng.integers(0, n, size=n)
        sampled_contributions = contributions[indices]

        for name, policy in policies.items():
            sampled_policy = policy[indices]
            bootstrap_values[name][b] = np.mean(
                sampled_policy * sampled_contributions
            )

    rows = []

    for name, policy in policies.items():
        samples = bootstrap_values[name]

        rows.append({
            "policy": name,
            "customers_targeted": int(policy.sum()),
            "targeting_rate_pct": float(policy.mean() * 100),
            "incremental_profit_per_customer_eur": point_values[name],
            "ci_lower_eur_per_customer": float(
                np.quantile(samples, 0.025)
            ),
            "ci_upper_eur_per_customer": float(
                np.quantile(samples, 0.975)
            ),
            "scaled_profit_per_10000_customers_eur":
                point_values[name] * 10000,
            "scaled_ci_lower_per_10000_eur":
                float(np.quantile(samples, 0.025) * 10000),
            "scaled_ci_upper_per_10000_eur":
                float(np.quantile(samples, 0.975) * 10000),
        })

    return pd.DataFrame(rows), bootstrap_values


def bootstrap_paired_policy_differences(
    data: pd.DataFrame,
    policies: dict[str, np.ndarray],
    reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    """
    Compare policies using paired bootstrap differences.

    Each reported difference is policy A's IPW incremental value
    minus policy B's IPW incremental value, per eligible customer.
    """
    rng = np.random.default_rng(SEED + 1)
    contributions = ipw_incremental_contributions(data)
    n = len(data)

    comparisons = [
        ("Practical targeting", "Oracle same budget"),
        ("Practical targeting", "Oracle own budget"),
        ("Oracle same budget", "Oracle own budget"),
        ("Blanket €10 rewards", "Practical targeting"),
    ]

    rows = []

    for name_a, name_b in comparisons:
        if name_a not in policies or name_b not in policies:
            continue

        policy_a = policies[name_a]
        policy_b = policies[name_b]

        point = float(
            np.mean((policy_a - policy_b) * contributions)
        )

        samples = np.empty(reps, dtype=float)

        for b in range(reps):
            indices = rng.integers(0, n, size=n)
            difference = (
                (policy_a[indices] - policy_b[indices])
                * contributions[indices]
            )
            samples[b] = np.mean(difference)

        lower, upper = np.quantile(samples, [0.025, 0.975])

        rows.append({
            "comparison": f"{name_a} minus {name_b}",
            "difference_per_customer_eur": point,
            "ci_lower_eur_per_customer": float(lower),
            "ci_upper_eur_per_customer": float(upper),
            "difference_per_10000_customers_eur": point * 10000,
            "ci_lower_per_10000_eur": float(lower * 10000),
            "ci_upper_per_10000_eur": float(upper * 10000),
        })

    return pd.DataFrame(rows)


# ============================================================
# HELD-OUT TEST EVALUATION
# ============================================================

def evaluate_test_policies(
    test: pd.DataFrame,
    test_scores: pd.DataFrame,
    practical_budget: float,
    oracle_budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all policies on the untouched final test set.

    Includes:
      - No rewards
      - Blanket €10 rewards
      - Practical ML targeting
      - Oracle targeting at the practical model's budget
      - Oracle targeting at its own validation-selected budget
    """
    practical_scores = test_scores[
        "predicted_net_benefit"
    ].to_numpy()

    oracle_scores = true_customer_net_benefit(test)

    n = len(test)

    policies = {
        "No rewards": np.zeros(n, dtype=int),
        "Blanket €10 rewards": np.ones(n, dtype=int),
        "Practical targeting": select_top_customers(
            practical_scores, practical_budget
        ),
        "Oracle same budget": select_top_customers(
            oracle_scores, practical_budget
        ),
        "Oracle own budget": select_top_customers(
            oracle_scores, oracle_budget
        ),
    }

    results, _ = bootstrap_policy_values(test, policies)
    differences = bootstrap_paired_policy_differences(test, policies)

    # Simulation-only diagnostics: expected benefit under the
    # known probabilities. These are not observed profit estimates.
    true_benefits = true_customer_net_benefit(test)

    results["true_expected_incremental_net_benefit_eur"] = [
        float(np.sum(policies[name] * true_benefits))
        for name in results["policy"]
    ]

    results["true_expected_benefit_per_10000_eur"] = (
        results["true_expected_incremental_net_benefit_eur"]
        / n * 10000
    )

    return results, differences


# ============================================================
# RANKING DIAGNOSTICS
# ============================================================

def ranking_diagnostics(
    data: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare practical ranking scores with simulated ground truth.
    Correlation is descriptive and is not a causal-effect metric.
    """
    true_benefits = true_customer_net_benefit(data)
    true_p0 = data["purchase_probability_no_reward"].to_numpy(dtype=float)
    true_p1 = data["purchase_probability_with_reward_10"].to_numpy(dtype=float)
    true_uplift = true_p1 - true_p0

    predicted_uplift = scores["predicted_uplift"].to_numpy(dtype=float)
    predicted_benefit = scores["predicted_net_benefit"].to_numpy(dtype=float)

    rows = []

    for share in BUDGET_SHARES:
        practical_policy = select_top_customers(predicted_benefit, share)
        oracle_policy = select_top_customers(true_benefits, share)

        selected = practical_policy == 1

        rows.append({
            "targeting_share_pct": share * 100,
            "customers_selected": int(practical_policy.sum()),
            "mean_true_uplift_selected": (
                float(true_uplift[selected].mean())
                if selected.any() else 0.0
            ),
            "true_expected_benefit_selected_eur":
                float(np.sum(practical_policy * true_benefits)),
            "oracle_true_expected_benefit_same_budget_eur":
                float(np.sum(oracle_policy * true_benefits)),
            "predicted_vs_true_uplift_correlation": float(
                np.corrcoef(predicted_uplift, true_uplift)[0, 1]
            ),
            "predicted_vs_true_net_benefit_correlation": float(
                np.corrcoef(predicted_benefit, true_benefits)[0, 1]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("REWARD DEPENDENCY INTELLIGENCE SYSTEM")
    print("PRACTICAL ML VS ORACLE POLICY EVALUATION")
    print("=" * 72)

    data = load_data()

    train, validation, test = split_data(data)

    model_control, model_treatment = train_outcome_models(train)

    validation_scores = predict_economics(
        validation,
        model_control,
        model_treatment,
    )

    test_scores = predict_economics(
        test,
        model_control,
        model_treatment,
    )

    # Select budgets only on validation.
    practical_budget, practical_budget_table = select_practical_budget(
        validation,
        validation_scores,
    )

    oracle_budget, oracle_budget_table = select_oracle_budget(validation)

    print("\nVALIDATION: PRACTICAL POLICY BUDGET")
    print(practical_budget_table.round(4).to_string(index=False))
    print(f"\nSelected practical budget: {practical_budget:.0%}")

    print("\nVALIDATION: ORACLE BUDGET")
    print(oracle_budget_table.round(4).to_string(index=False))
    print(f"\nSelected oracle budget: {oracle_budget:.0%}")

    # Evaluate once on the untouched test set.
    test_results, policy_differences = evaluate_test_policies(
        test=test,
        test_scores=test_scores,
        practical_budget=practical_budget,
        oracle_budget=oracle_budget,
    )

    diagnostics = ranking_diagnostics(test, test_scores)

    # Save outputs.
    budget_path = METRICS_DIR / "ml_oracle_validation_budgets.csv"
    test_path = METRICS_DIR / "ml_oracle_test_policy_comparison.csv"
    difference_path = METRICS_DIR / "ml_oracle_paired_differences.csv"
    diagnostics_path = METRICS_DIR / "ml_oracle_ranking_diagnostics.csv"

    practical_budget_table.to_csv(budget_path, index=False)
    test_results.to_csv(test_path, index=False)
    policy_differences.to_csv(difference_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print("\nFINAL HELD-OUT TEST POLICY COMPARISON")
    print(test_results.round(4).to_string(index=False))

    print("\nPAIRED POLICY DIFFERENCES")
    print(policy_differences.round(4).to_string(index=False))

    print("\nTEST RANKING DIAGNOSTICS")
    print(diagnostics.round(4).to_string(index=False))

    print("\nSAVED OUTPUTS")
    print(f"Validation budgets: {budget_path}")
    print(f"Test policy comparison: {test_path}")
    print(f"Paired differences: {difference_path}")
    print(f"Ranking diagnostics: {diagnostics_path}")

    print("\nINTERPRETATION AND RESEARCH CAVEATS")
    print("- Practical models use observable features only.")
    print("- Ground-truth probabilities are excluded from practical model training.")
    print("- Budget selection uses validation data only.")
    print("- Final policy estimates use the untouched test set.")
    print("- IPW evaluation assumes randomized assignment and propensity 0.20.")
    print("- Reward cost in the observed outcome uses realized voucher costs.")
    print("- True expected benefit columns use simulation ground truth.")
    print("- Oracle same-budget results compare rankings at equal targeting rates.")
    print("- Oracle own-budget results use a budget selected from validation.")
    print("- Bootstrap intervals quantify sampling uncertainty, not model misspecification.")
    print("- Synthetic data cannot establish real-world causal effectiveness.")


if __name__ == "__main__":
    main()
