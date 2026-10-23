
from pathlib import Path
from collections import Counter

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
TREATMENT_PROPENSITY = 0.20

# Repeat budget selection across development-data splits.
N_REPEATS = 30
VALIDATION_SHARE = 0.25
BOOTSTRAP_REPS = 2000

BUDGET_SHARES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Conservative rule: a positive budget must be selected this often
# and have a positive lower percentile of repeated validation values.
MIN_POSITIVE_BUDGET_SELECTION_FREQUENCY = 0.70

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
        raise ValueError("The input dataset is empty.")

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
        raise ValueError("Infinite numeric values found.")

    if not df["purchase"].isin([0, 1]).all():
        raise ValueError("Purchase must contain only 0 and 1.")

    if (df["avg_order_value_eur"] <= 0).any():
        raise ValueError("Order values must be positive.")

    if not df["discount_sensitivity"].between(0, 1).all():
        raise ValueError("Discount sensitivity must be between 0 and 1.")

    control = df[
        (df["reward_offered"] == 0)
        & (df["reward_amount_eur"] == 0)
    ].copy()

    treatment = df[
        (df["reward_offered"] == 1)
        & np.isclose(df["reward_amount_eur"], REWARD_EUR)
    ].copy()

    eligible = pd.concat([control, treatment], ignore_index=True)

    if control.empty or treatment.empty:
        raise ValueError(
            "Both control and exactly €10 treatment groups are required."
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
            "Treatment share is inconsistent with the assumed "
            f"propensity of {TREATMENT_PROPENSITY:.2f}."
        )

    return eligible.reset_index(drop=True)


# ============================================================
# OUTCOME MODELS
# ============================================================

def train_outcome_models(
    train: pd.DataFrame,
) -> tuple[RandomForestClassifier, RandomForestClassifier]:
    control = train[train["treatment"] == 0]
    treatment = train[train["treatment"] == 1]

    if len(control) < 50 or len(treatment) < 50:
        raise ValueError(
            "Insufficient control or treatment observations in training."
        )

    if control["purchase"].nunique() < 2:
        raise ValueError("Control data needs both purchase classes.")

    if treatment["purchase"].nunique() < 2:
        raise ValueError("Treatment data needs both purchase classes.")

    params = {
        "n_estimators": 300,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
        "random_state": SEED,
        "n_jobs": -1,
    }

    model_control = RandomForestClassifier(**params)
    model_treatment = RandomForestClassifier(**params)

    model_control.fit(control[FEATURES], control["purchase"])
    model_treatment.fit(treatment[FEATURES], treatment["purchase"])

    return model_control, model_treatment


def predict_economics(
    data: pd.DataFrame,
    model_control: RandomForestClassifier,
    model_treatment: RandomForestClassifier,
) -> pd.DataFrame:
    p0 = model_control.predict_proba(data[FEATURES])[:, 1]
    p1 = model_treatment.predict_proba(data[FEATURES])[:, 1]

    order_value = data["avg_order_value_eur"].to_numpy(dtype=float)
    sensitivity = data["discount_sensitivity"].to_numpy(dtype=float)

    uplift = p1 - p0
    incremental_gross_profit = uplift * order_value * GROSS_MARGIN
    expected_reward_cost = p1 * sensitivity * REWARD_EUR

    return pd.DataFrame({
        "predicted_p0": p0,
        "predicted_p1": p1,
        "predicted_uplift": uplift,
        "predicted_net_benefit": (
            incremental_gross_profit - expected_reward_cost
        ),
    }, index=data.index)


# ============================================================
# ECONOMIC VALUE AND POLICY HELPERS
# ============================================================

def true_net_benefit(data: pd.DataFrame) -> np.ndarray:
    """
    Ground-truth expected incremental net benefit from the simulator.
    For oracle evaluation and diagnostics only.
    """
    p0 = data["purchase_probability_no_reward"].to_numpy(dtype=float)
    p1 = data["purchase_probability_with_reward_10"].to_numpy(dtype=float)
    order_value = data["avg_order_value_eur"].to_numpy(dtype=float)
    sensitivity = data["discount_sensitivity"].to_numpy(dtype=float)

    return (
        (p1 - p0) * order_value * GROSS_MARGIN
        - p1 * sensitivity * REWARD_EUR
    )


def select_top_customers(
    scores: np.ndarray,
    share: float,
) -> np.ndarray:
    n = len(scores)

    if share <= 0:
        return np.zeros(n, dtype=int)

    count = min(n, int(np.ceil(n * share)))
    selected = np.zeros(n, dtype=int)
    selected[np.argsort(-scores)[:count]] = 1

    return selected


def observed_outcomes(data: pd.DataFrame) -> np.ndarray:
    """
    Observed customer-level profit outcome.

    This is a one-period outcome, not lifetime customer value.
    """
    purchase = data["purchase"].to_numpy(dtype=float)
    order_value = data["avg_order_value_eur"].to_numpy(dtype=float)
    reward_cost = data["reward_cost_eur"].to_numpy(dtype=float)

    return purchase * order_value * GROSS_MARGIN - reward_cost


def ipw_contributions(data: pd.DataFrame) -> np.ndarray:
    """
    Individual inverse-propensity-weighted treatment-control contrast.

    Requires randomized assignment and the specified conditional
    treatment propensity within the eligible comparison population.
    """
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = observed_outcomes(data)
    e = TREATMENT_PROPENSITY

    return (
        treatment * outcome / e
        - (1 - treatment) * outcome / (1 - e)
    )


def ipw_policy_value(
    data: pd.DataFrame,
    policy: np.ndarray,
) -> float:
    contributions = ipw_contributions(data)
    return float(np.mean(policy * contributions))


# ============================================================
# REPEATED VALIDATION BUDGET SELECTION
# ============================================================

def repeated_budget_stability(
    development: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """
    Repeat model fitting and budget selection using development data only.

    Each repeat:
      1. Splits development into train and validation subsets.
      2. Fits outcome models on the training subset.
      3. Selects a budget using validation IPW value.
      4. Records selected budget and validation estimate.

    The final test set is not accessed here.
    """
    rng = np.random.default_rng(SEED)

    split_rows = []
    candidate_rows = []

    for repeat in range(N_REPEATS):
        train, validation = train_test_split(
            development,
            test_size=VALIDATION_SHARE,
            random_state=int(rng.integers(0, 2**31 - 1)),
            stratify=development["treatment"],
        )

        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)

        model_control, model_treatment = train_outcome_models(train)

        validation_scores = predict_economics(
            validation,
            model_control,
            model_treatment,
        )["predicted_net_benefit"].to_numpy()

        budget_results = []

        for share in BUDGET_SHARES:
            policy = select_top_customers(validation_scores, share)
            value = ipw_policy_value(validation, policy)

            row = {
                "repeat": repeat + 1,
                "budget_share_pct": share * 100,
                "customers_targeted": int(policy.sum()),
                "validation_ipw_value_per_customer_eur": value,
                "validation_value_per_10000_eur": value * 10000,
            }

            candidate_rows.append(row)
            budget_results.append(row)

        budget_table = pd.DataFrame(budget_results).sort_values(
            [
                "validation_ipw_value_per_customer_eur",
                "budget_share_pct",
            ],
            ascending=[False, True],
        )

        best = budget_table.iloc[0]

        split_rows.append({
            "repeat": repeat + 1,
            "selected_budget_pct": float(best["budget_share_pct"]),
            "selected_validation_value_per_customer_eur": float(
                best["validation_ipw_value_per_customer_eur"]
            ),
            "selected_validation_value_per_10000_eur": float(
                best["validation_value_per_10000_eur"]
            ),
            "training_rows": len(train),
            "validation_rows": len(validation),
        })

        print(
            f"Repeat {repeat + 1:02d}/{N_REPEATS}: "
            f"selected budget {best['budget_share_pct']:.0f}%, "
            f"validation estimate "
            f"€{best['validation_value_per_10000_eur']:.2f}/10,000"
        )

    split_results = pd.DataFrame(split_rows)
    candidate_results = pd.DataFrame(candidate_rows)

    frequencies = (
        split_results["selected_budget_pct"]
        .value_counts(normalize=True)
        .reindex([share * 100 for share in BUDGET_SHARES], fill_value=0)
        .sort_index()
    )

    frequency_table = frequencies.rename(
        "selection_frequency"
    ).reset_index()
    frequency_table.columns = [
        "budget_share_pct",
        "selection_frequency",
    ]

    # A conservative rule: choose the modal budget only if:
    # 1. A positive budget wins at least the required frequency.
    # 2. The selected validation estimates have a positive lower
    #    percentile across repeats.
    #
    # Because repeats overlap in their source data, this percentile
    # is a stability diagnostic, NOT a formal confidence interval.
    counts = Counter(split_results["selected_budget_pct"])
    modal_budget_pct, modal_count = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]

    selection_frequency = modal_count / N_REPEATS

    selected_values = split_results[
        "selected_validation_value_per_customer_eur"
    ].to_numpy()

    stability_lower = float(np.quantile(selected_values, 0.025))
    median_selected_value = float(np.median(selected_values))

    conservative_rule_passed = (
        modal_budget_pct > 0
        and selection_frequency
        >= MIN_POSITIVE_BUDGET_SELECTION_FREQUENCY
        and stability_lower > 0
    )

    if conservative_rule_passed:
        final_budget = modal_budget_pct / 100
    else:
        final_budget = 0.0

    summary = {
        "modal_budget_pct": float(modal_budget_pct),
        "modal_budget_selection_frequency": float(selection_frequency),
        "median_selected_validation_value_per_customer_eur":
            median_selected_value,
        "lower_2_5pct_selected_validation_value_per_customer_eur":
            stability_lower,
        "minimum_required_selection_frequency":
            MIN_POSITIVE_BUDGET_SELECTION_FREQUENCY,
        "conservative_rule_passed": bool(conservative_rule_passed),
        "final_conservative_budget_pct": final_budget * 100,
    }

    return (
        split_results,
        frequency_table,
        final_budget,
        {
            "summary": summary,
            "candidate_results": candidate_results,
        },
    )


# ============================================================
# ORACLE BUDGET SELECTION ON DEVELOPMENT DATA
# ============================================================

def select_oracle_budget(
    development: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    benefits = true_net_benefit(development)
    rows = []

    for share in BUDGET_SHARES:
        policy = select_top_customers(benefits, share)

        rows.append({
            "budget_share_pct": share * 100,
            "customers_targeted": int(policy.sum()),
            "true_expected_benefit_eur": float(
                np.sum(policy * benefits)
            ),
        })

    results = pd.DataFrame(rows).sort_values(
        ["true_expected_benefit_eur", "budget_share_pct"],
        ascending=[False, True],
    )

    best = results.iloc[0]
    return float(best["budget_share_pct"]) / 100, results


# ============================================================
# BOOTSTRAP TEST ESTIMATES
# ============================================================

def evaluate_test_policies(
    test: pd.DataFrame,
    test_scores: pd.DataFrame,
    practical_budget: float,
    oracle_budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate fixed policies on the held-out test set.

    Budgets were chosen using development data, not test data.
    """
    practical_scores = test_scores[
        "predicted_net_benefit"
    ].to_numpy()

    oracle_scores = true_net_benefit(test)

    n = len(test)

    policies = {
        "No rewards": np.zeros(n, dtype=int),
        "Blanket €10 rewards": np.ones(n, dtype=int),
        "Practical ML targeting": select_top_customers(
            practical_scores, practical_budget
        ),
        "Oracle same budget": select_top_customers(
            oracle_scores, practical_budget
        ),
        "Oracle own budget": select_top_customers(
            oracle_scores, oracle_budget
        ),
    }

    contributions = ipw_contributions(test)
    rng = np.random.default_rng(SEED + 100)
    bootstrap = {
        name: np.empty(BOOTSTRAP_REPS, dtype=float)
        for name in policies
    }

    point_values = {
        name: float(np.mean(policy * contributions))
        for name, policy in policies.items()
    }

    for b in range(BOOTSTRAP_REPS):
        indices = rng.integers(0, n, size=n)
        sampled_contributions = contributions[indices]

        for name, policy in policies.items():
            bootstrap[name][b] = np.mean(
                policy[indices] * sampled_contributions
            )

    results = []

    true_benefits = true_net_benefit(test)

    for name, policy in policies.items():
        samples = bootstrap[name]
        lower, upper = np.quantile(samples, [0.025, 0.975])

        true_total = float(np.sum(policy * true_benefits))

        results.append({
            "policy": name,
            "customers_targeted": int(policy.sum()),
            "targeting_rate_pct": float(policy.mean() * 100),
            "incremental_profit_per_customer_eur": point_values[name],
            "ci_lower_eur_per_customer": float(lower),
            "ci_upper_eur_per_customer": float(upper),
            "profit_per_10000_customers_eur":
                point_values[name] * 10000,
            "ci_lower_per_10000_eur": float(lower * 10000),
            "ci_upper_per_10000_eur": float(upper * 10000),
            "simulation_true_expected_benefit_eur": true_total,
            "simulation_true_benefit_per_10000_eur":
                true_total / n * 10000,
        })

    result_df = pd.DataFrame(results)

    # Paired bootstrap comparisons on the same test customers.
    comparisons = [
        ("Practical ML targeting", "Oracle same budget"),
        ("Practical ML targeting", "Oracle own budget"),
        ("Oracle same budget", "Oracle own budget"),
        ("Blanket €10 rewards", "Practical ML targeting"),
        ("Practical ML targeting", "No rewards"),
    ]

    difference_rows = []

    for name_a, name_b in comparisons:
        policy_a = policies[name_a]
        policy_b = policies[name_b]

        point = float(np.mean(
            (policy_a - policy_b) * contributions
        ))

        samples = np.empty(BOOTSTRAP_REPS, dtype=float)

        for b in range(BOOTSTRAP_REPS):
            indices = rng.integers(0, n, size=n)
            samples[b] = np.mean(
                (policy_a[indices] - policy_b[indices])
                * contributions[indices]
            )

        lower, upper = np.quantile(samples, [0.025, 0.975])

        difference_rows.append({
            "comparison": f"{name_a} minus {name_b}",
            "difference_per_customer_eur": point,
            "ci_lower_eur_per_customer": float(lower),
            "ci_upper_eur_per_customer": float(upper),
            "difference_per_10000_eur": point * 10000,
            "ci_lower_per_10000_eur": float(lower * 10000),
            "ci_upper_per_10000_eur": float(upper * 10000),
        })

    return result_df, pd.DataFrame(difference_rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("REWARD DEPENDENCY INTELLIGENCE SYSTEM")
    print("POLICY STABILITY AND CONSERVATIVE DECISION ANALYSIS")
    print("=" * 72)

    data = load_data()

    # One final test set, isolated before any budget-selection work.
    development, test = train_test_split(
        data,
        test_size=0.20,
        random_state=SEED,
        stratify=data["treatment"],
    )

    development = development.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("\nFINAL DATA PARTITION")
    print(f"Development: {len(development):,}")
    print(f"Final test:  {len(test):,}")

    print("\nREPEATED VALIDATION BUDGET SELECTION")
    split_results, frequency_table, practical_budget, stability = (
        repeated_budget_stability(development)
    )

    stability_summary = pd.DataFrame([stability["summary"]])

    print("\nBUDGET SELECTION FREQUENCY")
    print(frequency_table.round(4).to_string(index=False))

    print("\nSTABILITY SUMMARY")
    print(stability_summary.round(4).to_string(index=False))

    print(
        f"\nConservative practical budget: {practical_budget:.0%}"
    )

    print("\nORACLE BUDGET SELECTION ON DEVELOPMENT DATA")
    oracle_budget, oracle_budget_table = select_oracle_budget(development)
    print(oracle_budget_table.round(4).to_string(index=False))
    print(f"\nOracle selected budget: {oracle_budget:.0%}")

    # Train practical models on all development data after selecting budget.
    model_control, model_treatment = train_outcome_models(development)

    test_scores = predict_economics(
        test,
        model_control,
        model_treatment,
    )

    print("\nHELD-OUT TEST EVALUATION")
    print(f"Practical budget fixed before test: {practical_budget:.0%}")
    print(f"Oracle budget fixed before test: {oracle_budget:.0%}")

    test_results, paired_differences = evaluate_test_policies(
        test=test,
        test_scores=test_scores,
        practical_budget=practical_budget,
        oracle_budget=oracle_budget,
    )

    # Save reports.
    split_path = METRICS_DIR / "policy_stability_repeated_splits.csv"
    frequency_path = METRICS_DIR / "policy_stability_budget_frequency.csv"
    stability_path = METRICS_DIR / "policy_stability_summary.csv"
    candidates_path = METRICS_DIR / "policy_stability_all_candidates.csv"
    oracle_path = METRICS_DIR / "policy_stability_oracle_budget.csv"
    test_path = METRICS_DIR / "policy_stability_test_comparison.csv"
    differences_path = METRICS_DIR / "policy_stability_paired_differences.csv"

    split_results.to_csv(split_path, index=False)
    frequency_table.to_csv(frequency_path, index=False)
    stability_summary.to_csv(stability_path, index=False)
    stability["candidate_results"].to_csv(candidates_path, index=False)
    oracle_budget_table.to_csv(oracle_path, index=False)
    test_results.to_csv(test_path, index=False)
    paired_differences.to_csv(differences_path, index=False)

    print("\nFINAL HELD-OUT TEST POLICY COMPARISON")
    print(test_results.round(4).to_string(index=False))

    print("\nPAIRED POLICY DIFFERENCES")
    print(paired_differences.round(4).to_string(index=False))

    print("\nSAVED OUTPUTS")
    for path in [
        split_path,
        frequency_path,
        stability_path,
        candidates_path,
        oracle_path,
        test_path,
        differences_path,
    ]:
        print(path)

    print("\nRESEARCH CAVEATS")
    print("- Repeated validation splits overlap; their percentiles are stability")
    print("  diagnostics, not formal independent-sample confidence intervals.")
    print("- The final test set is not used to select the targeting budget.")
    print("- The practical policy uses observable customer features only.")
    print("- The oracle uses synthetic ground-truth probabilities.")
    print("- IPW assumes randomized assignment and propensity 0.20.")
    print("- Confidence intervals quantify sampling uncertainty, not all model")
    print("  misspecification or simulation uncertainty.")
    print("- All results depend on synthetic data and assumed unit economics.")


if __name__ == "__main__":
    main()
