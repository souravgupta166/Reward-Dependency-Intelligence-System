"""
Audit the Reward Dependency Intelligence System's synthetic data and economics.

Run from the repository root:
    .\.venv\Scripts\python.exe src\optimization\audit_simulation_economics.py

Reads:
    data/raw/reward_customer_data.csv

Writes audit CSVs and a Markdown report to:
    reports/metrics/simulation_economics_audit/
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "reward_customer_data.csv"
OUT_DIR = ROOT / "reports" / "metrics" / "simulation_economics_audit"

MARGINS = [0.20, 0.30, 0.40, 0.50]
REWARD_AMOUNTS = [3.0, 5.0, 10.0, 15.0]
BOOTSTRAP_REPS = 1000
SEED = 42
EPS = 1e-9


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def write_csv(df, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / name, index=False)


def standardised_mean_difference(x_t, x_c):
    x_t = pd.to_numeric(pd.Series(x_t), errors="coerce").dropna()
    x_c = pd.to_numeric(pd.Series(x_c), errors="coerce").dropna()
    if len(x_t) < 2 or len(x_c) < 2:
        return np.nan
    pooled = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return float((x_t.mean() - x_c.mean()) / pooled) if pooled > EPS else 0.0


def bootstrap_mean_ci(values, reps=BOOTSTRAP_REPS, seed=SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Required columns missing from the raw dataset: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns.astype(str))
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/data/generate_data.py first, or update DATA_PATH in this script."
        )

    df = pd.read_csv(DATA_PATH)
    required = [
        "customer_id", "avg_order_value_eur", "discount_sensitivity",
        "reward_offered", "reward_amount_eur", "reward_redeemed",
        "purchase", "reward_cost_eur", "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]
    require_columns(df, required)

    # Normalize binary flags, while retaining the source data unchanged.
    for col in ["reward_offered", "reward_redeemed", "purchase"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "avg_order_value_eur", "discount_sensitivity", "reward_amount_eur",
        "reward_cost_eur", "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    issues = []
    checks = []

    def check(name, passed, observed, expected, severity="INFO", note=""):
        checks.append({
            "check": name, "passed": bool(passed), "severity": severity,
            "observed": str(observed), "expected": str(expected), "note": note
        })
        if not passed and severity in ("ERROR", "WARNING"):
            issues.append(f"{severity}: {name}. Observed={observed}; expected={expected}. {note}")

    n = len(df)
    check("row_count", n == 10000, n, "10,000 synthetic customers", "WARNING")
    check("customer_id_unique", df["customer_id"].is_unique, int(df["customer_id"].nunique()), n, "ERROR")
    check("missing_required_values", not df[required].isna().any().any(),
          int(df[required].isna().sum().sum()), 0, "ERROR")
    check("binary_flags_valid",
          all(set(df[c].dropna().unique()).issubset({0, 1}) for c in ["reward_offered", "reward_redeemed", "purchase"]),
          "checked reward_offered/reward_redeemed/purchase", "values only 0/1", "ERROR")
    check("probabilities_in_range",
          df[["purchase_probability_no_reward", "purchase_probability_with_reward_10"]].apply(
              lambda s: s.between(0, 1).all()).all(),
          "checked", "all probabilities between 0 and 1", "ERROR")
    check("discount_sensitivity_in_range",
          df["discount_sensitivity"].between(0, 1).all(),
          f"min={df['discount_sensitivity'].min():.4f}, max={df['discount_sensitivity'].max():.4f}",
          "all values between 0 and 1", "ERROR")

    # Reward assignment and realized cost integrity.
    expected_cost = df["reward_redeemed"] * df["reward_amount_eur"]
    cost_diff = (df["reward_cost_eur"] - expected_cost).abs()
    check("reward_cost_matches_redemption",
          cost_diff.max() < 1e-6, f"max absolute difference={cost_diff.max():.8f}",
          "reward_cost_eur = reward_redeemed * reward_amount_eur", "ERROR")
    invalid_redemption = ((df["reward_redeemed"] == 1) &
                          ((df["reward_offered"] != 1) | (df["purchase"] != 1)))
    check("redemption_requires_offer_and_purchase",
          int(invalid_redemption.sum()) == 0, int(invalid_redemption.sum()), 0, "ERROR")
    invalid_offer_amount = ((df["reward_offered"] == 0) & (df["reward_amount_eur"] != 0))
    check("no_offer_has_zero_amount",
          int(invalid_offer_amount.sum()) == 0, int(invalid_offer_amount.sum()), 0, "ERROR")

    # Restrict causal comparisons to no-offer control vs exactly €10 offered treatment.
    control = df[(df["reward_offered"] == 0) & (df["reward_amount_eur"] == 0)].copy()
    treat10 = df[(df["reward_offered"] == 1) & np.isclose(df["reward_amount_eur"], 10.0)].copy()
    eligible = pd.concat([control, treat10], ignore_index=True)
    eligible["A"] = (eligible["reward_offered"] == 1).astype(int)
    n0, n1, ne = len(control), len(treat10), len(eligible)
    p_emp = n1 / ne if ne else np.nan

    # Given 50% offer assignment and four equally likely non-zero reward amounts,
    # the expected conditional probability of €10 treatment vs control is 0.125/(0.5+0.125)=0.20.
    check("eligible_sample_nonempty", ne > 0, ne, "> 0", "ERROR")
    check("empirical_treatment_propensity_near_0_20",
          abs(p_emp - 0.20) < 0.035, round(p_emp, 5), "approximately 0.20", "WARNING",
          "A large deviation may indicate changed assignment logic or a small/filtered sample.")

    # Assignment balance diagnostics: standardized differences should generally be small
    # under randomization, but finite-sample variation is expected.
    candidate_covars = [
        "tenure_months", "prior_orders", "avg_order_value_eur",
        "engagement_score", "discount_sensitivity"
    ]
    balance_rows = []
    for col in candidate_covars:
        if col not in eligible.columns:
            continue
        smd = standardised_mean_difference(
            eligible.loc[eligible["A"] == 1, col],
            eligible.loc[eligible["A"] == 0, col]
        )
        balance_rows.append({
            "feature": col,
            "control_mean": pd.to_numeric(eligible.loc[eligible["A"] == 0, col], errors="coerce").mean(),
            "treatment_mean": pd.to_numeric(eligible.loc[eligible["A"] == 1, col], errors="coerce").mean(),
            "standardized_mean_difference": smd,
            "absolute_smd": abs(smd) if pd.notna(smd) else np.nan,
            "flag_abs_smd_over_0_10": bool(abs(smd) > 0.10) if pd.notna(smd) else False,
        })
    balance = pd.DataFrame(balance_rows)
    if not balance.empty:
        write_csv(balance, "assignment_balance.csv")
        check("covariate_balance_abs_smd_under_0_10",
              not balance["flag_abs_smd_over_0_10"].any(),
              balance.loc[balance["flag_abs_smd_over_0_10"], "feature"].tolist(),
              "absolute SMD <= 0.10 for each available covariate", "WARNING",
              "Balance is a diagnostic, not proof of randomization.")

    # Check consistency between stored truth probabilities and expected economics.
    eligible["p0"] = eligible["purchase_probability_no_reward"].clip(0, 1)
    eligible["p1"] = eligible["purchase_probability_with_reward_10"].clip(0, 1)
    eligible["aov"] = eligible["avg_order_value_eur"].clip(lower=0)
    eligible["sensitivity"] = eligible["discount_sensitivity"].clip(0, 1)
    check("treatment_probability_not_below_control_on_average",
          eligible["p1"].mean() >= eligible["p0"].mean(),
          f"mean p0={eligible['p0'].mean():.4f}; mean p1={eligible['p1'].mean():.4f}",
          "mean p1 >= mean p0 for this generator", "WARNING",
          "Individual customers may still have zero or very small modeled uplift.")

    # Descriptive observed outcome comparison, unadjusted by design.
    eligible["observed_net_contribution_30pct"] = (
        eligible["purchase"] * eligible["aov"] * 0.30 - eligible["reward_cost_eur"]
    )
    y0 = eligible.loc[eligible["A"] == 0, "observed_net_contribution_30pct"]
    y1 = eligible.loc[eligible["A"] == 1, "observed_net_contribution_30pct"]
    observed_diff = float(y1.mean() - y0.mean()) if len(y0) and len(y1) else np.nan
    # Standard error for difference in independent group means.
    observed_se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))) if len(y0) > 1 and len(y1) > 1 else np.nan
    observed_ci = (observed_diff - 1.96 * observed_se, observed_diff + 1.96 * observed_se) if pd.notna(observed_se) else (np.nan, np.nan)

    # Expected net benefit of giving a standardized €10 reward to a customer:
    # incremental gross profit - expected redemption cost.
    # This uses known simulator probabilities, for audit only (not deployable policy input).
    economic_rows = []
    for margin in MARGINS:
        gross_profit_uplift = (eligible["p1"] - eligible["p0"]) * eligible["aov"] * margin
        for redemption_multiplier in [0.5, 0.75, 1.0, 1.25]:
            expected_cost = eligible["p1"] * eligible["sensitivity"] * 10.0 * redemption_multiplier
            net = gross_profit_uplift - expected_cost
            for budget_pct in [0, 5, 10, 15, 20, 25, 30, 50, 100]:
                k = int(np.floor(len(eligible) * budget_pct / 100))
                if k == 0:
                    total = 0.0
                else:
                    # Oracle ranking is intentionally labeled as such: it uses ground truth.
                    top_idx = net.nlargest(k).index
                    total = float(net.loc[top_idx].sum())
                economic_rows.append({
                    "gross_margin": margin,
                    "redemption_cost_multiplier": redemption_multiplier,
                    "budget_pct": budget_pct,
                    "selected_customers": k,
                    "expected_net_benefit_total_eur": total,
                    "expected_net_benefit_per_10000_eligible_eur": total / len(eligible) * 10000 if len(eligible) else np.nan,
                    "is_oracle_ground_truth_ranking": True,
                })
    economics = pd.DataFrame(economic_rows)
    write_csv(economics, "oracle_economic_sensitivity.csv")

    # True population-level expected treatment effect by gross margin / redemption assumption.
    treatment_effect_rows = []
    for margin in MARGINS:
        gross = (eligible["p1"] - eligible["p0"]) * eligible["aov"] * margin
        for redemption_multiplier in [0.5, 0.75, 1.0, 1.25]:
            cost = eligible["p1"] * eligible["sensitivity"] * 10.0 * redemption_multiplier
            per_customer = (gross - cost).mean()
            treatment_effect_rows.append({
                "gross_margin": margin,
                "redemption_cost_multiplier": redemption_multiplier,
                "expected_net_benefit_per_eligible_customer_eur": float(per_customer),
                "expected_net_benefit_per_10000_eligible_eur": float(per_customer * 10000),
                "expected_incremental_gross_profit_per_customer_eur": float(gross.mean()),
                "expected_reward_cost_per_customer_eur": float(cost.mean()),
                "fraction_customers_with_positive_expected_net_benefit": float(((gross - cost) > 0).mean()),
            })
    treatment_sensitivity = pd.DataFrame(treatment_effect_rows)
    write_csv(treatment_sensitivity, "blanket_treatment_economic_sensitivity.csv")

    # IPW estimator check: the policy value increment relative to no rewards, using
    # the observed net contribution and the known randomized propensity in this eligible cohort.
    ipw_rows = []
    if ne and 0 < p_emp < 1:
        A = eligible["A"].to_numpy()
        Y = eligible["observed_net_contribution_30pct"].to_numpy()
        ipw_contrib = A * Y / p_emp - (1 - A) * Y / (1 - p_emp)
        mean, lo, hi = bootstrap_mean_ci(ipw_contrib)
        ipw_rows.append({
            "estimand": "blanket_10_eur_reward_vs_no_reward",
            "gross_margin": 0.30,
            "eligible_n": ne,
            "control_n": n0,
            "treatment_n": n1,
            "empirical_propensity": p_emp,
            "ipw_incremental_net_benefit_per_customer_eur": mean,
            "ipw_ci_lower_per_customer_eur": lo,
            "ipw_ci_upper_per_customer_eur": hi,
            "ipw_per_10000_eligible_eur": mean * 10000,
            "ipw_ci_lower_per_10000_eur": lo * 10000,
            "ipw_ci_upper_per_10000_eur": hi * 10000,
            "truth_expected_benefit_per_customer_eur": float(
                ((eligible["p1"] - eligible["p0"]) * eligible["aov"] * 0.30
                 - eligible["p1"] * eligible["sensitivity"] * 10.0).mean()
            ),
            "note": "Bootstrap CI is sampling uncertainty; it does not cover generator or economic-model misspecification."
        })
    ipw_df = pd.DataFrame(ipw_rows)
    write_csv(ipw_df, "blanket_ipw_vs_truth.csv")

    # A compact report for interpretation.
    margin30 = treatment_sensitivity[
        (treatment_sensitivity["gross_margin"] == 0.30) &
        (treatment_sensitivity["redemption_cost_multiplier"] == 1.0)
    ].iloc[0]
    best_oracle = economics[
        (economics["gross_margin"] == 0.30) &
        (economics["redemption_cost_multiplier"] == 1.0)
    ].sort_values("expected_net_benefit_total_eur", ascending=False).iloc[0]

    report_lines = [
        "# Simulation and Economics Audit",
        "",
        "## Dataset and assignment",
        f"- Raw file: `{DATA_PATH.relative_to(ROOT)}`",
        f"- Rows: {n:,}; unique customer IDs: {df['customer_id'].nunique():,}.",
        f"- Eligible no-offer control: {n0:,}; exactly €10 treatment: {n1:,}; combined: {ne:,}.",
        f"- Empirical treatment propensity in this eligible cohort: {p_emp:.4f} (reference expectation approximately 0.20).",
        "",
        "## Key economic check (30% gross margin, baseline redemption assumption)",
        f"- Expected incremental net benefit of a blanket €10 reward: €{margin30['expected_net_benefit_per_10000_eligible_eur']:,.2f} per 10,000 eligible customers.",
        f"- Expected incremental gross profit per eligible customer: €{margin30['expected_incremental_gross_profit_per_customer_eur']:.4f}.",
        f"- Expected reward cost per eligible customer: €{margin30['expected_reward_cost_per_customer_eur']:.4f}.",
        f"- Fraction with positive modeled individual net benefit: {margin30['fraction_customers_with_positive_expected_net_benefit']:.3f}.",
        f"- Best oracle budget in the tested grid at these assumptions: {int(best_oracle['budget_pct'])}% (expected €{best_oracle['expected_net_benefit_per_10000_eligible_eur']:,.2f} per 10,000 eligible customers).",
        "",
        "## Observed randomized comparison",
        f"- Observed treatment-minus-control net contribution at 30% margin: €{observed_diff:.4f} per eligible customer.",
        f"- Approximate 95% normal CI: [€{observed_ci[0]:.4f}, €{observed_ci[1]:.4f}] per customer.",
        "- This comparison is noisy and should be read alongside the IPW estimate and simulator's known expected value.",
        "",
        "## Interpretation guardrails",
        "- The stored potential-outcome probabilities are simulator ground truth. They are appropriate for audit/oracle analysis, but must not be used as model features in a real deployment.",
        "- Oracle ranking is not a deployable policy; it ranks customers using their true expected net benefit.",
        "- Sensitivity analysis varies gross margin and redemption-cost assumptions. It does not prove which margin or redemption rate is commercially correct.",
        "- A proxy based on discount sensitivity is only valid if it matches the generator's redemption mechanism; validate the mechanism directly in `src/data/generate_data.py`.",
        "- If assignment-balance checks flag covariates, investigate before treating the IPW estimate as a randomized experiment result.",
        "",
        "## Automated checks",
    ]
    for row in checks:
        status = "PASS" if row["passed"] else f"FAIL ({row['severity']})"
        report_lines.append(f"- **{status} — {row['check']}**: observed `{row['observed']}`, expected `{row['expected']}`. {row['note']}")
    if issues:
        report_lines.extend(["", "## Issues to investigate"] + [f"- {issue}" for issue in issues])
    else:
        report_lines.extend(["", "No ERROR/WARNING checks failed. This does not eliminate model or economic-assumption uncertainty."])

    (OUT_DIR / "AUDIT_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_csv(pd.DataFrame(checks), "automated_checks.csv")

    summary = {
        "dataset_path": str(DATA_PATH),
        "n_rows": int(n),
        "unique_customer_ids": int(df["customer_id"].nunique()),
        "eligible_control_n": int(n0),
        "eligible_treatment_10_eur_n": int(n1),
        "eligible_n": int(ne),
        "empirical_treatment_propensity": float(p_emp),
        "observed_treatment_minus_control_net_contribution_per_customer_30pct_margin": float(observed_diff),
        "observed_difference_ci_lower": float(observed_ci[0]),
        "observed_difference_ci_upper": float(observed_ci[1]),
        "truth_expected_blanket_net_benefit_per_10000_eligible_30pct_margin": float(margin30["expected_net_benefit_per_10000_eligible_eur"]),
        "best_oracle_budget_pct_30pct_margin_baseline_redemption": int(best_oracle["budget_pct"]),
        "failed_checks": int(sum(not c["passed"] for c in checks)),
        "error_checks": int(sum((not c["passed"]) and c["severity"] == "ERROR" for c in checks)),
        "warning_checks": int(sum((not c["passed"]) and c["severity"] == "WARNING" for c in checks)),
    }
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSIMULATION AND ECONOMICS AUDIT COMPLETE")
    print(f"Dataset: {DATA_PATH}")
    print(f"Audit outputs: {OUT_DIR}")
    print(f"Eligible control={n0:,}, €10 treatment={n1:,}, propensity={p_emp:.4f}")
    print(f"Observed treatment-control net contribution: €{observed_diff:.4f}/customer "
          f"(approx. 95% CI €{observed_ci[0]:.4f} to €{observed_ci[1]:.4f})")
    print(f"Truth expected blanket benefit (30% margin, baseline redemption): "
          f"€{margin30['expected_net_benefit_per_10000_eligible_eur']:,.2f} per 10,000")
    print(f"Best oracle budget on tested grid under baseline assumptions: {int(best_oracle['budget_pct'])}%")
    print(f"Failed checks: {summary['failed_checks']} "
          f"(errors={summary['error_checks']}, warnings={summary['warning_checks']})")
    print(f"Read report: {OUT_DIR / 'AUDIT_REPORT.md'}")


if __name__ == "__main__":
    main()
