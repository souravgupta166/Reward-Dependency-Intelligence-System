
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "raw" / "reward_customer_data.csv"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REWARD_AMOUNTS_EUR = [3, 5, 10, 15]
GROSS_MARGINS = [0.20, 0.30, 0.40, 0.50]

# IMPORTANT: 0% is the no-reward baseline.
TARGETING_SHARES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run src/data/generate_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "customer_id",
        "avg_order_value_eur",
        "discount_sensitivity",
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]

    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    numeric_columns = [
        "avg_order_value_eur",
        "discount_sensitivity",
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]

    if df[numeric_columns].isna().any().any():
        raise ValueError("Missing values found in required numeric columns.")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Infinite or invalid numeric values found.")

    if (df["avg_order_value_eur"] <= 0).any():
        raise ValueError("Average order values must be positive.")

    for column in [
        "discount_sensitivity",
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    ]:
        if not df[column].between(0, 1).all():
            raise ValueError(f"{column} must be between 0 and 1.")

    if (
        df["purchase_probability_with_reward_10"]
        < df["purchase_probability_no_reward"]
    ).any():
        raise ValueError(
            "Found €10 reward probabilities below no-reward probabilities. "
            "This script assumes non-negative reward effects."
        )

    return df.reset_index(drop=True)


# ============================================================
# PROBABILITY SCENARIOS
# ============================================================

def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    return np.log(probability / (1 - probability))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def estimate_reward_probability(
    p0: np.ndarray,
    p10: np.ndarray,
    reward_amount: float,
) -> np.ndarray:
    """
    Reconstruct purchase probabilities for different reward amounts.

    Assumption:
    The log-odds effect scales linearly with reward amount up to €15.
    The €10 probabilities are ground truth from the synthetic generator.

    Probabilities for €3, €5, and €15 are reconstructed scenarios,
    not separately observed or independently validated outcomes.
    """
    reward_multiplier = min(reward_amount / 10.0, 1.5)

    logit_difference = logit(p10) - logit(p0)

    estimated_logit = logit(p0) + reward_multiplier * logit_difference

    return sigmoid(estimated_logit)


# ============================================================
# CUSTOMER-LEVEL UNIT ECONOMICS
# ============================================================

def calculate_customer_economics(
    df: pd.DataFrame,
    reward_amount: float,
    gross_margin: float,
) -> pd.DataFrame:
    """
    Calculate expected incremental net benefit per eligible customer.

    Expected incremental gross profit:
        (P1 - P0) * average order value * gross margin

    Expected reward cost:
        P1 * redemption sensitivity * reward amount

    Expected incremental net benefit:
        incremental gross profit - expected reward cost

    This is a simplified one-purchase economic model. It excludes
    operational costs, repeat purchases, and other business effects.
    """
    if reward_amount <= 0:
        raise ValueError("Reward amount must be positive.")

    if not 0 < gross_margin <= 1:
        raise ValueError("Gross margin must be greater than 0 and at most 1.")

    p0 = df["purchase_probability_no_reward"].to_numpy(dtype=float)
    p10 = df["purchase_probability_with_reward_10"].to_numpy(dtype=float)
    order_value = df["avg_order_value_eur"].to_numpy(dtype=float)
    redemption_probability = df["discount_sensitivity"].to_numpy(dtype=float)

    p1 = estimate_reward_probability(
        p0=p0,
        p10=p10,
        reward_amount=reward_amount,
    )

    uplift = p1 - p0

    incremental_gross_profit = uplift * order_value * gross_margin

    expected_reward_cost = (
        p1 * redemption_probability * reward_amount
    )

    result = pd.DataFrame(index=df.index)
    result["p0"] = p0
    result["p_reward"] = p1
    result["purchase_uplift"] = uplift
    result["incremental_gross_profit_eur"] = incremental_gross_profit
    result["expected_reward_cost_eur"] = expected_reward_cost
    result["net_benefit_eur"] = (
        incremental_gross_profit - expected_reward_cost
    )

    return result


# ============================================================
# SCENARIO EVALUATION
# ============================================================

def evaluate_scenarios(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate blanket rewards and oracle targeting.

    Blanket rewards:
        Offer the specified reward to every eligible customer.

    Oracle targeting:
        Rank customers by their true simulated expected net benefit.

    Oracle targeting is an optimistic simulation benchmark. It uses
    ground-truth probabilities and is not a deployable ML policy.
    """
    scenario_rows = []
    targeting_rows = []

    n_customers = len(df)

    for reward_amount in REWARD_AMOUNTS_EUR:
        for margin in GROSS_MARGINS:
            economics = calculate_customer_economics(
                df,
                reward_amount=reward_amount,
                gross_margin=margin,
            )

            benefits = economics["net_benefit_eur"].to_numpy(dtype=float)

            incremental_revenue = float(
                (
                    economics["purchase_uplift"]
                    * df["avg_order_value_eur"].to_numpy(dtype=float)
                ).sum()
            )

            total_incremental_gross_profit = float(
                economics["incremental_gross_profit_eur"].sum()
            )

            total_reward_cost = float(
                economics["expected_reward_cost_eur"].sum()
            )

            blanket_total = float(benefits.sum())

            scenario_rows.append({
                "reward_amount_eur": reward_amount,
                "gross_margin_pct": margin * 100,
                "policy": "Blanket rewards",
                "targeting_share_pct": 100.0,
                "customers_targeted": n_customers,
                "mean_purchase_uplift": float(
                    economics["purchase_uplift"].mean()
                ),
                "expected_incremental_revenue_eur": incremental_revenue,
                "expected_incremental_gross_profit_eur":
                    total_incremental_gross_profit,
                "expected_reward_cost_eur": total_reward_cost,
                "expected_incremental_net_benefit_eur": blanket_total,
                "net_benefit_per_eligible_customer_eur":
                    blanket_total / n_customers,
                "net_benefit_per_10000_customers_eur":
                    blanket_total / n_customers * 10000,
            })

            # Rank by true simulated economic benefit.
            # This is an oracle benchmark, not a practical model.
            order = np.argsort(-benefits)

            for share in TARGETING_SHARES:
                # At 0%, select nobody. At positive shares, use ceil
                # to ensure a nonzero budget selects at least one person.
                number_targeted = (
                    0 if share == 0
                    else min(n_customers, int(np.ceil(n_customers * share)))
                )

                selected_indices = order[:number_targeted]
                selected = economics.iloc[selected_indices]

                total_benefit = float(selected["net_benefit_eur"].sum())

                total_gross_profit = float(
                    selected["incremental_gross_profit_eur"].sum()
                )

                total_cost = float(
                    selected["expected_reward_cost_eur"].sum()
                )

                mean_uplift = (
                    float(selected["purchase_uplift"].mean())
                    if number_targeted > 0
                    else 0.0
                )

                targeting_rows.append({
                    "reward_amount_eur": reward_amount,
                    "gross_margin_pct": margin * 100,
                    "policy": "Oracle targeted rewards",
                    "targeting_share_pct": share * 100,
                    "customers_targeted": number_targeted,
                    "mean_purchase_uplift_selected": mean_uplift,
                    "expected_incremental_gross_profit_eur":
                        total_gross_profit,
                    "expected_reward_cost_eur": total_cost,
                    "expected_incremental_net_benefit_eur": total_benefit,
                    "net_benefit_per_eligible_customer_eur":
                        total_benefit / n_customers,
                    "net_benefit_per_10000_customers_eur":
                        total_benefit / n_customers * 10000,
                })

    return pd.DataFrame(scenario_rows), pd.DataFrame(targeting_rows)


# ============================================================
# BREAK-EVEN ANALYSIS
# ============================================================

def calculate_break_even_margins(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the gross margin required for blanket rewards to break even.

    Required gross margin:
        total expected reward cost / total incremental revenue

    If incremental revenue is effectively zero, break-even margin
    cannot be calculated reliably.
    """
    rows = []

    for reward_amount in REWARD_AMOUNTS_EUR:
        economics = calculate_customer_economics(
            df,
            reward_amount=reward_amount,
            gross_margin=1.0,
        )

        incremental_revenue = float(
            (
                economics["purchase_uplift"]
                * df["avg_order_value_eur"].to_numpy(dtype=float)
            ).sum()
        )

        reward_cost = float(
            economics["expected_reward_cost_eur"].sum()
        )

        if incremental_revenue > 1e-10:
            required_margin = reward_cost / incremental_revenue
            break_even_pct = required_margin * 100
        else:
            required_margin = np.nan
            break_even_pct = np.nan

        rows.append({
            "reward_amount_eur": reward_amount,
            "expected_incremental_revenue_eur": incremental_revenue,
            "expected_reward_cost_eur": reward_cost,
            "break_even_gross_margin_pct": break_even_pct,
            "break_even_within_100pct_margin": bool(
                np.isfinite(required_margin)
                and required_margin <= 1.0
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# BEST ORACLE POLICY PER SCENARIO
# ============================================================

def select_best_oracle_scenarios(
    targeting_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Choose the targeting share with the highest total expected benefit
    for each reward amount and gross margin.

    Since 0% is included, the best benefit cannot be below zero,
    except for numerical rounding.
    """
    group_columns = ["reward_amount_eur", "gross_margin_pct"]

    best_indices = targeting_df.groupby(group_columns)[
        "expected_incremental_net_benefit_eur"
    ].idxmax()

    best = targeting_df.loc[best_indices].copy()

    return best.sort_values(
        ["gross_margin_pct", "reward_amount_eur"]
    ).reset_index(drop=True)


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_visualizations(
    blanket_df: pd.DataFrame,
    targeting_df: pd.DataFrame,
) -> None:
    # --------------------------------------------------------
    # Figure 1: Blanket policy heatmap
    # --------------------------------------------------------
    heatmap_data = blanket_df.pivot(
        index="gross_margin_pct",
        columns="reward_amount_eur",
        values="net_benefit_per_10000_customers_eur",
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(
        [f"€{value:g}" for value in heatmap_data.columns]
    )

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(
        [f"{value:.0f}%" for value in heatmap_data.index]
    )

    ax.set_xlabel("Reward amount")
    ax.set_ylabel("Gross margin")
    ax.set_title(
        "Blanket reward policy: expected net benefit per 10,000 customers"
    )

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.values[i, j]
            ax.text(
                j,
                i,
                f"€{value:,.0f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="Expected net benefit (€)")
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "unit_economics_blanket_heatmap.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Figure 2: Oracle targeting sensitivity at 30% margin
    # --------------------------------------------------------
    selected_margin = 30.0

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for reward_amount in REWARD_AMOUNTS_EUR:
        subset = targeting_df[
            (targeting_df["reward_amount_eur"] == reward_amount)
            & (targeting_df["gross_margin_pct"] == selected_margin)
        ].sort_values("targeting_share_pct")

        ax.plot(
            subset["targeting_share_pct"],
            subset["net_benefit_per_10000_customers_eur"],
            marker="o",
            label=f"€{reward_amount:g} reward",
        )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("Share of customers targeted (%)")
    ax.set_ylabel("Expected net benefit per 10,000 customers (€)")
    ax.set_title(
        "Oracle targeting sensitivity at 30% gross margin"
    )
    ax.legend(title="Reward amount")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "unit_economics_targeting_sensitivity.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 68)
    print("REWARD DEPENDENCY INTELLIGENCE SYSTEM")
    print("UNIT-ECONOMICS SENSITIVITY ANALYSIS")
    print("=" * 68)

    df = load_data()

    print(f"\nEligible customers: {len(df):,}")
    print(
        "Average order value: "
        f"€{df['avg_order_value_eur'].mean():.2f}"
    )
    print(
        "Average discount sensitivity: "
        f"{df['discount_sensitivity'].mean():.3f}"
    )

    blanket_df, targeting_df = evaluate_scenarios(df)
    break_even_df = calculate_break_even_margins(df)
    best_targeting = select_best_oracle_scenarios(targeting_df)

    blanket_path = (
        METRICS_DIR / "unit_economics_blanket_sensitivity.csv"
    )
    targeting_path = (
        METRICS_DIR / "unit_economics_targeting_sensitivity.csv"
    )
    break_even_path = (
        METRICS_DIR / "unit_economics_break_even.csv"
    )
    best_targeting_path = (
        METRICS_DIR / "unit_economics_best_oracle_policy.csv"
    )

    blanket_df.to_csv(blanket_path, index=False)
    targeting_df.to_csv(targeting_path, index=False)
    break_even_df.to_csv(break_even_path, index=False)
    best_targeting.to_csv(best_targeting_path, index=False)

    create_visualizations(blanket_df, targeting_df)

    print("\nBLANKET POLICY: NET BENEFIT PER 10,000 CUSTOMERS")
    print(
        blanket_df[
            [
                "reward_amount_eur",
                "gross_margin_pct",
                "net_benefit_per_10000_customers_eur",
            ]
        ]
        .sort_values(["gross_margin_pct", "reward_amount_eur"])
        .round(2)
        .to_string(index=False)
    )

    print("\nBREAK-EVEN GROSS MARGIN")
    print(break_even_df.round(2).to_string(index=False))

    print("\nBEST ORACLE TARGETING SCENARIO PER ASSUMPTION")
    print(
        best_targeting[
            [
                "reward_amount_eur",
                "gross_margin_pct",
                "targeting_share_pct",
                "customers_targeted",
                "expected_incremental_net_benefit_eur",
                "net_benefit_per_10000_customers_eur",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    # Verify that the zero-budget baseline is present.
    zero_budget_rows = targeting_df[
        targeting_df["targeting_share_pct"] == 0
    ]

    if len(zero_budget_rows) != (
        len(REWARD_AMOUNTS_EUR) * len(GROSS_MARGINS)
    ):
        raise AssertionError(
            "The no-reward baseline is missing from some scenarios."
        )

    if not np.allclose(
        zero_budget_rows["expected_incremental_net_benefit_eur"],
        0.0,
    ):
        raise AssertionError(
            "The zero-targeting baseline should have zero incremental benefit."
        )

    print("\nVALIDATION CHECK")
    print("Zero-targeting scenarios: present for all assumptions.")
    print("Zero-targeting incremental net benefit: €0.00.")
    print("Oracle policy selection includes the no-reward option.")

    print("\nSAVED OUTPUTS")
    print(f"Blanket sensitivity: {blanket_path}")
    print(f"Targeting sensitivity: {targeting_path}")
    print(f"Break-even analysis: {break_even_path}")
    print(f"Best oracle policy: {best_targeting_path}")
    print(
        "Heatmap: "
        f"{FIGURES_DIR / 'unit_economics_blanket_heatmap.png'}"
    )
    print(
        "Targeting plot: "
        f"{FIGURES_DIR / 'unit_economics_targeting_sensitivity.png'}"
    )

    print("\nRESEARCH CAVEATS")
    print("- All results are based on synthetic data.")
    print("- €3, €5, and €15 probabilities are reconstructed scenarios.")
    print("- Redemption sensitivity is a proxy for redemption probability.")
    print("- Oracle targeting uses ground-truth simulated benefits.")
    print("- Oracle results are not deployable ML performance estimates.")
    print("- Gross margin, order value, and redemption assumptions matter.")
    print("- Operational costs and repeat purchases are not modeled.")
    print("- Expected benefits are modeled values, not guaranteed realized profit.")


if __name__ == "__main__":
    main()
