
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Reward Dependency Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "reward_customer_data.csv"
METRICS_PATH = ROOT / "reports" / "metrics"
AUDIT_PATH = METRICS_PATH / "simulation_economics_audit"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1500px;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf2;
        padding: 16px;
        border-radius: 12px;
    }
    .hero {
        background: linear-gradient(120deg, #edf4ff, #f7fbff);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e1eafa;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path)


@st.cache_data
def load_report(path, modified_time):
    return pd.read_csv(path)


def read_report(filename):
    path = METRICS_PATH / filename

    if not path.exists():
        return pd.DataFrame()

    return load_report(str(path), path.stat().st_mtime)


def money(value, decimals=0):
    return f"€{value:,.{decimals}f}"


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Check that data/raw/reward_customer_data.csv "
        "exists in your project."
    )
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

audit_file = AUDIT_PATH / "audit_summary.json"

if audit_file.exists():
    import json

    with open(audit_file, encoding="utf-8") as file:
        audit = json.load(file)
else:
    audit = {}

economics = read_report(
    "simulation_economics_audit/"
    "blanket_treatment_economic_sensitivity.csv"
)

policy_results = read_report(
    "policy_stability_test_comparison.csv"
)

budget_frequency = read_report(
    "policy_stability_budget_frequency.csv"
)

paired_results = read_report(
    "policy_stability_paired_differences.csv"
)

audit_checks = read_report(
    "simulation_economics_audit/automated_checks.csv"
)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.title("📊 Reward Intelligence")
    st.caption("Customer Analytics & Optimization")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Explorer",
            "Customer Segmentation",
            "Reward Simulator",
            "Policy Optimization",
            "Results & Methodology",
        ],
    )

    st.divider()

    st.info(
        "**Synthetic-data demo**\n\n"
        "This project uses simulated customer data. "
        "Results are for educational and portfolio purposes."
    )

    st.caption(
        f"{len(df):,} customer records · "
        f"{len(df.columns)} columns"
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1 style="color:#14213d;margin-bottom:5px;">
            Reward Dependency Intelligence System
        </h1>
        <p style="color:#526783;margin-bottom:0;">
            Explore customer behaviour, simulate reward economics,
            and evaluate targeting policies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.subheader("Executive Overview")

    eligible = audit.get("eligible_n", 6262)
    treated = audit.get("eligible_treatment_10_eur_n", 1265)
    propensity = audit.get("empirical_treatment_propensity", 0.202)
    blanket_impact = audit.get(
        "truth_expected_blanket_net_benefit_per_10000_eligible_30pct_margin",
        -16760.16,
    )
    oracle_budget = audit.get(
        "best_oracle_budget_pct_30pct_margin_baseline_redemption",
        5,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{len(df):,}",
    )

    col2.metric(
        "Eligible Customers",
        f"{eligible:,}",
    )

    col3.metric(
        "€10 Treatment Share",
        f"{propensity:.1%}",
        f"{treated:,} treated",
    )

    col4.metric(
        "Blanket Reward Impact",
        money(blanket_impact),
        "per 10,000 eligible customers",
    )

    st.caption(
        f"Oracle benchmark budget: {oracle_budget}% "
        "under baseline simulation assumptions. "
        "This is not a deployable ML recommendation."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Reward Economics")

        if not economics.empty:
            plot_data = economics[
                economics["redemption_cost_multiplier"].isin(
                    [0.5, 1.0]
                )
            ].copy()

            plot_data["Gross Margin"] = (
                plot_data["gross_margin"] * 100
            ).astype(int).astype(str) + "%"

            plot_data["Redemption Assumption"] = (
                plot_data["redemption_cost_multiplier"]
                .map({
                    0.5: "50% of baseline cost",
                    1.0: "Baseline cost",
                })
            )

            fig = px.line(
                plot_data,
                x="Gross Margin",
                y="expected_net_benefit_per_10000_eligible_eur",
                color="Redemption Assumption",
                markers=True,
                template="plotly_white",
                labels={
                    "expected_net_benefit_per_10000_eligible_eur":
                    "Net Benefit per 10,000 (€)"
                },
            )

            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(height=380)

            st.plotly_chart(
                fig,
                use_container_width=True,
            )
        else:
            st.warning(
                "Economics report not found. "
                "Run the economics audit first."
            )

    with right:
        st.markdown("### Policy Comparison")

        if not policy_results.empty:
            policy_col = "policy"
            value_col = "profit_per_10000_customers_eur"

            if (
                policy_col in policy_results.columns
                and value_col in policy_results.columns
            ):
                fig = px.bar(
                    policy_results,
                    x=policy_col,
                    y=value_col,
                    template="plotly_white",
                    labels={
                        policy_col: "Policy",
                        value_col: "Net Benefit per 10,000 (€)",
                    },
                )

                fig.add_hline(y=0, line_dash="dash")
                fig.update_layout(height=380)
                fig.update_xaxes(tickangle=-20)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )
            else:
                st.dataframe(
                    policy_results,
                    use_container_width=True,
                )
        else:
            st.warning(
                "Policy evaluation report not found."
            )

    st.warning(
        "Current finding: blanket €10 rewards are "
        "unprofitable under the baseline simulated economics. "
        "The practical targeting policy currently selects "
        "0% because budget selection was unstable."
    )


# ==================================================
# PAGE 2: DATA EXPLORER
# ==================================================

elif page == "Data Explorer":

    st.subheader("Customer Data Explorer")

    st.caption(
        "Ground-truth simulation columns are hidden by default "
        "to distinguish observable data from simulation-only information."
    )

    ground_truth_columns = {
        "potential_purchase_no_reward",
        "potential_purchase_with_reward",
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
    }

    show_truth = st.checkbox(
        "Show ground-truth simulation fields for audit",
        value=False,
    )

    visible_columns = [
        col for col in df.columns
        if show_truth or col not in ground_truth_columns
    ]

    view = df[visible_columns].copy()

    search = st.text_input(
        "Search customer ID",
        placeholder="Enter a customer ID",
    )

    if search and "customer_id" in view.columns:
        view = view[
            view["customer_id"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows displayed", f"{len(view):,}")
    c2.metric(
        "Missing cells",
        f"{view.isna().sum().sum():,}",
    )
    c3.metric(
        "Duplicate IDs",
        f"{df['customer_id'].duplicated().sum():,}"
        if "customer_id" in df.columns
        else "N/A",
    )

    st.dataframe(
        view.head(1000),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Export filtered data",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="reward_dependency_data.csv",
        mime="text/csv",
    )

    left, right = st.columns(2)

    with left:
        if "purchase" in df.columns:
            purchase_data = (
                df["purchase"]
                .map({0: "No purchase", 1: "Purchase"})
                .value_counts()
                .rename_axis("Outcome")
                .reset_index(name="Customers")
            )

            fig = px.bar(
                purchase_data,
                x="Outcome",
                y="Customers",
                template="plotly_white",
                title="Purchase Distribution",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with right:
        numeric_columns = [
            col for col in df.select_dtypes("number").columns
            if col not in ground_truth_columns
            and df[col].nunique(dropna=True) > 1
        ]

        if numeric_columns:
            selected_feature = st.selectbox(
                "Select a numeric feature",
                numeric_columns,
            )

            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=30,
                template="plotly_white",
                title=f"Distribution of {selected_feature}",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ==================================================
# PAGE 3: CUSTOMER SEGMENTATION
# ==================================================

elif page == "Customer Segmentation":

    st.subheader("Customer Segmentation")

    segment_column = (
        "simulation_segment"
        if "simulation_segment" in df.columns
        else None
    )

    if segment_column:

        segment_counts = (
            df[segment_column]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        segment_counts["Share (%)"] = (
            segment_counts["Customers"] / len(df) * 100
        )

        left, right = st.columns(2)

        with left:
            fig = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                color="Segment",
                template="plotly_white",
                title="Customer Count by Simulation Segment",
            )

            fig.update_layout(
                showlegend=False,
                height=400,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:
            fig = px.pie(
                segment_counts,
                names="Segment",
                values="Customers",
                hole=0.55,
                template="plotly_white",
                title="Customer Segment Share",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        profile_columns = [
            col for col in [
                "purchase",
                "retained_90d",
                "reward_redeemed",
                "discount_sensitivity",
                "engagement_score",
                "prior_orders",
                "avg_order_value_eur",
            ]
            if col in df.columns
        ]

        if profile_columns:
            st.markdown("### Segment Profiles")

            profile = (
                df.groupby(segment_column)[profile_columns]
                .mean(numeric_only=True)
                .reset_index()
            )

            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "These are simulated latent segments, not verified labels "
            "for real customers. Do not interpret them as proof of "
            "causal reward dependency."
        )

    else:
        st.warning(
            "The simulation_segment column is not present in the dataset."
        )


# ==================================================
# PAGE 4: REWARD SIMULATOR
# ==================================================

elif page == "Reward Simulator":

    st.subheader("Reward Strategy Simulator")

    st.warning(
        "This scenario tool uses simulator ground-truth probabilities. "
        "It is an oracle economic analysis, not the practical ML policy."
    )

    required_columns = {
        "purchase_probability_no_reward",
        "purchase_probability_with_reward_10",
        "discount_sensitivity",
        "avg_order_value_eur",
    }

    if not required_columns.issubset(df.columns):
        st.error(
            "Required simulation columns are missing. "
            "Check the generated dataset."
        )
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        reward_amount = st.slider(
            "Reward amount (€)",
            min_value=3,
            max_value=15,
            value=10,
            step=1,
        )

    with col2:
        gross_margin = st.slider(
            "Gross margin (%)",
            min_value=10,
            max_value=60,
            value=30,
            step=5,
        ) / 100

    with col3:
        redemption_multiplier = st.slider(
            "Redemption cost multiplier",
            min_value=0.25,
            max_value=1.50,
            value=1.00,
            step=0.05,
        )

    budget = st.slider(
        "Targeting budget (%)",
        min_value=0,
        max_value=100,
        value=5,
        step=5,
    )

    p0 = df["purchase_probability_no_reward"].clip(0, 1)
    p10 = df["purchase_probability_with_reward_10"].clip(0, 1)

    epsilon = 1e-6

    logit0 = np.log(
        p0.clip(epsilon, 1 - epsilon)
        / (1 - p0.clip(epsilon, 1 - epsilon))
    )

    logit10 = np.log(
        p10.clip(epsilon, 1 - epsilon)
        / (1 - p10.clip(epsilon, 1 - epsilon))
    )

    adjusted_logit = (
        logit0
        + reward_amount / 10 * (logit10 - logit0)
    )

    p_reward = 1 / (
        1 + np.exp(-np.clip(adjusted_logit, -35, 35))
    )

    aov = df["avg_order_value_eur"].clip(lower=0)
    sensitivity = df["discount_sensitivity"].clip(0, 1)

    expected_benefit = (
        (p_reward - p0) * aov * gross_margin
        - p_reward * sensitivity
        * reward_amount * redemption_multiplier
    )

    n = len(df)
    k = int(np.floor(n * budget / 100))

    if k > 0:
        selected = expected_benefit.nlargest(k).index
        targeted_total = expected_benefit.loc[selected].sum()
    else:
        targeted_total = 0.0

    blanket_total = expected_benefit.sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Customers Targeted", f"{k:,}")
    c2.metric("Targeted Net Benefit", money(targeted_total))
    c3.metric("Blanket Net Benefit", money(blanket_total))

    st.caption(
        "Oracle selection ranks customers using known simulated economics. "
        "Real deployment would require estimating incremental benefit from "
        "observable customer features."
    )

    rows = []

    for share in range(0, 101, 5):
        count = int(np.floor(n * share / 100))

        total = (
            expected_benefit.nlargest(count).sum()
            if count > 0
            else 0.0
        )

        rows.append({
            "Budget (%)": share,
            "Customers": count,
            "Net Benefit (€)": total,
            "Benefit per 10,000 (€)": total / n * 10000,
        })

    curve = pd.DataFrame(rows)

    fig = px.line(
        curve,
        x="Budget (%)",
        y="Benefit per 10,000 (€)",
        markers=True,
        template="plotly_white",
        title="Oracle Targeting Budget Sensitivity",
    )

    fig.add_hline(y=0, line_dash="dash")

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        curve,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# PAGE 5: POLICY OPTIMIZATION
# ==================================================

elif page == "Policy Optimization":

    st.subheader("Policy Optimization")

    st.caption(
        "These results are loaded from saved held-out test evaluations. "
        "The UI does not reselect a policy using test outcomes."
    )

    if not policy_results.empty:

        st.markdown("### Held-out Test Results")

        st.dataframe(
            policy_results,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "policy" in policy_results.columns
            and "profit_per_10000_customers_eur" in policy_results.columns
        ):

            fig = px.bar(
                policy_results,
                x="policy",
                y="profit_per_10000_customers_eur",
                template="plotly_white",
                title="Estimated Net Benefit per 10,000 Customers",
                labels={
                    "policy": "Policy",
                    "profit_per_10000_customers_eur":
                    "Net Benefit (€)",
                },
            )

            fig.add_hline(y=0, line_dash="dash")

            fig.update_layout(
                height=420,
                xaxis_tickangle=-20,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    else:
        st.warning(
            "Policy comparison file not found. "
            "Run the policy stability analysis."
        )

    left, right = st.columns(2)

    with left:
        st.markdown("### Budget Stability")

        if not budget_frequency.empty:

            if {
                "budget_share_pct",
                "selection_frequency",
            }.issubset(budget_frequency.columns):

                plot = budget_frequency.copy()

                plot["selection_frequency"] *= 100

                fig = px.bar(
                    plot,
                    x="budget_share_pct",
                    y="selection_frequency",
                    template="plotly_white",
                    title="Budget Selection Frequency",
                    labels={
                        "budget_share_pct": "Budget (%)",
                        "selection_frequency": "Selection Frequency (%)",
                    },
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:
                st.dataframe(
                    budget_frequency,
                    use_container_width=True,
                )

    with right:
        st.markdown("### Paired Policy Differences")

        if not paired_results.empty:
            st.dataframe(
                paired_results,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Paired policy difference results are not available."
            )


# ==================================================
# PAGE 6: RESULTS & METHODOLOGY
# ==================================================

elif page == "Results & Methodology":

    st.subheader("Results & Methodology")

    st.markdown("### Audit Summary")

    if audit:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Failed Checks",
            str(audit.get("failed_checks", "—")),
        )

        c2.metric(
            "Errors",
            str(audit.get("error_checks", "—")),
        )

        c3.metric(
            "Warnings",
            str(audit.get("warning_checks", "—")),
        )

    else:
        st.warning(
            "Audit summary not found. Run the economics audit."
        )

    st.markdown("### Automated Data Checks")

    if not audit_checks.empty:
        st.dataframe(
            audit_checks,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Automated check results not found.")

    report_path = AUDIT_PATH / "AUDIT_REPORT.md"

    if report_path.exists():
        with st.expander("Read full audit report"):
            st.markdown(
                report_path.read_text(encoding="utf-8")
            )

    st.markdown("### Methodology")

    st.markdown(
        """
        **Synthetic data:** all customer records are simulated.

        **Practical model:** must use only information that would be
        available when making a customer-level decision.

        **Oracle benchmark:** uses simulated ground-truth probabilities
        and is not deployable in a real-world setting.

        **Policy evaluation:** relies on the assignment mechanism and
        propensity assumptions documented by the generator.

        **Uncertainty:** confidence intervals capture sampling
        uncertainty under the chosen evaluation method, not every
        possible source of model or simulation error.

        **Unit economics:** gross margin and reward redemption costs
        are assumptions that should be tested through sensitivity
        analysis rather than treated as universal business facts.
        """
    )

    st.markdown("### Audit Assignment Balance")

    balance_path = AUDIT_PATH / "assignment_balance.csv"

    if balance_path.exists():
        balance_data = pd.read_csv(balance_path)

        st.dataframe(
            balance_data,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Available Metrics Files")

    if METRICS_PATH.exists():

        files = sorted(
            p for p in METRICS_PATH.rglob("*")
            if p.is_file()
        )

        file_data = pd.DataFrame([
            {
                "File": str(path.relative_to(ROOT)),
                "Size (KB)": round(path.stat().st_size / 1024, 1),
            }
            for path in files
        ])

        st.dataframe(
            file_data,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Reward Dependency Intelligence System | "
    "Streamlit + Plotly | Synthetic-data portfolio project"
)
