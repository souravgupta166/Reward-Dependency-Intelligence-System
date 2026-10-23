# Reward Dependency Intelligence System
### Customer Analytics · Causal Evaluation · Machine Learning · Reward Economics

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reward-dependency-intelligence-system-7uwsqwzhcackrennwyprxn.streamlit.app/)

**Live application:** [Explore the Reward Dependency Intelligence System](https://reward-dependency-intelligence-system-7uwsqwzhcackrennwyprxn.streamlit.app/)

**GitHub repository:** [Reward Dependency Intelligence System](https://github.com/souravgupta166/Reward-Dependency-Intelligence-System)

---

## 1. Executive Summary

The Reward Dependency Intelligence System is an interactive customer analytics and decision-support application designed to investigate a critical question in customer retention and incentive marketing:

> Are rewards generating genuinely incremental purchases, or are businesses paying customers for purchases that would have happened anyway?

Many loyalty and promotional strategies measure success through redemption rates, conversion rates, or total sales. These metrics alone cannot establish whether an incentive created additional value. A customer may redeem a reward while purchasing something they would have bought without it.

This project combines synthetic customer data, customer segmentation, machine learning, reward-response simulation, policy evaluation, statistical stability analysis, and unit economics to investigate the difference between **customer conversion and incremental profitability**.

The analysis demonstrates why a high-performing predictive model is not automatically a profitable targeting strategy, why simulated oracle results must be distinguished from deployable predictions, and why a reward policy should be evaluated across repeated samples and economic assumptions before being recommended.

### Main findings

- **Blanket rewards can destroy value:** Under the baseline synthetic economic assumptions, offering a €10 reward broadly produces negative expected net benefit.
- **Observed treatment-control results are unfavorable:** The estimated difference in net contribution is approximately −€1.64 per eligible customer in the analyzed comparison.
- **The oracle is not a deployable policy:** The best budget on the tested oracle evaluation grid is 5%, but this uses simulated ground-truth information unavailable in a real deployment.
- **The practical policy is conservative:** Repeated validation selected different reward budgets, so the stability rule selected a 0% reward budget rather than treating an unstable optimum as reliable.
- **Economics matter:** Profitability changes with gross margin, redemption behavior, reward cost, and targeting effectiveness.

These are findings from a **synthetic simulation**, not proven effects in a real customer population. The application is intended for analytical exploration, methodology demonstration, and decision support.

---

## 2. Business Problem

Businesses frequently use discounts, vouchers, loyalty points, and targeted promotions to increase purchase frequency. However, a reward has a cost, and a purchase following an offer does not necessarily imply that the offer caused the purchase.

Consider three customers:

- **Organic loyalist:** Likely to purchase without an incentive. A reward may subsidize an existing purchase.
- **Reward-responsive customer:** More likely to purchase because of an incentive. A carefully targeted reward may create incremental value.
- **Incentive-dependent customer:** Purchases may be strongly associated with receiving incentives, creating potential dependency and margin pressure.
- **Disengaged customer:** Unlikely to purchase even after receiving a reward, making incentives potentially wasteful.

The decision problem is therefore not simply:

*Who is likely to purchase?*

It is:

*Who is likely to generate enough additional contribution because of a reward to justify its cost?*

The system explores this distinction through segmentation, purchase modeling, counterfactual simulation, budget evaluation, and economic sensitivity analysis.

---

## 3. System Architecture

The project is organized into data, modeling, optimization, evaluation, and application layers.

| Component | Responsibility |
|---|---|
| `src/data/generate_data.py` | Generates the synthetic customer dataset |
| `src/segmentation/customer_segments.py` | Builds customer segments |
| `src/scoring/reward_dependency.py` | Calculates reward-dependency scores |
| `src/modeling/train_models.py` | Trains and evaluates purchase models |
| `src/optimization/reward_simulator.py` | Simulates reward policies and outcomes |
| `src/optimization/evaluate_policy.py` | Evaluates policy performance |
| `src/optimization/validate_policy.py` | Supports policy validation |
| `src/optimization/policy_stability_analysis.py` | Evaluates budget-selection stability |
| `src/optimization/compare_ml_oracle.py` | Compares practical model-based and oracle policies |
| `src/optimization/unit_economics_sensitivity.py` | Investigates economic assumptions |
| `src/optimization/audit_simulation_economics.py` | Audits the simulation economics |
| `app.py` | Provides the interactive Streamlit interface |

The repository also contains saved customer data, processed segments, a trained purchase model, evaluation metrics, audit outputs, and sensitivity-analysis figures.

---

## 4. Synthetic Dataset and Experimental Design

The project uses a synthetic dataset of **10,000 customers**, with 17 columns in the generated customer data.

The synthetic population contains four latent customer segments:

| Latent segment | Simulated share | Behavioral interpretation |
|---|---:|---|
| Organic loyalist | 30% | Relatively strong baseline purchasing behavior |
| Reward-responsive | 30% | Purchases respond to incentives |
| Incentive-dependent | 20% | Purchasing behavior is strongly associated with rewards |
| Disengaged | 20% | Relatively weak purchase propensity |

These are simulation-defined behavioral groups. Their proportions and behavior should not be interpreted as measured real-world customer characteristics.

### Customer features

The generated data includes features such as:

- `tenure_months`
- `prior_orders`
- `avg_order_value_eur`
- `engagement_score`
- `discount_sensitivity`

The data also contains treatment information, reward amounts, purchase outcomes, redemption outcomes, and simulated ground-truth fields used for evaluation.

### Reward assignment

Reward assignment is simulated using a Bernoulli distribution:

\[
T_i \sim \operatorname{Bernoulli}(0.5)
\]

where \(T_i=1\) indicates that customer \(i\) is offered a reward.

For customers receiving an offer, the simulated reward amount is drawn from:

\[
R_i \in \{3,5,10,15\}\text{ euros}
\]

The reward amounts are sampled uniformly in the generator. Customers not offered a reward receive an assigned reward amount of zero.

Actual redemption is modeled separately from the offer. A reward being offered does not mean it is redeemed or that a purchase occurs.

This separation allows the analysis to distinguish reward assignment, purchasing, redemption, and reward cost.

---

## 5. Mathematical Foundations

### 5.1 Logistic purchase-probability model

The simulation uses a logistic function to transform a linear predictor into a probability between zero and one.

The logistic function is:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

For each customer, the baseline purchase logit is:

\[
z_{i,0} =
-1.25
+2.1b_i
+0.025\log(1+n_i)
+0.45e_i
\]

where:

- \(b_i\) is the simulation's baseline purchase-profile component.
- \(n_i\) represents prior orders.
- \(e_i\) is the engagement-score component.
- The coefficients are predefined simulation parameters, not coefficients estimated from a real customer population.

The baseline purchase probability is:

\[
p_{i,0}=\sigma(z_{i,0})
\]

The logarithmic transformation \(\log(1+n_i)\) allows prior-order effects to increase more slowly as order counts become large.

### 5.2 Reward-adjusted purchase probability

The simulated purchase logit when a reward is offered is:

\[
z_{i,1}=z_{i,0}
+2s_i\min\left(\frac{R_i}{10},1.5\right)
\]

where \(s_i\) represents the simulation's reward-effect-strength component and \(R_i\) is the reward amount.

The corresponding probability is:

\[
p_{i,1}=\sigma(z_{i,1})
\]

This structure allows the simulated reward effect to vary by customer. It also caps the reward scaling term at 1.5, so increasing the reward beyond the cap does not continue to increase this component of the logit.

These probabilities are part of the synthetic data-generating process. They are not evidence that real customers respond to rewards according to the same function.

### 5.3 Standardized €10 reward

For comparisons involving a standardized €10 reward, the simulation defines:

\[
p_{i,10}=
\sigma(z_{i,0}+2s_i)
\]

The simulated probability difference is:

\[
\Delta p_i=p_{i,10}-p_{i,0}
\]

This difference represents the simulated change in purchase probability under the specified reward scenario.

However, an increase in purchase probability is not sufficient to establish profitability. The value of the additional purchases must be compared with reward costs and other economic assumptions.

### 5.4 Purchase, redemption, and reward cost

The simulation distinguishes between three events:

1. A reward is offered.
2. A customer purchases.
3. A customer redeems the reward.

Redemption is simulated only when the customer receives an offer and makes a purchase, with redemption probability governed by the customer's discount-sensitivity component.

The realized reward cost is:

\[
C_i=T_iD_iR_i
\]

where \(D_i=1\) if the reward is redeemed and \(D_i=0\) otherwise.

Consequently, an offer does not automatically incur its full nominal reward cost. The realized cost depends on redemption in the simulated process.

---

## 6. Predictive Modeling Versus Incremental Impact

A central methodological distinction in this project is the difference between predicting purchasing behavior and estimating the effect of an intervention.

### Purchase prediction

A purchase model estimates:

\[
P(Y_i=1\mid X_i)
\]

where \(Y_i\) is the purchase outcome and \(X_i\) contains the features available to the model.

Such a model helps identify customers with a high predicted probability of purchasing.

But a customer who is highly likely to purchase may not need a reward.

### Incremental reward response

The decision-relevant quantity is the difference between potential outcomes:

\[
\tau_i=
P(Y_i(1)=1\mid X_i)
-
P(Y_i(0)=1\mid X_i)
\]

where:

- \(Y_i(1)\) is the potential purchase outcome under a reward.
- \(Y_i(0)\) is the potential purchase outcome without a reward.
- \(\tau_i\) represents the conditional treatment effect on purchase probability.

In real observational or experimental data, both potential outcomes cannot be observed for the same customer at the same time. Estimating individual treatment effects therefore requires additional assumptions and appropriate causal or experimental methodology.

In this project, the synthetic generator defines both potential outcomes for evaluation. These ground-truth fields allow model-based decisions to be compared against a simulated oracle.

**Important:** Ground-truth fields are used for simulation audits and oracle analysis, not as ordinary features for a deployable predictive model. A real implementation must not train on information that would only be available after the decision or that reveals the counterfactual outcome.

---

## 7. Customer Segmentation and Reward Dependency

Customer segmentation helps interpret different purchasing and incentive behaviors rather than treating every customer as economically identical.

The project uses the following conceptual interpretations:

| Segment | Possible business implication |
|---|---|
| Organic loyalist | Avoid unnecessary reward subsidies where purchases are already likely |
| Reward-responsive | Investigate whether incremental contribution justifies the reward |
| Incentive-dependent | Monitor profitability and dependence on recurring promotions |
| Disengaged | Avoid assuming that a larger incentive will reliably create demand |

Segmentation is descriptive and decision-supporting. A segment label does not, by itself, establish that a customer has a causal response to rewards.

The reward-dependency score is intended to support analysis of customers' reliance on incentives. It should not be interpreted as a validated causal estimate unless independently supported by appropriate experimental evidence.

---

## 8. Reward Policy Economics

The project evaluates reward strategies in economic terms instead of relying exclusively on purchase conversion.

### 8.1 Incremental contribution

A simplified conceptual model of incremental net benefit is:

\[
\Delta \Pi =
\Delta \text{Contribution}
-
\Delta \text{Reward Cost}
\]

If the gross margin rate is \(m\), incremental revenue is \(\Delta S\), and incremental reward expense is \(\Delta C\), a simplified calculation is:

\[
\Delta \Pi=m\Delta S-\Delta C
\]

This is a conceptual unit-economics relationship. The precise contribution calculation in any evaluation must follow that evaluation's implementation, reward-cost definition, and baseline.

A policy is economically attractive only when the expected incremental contribution exceeds the associated incremental costs under the selected assumptions.

### 8.2 Blanket reward strategy

A blanket policy offers a reward broadly to eligible customers.

Its advantage is operational simplicity. Its disadvantage is that it can subsidize customers who would have purchased anyway, as well as customers who do not convert after receiving the offer.

Under the project's baseline synthetic economics, a blanket €10 reward strategy produces an estimated expected net benefit of approximately **−€16,760 per 10,000 eligible customers**.

This is a simulation result, not a forecast for an actual company.

### 8.3 Treatment-control comparison

The analysis compares eligible customers receiving the relevant reward treatment with those in the control group.

The recorded comparison contained:

- 6,262 eligible customers in the analyzed cohort.
- 4,997 control customers.
- 1,265 treated customers.
- An empirical treatment share of approximately 20.2% within this eligible comparison.

The empirical treatment share differs from the generator's initial 50% assignment probability because this comparison uses a restricted eligible cohort and a specific reward scenario.

The estimated treatment-control difference in net contribution was approximately:

\[
\widehat{\Delta \Pi}=-€1.6377
\]

per eligible customer, with an approximate 95% confidence interval of:

\[
[-€2.0629,\,-€1.2126]
\]

Because the reported interval is below zero, the estimate is unfavorable under the evaluated comparison and its assumptions.

The interval represents uncertainty in the estimated comparison under the analysis used; it does not account automatically for every possible source of model misspecification or simulation uncertainty.

### 8.4 Confidence intervals

For an estimated mean difference, a conventional large-sample confidence interval can be expressed as:

\[
\widehat{\Delta}\pm1.96\,SE(\widehat{\Delta})
\]

where \(SE(\widehat{\Delta})\) is the estimated standard error.

The exact interval depends on the estimator, variance calculation, sampling assumptions, and implementation. The formula above explains the general principle and should not be treated as a claim that every interval in the project was computed using that exact method.

A confidence interval helps assess estimation uncertainty; it does not by itself establish that a policy is commercially viable.

---

## 9. Policy Optimization and Budget Selection

The project evaluates reward policies across different targeting budgets.

A budget represents the share of eligible customers selected for a reward. For example, a 5% targeting budget selects a smaller subset than a 20% budget.

In principle, a targeting policy should prioritize customers expected to generate the largest incremental economic value, rather than simply selecting customers with the highest purchase probability.

### 9.1 Model-based policy

A practical policy uses information that could be available when a real decision is made. Its performance depends on model quality, feature availability, estimation error, and how the policy ranks customers.

### 9.2 Oracle policy

The oracle policy uses simulated ground-truth information to represent an idealized benchmark.

The best tested oracle budget in the project's audit grid was **5%** under the baseline assumptions.

This result is useful for understanding the potential value of ranking customers correctly. It is not a deployable recommendation because the oracle uses information that would not be known in an actual business decision.

An oracle is a benchmark, not a production model.

### 9.3 Why the highest-performing budget is not automatically selected

Selecting the budget with the highest estimated profit on a single validation sample can lead to overfitting. Small changes in customers, outcomes, or estimated treatment effects may change the apparent best budget.

The project therefore includes repeated validation and policy-stability analysis.

Instead of trusting a single best result, the analysis examines how selected budgets change across repeated splits.

---

## 10. Policy Stability and the Final Decision

The policy-stability analysis found that the preferred reward budget varied across repeated validation runs.

The selected budgets ranged from **0% to 30%**, and the modal budget was **10%**, selected in only **23.33%** of the repeated runs.

The project uses a 70% selection-frequency threshold as its stability criterion. The observed modal frequency falls well below that threshold.

The decision rule therefore favors a conservative 0% practical reward budget under the tested baseline assumptions.

### Why 0% was selected

The decision was not based on a claim that rewards can never work. It reflects the evidence available in this particular simulation:

1. Blanket €10 rewards had negative estimated economics.
2. The best oracle result relied on unavailable ground-truth information.
3. The preferred budget varied substantially across validation runs.
4. The modal selected budget did not satisfy the predefined stability threshold.
5. A conservative policy was preferable to presenting an unstable budget as a reliable business recommendation.

**Final interpretation:** The analysis did not establish a stable, reliably profitable nonzero reward budget under the tested baseline setup.

The 0% result is a conservative policy-selection outcome, not proof that no reward strategy can ever generate incremental profit. Different margins, redemption behavior, targeting quality, or real-world treatment effects could lead to a different decision.

---

## 11. Unit Economics and Sensitivity Analysis

A policy can be unprofitable under one set of assumptions and profitable under another. Sensitivity analysis helps identify which assumptions drive the result.

The project includes economic sensitivity outputs for blanket rewards, targeted policies, oracle policies, and related break-even calculations.

Key considerations include:

- **Gross margin:** A higher margin can increase the contribution earned from incremental purchases.
- **Reward amount:** Larger rewards may increase response in the simulation, but they can also increase the cost of redeemed incentives.
- **Redemption behavior:** Redemption affects realized reward expense.
- **Targeting quality:** A policy that identifies genuinely incremental customers may perform differently from one that simply identifies likely purchasers.
- **Targeting budget:** Selecting more customers increases reach but may also include customers with lower incremental value.
- **Model uncertainty:** An estimated uplift or treatment effect can be noisy, changing the ranking and the selected budget.

### Break-even principle

A simplified break-even condition is:

\[
\Delta \text{Contribution}
\geq
\Delta \text{Reward Cost}
\]

The exact break-even point depends on the implementation's definition of contribution, the reward expense, and the policy being evaluated.

The sensitivity analyses are designed to reveal when the policy's economics change sign and which assumptions warrant further validation.

They do not establish that a scenario will occur in a real business.

---

## 12. Audit and Validation

The project includes a simulation-economics audit and supporting outputs, including:

- Assignment-balance checks.
- Automated simulation checks.
- Comparisons between inverse-propensity-weighted estimates and simulated truth.
- Blanket-treatment economic sensitivity.
- Oracle economic sensitivity.
- Policy-stability summaries and paired comparisons.
- Model-ranking diagnostics.

The audit outputs help examine whether the simulation's assignments and economic calculations behave as expected and whether estimated policy results align with the simulation's defined ground truth.

An audit passing its automated checks establishes only that the checked conditions passed. It does not independently validate the simulation assumptions against real customer behavior.

### Inverse probability weighting

Inverse probability weighting (IPW) is a method for estimating treatment effects by weighting observations according to their probability of receiving the treatment they actually received.

For treatment indicator \(T_i\) and estimated propensity score \(e(X_i)\), a common inverse-probability weight has the form:

\[
w_i =
\frac{T_i}{e(X_i)}
+
\frac{1-T_i}{1-e(X_i)}
\]

The precise estimator, normalization, and outcome calculation matter. This expression illustrates the basic weighting principle rather than specifying every implementation detail of the project's audit.

IPW can help address treatment-assignment differences under appropriate assumptions, including a correctly specified propensity model, positivity, and conditional exchangeability. It does not automatically remove unmeasured confounding or other forms of bias.

In a synthetic setting, estimates can also be compared against the known simulated truth, providing a valuable additional diagnostic.

---

## 13. Interactive Application

The Streamlit application presents the analysis through six pages:

1. **Overview** — summarizes the project, key findings, and policy conclusions.
2. **Data Explorer** — examines the synthetic customer dataset and its features.
3. **Customer Segmentation** — explores customer groups and their characteristics.
4. **Reward Simulator** — investigates how simulated reward assumptions affect outcomes.
5. **Policy Optimization** — compares targeting policies and budget choices.
6. **Results & Methodology** — provides evaluation outputs, economic sensitivity, audit results, and methodological context.

The interface is intended to make the relationship between prediction, incremental response, policy selection, and economics easier to inspect.

Some views use simulation ground truth for oracle or evaluation purposes. Such views must not be confused with a model's actual predictive capability in a real deployment.

---

## 14. Repository Structure

```text
Reward-Dependency-Intelligence-System/
├── app.py
├── requirements.txt
├── README.md
├── config/
├── data/
│   ├── raw/
│   │   └── reward_customer_data.csv
│   └── processed/
│       └── reward_customer_segments.csv
├── models/
│   └── purchase_model.joblib
├── notebooks/
├── reports/
│   ├── figures/
│   └── metrics/
│       └── simulation_economics_audit/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── modeling/
│   ├── optimization/
│   ├── scoring/
│   └── segmentation/
└── tests/
```

This tree highlights the principal application and analysis components. Some folders contain additional files and outputs.

---

## 15. Technology Stack

- **Python** — data processing, modeling, simulation, and evaluation.
- **Pandas** — dataset manipulation and metric analysis.
- **NumPy** — numerical calculations and simulation.
- **scikit-learn** — machine-learning workflows and model evaluation.
- **Streamlit** — interactive application.
- **Plotly** — interactive visualizations.
- **Git and GitHub** — version control and source-code hosting.
- **Streamlit Community Cloud** — application deployment.

---

## 16. Run Locally

### Prerequisites

Install a compatible Python version and Git.

### Setup

Clone the repository:

```bash
git clone https://github.com/souravgupta166/Reward-Dependency-Intelligence-System.git
cd Reward-Dependency-Intelligence-System
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local address where you can open the application in your browser.

---

## 17. Limitations and Responsible Interpretation

This project is a synthetic analytical demonstration, not a validated production reward-optimization system.

Important limitations include:

1. **Synthetic data:** Customer behavior, segment proportions, and reward effects are generated using predefined assumptions.
2. **Assumption dependence:** Estimated economics change when the simulation's margin, reward response, and redemption assumptions change.
3. **Ground-truth availability:** Oracle policies use simulated information that would not ordinarily be available for real customers.
4. **Prediction versus causation:** Purchase prediction alone does not estimate incremental reward impact.
5. **Validation uncertainty:** A stable result on simulated data does not guarantee stability in a different population or business environment.
6. **Economic simplification:** Real reward programs may involve operational costs, long-term retention effects, customer lifetime value, competitive responses, and other factors not fully represented by a simplified simulation.
7. **No production claim:** The results do not establish that a policy will improve profitability for a real company.

Before using this methodology in a business, the approach should be validated using appropriately collected experimental or observational data, clearly defined contribution economics, leakage-safe modeling, and out-of-sample evaluation.

---

## 18. Future Improvements

Potential next steps include:

- Testing on a real, appropriately governed customer dataset.
- Estimating heterogeneous treatment effects using leakage-safe causal methods.
- Comparing uplift-ranking and incremental-profit-ranking strategies.
- Evaluating uncertainty-aware policy selection.
- Adding additional temporal and customer-retention outcomes.
- Modeling longer-term customer lifetime value and repeated reward exposure.
- Introducing production monitoring and drift detection.
- Comparing policies through prospective randomized experiments.

These are potential extensions, not claims about features already implemented.

---

## 19. Key Takeaway

The central lesson of the Reward Dependency Intelligence System is that **a successful reward strategy must create incremental economic value, not merely correlate with purchases**.

Predictive accuracy, customer segmentation, causal response, reward cost, budget selection, and policy stability answer different questions. A responsible decision process considers them together.

Under the baseline synthetic assumptions, blanket rewards were economically unfavorable, oracle results were not directly deployable, and repeated budget selection was too unstable to justify a nonzero practical reward budget under the chosen stability rule.

The resulting conservative decision is a useful analytical outcome: sometimes the strongest data-driven recommendation is to avoid deploying an incentive policy until the evidence becomes more reliable.

---

**Explore the live application:** [Reward Dependency Intelligence System](https://reward-dependency-intelligence-system-7uwsqwzhcackrennwyprxn.streamlit.app/)

**View the source code:** [GitHub Repository](https://github.com/souravgupta166/Reward-Dependency-Intelligence-System)
