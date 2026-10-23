# Simulation and Economics Audit

## Dataset and assignment
- Raw file: `data\raw\reward_customer_data.csv`
- Rows: 10,000; unique customer IDs: 10,000.
- Eligible no-offer control: 4,997; exactly €10 treatment: 1,265; combined: 6,262.
- Empirical treatment propensity in this eligible cohort: 0.2020 (reference expectation approximately 0.20).

## Key economic check (30% gross margin, baseline redemption assumption)
- Expected incremental net benefit of a blanket €10 reward: €-16,760.16 per 10,000 eligible customers.
- Expected incremental gross profit per eligible customer: €1.1758.
- Expected reward cost per eligible customer: €2.8518.
- Fraction with positive modeled individual net benefit: 0.030.
- Best oracle budget in the tested grid at these assumptions: 5% (expected €39.08 per 10,000 eligible customers).

## Observed randomized comparison
- Observed treatment-minus-control net contribution at 30% margin: €-1.6377 per eligible customer.
- Approximate 95% normal CI: [€-2.0629, €-1.2126] per customer.
- This comparison is noisy and should be read alongside the IPW estimate and simulator's known expected value.

## Interpretation guardrails
- The stored potential-outcome probabilities are simulator ground truth. They are appropriate for audit/oracle analysis, but must not be used as model features in a real deployment.
- Oracle ranking is not a deployable policy; it ranks customers using their true expected net benefit.
- Sensitivity analysis varies gross margin and redemption-cost assumptions. It does not prove which margin or redemption rate is commercially correct.
- A proxy based on discount sensitivity is only valid if it matches the generator's redemption mechanism; validate the mechanism directly in `src/data/generate_data.py`.
- If assignment-balance checks flag covariates, investigate before treating the IPW estimate as a randomized experiment result.

## Automated checks
- **PASS — row_count**: observed `10000`, expected `10,000 synthetic customers`. 
- **PASS — customer_id_unique**: observed `10000`, expected `10000`. 
- **PASS — missing_required_values**: observed `0`, expected `0`. 
- **PASS — binary_flags_valid**: observed `checked reward_offered/reward_redeemed/purchase`, expected `values only 0/1`. 
- **PASS — probabilities_in_range**: observed `checked`, expected `all probabilities between 0 and 1`. 
- **PASS — discount_sensitivity_in_range**: observed `min=0.0100, max=0.9900`, expected `all values between 0 and 1`. 
- **PASS — reward_cost_matches_redemption**: observed `max absolute difference=0.00000000`, expected `reward_cost_eur = reward_redeemed * reward_amount_eur`. 
- **PASS — redemption_requires_offer_and_purchase**: observed `0`, expected `0`. 
- **PASS — no_offer_has_zero_amount**: observed `0`, expected `0`. 
- **PASS — eligible_sample_nonempty**: observed `6262`, expected `> 0`. 
- **PASS — empirical_treatment_propensity_near_0_20**: observed `0.20201`, expected `approximately 0.20`. A large deviation may indicate changed assignment logic or a small/filtered sample.
- **PASS — covariate_balance_abs_smd_under_0_10**: observed `[]`, expected `absolute SMD <= 0.10 for each available covariate`. Balance is a diagnostic, not proof of randomization.
- **PASS — treatment_probability_not_below_control_on_average**: observed `mean p0=0.4741; mean p1=0.5548`, expected `mean p1 >= mean p0 for this generator`. Individual customers may still have zero or very small modeled uplift.

No ERROR/WARNING checks failed. This does not eliminate model or economic-assumption uncertainty.