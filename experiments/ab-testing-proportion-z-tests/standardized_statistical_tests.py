#!/usr/bin/env python3
# coding: utf-8
"""Five statistical tests in the sequential, inline style of Untitled.py.
All datasets and planning assumptions are illustrative, not company records.
Dependencies: numpy, pandas, scipy, statsmodels.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.stats.power import TTestPower, TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

alpha = 0.05
confidence_level = 0.95
target_power = 0.80

# Shared one-line definitions printed inline, without helper functions.
print("STATISTICAL TERMS — ONE-LINE DEFINITIONS")
print("Null hypothesis (H0): the reference claim tested using the sample data.")
print("Alternative hypothesis (H1): the difference or direction being investigated.")
print("Alpha: the tolerated Type I error rate, commonly 5%.")
print("Mean: the arithmetic average of observed values.")
print("Variance: the average squared spread around the mean, with sample correction when needed.")
print("Standard deviation: the typical spread of observations around their mean.")
print("Standard error: estimated sampling uncertainty of a statistic.")
print("Test statistic: the observed effect expressed in standard-error units.")
print("Degrees of freedom: the quantity determining the relevant t-distribution.")
print("P-value: assuming H0, the probability of a result at least this extreme.")
print("Critical value: the test-statistic cutoff that defines the rejection region.")
print("Confidence interval: a range produced by a method with stated long-run coverage.")
print("Effect size: the magnitude of an observed or planned difference.")
print("Power: probability of rejecting H0 if a specified alternative effect is true.")
print("MDE: the smallest effect detectable at chosen alpha and target power.")
print("Fail to reject H0 does not prove H0 is true.")
print("Power and MDE below are prospective estimates based on stated assumptions.")

# ============================================================================
# TEST 1 — ONE-PROPORTION Z-TEST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 1 — ONE-PROPORTION Z-TEST")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Check whether a landing page conversion rate exceeds a 5% benchmark.")
benchmark_conversions = 50
conversions = 65
population = 1000
benchmark_rate = benchmark_conversions / population
observed_conversion_rate = conversions / population

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: population conversion rate p = 5%.")
print("H1: population conversion rate p > 5% (one-sided, specified in advance).")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Each visitor has a binary outcome: converted or not converted.")
print("- Visitor outcomes are independent and represent the target population.")
print("- Expected successes and failures under H0 should be sufficiently large.")
print("- Comparing with a benchmark alone does not establish treatment causality.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
assert isinstance(conversions, int) and isinstance(population, int)
assert population > 0 and 0 <= conversions <= population
assert 0 < benchmark_rate < 1
assert population * benchmark_rate >= 5
assert population * (1 - benchmark_rate) >= 5
print(f"Validated {conversions} conversions from {population} visitors.")
print(f"Missing values: 0 (aggregate count inputs); invalid counts: 0.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
absolute_lift = observed_conversion_rate - benchmark_rate
relative_lift = absolute_lift / benchmark_rate
standard_error_under_null = np.sqrt(benchmark_rate * (1 - benchmark_rate) / population)
z_statistic = absolute_lift / standard_error_under_null
p_value = norm.sf(z_statistic)  # right-tailed test
z_critical = norm.ppf(1 - alpha)
standard_error_ci = np.sqrt(observed_conversion_rate * (1 - observed_conversion_rate) / population)
z_ci = norm.ppf(1 - (1 - confidence_level) / 2)
ci_lower = observed_conversion_rate - z_ci * standard_error_ci
ci_upper = observed_conversion_rate + z_ci * standard_error_ci
print(f"Benchmark rate: {benchmark_rate:.2%}")
print(f"Observed conversion rate: {observed_conversion_rate:.2%}")
print(f"Absolute lift: {absolute_lift * 100:.2f} percentage points")
print(f"Relative lift: {relative_lift:.2%}")
print(f"Null standard error: {standard_error_under_null:.6f}")
print(f"Z-statistic: {z_statistic:.4f}; critical Z: {z_critical:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"{confidence_level:.0%} CI for observed rate: {ci_lower:.2%} to {ci_upper:.2%}")
print("P-value context: evidence against the 5% benchmark in the hypothesised direction.")

# 6. POWER AND MDE — normal approximation, one-sided
print("\n6. STATISTICAL POWER AND MDE")
planning_lift = 0.015
planning_rate = benchmark_rate + planning_lift
se_target = np.sqrt(planning_rate * (1 - planning_rate) / population)
critical_rate = benchmark_rate + z_critical * standard_error_under_null
power = norm.sf((critical_rate - planning_rate) / se_target)
# Approximate one-sided MDE using null and target-rate standard errors.
z_power = norm.ppf(target_power)
mde_rate = (z_critical * np.sqrt(benchmark_rate * (1 - benchmark_rate) / population)
            + z_power * np.sqrt(benchmark_rate * (1 - benchmark_rate) / population))
print(f"Planning rate: {planning_rate:.2%}; planned lift: {planning_lift * 100:.2f} percentage points")
print(f"Approximate power for planned lift: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: {mde_rate * 100:.2f} percentage points")
print("Context: power is the chance this design detects the planned lift if it is real.")
print("Context: MDE is a planning approximation; exact binomial power can differ.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: the conversion rate is statistically significantly above 5%.")
else:
    print("Fail to reject H0: evidence is insufficient to conclude the rate exceeds 5%.")

# ============================================================================
# TEST 2 — TWO-PROPORTION Z-TEST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 2 — TWO-PROPORTION Z-TEST")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Compare conversion rates between independent control and treatment groups.")
control_population = 1000
control_conversions = 50
treatment_population = 1000
treatment_conversions = 85
control_rate = control_conversions / control_population
treatment_rate = treatment_conversions / treatment_population

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: p_treatment - p_control = 0.")
print("H1: p_treatment - p_control != 0 (two-sided).")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Groups are independent; each person belongs to one group only.")
print("- Outcomes are binary and independent within each group.")
print("- Expected successes and failures should support the normal approximation.")
print("- Random assignment supports causal interpretation; significance alone does not.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
assert control_population > 0 and treatment_population > 0
assert 0 <= control_conversions <= control_population
assert 0 <= treatment_conversions <= treatment_population
pooled_rate = (control_conversions + treatment_conversions) / (control_population + treatment_population)
assert min(control_population * pooled_rate,
           control_population * (1 - pooled_rate),
           treatment_population * pooled_rate,
           treatment_population * (1 - pooled_rate)) >= 5
print(f"Control validated: {control_conversions}/{control_population}.")
print(f"Treatment validated: {treatment_conversions}/{treatment_population}.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
absolute_lift = treatment_rate - control_rate
relative_lift = absolute_lift / control_rate
standard_error_under_null = np.sqrt(pooled_rate * (1 - pooled_rate) *
                                    (1 / control_population + 1 / treatment_population))
z_statistic = absolute_lift / standard_error_under_null
p_value = 2 * norm.sf(abs(z_statistic))
z_critical = norm.ppf(1 - alpha / 2)
standard_error_ci = np.sqrt(control_rate * (1 - control_rate) / control_population
                            + treatment_rate * (1 - treatment_rate) / treatment_population)
ci_lower = absolute_lift - z_critical * standard_error_ci
ci_upper = absolute_lift + z_critical * standard_error_ci
print(f"Control conversion rate: {control_rate:.2%}")
print(f"Treatment conversion rate: {treatment_rate:.2%}")
print(f"Absolute lift: {absolute_lift * 100:.2f} percentage points")
print(f"Relative lift: {relative_lift:.2%}")
print(f"Pooled rate under H0: {pooled_rate:.4%}")
print(f"Null standard error: {standard_error_under_null:.6f}")
print(f"Z-statistic: {z_statistic:.4f}; critical Z: +/-{z_critical:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"95% CI for treatment-control lift: {ci_lower * 100:.2f} to {ci_upper * 100:.2f} percentage points")
print("P-value context: evidence against equal population conversion rates.")

# 6. POWER AND MDE — two-sided; Cohen's h planning approximation
print("\n6. STATISTICAL POWER AND MDE")
planning_control_rate = 0.05
meaningful_lift = 0.02
planning_treatment_rate = planning_control_rate + meaningful_lift
cohens_h = abs(proportion_effectsize(planning_treatment_rate, planning_control_rate))
power_model = NormalIndPower()
power = power_model.power(effect_size=cohens_h, nobs1=control_population,
                          alpha=alpha, ratio=treatment_population / control_population,
                          alternative="two-sided")
mde_h = power_model.solve_power(effect_size=None, nobs1=control_population,
                                alpha=alpha, power=target_power,
                                ratio=treatment_population / control_population,
                                alternative="two-sided")
mde_treatment_rate = np.sin(np.arcsin(np.sqrt(planning_control_rate)) + mde_h / 2) ** 2
mde_lift = abs(mde_treatment_rate - planning_control_rate)
print(f"Planning control rate: {planning_control_rate:.2%}")
print(f"Planning treatment rate: {planning_treatment_rate:.2%}")
print(f"Approximate power for {meaningful_lift * 100:.1f}-point lift: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: {mde_lift * 100:.2f} percentage points")
print("Context: power and MDE depend on baseline rate, group sizes, alpha, and target power.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: the control and treatment conversion rates differ significantly.")
else:
    print("Fail to reject H0: evidence is insufficient to establish a conversion-rate difference.")

# ============================================================================
# TEST 3 — ONE-SAMPLE T-TEST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 3 — ONE-SAMPLE T-TEST")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Test whether average order value differs from a benchmark of ₹800.")
order_values = np.array([
    820, 910, 760, 845, 790, 880, 940, 810, 860, 735,
    895, 825, 780, 920, 850, 805, 870, 750, 960, 830,
    890, 795, 840, 915, 770, 855, 800, 935, 815, 875
], dtype=float)
benchmark = 800.0

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: population mean AOV = ₹800.")
print("H1: population mean AOV != ₹800 (two-sided).")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Observations are independent and representative of the target population.")
print("- Outcome is quantitative; data are approximately normal or sample size is adequate.")
print("- Outliers can strongly affect the mean and t-test.")
print("- This test compares a sample mean with a benchmark, not a control group.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
df = pd.DataFrame({"order_value": order_values})
x = df["order_value"]
assert len(x) >= 2
assert x.notna().all()
assert pd.api.types.is_numeric_dtype(x)
assert np.isfinite(x).all()
assert (x > 0).all()
print(f"Sample size: {len(x)}; missing: {x.isna().sum()}; infinite: {(~np.isfinite(x)).sum()}")
print(f"Minimum: ₹{x.min():.2f}; maximum: ₹{x.max():.2f}")
print("Review the distribution and outliers before relying on the t-test.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
n = len(x)
sample_mean = x.mean()
sample_sd = x.std(ddof=1)
standard_error = sample_sd / np.sqrt(n)
degrees_freedom = n - 1
difference = sample_mean - benchmark
t_statistic = difference / standard_error
p_value = 2 * t.sf(abs(t_statistic), df=degrees_freedom)
t_critical = t.ppf(1 - alpha / 2, df=degrees_freedom)
ci_lower = sample_mean - t_critical * standard_error
ci_upper = sample_mean + t_critical * standard_error
cohens_d = difference / sample_sd
print(f"Benchmark: ₹{benchmark:.2f}; sample mean: ₹{sample_mean:.2f}")
print(f"Difference from benchmark: ₹{difference:.2f}")
print(f"Sample SD: ₹{sample_sd:.2f}; standard error: ₹{standard_error:.2f}")
print(f"Degrees of freedom: {degrees_freedom}")
print(f"t-statistic: {t_statistic:.4f}; critical t: +/-{t_critical:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"95% CI for population mean: ₹{ci_lower:.2f} to ₹{ci_upper:.2f}")
print(f"Cohen's d: {cohens_d:.3f}")
print("P-value context: evidence against a population mean of ₹800.")

# 6. POWER AND MDE
print("\n6. STATISTICAL POWER AND MDE")
planning_sd = 100.0
meaningful_difference = 50.0
power_model = TTestPower()
power = power_model.power(effect_size=meaningful_difference / planning_sd,
                          nobs=n, alpha=alpha, alternative="two-sided")
mde_standardized = power_model.solve_power(effect_size=None, nobs=n,
                                           alpha=alpha, power=target_power,
                                           alternative="two-sided")
mde = mde_standardized * planning_sd
print(f"Planning SD: ₹{planning_sd:.2f}; meaningful difference: ₹{meaningful_difference:.2f}")
print(f"Approximate power for ₹{meaningful_difference:.0f} difference: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: ₹{mde:.2f}")
print("Context: estimates assume a planning SD of ₹100 and a two-sided test.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: the population mean AOV differs significantly from ₹800.")
else:
    print("Fail to reject H0: evidence is insufficient to establish a difference from ₹800.")

# ============================================================================
# TEST 4 — INDEPENDENT TWO-SAMPLE WELCH T-TEST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 4 — INDEPENDENT TWO-SAMPLE WELCH T-TEST")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Compare average order value between independent control and treatment groups.")
control = np.array([780, 820, 760, 810, 790, 850, 740, 800, 830, 770, 815, 795], dtype=float)
treatment = np.array([850, 880, 820, 910, 860, 840, 950, 870, 890, 830, 920, 860], dtype=float)

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: μ_treatment - μ_control = ₹0.")
print("H1: μ_treatment - μ_control != ₹0 (two-sided).")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Groups and observations within groups are independent.")
print("- Outcome is quantitative; inspect distribution and influential outliers.")
print("- Welch's test does not require equal population variances.")
print("- Random assignment is needed for a causal treatment interpretation.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
control = np.asarray(control, dtype=float)
treatment = np.asarray(treatment, dtype=float)
assert control.ndim == 1 and treatment.ndim == 1
assert len(control) >= 2 and len(treatment) >= 2
assert np.isfinite(control).all() and np.isfinite(treatment).all()
print(f"Control n={len(control)}; treatment n={len(treatment)}")
print("Missing/infinite values: none; groups are represented as separate samples.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
n_control, n_treatment = len(control), len(treatment)
mean_control, mean_treatment = control.mean(), treatment.mean()
sd_control, sd_treatment = control.std(ddof=1), treatment.std(ddof=1)
difference = mean_treatment - mean_control
standard_error = np.sqrt(sd_control**2 / n_control + sd_treatment**2 / n_treatment)
degrees_freedom = (sd_control**2 / n_control + sd_treatment**2 / n_treatment)**2 / (
    (sd_control**2 / n_control)**2 / (n_control - 1)
    + (sd_treatment**2 / n_treatment)**2 / (n_treatment - 1)
)
t_statistic = difference / standard_error
p_value = 2 * t.sf(abs(t_statistic), df=degrees_freedom)
t_critical = t.ppf(1 - alpha / 2, df=degrees_freedom)
ci_lower = difference - t_critical * standard_error
ci_upper = difference + t_critical * standard_error
pooled_sd = np.sqrt(((n_control - 1) * sd_control**2 + (n_treatment - 1) * sd_treatment**2)
                    / (n_control + n_treatment - 2))
cohens_d = difference / pooled_sd
print(f"Control mean: ₹{mean_control:.2f}; SD: ₹{sd_control:.2f}")
print(f"Treatment mean: ₹{mean_treatment:.2f}; SD: ₹{sd_treatment:.2f}")
print(f"Difference (treatment - control): ₹{difference:.2f}")
print(f"Standard error: ₹{standard_error:.2f}; Welch df: {degrees_freedom:.2f}")
print(f"t-statistic: {t_statistic:.4f}; critical t: +/-{t_critical:.4f}")
print(f"P-value: {p_value:.8f}")
print(f"95% CI for mean difference: ₹{ci_lower:.2f} to ₹{ci_upper:.2f}")
print(f"Cohen's d (descriptive): {cohens_d:.3f}")
print("P-value context: evidence against equal population mean order values.")

# 6. POWER AND MDE — approximate common-SD planning model
print("\n6. STATISTICAL POWER AND MDE")
planning_sd = 40.0
meaningful_difference = 50.0
power_model = TTestIndPower()
power = power_model.power(effect_size=meaningful_difference / planning_sd,
                          nobs1=n_control, alpha=alpha,
                          ratio=n_treatment / n_control, alternative="two-sided")
mde_standardized = power_model.solve_power(effect_size=None, nobs1=n_control,
                                           alpha=alpha, power=target_power,
                                           ratio=n_treatment / n_control,
                                           alternative="two-sided")
mde = mde_standardized * planning_sd
print(f"Planning SD: ₹{planning_sd:.2f}; meaningful difference: ₹{meaningful_difference:.2f}")
print(f"Approximate power for ₹{meaningful_difference:.0f} difference: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: ₹{mde:.2f}")
print("Context: this is an approximate planning model, not exact Welch-specific power.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: the population mean order values differ significantly.")
else:
    print("Fail to reject H0: evidence is insufficient to establish a mean difference.")

# ============================================================================
# TEST 5 — PAIRED T-TEST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 5 — PAIRED T-TEST")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Check whether a redesigned checkout changes completion time for the same customers.")
before = np.array([12, 15, 11, 14, 10, 16, 13, 12], dtype=float)
after = np.array([10, 13, 10, 12, 9, 13, 11, 10], dtype=float)

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("Define each paired difference as after - before.")
print("H0: mean paired difference μ_d = 0 minutes.")
print("H1: mean paired difference μ_d != 0 minutes (two-sided).")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Each before value must match the same customer's after value.")
print("- Different customers' paired differences must be independent.")
print("- Paired differences should be approximately normal, especially for small samples.")
print("- The test evaluates the mean change, not whether every customer improved.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
assert before.ndim == 1 and after.ndim == 1
assert len(before) == len(after) and len(before) >= 2
assert np.isfinite(before).all() and np.isfinite(after).all()
print(f"Validated {len(before)} before-after pairs; no missing or infinite values.")
print("Confirm subject IDs match before interpreting paired results.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
differences = after - before
n = len(differences)
mean_difference = differences.mean()
sd_difference = differences.std(ddof=1)
standard_error = sd_difference / np.sqrt(n)
degrees_freedom = n - 1
t_statistic = mean_difference / standard_error
p_value = 2 * t.sf(abs(t_statistic), df=degrees_freedom)
t_critical = t.ppf(1 - alpha / 2, df=degrees_freedom)
ci_lower = mean_difference - t_critical * standard_error
ci_upper = mean_difference + t_critical * standard_error
cohens_dz = mean_difference / sd_difference
print(f"Before mean: {before.mean():.2f} minutes; after mean: {after.mean():.2f} minutes")
print(f"Mean difference (after - before): {mean_difference:.3f} minutes")
print(f"SD of differences: {sd_difference:.3f}; standard error: {standard_error:.3f}")
print(f"Degrees of freedom: {degrees_freedom}")
print(f"t-statistic: {t_statistic:.4f}; critical t: +/-{t_critical:.4f}")
print(f"P-value: {p_value:.8f}")
print(f"95% CI for mean difference: {ci_lower:.3f} to {ci_upper:.3f} minutes")
print(f"Cohen's dz: {cohens_dz:.3f}")
print("P-value context: evidence against zero average before-after change.")
print("A negative difference means checkout completion time decreased.")

# 6. POWER AND MDE
print("\n6. STATISTICAL POWER AND MDE")
planning_sd_difference = 2.0
meaningful_difference = 1.5
power_model = TTestPower()
power = power_model.power(effect_size=meaningful_difference / planning_sd_difference,
                          nobs=n, alpha=alpha, alternative="two-sided")
mde_standardized = power_model.solve_power(effect_size=None, nobs=n,
                                           alpha=alpha, power=target_power,
                                           alternative="two-sided")
mde = mde_standardized * planning_sd_difference
print(f"Planning SD of paired differences: {planning_sd_difference:.2f} minutes")
print(f"Meaningful difference: {meaningful_difference:.2f} minutes")
print(f"Approximate power for that difference: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: {mde:.3f} minutes")
print("Context: power depends on sample size and variability of paired differences.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: the average before-after completion time changed significantly.")
else:
    print("Fail to reject H0: evidence is insufficient to establish an average change.")

print("\nEND OF FIVE-TEST PLAYBOOK")
