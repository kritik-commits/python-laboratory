#!/usr/bin/env python3
# coding: utf-8
"""Five statistical tests in the sequential, inline style of Untitled.py.
All datasets and planning assumptions are illustrative, not company records.
Dependencies: numpy, pandas, scipy, statsmodels.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2, chi2_contingency, chisquare, fisher_exact, f_oneway, f
from statsmodels.stats.power import TTestPower, TTestIndPower, NormalIndPower, GofChisquarePower, FTestAnovaPower
from statsmodels.stats.proportion import proportion_effectsize

alpha = 0.05
confidence_level = 0.95
target_power = 0.80

# Shared one-line definitions printed inline, without helper functions.
print("STATISTICAL TERMS — ONE-LINE DEFINITIONS")

print("Null hypothesis (H0): the default claim tested using sample data.")
print("Alternative hypothesis (H1): the effect or difference being investigated.")
print("Type I error - false positive: rejecting a true null hypothesis.")
print("Type II error - false negative: failing to reject a false null hypothesis.")
print("Alpha: the maximum acceptable Type I error rate, commonly 5%.")
print("Beta: the probability of making a Type II error.")
print("Mean: the arithmetic average of observed values.")
print("Median: the middle value when observations are arranged in order.")
print("Mode: the most frequently occurring value in a dataset.")
print("Range: the difference between the maximum and minimum values.")
print("IQR: the difference between the 75th and 25th percentiles.")
print("Variance: the average squared deviation from the mean; sample variance uses (n-1) in the denominator.")
print("Standard deviation: the typical spread around the mean, measured in the original data units.")
print("Standard error: measures sampling uncertainty, often calculated by dividing standard deviation by the square root of sample size.")
print("Central Limit Theorem (CLT): the sample mean's distribution approaches normality as sample size increases, under suitable conditions.")
print("Test statistic: a value summarizing the sample evidence against the null hypothesis.")
print("Degrees of freedom: determines the t-distribution's shape; the standard normal (z) distribution requires none.")
print("P-value: assuming H0 is true, the probability of getting a result this extreme or more extreme.")
print("Critical value: the test-statistic cutoff that defines the rejection region.")
print("Confidence interval: a range of plausible values for a population parameter.")
print("95% confidence interval: across repeated samples, approximately 95% of calculated intervals contain the true population parameter.")
print("Effect size: measures the magnitude of an observed or planned difference.")
print("Power (1 - beta): the probability of rejecting H0 when a specified alternative is true.")
print("MDE: the smallest effect detectable at a chosen significance level and target power.")
print("Fail to reject H0: does not prove that H0 is true.")
print("Statistical significance: evidence against H0 strong enough to cross the chosen significance threshold.")
print("Practical significance: whether an effect is large enough to matter in practice.")
print("Probability: the likelihood of an event occurring.")
print("Probability Mass Function (PMF): gives probabilities for possible values of a discrete random variable.")
print("Probability Density Function (PDF): describes a continuous distribution; probabilities come from areas over intervals.")
print("Cumulative Distribution Function (CDF): gives the probability that a random variable is at most a specified value.")

print("\\nSTATISTICAL TESTS — ONE-LINE DEFINITIONS")

print("One-sided test: tests for an effect in one specified direction.")
print("Two-sided test: tests for an effect in either direction.")
print("One-proportion z-test: tests whether a population proportion differs from a specified value.")
print("Two-proportion z-test: tests whether two independent population proportions differ.")
print("One-sample t-test: tests whether a population mean differs from a specified value.")
print("Independent two-sample t-test: compares the means of two independent groups.")
print("Welch's t-test: compares two independent means without assuming equal population variances.")
print("Paired t-test: tests the mean difference between matched or repeated measurements.")
print("Chi-square test of independence: tests whether two categorical variables are associated.")
print("Chi-square goodness-of-fit test: tests whether observed category counts match expected proportions.")
print("Fisher's exact test: tests the association between two categorical variables, often with small sample counts.")
print("One-way ANOVA: tests whether three or more independent group means differ.")
print("Z-test: tests a mean or proportion using a standard normal reference distribution when its assumptions are satisfied.")
print("T-test: tests mean differences using the t-distribution when population variability is estimated from sample data.")

print("\\nEXPERIMENT DESIGN — ONE-LINE DEFINITIONS")

print("Sample size: the number of observations included in a study.")
print("Significance level (alpha): the chosen threshold for rejecting the null hypothesis.")
print("Statistical power: the chance of detecting a specified effect when it truly exists.")
print("Minimum detectable effect (MDE): the smallest target effect detectable at the chosen alpha and power.")
print("Confidence level: the long-run percentage of intervals expected to contain the true parameter.")
print("Assumption: a condition that should hold for a statistical method to be reliable.")
print("Independent observations: observations that do not depend on one another.")
print("Contingency table: a table showing counts for combinations of categorical variables.")
print("Expected count: the count predicted under the null hypothesis.")
print("Post-hoc test: a follow-up test identifying which groups differ after a significant overall test.")
print("Cramer's V: measures the strength of association between categorical variables.")
print("Odds ratio: compares the odds of an outcome between two groups.")
print("Cohen's d: measures the difference between two means in standard deviation units.")
print("Eta-squared: measures the proportion of outcome variability explained by group membership in ANOVA.")
print("Multiple testing: performing several tests, which increases the risk of false positives.")
print("Bonferroni correction: reduces the significance threshold to account for multiple tests.")



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






# ============================================================================
# TEST 6 — CHI-SQUARE TEST OF INDEPENDENCE
# ============================================================================
print("\n" + "=" * 72)
print("TEST 6 — CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Check whether device type and purchase completion are associated.")
print("Rows are device groups; columns are purchased and did not purchase.")
observed_table = np.array([[120, 880], [90, 910], [150, 850]], dtype=int)
row_labels = ["Mobile", "Desktop", "Tablet"]
column_labels = ["Purchased", "Did not purchase"]

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: device type and purchase completion are independent.")
print("H1: device type and purchase completion are associated.")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Each observation belongs to exactly one cell in the contingency table.")
print("- Observations are independent; repeated users require a different design.")
print("- Expected cell counts should generally be at least 5.")
print("- Association does not prove that device type causes purchasing behavior.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
assert observed_table.ndim == 2 and observed_table.shape == (3, 2)
assert np.issubdtype(observed_table.dtype, np.integer)
assert (observed_table >= 0).all() and observed_table.sum() > 0
print(f"Validated contingency table: {observed_table.sum()} observations across 6 cells.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL / EFFECT SIZE
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
chi2_statistic, p_value, degrees_freedom, expected_table = chi2_contingency(
    observed_table, correction=False
)
critical_value = chi2.ppf(1 - alpha, df=degrees_freedom)
minimum_expected = expected_table.min()
cramers_v = np.sqrt(chi2_statistic / (observed_table.sum() * min(observed_table.shape[0] - 1,
                                                                  observed_table.shape[1] - 1)))
print("Observed counts:")
print(pd.DataFrame(observed_table, index=row_labels, columns=column_labels).to_string())
print("Expected counts under H0:")
print(pd.DataFrame(expected_table, index=row_labels, columns=column_labels).round(2).to_string())
print(f"Chi-square statistic: {chi2_statistic:.4f}; df: {degrees_freedom}")
print(f"Critical chi-square: {critical_value:.4f}; p-value: {p_value:.8f}")
print(f"Minimum expected cell count: {minimum_expected:.2f}")
print(f"Cramer's V effect size: {cramers_v:.4f}")
print("A single mean-difference confidence interval is not defined for this association test.")
print("Use cell proportions and their intervals to quantify specific group differences.")

# 6. POWER AND MDE — chi-square approximation
print("\n6. STATISTICAL POWER AND MDE")
planning_effect_size = 0.10  # Cohen's w, illustrative association effect
power_model = GofChisquarePower()
power = power_model.power(effect_size=planning_effect_size, nobs=observed_table.sum(),
                          alpha=alpha, n_bins=observed_table.size - 1)
mde_effect_size = power_model.solve_power(effect_size=None, nobs=observed_table.sum(),
                                          alpha=alpha, power=target_power,
                                          n_bins=observed_table.size - 1)
print(f"Planning effect size (Cohen's w): {planning_effect_size:.2f}")
print(f"Approximate power for planned effect: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: Cohen's w = {mde_effect_size:.3f}")
print("Planning power is approximate; contingency-table geometry affects the result.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if minimum_expected < 5:
    print("Warning: some expected counts are below 5; consider an exact or simulation-based method.")
if p_value < alpha:
    print("Reject H0: evidence indicates an association between device and purchase completion.")
else:
    print("Fail to reject H0: evidence is insufficient to establish an association.")






# ============================================================================
# TEST 7 — CHI-SQUARE GOODNESS-OF-FIT TEST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 7 — CHI-SQUARE GOODNESS-OF-FIT TEST")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Check whether four equally promoted campaign variants receive equal clicks.")
observed_counts = np.array([120, 95, 105, 80], dtype=int)
expected_probabilities = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: category probabilities match the prespecified expected proportions.")
print("H1: at least one category probability differs from its expected proportion.")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Categories are mutually exclusive and cover all observations.")
print("- Observations are independent and expected counts are sufficiently large.")
print("- Expected probabilities must be specified independently of observed counts.")
print("- The omnibus test does not identify which category differs by itself.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
assert observed_counts.ndim == 1 and len(observed_counts) >= 2
assert np.issubdtype(observed_counts.dtype, np.integer)
assert (observed_counts >= 0).all() and observed_counts.sum() > 0
assert len(expected_probabilities) == len(observed_counts)
assert (expected_probabilities > 0).all()
assert np.isclose(expected_probabilities.sum(), 1.0)
expected_counts = observed_counts.sum() * expected_probabilities
assert (expected_counts >= 5).all(), "Expected counts below 5; consider an exact/Monte Carlo approach."
print(f"Validated {observed_counts.sum()} observations across {len(observed_counts)} categories.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL / EFFECT SIZE
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
chi2_statistic, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
degrees_freedom = len(observed_counts) - 1
critical_value = chi2.ppf(1 - alpha, df=degrees_freedom)
cohens_w = np.sqrt(chi2_statistic / observed_counts.sum())
print(f"Observed counts: {observed_counts.tolist()}")
print(f"Expected counts: {expected_counts.round(2).tolist()}")
print(f"Chi-square statistic: {chi2_statistic:.4f}; df: {degrees_freedom}")
print(f"Critical chi-square: {critical_value:.4f}; p-value: {p_value:.8f}")
print(f"Cohen's w effect size: {cohens_w:.4f}")
print("A single mean-difference confidence interval is not defined for this omnibus test.")
print("Inspect category proportions and intervals if you need category-level estimates.")

# 6. POWER AND MDE — chi-square approximation
print("\n6. STATISTICAL POWER AND MDE")
power_model = GofChisquarePower()
planning_effect_size = 0.20
power = power_model.power(effect_size=planning_effect_size, nobs=observed_counts.sum(),
                          alpha=alpha, n_bins=len(observed_counts))
mde_effect_size = power_model.solve_power(effect_size=None, nobs=observed_counts.sum(),
                                          alpha=alpha, power=target_power,
                                          n_bins=len(observed_counts))
print(f"Planning effect size (Cohen's w): {planning_effect_size:.2f}")
print(f"Approximate power for planned effect: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: Cohen's w = {mde_effect_size:.3f}")
print("Power estimates assume the specified category count and chi-square approximation.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: observed category frequencies differ from the expected distribution.")
else:
    print("Fail to reject H0: evidence is insufficient to establish a distribution mismatch.")






# ============================================================================
# TEST 8 — FISHER'S EXACT TEST
# ============================================================================
print("\n" + "=" * 72)
print("TEST 8 — FISHER'S EXACT TEST")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Compare purchase odds for two landing pages with small samples.")
fisher_table = np.array([[8, 2], [1, 5]], dtype=int)

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: the two categorical variables are independent; odds ratio = 1.")
print("H1: the variables are associated; odds ratio differs from 1 (two-sided).")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Use a 2x2 table of independent observations and mutually exclusive outcomes.")
print("- Fisher's exact test is useful when expected counts are small.")
print("- The test conditions on observed row and column margins.")
print("- Exact p-values can be conservative; small samples may have low power.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
assert fisher_table.shape == (2, 2)
assert np.issubdtype(fisher_table.dtype, np.integer)
assert (fisher_table >= 0).all() and fisher_table.sum() > 0
assert (fisher_table.sum(axis=0) > 0).all() and (fisher_table.sum(axis=1) > 0).all()
print(f"Validated 2x2 table with {fisher_table.sum()} observations.")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL / EFFECT SIZE
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
conditional_odds_ratio, p_value = fisher_exact(fisher_table, alternative="two-sided")
# Sample odds ratio and Wald interval are descriptive approximations, not exact intervals.
a, b, c, d = fisher_table.ravel()
if min(a, b, c, d) > 0:
    sample_odds_ratio = (a * d) / (b * c)
    log_or_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z_ci = norm.ppf(1 - (1 - confidence_level) / 2)
    or_ci_lower = np.exp(np.log(sample_odds_ratio) - z_ci * log_or_se)
    or_ci_upper = np.exp(np.log(sample_odds_ratio) + z_ci * log_or_se)
else:
    sample_odds_ratio = np.inf if (a * d > b * c) else 0.0
    or_ci_lower, or_ci_upper = np.nan, np.nan
print(f"Sample odds ratio: {sample_odds_ratio:.4f}")
print(f"Fisher exact two-sided p-value: {p_value:.8f}")
if np.isfinite(or_ci_lower):
    print(f"Approximate Wald 95% CI for odds ratio: {or_ci_lower:.3f} to {or_ci_upper:.3f}")
else:
    print("Wald odds-ratio interval unavailable with zero cells; use an exact interval method.")
print("SciPy's reported conditional odds ratio and the sample cross-product ratio can differ.")
print("The odds-ratio interval above is approximate, not a Fisher exact confidence interval.")

# 6. POWER AND MDE
print("\n6. STATISTICAL POWER AND MDE")
print("Exact Fisher power/MDE requires specified margins and an alternative odds ratio.")
print("Use design-specific exact enumeration or simulation for formal planning.")
print("A chi-square approximation may be used for rough planning, not as exact Fisher power.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: evidence indicates an association between page and purchase outcome.")
else:
    print("Fail to reject H0: evidence is insufficient to establish an association.")






# ============================================================================
# TEST 9 — ONE-WAY ANOVA
# ============================================================================
print("\n" + "=" * 72)
print("TEST 9 — ONE-WAY ANOVA")
print("=" * 72)

# 1. BUSINESS PROBLEM
print("\n1. BUSINESS PROBLEM")
print("Compare mean order values across three independent pricing strategies.")
price_a = np.array([780, 810, 790, 820, 805, 795, 815, 800], dtype=float)
price_b = np.array([830, 850, 840, 860, 845, 855, 835, 865], dtype=float)
price_c = np.array([800, 790, 810, 805, 795, 815, 785, 805], dtype=float)
groups = [price_a, price_b, price_c]
group_names = ["Strategy A", "Strategy B", "Strategy C"]

# 2. HYPOTHESES
print("\n2. HYPOTHESES")
print("H0: all population mean order values are equal.")
print("H1: at least one population mean differs.")
print(f"Alpha: {alpha:.2f}")

# 3. ASSUMPTIONS AND LIMITATIONS
print("\n3. ASSUMPTIONS AND LIMITATIONS")
print("- Outcome is continuous; groups are independent and observations are independent.")
print("- Within-group residuals should be approximately normal, especially for small samples.")
print("- Population variances should be reasonably similar for classical one-way ANOVA.")
print("- A significant omnibus result does not identify which pairs differ.")
print("- Use planned contrasts or multiplicity-adjusted post-hoc tests after an omnibus result.")

# 4. DATA VALIDATION
print("\n4. DATA VALIDATION")
assert len(groups) >= 3
assert all(group.ndim == 1 and len(group) >= 2 for group in groups)
assert all(np.isfinite(group).all() for group in groups)
print(f"Validated {len(groups)} groups and {sum(len(group) for group in groups)} observations.")
for group_name, group in zip(group_names, groups):
    print(f"{group_name}: n={len(group)}, mean=₹{group.mean():.2f}, SD=₹{group.std(ddof=1):.2f}")

# 5. TEST, P-VALUE, CONFIDENCE INTERVAL / EFFECT SIZE
print("\n5. TEST CALCULATIONS AND INTERPRETATION")
f_statistic, p_value = f_oneway(*groups)
all_values = np.concatenate(groups)
number_groups = len(groups)
total_n = len(all_values)
degrees_freedom_between = number_groups - 1
degrees_freedom_within = total_n - number_groups
grand_mean = all_values.mean()
ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
ss_total = ss_between + ss_within
ms_between = ss_between / degrees_freedom_between
ms_within = ss_within / degrees_freedom_within
critical_f = f.ppf(1 - alpha, degrees_freedom_between, degrees_freedom_within)
eta_squared = ss_between / ss_total
omega_squared = max(0.0, (ss_between - degrees_freedom_between * ms_within) /
                    (ss_total + ms_within))
print(f"Grand mean: ₹{grand_mean:.2f}")
print(f"Between-group SS: {ss_between:.3f}; within-group SS: {ss_within:.3f}")
print(f"MS between: {ms_between:.3f}; MS within: {ms_within:.3f}")
print(f"F-statistic: {f_statistic:.4f}; df=({degrees_freedom_between}, {degrees_freedom_within})")
print(f"Critical F: {critical_f:.4f}; p-value: {p_value:.8f}")
print(f"Eta-squared: {eta_squared:.4f}; omega-squared estimate: {omega_squared:.4f}")
print("ANOVA tests the omnibus mean difference; it does not provide one CI for all groups.")
print("Use adjusted pairwise confidence intervals to locate and quantify group differences.")

# 6. POWER AND MDE — Cohen's f planning approximation
print("\n6. STATISTICAL POWER AND MDE")
planning_cohens_f = 0.25
power_model = FTestAnovaPower()
power = power_model.power(effect_size=planning_cohens_f, nobs=total_n,
                          alpha=alpha, k_groups=number_groups)
mde_cohens_f = power_model.solve_power(effect_size=None, nobs=total_n,
                                       alpha=alpha, power=target_power,
                                       k_groups=number_groups)
print(f"Planning effect size (Cohen's f): {planning_cohens_f:.2f}")
print(f"Approximate power for planned effect: {power:.1%}")
print(f"Approximate MDE at {target_power:.0%} power: Cohen's f = {mde_cohens_f:.3f}")
print("Translate Cohen's f into business units using planned group means and pooled variability.")

# 7. DECISION
print("\n7. STATISTICAL DECISION")
if p_value < alpha:
    print("Reject H0: at least one pricing strategy has a different population mean.")
    print("Next: run an appropriate multiplicity-adjusted post-hoc comparison.")
else:
    print("Fail to reject H0: evidence is insufficient to establish a mean difference.")

print("\nEND OF NINE-TEST PLAYBOOK")


# ============================================================================
# ============================================================================
