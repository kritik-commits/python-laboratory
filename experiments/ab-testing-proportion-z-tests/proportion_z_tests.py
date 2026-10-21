#!/usr/bin/env python
# coding: utf-8

"""
Proportion Z-tests for conversion-rate analysis.

Part 1: One-proportion, right-tailed Z-test against a benchmark.
Part 2: Two-proportion, two-sided Z-test comparing independent groups.

Requirements: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# PART 1 — ONE-PROPORTION Z-TEST (RIGHT-TAILED)
# ============================================================
# Business question:
# Is the observed conversion rate greater than the 5% benchmark?
#
# H0: p = 0.05
# H1: p > 0.05

benchmark_conversions = 50
conversions = 65
population = 1000

alpha = 0.05
confidence_level = 0.95

observed_conversion_rate = conversions / population
benchmark_rate = benchmark_conversions / population

absolute_lift = observed_conversion_rate - benchmark_rate
relative_lift = absolute_lift / benchmark_rate

# Under H0, the standard error is calculated using the benchmark rate.
standard_error_under_null = np.sqrt(
    benchmark_rate * (1 - benchmark_rate) / population
)
z_statistic = absolute_lift / standard_error_under_null

# Right-tailed p-value: do not take abs(z) for a directional test.
p_value = norm.sf(z_statistic)
z_critical_test = norm.ppf(1 - alpha)

# Two-sided Wald confidence interval for the observed rate.
# This interval is not the matching interval for the one-sided test.
standard_error_ci = np.sqrt(
    observed_conversion_rate * (1 - observed_conversion_rate) / population
)
z_critical_ci = norm.ppf(1 - (1 - confidence_level) / 2)
margin_of_error = z_critical_ci * standard_error_ci
lower_ci_limit = observed_conversion_rate - margin_of_error
upper_ci_limit = observed_conversion_rate + margin_of_error

print("=" * 58)
print("ONE-PROPORTION Z-TEST — RIGHT-TAILED")
print("=" * 58)
print("H0: Population conversion rate equals the 5% benchmark.")
print("H1: Population conversion rate is greater than 5%.")
print(f"Population size:              {population:,}")
print(f"Conversions:                  {conversions:,}")
print(f"Benchmark conversion rate:    {benchmark_rate:.2%}")
print(f"Observed conversion rate:     {observed_conversion_rate:.2%}")
print(f"Absolute lift:                {absolute_lift * 100:.2f} percentage points")
print(f"Relative lift:                {relative_lift:.2%}")
print(f"Null standard error:          {standard_error_under_null:.6f}")
print(f"Z-statistic:                  {z_statistic:.4f}")
print(f"One-sided p-value:            {p_value:.5f}")
print(f"One-sided critical Z:         {z_critical_test:.4f}")
print(f"Significance level:           {alpha:.2%}")
print(
    f"Two-sided {confidence_level:.0%} CI for observed rate: "
    f"{lower_ci_limit:.2%} to {upper_ci_limit:.2%}"
)

if p_value < alpha:
    print("Decision: REJECT H0.")
    print("Evidence supports a conversion rate above the benchmark.")
else:
    print("Decision: FAIL TO REJECT H0.")
    print("There is insufficient evidence that conversion exceeds the benchmark.")

# Visualise the right-tailed null distribution.
z_values = np.linspace(-4, 4, 1000)
density = norm.pdf(z_values)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(z_values, density, label="Null distribution")
ax.fill_between(
    z_values,
    density,
    where=(z_values >= z_critical_test),
    alpha=0.3,
    label=f"Right rejection region (Z ≥ {z_critical_test:.2f})",
)
ax.axvline(
    z_critical_test,
    linestyle="--",
    label=f"Critical Z = {z_critical_test:.2f}",
)
ax.axvline(
    z_statistic,
    linestyle="-.",
    linewidth=2,
    label=f"Observed Z = {z_statistic:.2f}",
)
ax.set_title("One-Proportion Z-Test: Right-Tailed Null Distribution")
ax.set_xlabel("Z-statistic")
ax.set_ylabel("Probability density")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ============================================================
# PART 2 — TWO-PROPORTION Z-TEST (TWO-SIDED)
# ============================================================
# Business question:
# Do the treatment and control conversion rates differ?
#
# H0: p_treatment = p_control
# H1: p_treatment != p_control

control_population = 1000
control_conversions = 50

treatment_population = 1000
treatment_conversions = 85

control_rate = control_conversions / control_population
treatment_rate = treatment_conversions / treatment_population

absolute_lift = treatment_rate - control_rate
relative_lift = absolute_lift / control_rate

# Pooled rate and standard error are used for the null-hypothesis test.
pooled_rate = (
    (control_conversions + treatment_conversions)
    / (control_population + treatment_population)
)
standard_error_under_null = np.sqrt(
    pooled_rate
    * (1 - pooled_rate)
    * (1 / control_population + 1 / treatment_population)
)
z_statistic = absolute_lift / standard_error_under_null
p_value = 2 * norm.sf(abs(z_statistic))

# Unpooled standard error for the Wald confidence interval of the lift.
standard_error_ci = np.sqrt(
    control_rate * (1 - control_rate) / control_population
    + treatment_rate * (1 - treatment_rate) / treatment_population
)
z_critical_ci = norm.ppf(1 - alpha / 2)
margin_of_error = z_critical_ci * standard_error_ci
lower_ci_limit = absolute_lift - margin_of_error
upper_ci_limit = absolute_lift + margin_of_error

print("\n" + "=" * 58)
print("TWO-PROPORTION Z-TEST — TWO-SIDED")
print("=" * 58)
print("H0: Control and treatment population conversion rates are equal.")
print("H1: Control and treatment population conversion rates differ.")
print(f"Control conversions / population:   {control_conversions:,} / {control_population:,}")
print(f"Treatment conversions / population: {treatment_conversions:,} / {treatment_population:,}")
print(f"Control conversion rate:            {control_rate:.2%}")
print(f"Treatment conversion rate:          {treatment_rate:.2%}")
print(f"Absolute lift:                      {absolute_lift * 100:.2f} percentage points")
print(f"Relative lift:                      {relative_lift:.2%}")
print(f"Pooled conversion rate:             {pooled_rate:.4%}")
print(f"Null standard error:                {standard_error_under_null:.6f}")
print(f"Z-statistic:                        {z_statistic:.4f}")
print(f"Two-sided p-value:                  {p_value:.5f}")
print(f"Significance level:                 {alpha:.2%}")
print(
    f"95% CI for treatment − control lift: "
    f"[{lower_ci_limit * 100:.2f}, {upper_ci_limit * 100:.2f}] percentage points"
)

if p_value < alpha:
    direction = "higher" if absolute_lift > 0 else "lower"
    print("Decision: REJECT H0.")
    print(f"The treatment conversion rate is statistically significantly {direction}.")
else:
    print("Decision: FAIL TO REJECT H0.")
    print("There is insufficient evidence of a conversion-rate difference.")

if lower_ci_limit <= 0 <= upper_ci_limit:
    print("The 95% confidence interval includes zero lift.")
else:
    print("The 95% confidence interval excludes zero lift.")

# Visualise the two-sided null distribution.
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
z_critical_test = norm.ppf(1 - alpha / 2)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(x, y, label="Null distribution")
ax.fill_between(
    x, y, where=(x <= -z_critical_test), alpha=0.3,
    label="Left rejection region",
)
ax.fill_between(
    x, y, where=(x >= z_critical_test), alpha=0.3,
    label="Right rejection region",
)
ax.axvline(
    -z_critical_test, linestyle="--",
    label=f"Critical Z = ±{z_critical_test:.2f}",
)
ax.axvline(z_critical_test, linestyle="--")
ax.axvline(
    z_statistic, linestyle="-.", linewidth=2,
    label=f"Observed Z = {z_statistic:.2f}",
)
ax.set_title("Two-Proportion Z-Test: Two-Sided Rejection Regions")
ax.set_xlabel("Z-statistic")
ax.set_ylabel("Probability density")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Visualise the confidence interval for conversion-rate lift.
point_pp = absolute_lift * 100
lower_pp = lower_ci_limit * 100
upper_pp = upper_ci_limit * 100

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.errorbar(
    point_pp,
    0,
    xerr=[[point_pp - lower_pp], [upper_pp - point_pp]],
    fmt="o",
    capsize=7,
    markersize=8,
    label="Observed lift with 95% CI",
)
ax.axvline(0, linestyle="--", label="Zero difference")
ax.set_yticks([])
ax.set_xlabel("Treatment − control (percentage points)")
ax.set_title("95% Confidence Interval for Conversion Lift")
ax.legend()
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
