"""
PROJECT: A/B Testing — Landing Page Conversion Experiment

PROBLEM STATEMENT
-----------------
An e-commerce company wants to evaluate a redesigned landing page.

The existing landing page is the controlled group.
The redesigned landing page is the treatment group.

The experiment collected the following data:
- Controlled visitors: 10,000
- Controlled conversions: 500
- Treatment visitors: 10,000
- Treatment conversions: 600

OBJECTIVES
----------
1. Calculate conversion rates for both groups.
2. Calculate absolute and relative conversion lift.
3. Calculate the pooled conversion rate and standard error.
4. Conduct a two-sided Z-test for two proportions.
5. Calculate the p-value and test statistical significance.
6. Calculate the 95% confidence interval for conversion lift.
7. Translate statistical results into business insights.

HYPOTHESES
----------
H0: Treatment and control conversion rates are equal.
H1: Treatment and control conversion rates are different.

SIGNIFICANCE LEVEL
------------------
Alpha = 0.05

DECISION RULE
-------------
If p-value < alpha, reject the null hypothesis.
Otherwise, fail to reject the null hypothesis.

BUSINESS OBJECTIVE
------------------
Determine whether the redesigned landing page improves
conversion rates sufficiently to justify further evaluation
and potential rollout.

ASSUMPTIONS
-----------
- Visitors are randomly assigned to the two groups.
- Each visitor contributes one independent binary outcome.
- The groups are independent.
- The experiment is evaluated using a two-sided test.

TOOLS
-----
Python, NumPy, SciPy

IMPORTANT
---------
Statistical significance does not guarantee profitability.
Business rollout decisions should also consider revenue,
costs, customer quality, and experiment validity.
"""

import numpy as np
from scipy.stats import norm

# --------------------------------------------------
# 1. EXPERIMENT DATA
# --------------------------------------------------

# Controlled group: existing landing page
controlled_visitors = 10_000
controlled_conversions = 500

# Treatment group: redesigned landing page
treatment_visitors = 10_000
treatment_conversions = 600

# Statistical testing parameters
null_difference = 0
significance_level = 0.05
confidence_level = 0.95

# --------------------------------------------------
# 2. CONVERSION RATES
# --------------------------------------------------

controlled_conv_rate = (
    controlled_conversions / controlled_visitors
)

treatment_conv_rate = (
    treatment_conversions / treatment_visitors
)

print("CONTROLLED CONVERSION RATE:",
      f"{controlled_conv_rate:.2%}")

print("TREATMENT CONVERSION RATE:",
      f"{treatment_conv_rate:.2%}")

# --------------------------------------------------
# 3. ABSOLUTE AND RELATIVE CONVERSION LIFT
# --------------------------------------------------

# Absolute lift measures the difference in proportions.
absolute_conv_lift = (
    treatment_conv_rate - controlled_conv_rate
)

# Relative lift measures improvement against control.
relative_conv_lift = (
    absolute_conv_lift / controlled_conv_rate
)

additional_conversions = (
    treatment_conversions - controlled_conversions
)

print("\nABSOLUTE LIFT:",
      f"{absolute_conv_lift * 100:.2f} percentage points")

print("RELATIVE LIFT:",
      f"{relative_conv_lift:.2%}")

print("ADDITIONAL OBSERVED CONVERSIONS:",
      additional_conversions)

# --------------------------------------------------
# 4. POOLED CONVERSION RATE
# --------------------------------------------------

# Under H0, both groups share a common conversion rate.
pooled_conversion_rate = (
    controlled_conversions + treatment_conversions
) / (
    controlled_visitors + treatment_visitors
)

print("\nPOOLED CONVERSION RATE:",
      f"{pooled_conversion_rate:.2%}")

# --------------------------------------------------
# 5. POOLED STANDARD ERROR
# --------------------------------------------------

# Used for the two-proportion Z-test under H0.
pooled_standard_error = np.sqrt(
    pooled_conversion_rate
    * (1 - pooled_conversion_rate)
    * (
        1 / controlled_visitors
        + 1 / treatment_visitors
    )
)

print("POOLED STANDARD ERROR:",
      f"{pooled_standard_error:.6f}")

# --------------------------------------------------
# 6. Z-STATISTIC AND P-VALUE
# --------------------------------------------------

z_statistic = (
    absolute_conv_lift - null_difference
) / pooled_standard_error

# Two-sided p-value: account for both tails.
p_value = 2 * norm.sf(abs(z_statistic))

print("\nZ-STATISTIC:", f"{z_statistic:.4f}")
print("P-VALUE:", f"{p_value:.6f}")

# Statistical decision at the chosen significance level.
if p_value < significance_level:
    print("DECISION: Reject the null hypothesis.")
    print("RESULT: Statistically significant difference.")
else:
    print("DECISION: Fail to reject the null hypothesis.")
    print("RESULT: Insufficient evidence of a difference.")

# --------------------------------------------------
# 7. 95% CONFIDENCE INTERVAL
# --------------------------------------------------

# Estimate uncertainty separately in each group.
unpooled_standard_error = np.sqrt(
    controlled_conv_rate
    * (1 - controlled_conv_rate)
    / controlled_visitors
    +
    treatment_conv_rate
    * (1 - treatment_conv_rate)
    / treatment_visitors
)

alpha = 1 - confidence_level

critical_z = norm.ppf(1 - alpha / 2)

ci_lower_lift = (
    absolute_conv_lift
    - critical_z * unpooled_standard_error
)

ci_upper_lift = (
    absolute_conv_lift
    + critical_z * unpooled_standard_error
)

print("\n95% CONFIDENCE INTERVAL")
print(
    f"Lower bound: {ci_lower_lift * 100:.2f} percentage points"
)
print(
    f"Upper bound: {ci_upper_lift * 100:.2f} percentage points"
)

# --------------------------------------------------
# 8. BUSINESS INTERPRETATION
# --------------------------------------------------

print("\nBUSINESS INTERPRETATION")
print("-----------------------")

print(
    f"Observed relative conversion improvement: "
    f"{relative_conv_lift:.2%}"
)

print(
    f"Observed additional conversions: "
    f"{additional_conversions}"
)

if ci_lower_lift > 0:
    print(
        "The confidence interval indicates a positive "
        "conversion effect at the 95% confidence level."
    )
elif ci_upper_lift < 0:
    print(
        "The confidence interval indicates a negative "
        "conversion effect at the 95% confidence level."
    )
else:
    print(
        "The confidence interval includes zero. "
        "A zero conversion effect remains plausible."
    )

print(
    "Final decision should also consider revenue, costs, "
    "customer quality, and experiment validity."
)
