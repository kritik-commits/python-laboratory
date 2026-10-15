"""
PROJECT: A/B Testing — Average Order Value Experiment

OBJECTIVE
---------
Determine whether a redesigned pricing page changes average
order value using Welch's two-sample t-test.

CONTROL GROUP
-------------
Orders: 100
Mean AOV: ₹800
Standard deviation: ₹200

TREATMENT GROUP
---------------
Orders: 100
Mean AOV: ₹850
Standard deviation: ₹220

SIGNIFICANCE LEVEL: 0.05
CONFIDENCE LEVEL: 95%
"""

import numpy as np
from scipy.stats import t


# 1. Experiment data
controlled_orders = 100
controlled_aov = 800
controlled_std = 200

treatment_orders = 100
treatment_aov = 850
treatment_std = 220

alpha = 0.05


# 2. Calculate the observed difference in means
observed_aov_lift = treatment_aov - controlled_aov


# 3. Calculate the standard error
controlled_variance = controlled_std**2 / controlled_orders
treatment_variance = treatment_std**2 / treatment_orders

standard_error = np.sqrt(
    controlled_variance + treatment_variance
)


# 4. Calculate Welch's degrees of freedom
degrees_of_freedom = (
    (controlled_variance + treatment_variance)**2
    / (
        controlled_variance**2 / (controlled_orders - 1)
        + treatment_variance**2 / (treatment_orders - 1)
    )
)


# 5. Calculate the t-statistic and two-sided p-value
t_statistic = observed_aov_lift / standard_error

p_value = 2 * t.sf(
    abs(t_statistic),
    df=degrees_of_freedom
)


# 6. Calculate the 95% confidence interval
critical_t = t.ppf(
    1 - alpha / 2,
    df=degrees_of_freedom
)

confidence_interval_lower = (
    observed_aov_lift - critical_t * standard_error
)

confidence_interval_upper = (
    observed_aov_lift + critical_t * standard_error
)


# 7. Display the results
print("A/B TESTING RESULTS — AVERAGE ORDER VALUE")
print("-" * 42)

print(f"Controlled AOV: ₹{controlled_aov:.2f}")
print(f"Treatment AOV: ₹{treatment_aov:.2f}")
print(f"Observed AOV Lift: ₹{observed_aov_lift:.2f}")
print(f"Standard Error: ₹{standard_error:.4f}")
print(f"Degrees of Freedom: {degrees_of_freedom:.2f}")
print(f"T-Statistic: {t_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

print(
    "95% Confidence Interval: "
    f"₹{confidence_interval_lower:.2f} to "
    f"₹{confidence_interval_upper:.2f}"
)

print("-" * 42)

if p_value < alpha:
    print("Decision: Reject the null hypothesis.")
    print("Result: Statistically significant difference.")
else:
    print("Decision: Fail to reject the null hypothesis.")
    print("Result: No statistically significant difference.")

print("-" * 42)
