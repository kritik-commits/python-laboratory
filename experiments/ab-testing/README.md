# A/B Testing with Python

**Can you prove that a redesigned landing page converts better than the original?**

In this project, you’ll explore how hypothesis testing helps businesses make evidence-based decisions. You’ll use Python to compare conversion rates, measure uncertainty, and determine whether an observed improvement is statistically significant.

## What You’ll Explore

Imagine an e-commerce company redesigns its landing page to increase purchases. The new version generates more conversions, but is the improvement genuine or simply random variation?

This project walks you through the statistical analysis needed to answer that question.

By working through this experiment, you’ll learn how to:

- Calculate conversion rates and measure absolute and relative lift.
- Formulate null and alternative hypotheses.
- Calculate a pooled conversion rate and standard error.
- Perform a two-proportion Z-test.
- Interpret Z-statistics and two-sided p-values.
- Calculate and interpret a 95% confidence interval.
- Translate statistical findings into practical business insights.

## The Business Scenario

An e-commerce company wants to evaluate a redesigned landing page before rolling it out to all visitors.

The experiment randomly assigns visitors to two groups:

- **Control Group:** Visitors see the existing landing page.
- **Variant Group:** Visitors see the redesigned landing page.

The objective is to determine whether the redesigned page produces a statistically significant difference in conversion rates.

## Meet the Experiment Data

| Metric | Control Group | Variant Group |
|:--|--:|--:|
| Visitors | 1,000 | 1,000 |
| Conversions | 120 | 145 |
| Conversion Rate | 12.0% | 14.5% |

The variant generates 25 additional observed conversions.

Its conversion rate is also 2.5 percentage points higher than the control rate.

But an observed improvement alone does not establish statistical significance. That’s where hypothesis testing comes in.

## How the Analysis Works

The experiment follows a sequence commonly used in A/B testing.

**1. Measure the observed effect**

Calculate both conversion rates, then compare them.

- Absolute lift measures the difference in percentage points.
- Relative lift measures improvement against the control rate.

**2. Establish the hypotheses**

Define the null hypothesis as equal population conversion rates. Define the alternative hypothesis as different population conversion rates.

**3. Quantify uncertainty**

Calculate the pooled conversion rate and standard error under the null hypothesis.

These values help determine whether the observed difference is large relative to expected random variation.

**4. Calculate the Z-statistic and p-value**

Use a two-proportion Z-test to evaluate the observed difference.

Compare the two-sided p-value against the predefined significance level of 0.05.

**5. Estimate the confidence interval**

Calculate a 95% confidence interval for the difference between conversion rates.

This provides an estimate of the effect's uncertainty and the range of values compatible with the confidence-interval procedure.

**6. Translate the findings into a business decision**

Statistical evidence is one part of the decision. Practical impact, implementation costs, revenue, and experiment quality matter too.

## Tools and Technologies

- **Python:** Implements the experiment analysis.
- **NumPy:** Handles numerical calculations.
- **SciPy:** Provides statistical distribution functions.
- **unittest:** Supports automated testing.

## Explore the Project

```text
ab-testing/
├── README.md
├── ab_test.py
└── test_ab_test.py
```

- `ab_test.py` contains the statistical analysis.
- `test_ab_test.py` contains unit tests.
- `README.md` explains the experiment and methodology.

## Run It Yourself

**1. Install the dependencies**

```bash
py -m pip install numpy scipy
```

**2. Run the experiment**

```bash
py ab_test.py
```

**3. Run the unit tests**

```bash
py -m unittest -v
```

## What Should You Take Away?

An observed improvement and a statistically significant improvement are not the same thing.

Hypothesis testing helps you evaluate whether the evidence supports a difference between two experiences. Confidence intervals help communicate the estimated effect and its uncertainty.

Neither guarantees profitability, and a non-significant result does not prove that two versions are equivalent.

The goal is to make better business decisions by combining statistical evidence with practical context.

---

**Topics:** A/B Testing · Hypothesis Testing · Statistical Significance · Confidence Intervals · Conversion Rate Optimization · Business Analytics
