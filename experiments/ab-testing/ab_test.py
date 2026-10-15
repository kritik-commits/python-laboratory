from math import sqrt, erf


def normal_cdf(z: float) -> float:
    """Calculate the cumulative probability of the standard normal distribution."""
    return 0.5 * (1 + erf(z / sqrt(2)))


def two_proportion_z_test(
    control_conversions: int,
    control_visitors: int,
    variant_conversions: int,
    variant_visitors: int,
) -> dict:
    """Compare two conversion rates using a two-sided pooled z-test."""

    if control_visitors <= 0 or variant_visitors <= 0:
        raise ValueError("Visitor counts must be positive.")

    if not 0 <= control_conversions <= control_visitors:
        raise ValueError("Invalid control conversion count.")

    if not 0 <= variant_conversions <= variant_visitors:
        raise ValueError("Invalid variant conversion count.")

    p_control = control_conversions / control_visitors
    p_variant = variant_conversions / variant_visitors

    pooled_rate = (
        control_conversions + variant_conversions
    ) / (control_visitors + variant_visitors)

    standard_error = sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * (1 / control_visitors + 1 / variant_visitors)
    )

    if standard_error == 0:
        raise ValueError("Cannot calculate the test with zero standard error.")

    z_score = (p_variant - p_control) / standard_error
    p_value = 2 * (1 - normal_cdf(abs(z_score)))

    return {
        "control_rate": p_control,
        "variant_rate": p_variant,
        "absolute_lift": p_variant - p_control,
        "z_score": z_score,
        "p_value": p_value,
    }


if __name__ == "__main__":
    results = two_proportion_z_test(120, 1000, 145, 1000)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")