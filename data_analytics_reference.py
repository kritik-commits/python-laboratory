"""
DATA ANALYTICS REFERENCE
Reusable examples for NumPy, pandas, descriptive statistics, outliers,
Matplotlib, Seaborn, and time-series exploration.

Setup:
    pip install numpy pandas scipy matplotlib seaborn

Place a CSV at data/retail_transactions_outlier_practice.csv, or update DATA_PATH.
The examples use the retail dataset columns from the learning exercises.
"""

from pathlib import Path
from typing import Iterable
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore


DATA_PATH = Path("data/retail_transactions_outlier_practice.csv")
REQUIRED_COLUMNS = [
    "transaction_id",
    "transaction_date",
    "customer_segment",
    "payment_method",
    "items_purchased",
    "discount_pct",
    "transaction_amount",
]

sns.set_theme(style="whitegrid", palette="deep")


# ---------------------------------------------------------------------------
# 1. DATA INGESTION AND INSPECTION
# ---------------------------------------------------------------------------

def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV and parse the transaction date when available."""
    df = pd.read_csv(file_path)
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"], errors="coerce"
        )
    return df


def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize types, missingness, and unique values for every column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(2),
            "unique_count": df.nunique(dropna=True),
        }
    ).sort_values("missing_pct", ascending=False)


def validate_schema(
    df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim text columns without modifying the original DataFrame."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    return cleaned


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rank columns by missing-value count and percentage."""
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(2),
        }
    )
    return report.sort_values("missing_count", ascending=False)


def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate rows for inspection before removal."""
    return df.loc[df.duplicated(keep=False)].sort_values(
        by=list(df.columns), na_position="last"
    )


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative, reusable cleaning steps to the retail dataset."""
    cleaned = clean_text_columns(df)

    if "transaction_date" in cleaned.columns:
        cleaned["transaction_date"] = pd.to_datetime(
            cleaned["transaction_date"], errors="coerce"
        )

    if "discount_pct" in cleaned.columns:
        cleaned["discount_pct"] = pd.to_numeric(
            cleaned["discount_pct"], errors="coerce"
        )

    if "items_purchased" in cleaned.columns:
        cleaned["items_purchased"] = pd.to_numeric(
            cleaned["items_purchased"], errors="coerce"
        )

    if "transaction_amount" in cleaned.columns:
        cleaned["transaction_amount"] = pd.to_numeric(
            cleaned["transaction_amount"], errors="coerce"
        )

    return cleaned


# ---------------------------------------------------------------------------
# 2. PANDAS: SELECT, FILTER, TRANSFORM, AND SUMMARIZE
# ---------------------------------------------------------------------------

def filter_examples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Demonstrate column selection and single- or multi-condition filtering."""
    examples = {
        "selected_columns": df.loc[
            :, [column for column in ["transaction_id", "payment_method",
                                      "transaction_amount"] if column in df]
        ]
    }

    if {"payment_method", "transaction_amount"}.issubset(df.columns):
        examples["high_value_transactions"] = df.loc[
            df["transaction_amount"] > df["transaction_amount"].quantile(0.90)
        ]

    if {"customer_segment", "transaction_amount"}.issubset(df.columns):
        examples["segment_filter"] = df.loc[
            (df["customer_segment"].notna())
            & (df["transaction_amount"].notna())
        ]

    return examples


def add_category_bins(
    df: pd.DataFrame,
    column: str,
    bins: list[float],
    labels: list[str],
    new_column: str,
) -> pd.DataFrame:
    """Create ordered categories from numeric intervals."""
    result = df.copy()
    result[new_column] = pd.cut(
        result[column], bins=bins, labels=labels, right=False
    )
    return result


def add_datetime_features(
    df: pd.DataFrame, date_column: str = "transaction_date"
) -> pd.DataFrame:
    """Extract calendar features from a datetime column."""
    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    dates = result[date_column]
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["weekday"] = dates.dt.day_name()
    result["hour"] = dates.dt.hour
    return result


def grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction performance by payment method."""
    return (
        df.groupby("payment_method", dropna=False)["transaction_amount"]
        .agg(transaction_count="count", mean="mean", median="median", total="sum")
        .round(2)
        .sort_values("total", ascending=False)
    )


def conversion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize visitor and conversion metrics when those columns exist."""
    required = {"visitor_id", "converted"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns: {sorted(required)}")

    group_columns = [
        column for column in ["device", "experiment_group"] if column in df.columns
    ]
    if not group_columns:
        group_columns = ["converted"]

    return (
        df.groupby(group_columns, dropna=False)
        .agg(
            visitors=("visitor_id", "nunique"),
            conversions=("converted", "sum"),
            conversion_rate=("converted", "mean"),
        )
        .reset_index()
    )


def make_pivot(
    df: pd.DataFrame,
    *,
    values: str,
    index: str,
    columns: str,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Reshape long data into a comparison table."""
    return pd.pivot_table(
        df, values=values, index=index, columns=columns, aggfunc=aggfunc
    )


def make_long(wide_df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """Convert wide-format measurements into tidy long-format data."""
    return wide_df.melt(
        id_vars=id_columns, var_name="measure", value_name="value"
    )


def merge_examples(
    left: pd.DataFrame, right: pd.DataFrame, key: str
) -> dict[str, pd.DataFrame]:
    """Compare common join types and validate expected key relationships."""
    return {
        "inner": pd.merge(left, right, on=key, how="inner"),
        "left": pd.merge(left, right, on=key, how="left"),
        "right": pd.merge(left, right, on=key, how="right"),
        "outer": pd.merge(left, right, on=key, how="outer"),
    }


# ---------------------------------------------------------------------------
# 3. NUMPY: ARRAYS, VECTORIZATION, AND RANDOM SAMPLING
# ---------------------------------------------------------------------------

def numpy_summary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Calculate core numerical summaries while ignoring missing values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        raise ValueError("At least one finite numeric value is required.")

    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_population": float(np.std(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def revenue_scenarios(
    revenue: pd.Series | np.ndarray,
    adjustment_rates: Iterable[float] = (0.05, 0.10, 0.15),
) -> np.ndarray:
    """Use broadcasting to model multiple revenue-adjustment scenarios."""
    values = np.asarray(revenue, dtype=float)
    rates = np.asarray(list(adjustment_rates), dtype=float)
    return values[:, np.newaxis] * (1 + rates)


def reproducible_sample(
    values: Iterable, size: int, seed: int = 42, replace: bool = False
) -> np.ndarray:
    """Sample reproducibly using NumPy's modern random generator."""
    array = np.asarray(list(values))
    if size > len(array) and not replace:
        raise ValueError("Sample size exceeds population when replace=False.")
    return np.random.default_rng(seed).choice(array, size=size, replace=replace)


# ---------------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS AND OUTLIER DETECTION
# ---------------------------------------------------------------------------

def distribution_summary(values: pd.Series) -> pd.Series:
    """Review spread, percentiles, skewness, and kurtosis."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    summary = numeric.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    summary.loc["skewness"] = numeric.skew()
    summary.loc["kurtosis"] = numeric.kurtosis()
    summary.loc["iqr"] = numeric.quantile(0.75) - numeric.quantile(0.25)
    return summary


def outlier_flags(values: pd.Series) -> pd.DataFrame:
    """Compare IQR, standard Z-score, and modified Z-score flags."""
    numeric = pd.to_numeric(values, errors="coerce")
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_flag = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)

    standard_z = pd.Series(
        zscore(numeric, nan_policy="omit"), index=numeric.index, dtype=float
    )
    z_flag = standard_z.abs() > 3

    median = numeric.median()
    mad = (numeric - median).abs().median()
    if pd.notna(mad) and mad > 0:
        modified_z = 0.6745 * (numeric - median) / mad
        modified_flag = modified_z.abs() > 3.5
    else:
        modified_z = pd.Series(np.nan, index=numeric.index)
        modified_flag = pd.Series(False, index=numeric.index)

    return pd.DataFrame(
        {
            "value": numeric,
            "iqr_flag": iqr_flag.fillna(False),
            "z_score": standard_z,
            "z_flag": z_flag.fillna(False),
            "modified_z_score": modified_z,
            "modified_z_flag": modified_flag.fillna(False),
        }
    )


def correlation_matrix(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Calculate pairwise Pearson correlations for selected numeric columns."""
    return df[columns].corr(numeric_only=True)


# ---------------------------------------------------------------------------
# 5. MATPLOTLIB: REUSABLE PLOT PATTERNS
# ---------------------------------------------------------------------------

def plot_histogram(
    df: pd.DataFrame, column: str, bins: int = 25
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a distribution with a labelled axis and consistent layout."""
    values = df[column].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set(title=f"Distribution of {column}", xlabel=column, ylabel="Frequency")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_boxplot(
    df: pd.DataFrame, column: str
) -> tuple[plt.Figure, plt.Axes]:
    """Compare spread, median, mean, and potential outliers."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(
        df[column].dropna(),
        vert=False,
        showmeans=True,
        showfliers=True,
        widths=0.5,
    )
    ax.set(title=f"Spread and Outliers: {column}", xlabel=column)
    fig.tight_layout()
    return fig, ax


def plot_scatter(
    df: pd.DataFrame, x: str, y: str
) -> tuple[plt.Figure, plt.Axes]:
    """Inspect the relationship between two numeric variables."""
    data = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(data[x], data[y], color="teal", alpha=0.65)
    ax.set(title=f"{x} vs {y}", xlabel=x, ylabel=y)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_category_totals(
    df: pd.DataFrame, category: str, value: str
) -> tuple[plt.Figure, plt.Axes]:
    """Compare totals across categories with a horizontal bar chart."""
    summary = (
        df.groupby(category, dropna=False)[value]
        .sum(min_count=1)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(summary.index.astype(str), summary.values, color="darkorange")
    ax.bar_label(bars, fmt="%.0f", padding=4)
    ax.set(title=f"Total {value} by {category}", xlabel=f"Total {value}")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. SEABORN: DISTRIBUTIONS AND FACETED COMPARISONS
# ---------------------------------------------------------------------------

def plot_seaborn_distribution(
    df: pd.DataFrame, column: str
) -> tuple[plt.Figure, plt.Axes]:
    """Combine a histogram and kernel density estimate."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df, x=column, bins="auto", stat="density",
        kde=True, color="steelblue", alpha=0.55, ax=ax
    )
    ax.set(title=f"Distribution of {column}", xlabel=column, ylabel="Density")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_distribution_by_category(
    df: pd.DataFrame,
    column: str,
    category: str,
    columns: int = 2,
):
    """Use displot to create an automatically managed facet grid."""
    grid = sns.displot(
        data=df,
        x=column,
        col=category,
        col_wrap=columns,
        kind="hist",
        bins="auto",
        kde=True,
        height=4,
        aspect=1.25,
        facet_kws={"sharex": True, "sharey": True},
    )
    grid.set_axis_labels(column, "Count")
    grid.set_titles("{col_name}")
    grid.figure.tight_layout()
    return grid


def plot_correlation_heatmap(
    df: pd.DataFrame, columns: list[str]
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize correlations among selected numeric features."""
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="YlGnBu",
        vmin=-1, vmax=1, center=0, square=True, ax=ax
    )
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7. TIME SERIES: RESAMPLING, ROLLING WINDOWS, AND CUMULATIVE METRICS
# ---------------------------------------------------------------------------

def daily_metric(
    df: pd.DataFrame,
    date_column: str = "transaction_date",
    value_column: str = "transaction_amount",
) -> pd.DataFrame:
    """Aggregate a numeric metric by day and calculate trend features."""
    data = df[[date_column, value_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna(subset=[date_column])

    daily = (
        data.groupby(date_column)[value_column]
        .sum(min_count=1)
        .sort_index()
        .asfreq("D")
    )
    daily = daily.fillna(0).rename("daily_value").to_frame()
    daily["cumulative_total"] = daily["daily_value"].cumsum()
    daily["cumulative_average"] = daily["daily_value"].expanding().mean()
    daily["rolling_mean_7d"] = daily["daily_value"].rolling(7, min_periods=7).mean()
    daily["rolling_std_7d"] = daily["daily_value"].rolling(7, min_periods=7).std()
    return daily


def plot_time_series(daily: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    """Plot daily values and rolling volatility in separate panels."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(
        daily.index, daily["daily_value"], color="gray",
        linewidth=1.5, label="Daily value"
    )
    axes[0].plot(
        daily.index, daily["rolling_mean_7d"], color="navy",
        linewidth=2, label="7-day rolling mean"
    )
    axes[0].set(title="Daily Metric and Rolling Mean", ylabel="Value")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        daily.index, daily["rolling_std_7d"], color="darkorange",
        linewidth=2, label="7-day rolling standard deviation"
    )
    axes[1].set(title="Rolling Volatility", xlabel="Date", ylabel="Standard deviation")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig, axes


def plot_weekday_heatmap(df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Compare daily totals across weekdays and calendar weeks."""
    data = df[["transaction_date", "transaction_amount"]].copy()
    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"], errors="coerce"
    )
    data = data.dropna(subset=["transaction_date"])

    daily = (
        data.groupby("transaction_date")["transaction_amount"]
        .sum()
        .rename("revenue")
        .reset_index()
    )
    daily["weekday"] = daily["transaction_date"].dt.day_name()
    daily["week_start"] = (
        daily["transaction_date"]
        - pd.to_timedelta(daily["transaction_date"].dt.weekday, unit="D")
    )
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    daily["weekday"] = pd.Categorical(
        daily["weekday"], categories=weekday_order, ordered=True
    )
    matrix = daily.pivot(index="week_start", columns="weekday", values="revenue")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        matrix, cmap="YlGnBu", linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Revenue"}, ax=ax
    )
    ax.set(title="Daily Revenue by Weekday", xlabel="Weekday", ylabel="Week starting")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 8. RUN A SMALL, SAFE FIRST-PASS REPORT
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the configured dataset and print core data-quality summaries."""
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place the CSV at that path or update DATA_PATH at the top.")
        print("All reusable functions remain available for import and practice.")
        return

    raw = load_data(DATA_PATH)
    validate_schema(raw)
    df = clean_retail_data(raw)

    print("\nDATASET OVERVIEW")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")
    print(df.head())
    print("\nDATA QUALITY REPORT")
    print(inspect_data(df))
    print("\nDESCRIPTIVE STATISTICS")
    print(distribution_summary(df["transaction_amount"]))
    print("\nOUTLIER COUNTS")
    flags = outlier_flags(df["transaction_amount"])
    print(flags[["iqr_flag", "z_flag", "modified_z_flag"]].sum())
    print("\nPAYMENT METHOD SUMMARY")
    print(grouped_summary(df))


if __name__ == "__main__":
    main()
