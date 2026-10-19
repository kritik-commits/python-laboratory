"""
PROJECT: Retail Transactions — Time-Series EDA and Forecast Readiness
====================================================================

Dataset:
    data/retail_transactions_outlier_practice.csv

Expected columns:
    transaction_id, transaction_date, customer_segment, payment_method,
    items_purchased, discount_pct, transaction_amount

Run from the project root after installing:
    numpy pandas matplotlib seaborn statsmodels scipy

This script is intentionally educational:
- validates data before analysis
- demonstrates datetime transformations
- builds daily and weekly series
- plots time-series diagnostics
- runs stationarity and residual-dependence tests
- makes an explicit forecasting-readiness decision

Important interpretation:
A model can be fitted to a short series. That does not mean its forecast
is reliable. The dataset's short history cannot establish annual seasonality.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("retail_transactions_outlier_practice.csv")
OUTPUT_DIR = Path("time_series_outputs")
DATE_COLUMN = "transaction_date"
TARGET_COLUMN = "transaction_amount"
WEEK_RULE = "W-SUN"  # Weeks ending Sunday; weekly index labels are Sundays.
FORECAST_HORIZON_WEEKS = 4

# Readiness thresholds are explicit heuristics, not universal laws.
MIN_WEEKS_FOR_EXPLORATORY_BASELINE = 20
MIN_WEEKS_FOR_SEASONAL_WEEKLY_MODEL = 104  # 2 annual cycles of 52 weeks.
MAX_MISSING_WEEK_RATE = 0.10

sns.set_theme(style="whitegrid", context="notebook")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Load, validate, and inspect
# ---------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and fail early when required columns are absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}\n"
            "Place retail_transactions_outlier_practice.csv beside this script "
            "or update DATA_PATH."
        )

    df = pd.read_csv(path)
    required = {
        "transaction_id", DATE_COLUMN, "customer_segment", "payment_method",
        "items_purchased", "discount_pct", TARGET_COLUMN,
    }
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Required columns missing: {sorted(missing_columns)}")

    print("\n" + "=" * 78)
    print("1. DATA INGESTION AND SCHEMA")
    print("=" * 78)
    print(f"Shape: {df.shape}")
    print("\nDtypes before conversion:\n", df.dtypes)
    print("\nSample:\n", df.head())
    print("\nMissing values:\n", df.isna().sum())
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate transaction IDs: {df['transaction_id'].duplicated().sum()}")

    # Explicit, safe datetime conversion. Invalid values become NaT for auditing.
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    invalid_dates = int(df[DATE_COLUMN].isna().sum())
    print(f"Invalid or missing dates after parsing: {invalid_dates}")

    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent quality rules; preserve a report of what was changed."""
    print("\n" + "=" * 78)
    print("2. DATA QUALITY CHECKS")
    print("=" * 78)

    work = df.copy()
    initial_rows = len(work)

    # Duplicate transaction IDs usually represent duplicate business events.
    duplicate_id_rows = int(work["transaction_id"].duplicated(keep="first").sum())
    work = work.drop_duplicates(subset="transaction_id", keep="first")

    # A row without a valid transaction date cannot enter time aggregation.
    invalid_date_rows = int(work[DATE_COLUMN].isna().sum())
    work = work.dropna(subset=[DATE_COLUMN])

    # Missing target values cannot contribute to revenue totals.
    missing_target_rows = int(work[TARGET_COLUMN].isna().sum())
    work = work.dropna(subset=[TARGET_COLUMN])

    # Negative transaction amounts may be refunds, reversals, or errors.
    # Keep them unless business rules establish that they are invalid.
    negative_amounts = int((work[TARGET_COLUMN] < 0).sum())

    print(f"Rows at start: {initial_rows}")
    print(f"Duplicate transaction IDs removed: {duplicate_id_rows}")
    print(f"Rows removed for invalid/missing date: {invalid_date_rows}")
    print(f"Rows removed for missing target: {missing_target_rows}")
    print(f"Negative transaction amounts retained for review: {negative_amounts}")
    print("\nRemaining missing values:\n", work.isna().sum())

    # Numeric validation. Do not silently impute unknown business values.
    for col in ["items_purchased", "discount_pct", TARGET_COLUMN]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    print("\nNumeric summary:\n", work[["items_purchased", "discount_pct", TARGET_COLUMN]].describe())
    return work.sort_values(DATE_COLUMN).reset_index(drop=True)


# ---------------------------------------------------------------------
# 3. Datetime transformations: extract, format, add, subtract, compare
# ---------------------------------------------------------------------

def engineer_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful date parts and demonstrate datetime arithmetic."""
    print("\n" + "=" * 78)
    print("3. DATETIME TRANSFORMATIONS")
    print("=" * 78)

    work = df.copy()
    dt = work[DATE_COLUMN]

    # Extraction into integer/string/categorical representations.
    work["date_only"] = dt.dt.date
    work["year"] = dt.dt.year
    work["quarter"] = dt.dt.quarter
    work["month"] = dt.dt.month
    work["month_name"] = dt.dt.month_name()
    work["day"] = dt.dt.day
    work["day_name"] = dt.dt.day_name()
    work["day_of_week"] = dt.dt.dayofweek  # Monday=0, Sunday=6
    work["day_of_year"] = dt.dt.dayofyear
    work["week_of_year"] = dt.dt.isocalendar().week.astype("Int64")
    work["iso_year"] = dt.dt.isocalendar().year.astype("Int64")
    work["is_weekend"] = work["day_of_week"].isin([5, 6])
    work["year_month"] = dt.dt.to_period("M").astype(str)
    work["year_quarter"] = dt.dt.to_period("Q").astype(str)
    work["iso_week_label"] = (
        work["iso_year"].astype(str) + "-W" +
        work["week_of_year"].astype(str).str.zfill(2)
    )

    # Formatting as strings. These are display keys, not timestamps.
    work["date_iso"] = dt.dt.strftime("%Y-%m-%d")
    work["date_us_style"] = dt.dt.strftime("%m/%d/%Y")
    work["date_readable"] = dt.dt.strftime("%d %b %Y")

    # Date arithmetic: add/subtract timedeltas and calendar offsets.
    work["date_plus_7_days"] = dt + pd.Timedelta(days=7)
    work["date_minus_30_days"] = dt - pd.Timedelta(days=30)
    work["date_plus_1_month"] = dt + pd.DateOffset(months=1)
    work["days_from_first_transaction"] = (dt - dt.min()).dt.days
    work["days_until_last_transaction"] = (dt.max() - dt).dt.days

    # Boundary calculations and normalization.
    work["week_start_monday"] = dt.dt.to_period("W-SUN").dt.start_time
    work["week_end_sunday"] = dt.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    work["month_start"] = dt.dt.to_period("M").dt.start_time
    work["month_end"] = dt.dt.to_period("M").dt.end_time.dt.normalize()
    work["is_month_start"] = dt.dt.is_month_start
    work["is_month_end"] = dt.dt.is_month_end

    print(f"Date minimum: {dt.min()}")
    print(f"Date maximum: {dt.max()}")
    print(f"Calendar span: {(dt.max() - dt.min()).days + 1} days")
    print(f"Unique transaction dates: {dt.dt.normalize().nunique()}")
    print("\nDatetime feature sample:\n", work[
        [DATE_COLUMN, "date_iso", "year_month", "day_name", "week_of_year",
         "week_start_monday", "date_plus_7_days", "days_from_first_transaction"]
    ].head())

    work.to_csv(OUTPUT_DIR / "transactions_with_datetime_features.csv", index=False)
    return work


# ---------------------------------------------------------------------
# 4. EDA: transaction-level patterns and data checks
# ---------------------------------------------------------------------

def run_transaction_eda(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("4. TRANSACTION-LEVEL EDA")
    print("=" * 78)

    print("\nCategorical distributions:")
    for col in ["customer_segment", "payment_method", "month_name", "day_name"]:
        print(f"\n{col}:\n{df[col].value_counts(dropna=False)}")

    print("\nNumeric summary:")
    print(df[["items_purchased", "discount_pct", TARGET_COLUMN]].describe().T)

    print("\nAmount quantiles:")
    print(df[TARGET_COLUMN].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]))

    # Plausibility checks should be investigated, not automatically deleted.
    checks = {
        "negative transaction amount": df[TARGET_COLUMN] < 0,
        "zero transaction amount": df[TARGET_COLUMN] == 0,
        "negative item count": df["items_purchased"] < 0,
        "discount below 0": df["discount_pct"] < 0,
        "discount above 100": df["discount_pct"] > 100,
    }
    print("\nPlausibility-check counts:")
    for label, mask in checks.items():
        print(f"{label}: {int(mask.fillna(False).sum())}")

    # Transaction amount distribution and business segment comparisons.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(data=df, x=TARGET_COLUMN, kde=True, ax=axes[0, 0], color="C0")
    axes[0, 0].set_title("Transaction amount distribution")

    sns.boxplot(data=df, x="customer_segment", y=TARGET_COLUMN, ax=axes[0, 1], color="C0")
    axes[0, 1].set_title("Transaction amount by customer segment")

    sns.countplot(data=df, x="payment_method", ax=axes[1, 0], color="C0")
    axes[1, 0].set_title("Transactions by payment method")

    sns.scatterplot(
        data=df, x="discount_pct", y=TARGET_COLUMN,
        hue="customer_segment", ax=axes[1, 1]
    )
    axes[1, 1].set_title("Discount versus transaction amount")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "transaction_level_eda.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# 5. Build daily and weekly time series
# ---------------------------------------------------------------------

def build_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw transactions; create continuous daily and weekly indexes."""
    print("\n" + "=" * 78)
    print("5. DAILY AND WEEKLY AGGREGATION")
    print("=" * 78)

    dated = df.set_index(DATE_COLUMN).sort_index()

    # Daily sum/count/mean: each answers a different business question.
    daily = dated.resample("D").agg(
        revenue=(TARGET_COLUMN, "sum"),
        transactions=(TARGET_COLUMN, "count"),
        average_transaction_value=(TARGET_COLUMN, "mean"),
        items_sold=("items_purchased", "sum"),
    )

    # A date with no transactions has zero revenue and zero transaction count.
    # Average transaction value remains NaN when there were no transactions.
    daily["revenue"] = daily["revenue"].fillna(0)
    daily["transactions"] = daily["transactions"].fillna(0)
    daily["items_sold"] = daily["items_sold"].fillna(0)

    weekly = dated.resample(WEEK_RULE, label="right", closed="right").agg(
        revenue=(TARGET_COLUMN, "sum"),
        transactions=(TARGET_COLUMN, "count"),
        average_transaction_value=(TARGET_COLUMN, "mean"),
        items_sold=("items_purchased", "sum"),
    )

    # Reindex to every week between first and last observed week.
    # Zero revenue is valid only if no transactions occurred in that week.
    full_week_index = pd.date_range(
        weekly.index.min(), weekly.index.max(), freq=WEEK_RULE
    )
    weekly = weekly.reindex(full_week_index)
    weekly.index.name = "week_ending"
    weekly["revenue"] = weekly["revenue"].fillna(0)
    weekly["transactions"] = weekly["transactions"].fillna(0)
    weekly["items_sold"] = weekly["items_sold"].fillna(0)

    # Derived weekly features.
    weekly["revenue_lag_1"] = weekly["revenue"].shift(1)
    weekly["revenue_lag_2"] = weekly["revenue"].shift(2)
    weekly["revenue_lag_4"] = weekly["revenue"].shift(4)
    weekly["revenue_change"] = weekly["revenue"].diff()
    weekly["revenue_pct_change"] = weekly["revenue"].pct_change()
    weekly["revenue_rolling_mean_4w"] = weekly["revenue"].rolling(4, min_periods=2).mean()
    weekly["revenue_rolling_std_4w"] = weekly["revenue"].rolling(4, min_periods=2).std()
    weekly["revenue_rolling_mean_8w"] = weekly["revenue"].rolling(8, min_periods=4).mean()
    weekly["revenue_cumulative"] = weekly["revenue"].cumsum()
    weekly["week_number"] = np.arange(1, len(weekly) + 1)
    weekly["month"] = weekly.index.month
    weekly["quarter"] = weekly.index.quarter

    # Flag incomplete edge weeks. The source begins/ends mid-week, so these
    # partial sums are not comparable with full seven-day totals.
    first_date = dated.index.min().normalize()
    last_date = dated.index.max().normalize()
    week_starts = weekly.index - pd.Timedelta(days=6)
    weekly["is_partial_week"] = (week_starts < first_date) | (weekly.index > last_date)

    print(f"Daily observations: {len(daily)}")
    print(f"Weekly bins including partial edge weeks: {len(weekly)}")
    print(f"Complete weekly observations: {(~weekly['is_partial_week']).sum()}")
    print(f"Partial weekly bins: {weekly['is_partial_week'].sum()}")
    print(f"Weekly index continuous: {weekly.index.is_monotonic_increasing and weekly.index.is_unique}")
    print("\nWeekly summary:\n", weekly[["revenue", "transactions", "items_sold"]].describe())
    print("\nWeekly observations:\n", weekly[["revenue", "transactions"]].to_string())

    daily.to_csv(OUTPUT_DIR / "daily_time_series.csv")
    weekly.to_csv(OUTPUT_DIR / "weekly_time_series.csv")
    return daily, weekly


# ---------------------------------------------------------------------
# 6. Time-series visualizations with Matplotlib and Seaborn
# ---------------------------------------------------------------------

def plot_time_series(daily: pd.DataFrame, weekly: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("6. TIME-SERIES VISUALIZATIONS")
    print("=" * 78)

    # Daily series: raw variability and smoothed trend.
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(daily.index, daily["revenue"], alpha=0.55, label="Daily revenue")
    ax.plot(
        daily.index, daily["revenue"].rolling(7, min_periods=3).mean(),
        linewidth=2, label="7-day rolling mean"
    )
    ax.set(title="Daily revenue and 7-day rolling mean", xlabel="Date", ylabel="Revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_revenue_trend.png", dpi=150)
    plt.close(fig)

    # Weekly revenue and transactions are separate plots due to different scales.
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(weekly.index, weekly["revenue"], marker="o")
    axes[0].set(title="Weekly revenue", ylabel="Revenue")
    axes[1].bar(weekly.index, weekly["transactions"], width=5)
    axes[1].set(title="Weekly transaction count", ylabel="Transactions", xlabel="Week ending")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weekly_revenue_and_count.png", dpi=150)
    plt.close(fig)

    # Trend, rolling variability, cumulative amount.
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(weekly.index, weekly["revenue"], label="Weekly revenue")
    axes[0].plot(weekly.index, weekly["revenue_rolling_mean_4w"], label="4-week mean")
    axes[0].legend()
    axes[0].set_ylabel("Revenue")
    axes[1].plot(weekly.index, weekly["revenue_rolling_std_4w"])
    axes[1].set_ylabel("4-week rolling SD")
    axes[2].plot(weekly.index, weekly["revenue_cumulative"])
    axes[2].set_ylabel("Cumulative revenue")
    axes[2].set_xlabel("Week ending")
    fig.suptitle("Weekly trend, variability, and cumulative revenue")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weekly_trend_variability_cumulative.png", dpi=150)
    plt.close(fig)

    # Calendar effects: average daily revenue by weekday and month.
    daily_calendar = daily.copy()
    daily_calendar["day_name"] = daily_calendar.index.day_name()
    daily_calendar["weekday_num"] = daily_calendar.index.dayofweek
    daily_calendar["month_name"] = daily_calendar.index.month_name()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.boxplot(data=daily_calendar, x="day_name", y="revenue", order=weekday_order, ax=axes[0], color="C0")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Daily revenue distribution by weekday")

    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    observed_months = [m for m in month_order if m in daily_calendar["month_name"].unique()]
    sns.boxplot(data=daily_calendar, x="month_name", y="revenue", order=observed_months, ax=axes[1], color="C0")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Daily revenue distribution by month")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calendar_effects.png", dpi=150)
    plt.close(fig)

    # Weekly revenue distributions across calendar months.
    weekly_plot = weekly.reset_index()
    weekly_plot["month_name"] = weekly_plot["week_ending"].dt.month_name()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=weekly_plot, x="month_name", y="revenue", ax=ax, color="C0")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Weekly revenue by week-ending month")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weekly_revenue_by_month.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# 7. Stationarity tests and autocorrelation diagnostics
# ---------------------------------------------------------------------

def run_stationarity_tests(series: pd.Series, label: str) -> dict:
    """
    ADF null: unit root / non-stationarity.
    KPSS null: level-stationarity (regression='c').
    Use both because their null hypotheses are opposite.
    """
    y = series.dropna().astype(float)
    print(f"\n--- Stationarity diagnostics: {label} ---")

    results = {"series": label, "n": len(y)}
    if len(y) < 10 or y.nunique() < 2:
        print("Insufficient variation/observations for reliable stationarity tests.")
        results.update({"adf_p": np.nan, "kpss_p": np.nan, "interpretation": "INSUFFICIENT"})
        return results

    try:
        adf_result = adfuller(y, autolag="AIC")
        results["adf_stat"] = float(adf_result[0])
        results["adf_p"] = float(adf_result[1])
        results["adf_lags"] = int(adf_result[2])
        print(f"ADF statistic={adf_result[0]:.4f}, p={adf_result[1]:.4f}, lags={adf_result[2]}")
    except Exception as exc:
        results["adf_p"] = np.nan
        print(f"ADF could not run: {exc}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(y, regression="c", nlags="auto")
        results["kpss_stat"] = float(kpss_result[0])
        results["kpss_p"] = float(kpss_result[1])
        print(f"KPSS statistic={kpss_result[0]:.4f}, p={kpss_result[1]:.4f}")
    except Exception as exc:
        results["kpss_p"] = np.nan
        print(f"KPSS could not run: {exc}")

    adf_p, kpss_p = results.get("adf_p", np.nan), results.get("kpss_p", np.nan)
    if pd.notna(adf_p) and pd.notna(kpss_p):
        if adf_p < 0.05 and kpss_p >= 0.05:
            interpretation = "EVIDENCE_SUPPORTS_STATIONARITY"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            interpretation = "EVIDENCE_SUPPORTS_NONSTATIONARITY"
        else:
            interpretation = "MIXED_OR_INCONCLUSIVE"
    else:
        interpretation = "INCOMPLETE_TESTS"
    results["interpretation"] = interpretation
    print("Combined interpretation:", interpretation)
    return results


def plot_acf_pacf_diagnostics(weekly: pd.DataFrame) -> None:
    """Plot ACF/PACF for weekly revenue and its first difference."""
    print("\n" + "=" * 78)
    print("7. ACF AND PACF")
    print("=" * 78)

    y = weekly["revenue"].dropna().astype(float)
    if len(y) < 8 or y.nunique() < 2:
        print("Not enough variable weekly observations to plot ACF/PACF responsibly.")
        return

    # statsmodels PACF commonly requires lags < half the sample size.
    max_lags = max(1, min(12, len(y) // 2 - 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_acf(y, lags=max_lags, ax=axes[0, 0], zero=False)
    axes[0, 0].set_title("ACF — weekly revenue")
    plot_pacf(y, lags=max_lags, ax=axes[0, 1], method="ywm", zero=False)
    axes[0, 1].set_title("PACF — weekly revenue")

    diff = y.diff().dropna()
    diff_lags = max(1, min(12, len(diff) // 2 - 1))
    plot_acf(diff, lags=diff_lags, ax=axes[1, 0], zero=False)
    axes[1, 0].set_title("ACF — first-differenced revenue")
    plot_pacf(diff, lags=diff_lags, ax=axes[1, 1], method="ywm", zero=False)
    axes[1, 1].set_title("PACF — first-differenced revenue")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "acf_pacf_diagnostics.png", dpi=150)
    plt.close(fig)


def run_other_statistical_tests(weekly: pd.DataFrame) -> pd.DataFrame:
    """Run additional tests, reporting p-values without overclaiming."""
    y = weekly["revenue"].dropna().astype(float)
    results = []

    print("\n" + "=" * 78)
    print("8. ADDITIONAL STATISTICAL TESTS")
    print("=" * 78)

    # ADF and KPSS on original and first-differenced weekly revenue.
    results.append(run_stationarity_tests(y, "weekly revenue"))
    results.append(run_stationarity_tests(y.diff().dropna(), "first difference"))

    # Ljung-Box: null says no autocorrelation through the selected lag.
    if len(y) >= 12 and y.nunique() > 1:
        lag = min(10, max(1, len(y) // 4))
        try:
            lb = acorr_ljungbox(y, lags=[lag], return_df=True)
            p_value = float(lb["lb_pvalue"].iloc[0])
            print(f"\nLjung-Box (lag {lag}): statistic={lb['lb_stat'].iloc[0]:.4f}, p={p_value:.4f}")
            print("Interpretation:", "evidence of autocorrelation" if p_value < .05 else "insufficient evidence of autocorrelation")
            results.append({"series": "weekly revenue", "test": "Ljung-Box",
                            "statistic": float(lb["lb_stat"].iloc[0]), "p_value": p_value})
        except Exception as exc:
            print("Ljung-Box unavailable:", exc)

    # Normality tests are descriptive only; revenue need not be normally distributed.
    if len(y) >= 8:
        sample = y.sample(min(len(y), 5000), random_state=42)
        if sample.nunique() > 2:
            try:
                shapiro = stats.shapiro(sample)
                print(f"\nShapiro-Wilk (weekly revenue): statistic={shapiro.statistic:.4f}, p={shapiro.pvalue:.4f}")
                print("This tests distributional normality, not forecast validity.")
                results.append({"series": "weekly revenue", "test": "Shapiro-Wilk",
                                "statistic": float(shapiro.statistic), "p_value": float(shapiro.pvalue)})
            except Exception as exc:
                print("Shapiro-Wilk unavailable:", exc)

    # Jarque-Bera: skewness/kurtosis departure from normality.
    if len(y) >= 8 and y.nunique() > 2:
        try:
            jb = stats.jarque_bera(y)
            print(f"Jarque-Bera: statistic={jb.statistic:.4f}, p={jb.pvalue:.4f}")
            results.append({"series": "weekly revenue", "test": "Jarque-Bera",
                            "statistic": float(jb.statistic), "p_value": float(jb.pvalue)})
        except Exception as exc:
            print("Jarque-Bera unavailable:", exc)

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "statistical_test_results.csv", index=False)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# 9. Seasonal decomposition
# ---------------------------------------------------------------------

def run_seasonal_decomposition(weekly: pd.DataFrame) -> None:
    """
    Additive: observed = trend + seasonal + residual.
    Multiplicative: observed = trend * seasonal * residual.
    Annual weekly period=52 needs multiple years of data. Never imply annual
    seasonality has been validated from a few months of observations.
    """
    print("\n" + "=" * 78)
    print("9. SEASONAL DECOMPOSITION")
    print("=" * 78)

    y = weekly["revenue"].astype(float).asfreq(WEEK_RULE)
    y = y.interpolate(limit_direction="both")

    if len(y) >= 2 * 52:
        period = 52
        period_label = "52-week annual cycle"
    elif len(y) >= 8:
        period = 4
        period_label = "4-week exploratory cycle (NOT annual seasonality)"
    else:
        print("Too few weekly observations for a useful seasonal decomposition.")
        return

    print(f"Decomposition period: {period} ({period_label})")
    print("Caution: a 4-week cycle is only an exploratory pattern, not proof of monthly seasonality.")

    try:
        result = seasonal_decompose(y, model="additive", period=period, extrapolate_trend="freq")
        fig = result.plot()
        fig.set_size_inches(14, 9)
        fig.suptitle(f"Additive decomposition — {period_label}", y=1.01)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "seasonal_decomposition_additive.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        components = pd.DataFrame({
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        })
        components.to_csv(OUTPUT_DIR / "seasonal_decomposition_components.csv")
    except Exception as exc:
        print(f"Decomposition could not be computed: {exc}")


# ---------------------------------------------------------------------
# 10. Binary forecast-readiness judgement
# ---------------------------------------------------------------------

def forecast_readiness(weekly: pd.DataFrame) -> dict:
    """
    Two distinct decisions:
    1) Can we explore a short-horizon baseline?
    2) Is there enough history for a weekly annual-seasonal model?

    The second rule requires at least two 52-week cycles. This is a practical
    screening rule, not a guarantee of forecast accuracy.
    """
    print("\n" + "=" * 78)
    print("10. FORECAST READINESS — BINARY JUDGEMENT")
    print("=" * 78)

    y = weekly["revenue"].astype(float)
    n_weeks = len(y)
    missing_week_rate = float(weekly["transactions"].eq(0).mean()) if n_weeks else 1.0
    date_span_days = (weekly.index.max() - weekly.index.min()).days if n_weeks else 0
    distinct_revenue_values = int(y.nunique())

    baseline_checks = {
        "at_least_20_weekly_points": n_weeks >= MIN_WEEKS_FOR_EXPLORATORY_BASELINE,
        "revenue_has_variation": distinct_revenue_values > 1,
        "missing_or_zero_transaction_weeks_not_over_10pct": missing_week_rate <= MAX_MISSING_WEEK_RATE,
        "enough_points_for_4_week_horizon_screen": n_weeks >= FORECAST_HORIZON_WEEKS * 3,
    }
    baseline_ready = all(baseline_checks.values())

    seasonal_checks = {
        "at_least_104_weekly_points": n_weeks >= MIN_WEEKS_FOR_SEASONAL_WEEKLY_MODEL,
        "at_least_two_years_of_weekly_history": date_span_days >= 365 * 2 - 7,
        "revenue_has_variation": distinct_revenue_values > 1,
    }
    seasonal_ready = all(seasonal_checks.values())

    print(f"Weekly observations: {n_weeks}")
    print(f"Weekly date span: {date_span_days} days")
    print(f"Weeks with zero transactions: {missing_week_rate:.1%}")
    print("\nShort-horizon baseline checks:")
    for name, passed in baseline_checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"SHORT-HORIZON BASELINE READY: {'YES' if baseline_ready else 'NO'}")

    print("\nAnnual seasonal-model checks:")
    for name, passed in seasonal_checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"ANNUAL-SEASONAL WEEKLY FORECAST READY: {'YES' if seasonal_ready else 'NO'}")

    result = {
        "weekly_observations": n_weeks,
        "date_span_days": date_span_days,
        "zero_transaction_week_rate": missing_week_rate,
        "short_horizon_baseline_ready": "YES" if baseline_ready else "NO",
        "annual_seasonal_model_ready": "YES" if seasonal_ready else "NO",
        "baseline_checks": baseline_checks,
        "seasonal_checks": seasonal_checks,
    }
    pd.DataFrame([
        {"scope": "short_horizon_baseline", "check": k, "passed": v}
        for k, v in baseline_checks.items()
    ] + [
        {"scope": "annual_seasonal_model", "check": k, "passed": v}
        for k, v in seasonal_checks.items()
    ]).to_csv(OUTPUT_DIR / "forecast_readiness_checks.csv", index=False)

    return result


# ---------------------------------------------------------------------
# 11. Demonstration forecasts and time-ordered validation
# ---------------------------------------------------------------------

def run_baseline_forecasts(weekly: pd.DataFrame, readiness: dict) -> None:
    """Compare naive and smoothing baselines; do not use random train/test split."""
    print("\n" + "=" * 78)
    print("11. BASELINE FORECASTS")
    print("=" * 78)

    y = weekly["revenue"].astype(float).asfreq(WEEK_RULE)
    if len(y) < 12:
        print("Not enough observations for a minimally useful time-ordered demonstration.")
        return

    horizon = FORECAST_HORIZON_WEEKS
    train = y.iloc[:-horizon]
    test = y.iloc[-horizon:]

    # Naive forecast: repeat the final training observation.
    naive_forecast = pd.Series(train.iloc[-1], index=test.index)

    # Exponential smoothing is a baseline, not a seasonal model.
    try:
        fit = SimpleExpSmoothing(train, initialization_method="estimated").fit()
        ses_forecast = fit.forecast(horizon)
    except Exception as exc:
        print("Simple exponential smoothing failed:", exc)
        ses_forecast = naive_forecast.copy()

    def metric_row(name: str, actual: pd.Series, predicted: pd.Series) -> dict:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        denominator = np.where(np.abs(actual) < 1e-9, np.nan, np.abs(actual))
        mape = float(np.nanmean(np.abs((actual - predicted) / denominator)) * 100)
        return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE_percent": mape}

    metrics = pd.DataFrame([
        metric_row("Naive", test, naive_forecast),
        metric_row("Simple exponential smoothing", test, ses_forecast),
    ])
    print(f"Holdout: last {horizon} weeks; training: prior {len(train)} weeks")
    print(metrics.to_string(index=False))
    metrics.to_csv(OUTPUT_DIR / "baseline_holdout_metrics.csv", index=False)

    # Fit baselines on all available data to create the next-horizon illustration.
    # Skip the current incomplete calendar week; forecast the next complete weeks.
    future_index = pd.date_range(
        y.index[-1] + pd.Timedelta(days=14), periods=horizon, freq=WEEK_RULE
    )
    future_naive = pd.Series(y.iloc[-1], index=future_index, name="naive_forecast")
    try:
        full_fit = SimpleExpSmoothing(y, initialization_method="estimated").fit()
        future_ses = full_fit.forecast(horizon)
        future_ses.index = future_index
        future_ses.name = "ses_forecast"
    except Exception:
        future_ses = future_naive.rename("ses_forecast")

    forecast_table = pd.concat([future_naive, future_ses], axis=1)
    forecast_table.index.name = "week_ending"
    forecast_table.to_csv(OUTPUT_DIR / "illustrative_next_4_week_forecast.csv")
    print("\nIllustrative next 4 weeks (not a validated production forecast):")
    print(forecast_table)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y.index, y, marker="o", label="Observed weekly revenue")
    ax.plot(future_naive.index, future_naive, marker="o", linestyle="--", label="Naive forecast")
    ax.plot(future_ses.index, future_ses, marker="o", linestyle="--", label="SES forecast")
    ax.axvline(y.index[-1], linestyle=":", linewidth=1)
    ax.set(title="Illustrative 4-week forecasts", xlabel="Week ending", ylabel="Revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "illustrative_forecasts.png", dpi=150)
    plt.close(fig)

    print("\nValidation note:")
    print("- Last-four-week holdout is a single test window, so metrics are unstable.")
    print("- Prefer rolling-origin backtesting over several forecast origins.")
    print("- Compare against naive/seasonal-naive baselines before choosing a complex model.")
    print("- Never use a random split for time series.")


# ---------------------------------------------------------------------
# 12. Run project
# ---------------------------------------------------------------------

def main() -> None:
    print("RETAIL TIME-SERIES ANALYSIS ROADMAP")
    print("Outputs will be saved in:", OUTPUT_DIR.resolve())

    raw = load_data(DATA_PATH)
    clean = validate_and_clean(raw)
    featured = engineer_datetime_features(clean)
    run_transaction_eda(featured)
    daily, weekly_all = build_time_series(featured)
    # Use complete weeks for statistical diagnostics and model evaluation.
    # Keep partial bins in the saved raw weekly table for transparency.
    weekly = weekly_all.loc[~weekly_all["is_partial_week"]].copy()
    weekly = weekly.drop(columns=["is_partial_week"])
    print(f"\nUsing {len(weekly)} complete weeks for diagnostics and forecasting.")
    plot_time_series(daily, weekly)
    plot_acf_pacf_diagnostics(weekly)
    run_other_statistical_tests(weekly)
    run_seasonal_decomposition(weekly)
    readiness = forecast_readiness(weekly)
    run_baseline_forecasts(weekly, readiness)

    print("\n" + "=" * 78)
    print("PROJECT COMPLETE")
    print("=" * 78)
    print(f"Review generated CSVs, statistical results, and plots in: {OUTPUT_DIR.resolve()}")
    print("Interpret the seasonal-model verdict conservatively.")
    print("Short history can support a baseline experiment, not robust annual seasonality.")


if __name__ == "__main__":
    main()