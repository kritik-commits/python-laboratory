# Retail Time-Series Analysis Roadmap

## Project objective

Analyze retail transaction history and decide whether weekly revenue is forecastable.
The workflow covers datetime engineering, data quality, EDA, time-series plots,
autocorrelation, stationarity tests, decomposition, and baseline prediction.

## Dataset facts

The supplied file has 500 transaction rows and these columns:

- `transaction_id`
- `transaction_date`
- `customer_segment`
- `payment_method`
- `items_purchased`
- `discount_pct`
- `transaction_amount`

The observed dates run from **2026-01-01 through 2026-06-29**. The file contains
missing values in `customer_segment`, `items_purchased`, and `discount_pct`.
There are approximately 27 Sunday-ending weekly bins, including partial edge weeks.

## Roadmap

1. **Ingest and validate**
   - Inspect dimensions, dtypes, missingness, duplicates, invalid dates, and numeric ranges.
   - Do not automatically delete negative amounts; they may be refunds.
   - Keep a documented distinction between missing values and true zero activity.

2. **Engineer datetime features**
   - Extract year, quarter, month, day, weekday, day-of-year, ISO week, and ISO year.
   - Format dates for reporting with `strftime`.
   - Add/subtract `Timedelta` and `DateOffset` values.
   - Compute elapsed days, week starts/ends, month boundaries, weekend flags, and periods.

3. **Perform transaction-level EDA**
   - Inspect target distribution, quantiles, outliers, category frequencies, and relationships.
   - Treat outlier flags as investigation prompts, not automatic deletion rules.

4. **Aggregate the time series**
   - Daily: revenue, transaction count, average transaction value, and items sold.
   - Weekly: the same measures, indexed by week-ending date.
   - Build a continuous date index and inspect weeks with zero transactions.
   - Keep revenue, transaction count, and average order value separate: they answer different questions.

5. **Visualize temporal behavior**
   - Daily line plot and 7-day rolling mean.
   - Weekly revenue and transaction-count plots.
   - Rolling mean, rolling standard deviation, cumulative revenue.
   - Weekday/month distributions and weekly calendar comparisons.
   - ACF/PACF plots for levels and first differences.

6. **Test time-series properties**
   - ADF and KPSS stationarity tests.
   - Ljung–Box test for serial autocorrelation.
   - Shapiro–Wilk/Jarque–Bera as distribution checks, not forecast-readiness tests.
   - Interpret p-values with plots, sample size, and business context.

7. **Decompose the series**
   - Trend: slower movement in the level of the series.
   - Seasonal: repeated pattern at a specified, fixed period.
   - Residual: variation not captured by trend and seasonality.
   - Additive model: `Observed = Trend + Seasonal + Residual`.
   - Multiplicative model: `Observed = Trend × Seasonal × Residual`.
   - Weekly annual seasonality has a period of about 52 observations. Two cycles
     (roughly 104 weeks) are a practical minimum for inspecting annual repetition.
   - With only around half a year, annual seasonality cannot be validated.
     A 4-week decomposition can be demonstrated as exploratory only; it does not
     prove a monthly seasonal effect.

8. **Make the binary readiness decision**
   - **Short-horizon baseline: YES/NO** based on minimum observations, variability,
     zero-activity weeks, and horizon length.
   - **Annual-seasonal weekly model: YES/NO** based on at least 104 weekly points,
     two years of span, and a non-constant target.
   - These are explicit screening rules, not universal statistical laws.

9. **Forecast and validate**
   - Compare naive and simple exponential-smoothing baselines.
   - Hold out the latest weeks in chronological order.
   - Prefer rolling-origin backtesting across multiple cutoffs.
   - Report MAE and RMSE; treat MAPE cautiously when actual values can be zero.
   - Never use a random train/test split for time series.
   - Do not choose a complex seasonal model until it beats simple baselines consistently.

## Concepts in plain language

### ACF — autocorrelation function

ACF measures how strongly the series relates to its own past values at different lags.
Lag 1 compares each week with the previous week; lag 4 compares values four weeks apart.
A slow ACF decay can indicate trend or non-stationarity. Repeated peaks may suggest
repeating cycles, but a short series can produce misleading peaks.

### PACF — partial autocorrelation function

PACF estimates the relationship at a given lag after accounting for shorter lags.
As a rough identification aid, a PACF that cuts off after lag `p` can suggest an AR(p)
component; an ACF that cuts off after lag `q` can suggest an MA(q) component.
Real data rarely follow these patterns perfectly, so ACF/PACF guide model candidates;
they do not choose a model automatically.

### Seasonal decomposition

Decomposition separates a series into trend, seasonal, and residual components.
Additive decomposition assumes the seasonal swing is roughly constant in size.
Multiplicative decomposition assumes the swing scales with the level of the series.
Multiplicative decomposition generally requires positive values. Decomposition describes
patterns; it does not itself forecast or prove that a pattern will repeat in the future.

### ADF — Augmented Dickey–Fuller

- Null hypothesis: the series has a unit root (non-stationary under the test setup).
- Small p-value (commonly `< 0.05`): reject the unit-root null.
- Large p-value: fail to reject it; this is not proof of non-stationarity.
- The result depends on sample size, deterministic terms, and lag selection.

### KPSS

- Null hypothesis: the series is stationary around a specified level or trend.
- Small p-value: evidence against the stationarity null.
- ADF and KPSS have opposite null hypotheses, so using both can be informative.
- Conflicting results should be described as mixed/inconclusive, not forced into a binary claim.

### Ljung–Box

Tests whether a group of autocorrelations up to selected lags is jointly zero.
A small p-value suggests serial dependence remains. For model diagnostics, apply it to
model residuals; applying it to raw data only establishes that the raw series is dependent.

### Other tests and metrics

- **Shapiro–Wilk / Jarque–Bera:** test distributional normality. Normality is not a
  prerequisite for every forecasting model.
- **MAE:** average absolute forecast error, in the target's units.
- **RMSE:** gives larger errors more weight.
- **MAPE:** percentage error; unstable or undefined when actual values are zero or near zero.
- **Rolling-origin backtest:** repeatedly trains on the past and tests on the next period.

## Expected judgement for this dataset

The supplied file covers only about six months and roughly 27 weekly bins.
Therefore, the expected result is:

- **Annual-seasonal weekly forecasting: NO.** There is not enough history to establish
  annual seasonality or evaluate a 52-week seasonal pattern.
- **Exploratory short-horizon baseline: potentially YES.** A naive or smoothing baseline
  can be demonstrated, but forecast quality must be evaluated with chronological
  backtesting and should not be presented as production-grade.

The script computes and prints the final checks from the actual data rather than
hardcoding these judgements.

## Outputs

The script creates `time_series_outputs/` with cleaned/featured data, daily and weekly
series, statistical-test results, readiness checks, forecast metrics, and PNG plots.