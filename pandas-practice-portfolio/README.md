# Pandas Practice Portfolio

A commented, executable reference script covering basic, intermediate, and advanced Pandas.

## Requirements

- Python 3.9+
- pandas
- NumPy

Install dependencies:

```bash
python -m pip install pandas numpy
```

## Run the script

From this directory:

```bash
python pandas_practice.py
```

The script uses synthetic retail orders and generated city-level sales data. It does not require external datasets.

## Topics covered

### Basic Pandas
- Create DataFrames and inspect shape, columns, types, and descriptive statistics
- Find missing values, unique values, and category frequencies
- Select columns and rows with `[]`, `.loc`, and `.iloc`
- Filter with conditions, `.isin()`, `.between()`, and `.query()`
- Sort records and find top/bottom values
- Create, rename, reorder, and drop columns
- Read sample CSV and JSON data; handle separators, decimal commas, and identifier types

### Intermediate Pandas
- Missing-value handling: `.dropna()`, `.fillna()`, forward/backward fill, interpolation
- Detect and remove duplicates
- Convert data types and validate assumptions
- Combine data with `concat()` and `merge()`
- Group and aggregate business metrics with `groupby()`
- Use `.map()`, `.apply()`, and `.transform()`
- Build pivot tables and reshape data with `.melt()`
- Use categorical data types

### String manipulation and regular expressions
- Normalize whitespace and letter case
- Extract product categories, IDs, and email domains
- Detect prefixes, suffixes, digits, and Gmail addresses
- Use the Pandas `.str` accessor and regex patterns

### Datetime and time series
- Parse dates and handle invalid values
- Extract date parts with `.dt`
- Calculate delivery intervals
- Resample daily and monthly metrics
- Calculate rolling seven-day revenue and month-over-month changes
- Review timezone localization and conversion examples

### Advanced Pandas and feature engineering
- Generate reproducible sample data with NumPy's random generator
- Calculate ranks, row numbers, lags, leads, deltas, and cumulative totals
- Use grouped transforms, rolling windows, and expanding averages
- Compare actual performance against targets
- Apply vectorized conditions with `np.select()` and cap values with `.clip()`
- Reshape with `pivot()`, `stack()`, and `unstack()`
- Inspect DataFrame memory usage
- Create derived features, revenue bands, and missingness flags
- Validate data quality with assertions
- Understand the risk of future-data leakage in predictive modeling

## Notes

- The datasets are synthetic and intended for learning.
- Several examples print their outputs to make experimentation easier.
- File input/output paths are shown as commented examples.
- Review business meaning before filling missing values or removing records.
- Avoid data leakage when engineering features for predictive models.
