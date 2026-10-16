# Pandas Practice
# Basic, Intermediate, and Advanced Pandas examples.
# Install the required libraries with: pip install pandas numpy
# Run this file with: python pandas_practice.py

# Python provides the language and built-in data structures.
# Pandas is used to work with tables, columns, rows, and datasets.
# NumPy supports numerical calculations, missing values, and array operations.
# StringIO lets us read sample CSV/JSON text without creating files.

from io import StringIO
import pandas as pd
import numpy as np


# BASIC PANDAS
# Create a small retail orders dataset.
# Each row represents one order; the data is synthetic for practice.

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "customer": ["Aman", "Priya", "Rahul", "Neha", "Aman", "Priya", None, "Neha"],
    "city": ["Bengaluru", "Mumbai", "Delhi", None, "Bengaluru", "Mumbai", "Delhi", "Pune"],
    "product": ["Laptop", "Phone", "Laptop", "Keyboard", "Phone", "Laptop", "Phone", "Monitor"],
    "quantity": [1, 2, 1, 3, 1, 1, 2, 1],
    "unit_price": [65000, 25000, 62000, 3000, 24000, 64000, 23000, 18000],
    "payment_method": ["UPI", "Card", "UPI", "Cash", "Card", "UPI", "Wallet", "UPI"],
    "status": ["Delivered", "Delivered", "Cancelled", "Delivered", "Returned", "Delivered", "Delivered", "Delivered"],
    "order_date": ["2026-01-02", "2026-01-03", "2026-01-03", "2026-01-05",
                   "2026-01-07", "2026-01-08", "2026-01-10", "2026-01-12"]
})

# Preview the data and inspect its structure.
print(orders.head())                 # First five rows
print(orders.head(2))                # First two rows
print(orders.tail(3))                # Last three rows
print(orders.shape)                  # Number of rows and columns
print(orders.columns.tolist())       # Column names as a Python list
print(orders.dtypes)                 # Data type of each column
orders.info()                        # Compact structural summary
print(orders.describe())             # Numeric summary statistics
print(orders.describe(include="all"))  # Summary for numeric and non-numeric columns

# Find missing values and understand column distributions.
print(orders.isna())                 # True where a value is missing
print(orders.isna().sum())           # Missing values per column
print(orders.isna().sum().sum())     # Total missing cells in the DataFrame
print(orders.isna().mean().mul(100).round(2))  # Missing percentage per column
print(orders["city"].unique())       # Distinct city values, including missing values
print(orders["city"].nunique())      # Number of unique non-missing cities
print(orders["city"].nunique(dropna=False))  # Count missing as a distinct value
print(orders["payment_method"].value_counts())
print(orders["city"].value_counts(dropna=False))

# Select one column, multiple columns, rows, or row-and-column combinations.
print(orders["unit_price"])          # Returns a Series
print(orders[["customer", "city", "unit_price"]])  # Returns a DataFrame
print(orders.iloc[0:3, 0:4])          # Select by integer position; end is excluded
print(orders.loc[0:2, ["customer", "city", "unit_price"]])  # Select by labels

# Filter rows using conditions, membership, ranges, and query syntax.
high_value_orders = orders[orders["unit_price"] >= 50000]
bengaluru_high_value = orders[(orders["city"] == "Bengaluru") & (orders["unit_price"] >= 50000)]
selected_cities = orders[orders["city"].isin(["Bengaluru", "Mumbai"])]
mid_price_orders = orders[orders["unit_price"].between(10000, 30000)]
delivered_orders = orders.query("status == 'Delivered' and quantity >= 1")
print(high_value_orders)
print(bengaluru_high_value)
print(selected_cities)
print(mid_price_orders)
print(delivered_orders)

# Sort rows and find the highest- or lowest-value records.
print(orders.sort_values("unit_price", ascending=False))
print(orders.sort_values(["city", "unit_price"], ascending=[True, False]))
print(orders.nlargest(3, "unit_price"))
print(orders.nsmallest(2, "unit_price"))

# Create new columns, rename columns, reorder columns, and remove columns.
orders["revenue"] = orders["quantity"] * orders["unit_price"]
orders = orders.rename(columns={"customer": "customer_name"})
orders = orders[["order_id", "customer_name", "city", "product", "quantity",
                 "unit_price", "revenue", "payment_method", "status", "order_date"]]
print(orders.head())
print(orders.drop(columns=["status"]).head())  # Drop a column without changing orders

# Read CSV data. dtype preserves leading zeroes in identifiers.
csv_text = """order_id,revenue,city
00101,1200,Bengaluru
00102,NA,Mumbai
00103,2100,Delhi
"""
imported_csv = pd.read_csv(StringIO(csv_text), dtype={"order_id": "string"}, na_values=["NA"])
print(imported_csv)
print(imported_csv.dtypes)
print(imported_csv.isna().sum())

# Handle semicolon separators and decimal commas in some regional CSV files.
semicolon_csv = """order_id;revenue
101;1200,50
102;850,75
103;2100,25
"""
regional_csv = pd.read_csv(StringIO(semicolon_csv), sep=";", decimal=",")
print(regional_csv)

# Read JSON records from text.
json_text = """[
    {"order_id": 201, "revenue": 1200, "city": "Bengaluru"},
    {"order_id": 202, "revenue": 850, "city": "Mumbai"}
]"""
print(pd.read_json(StringIO(json_text)))

# Typical file input/output examples; replace paths with your own paths.
# data = pd.read_csv("data/orders.csv")
# data.to_csv("output/clean_orders.csv", index=False)
# data = pd.read_json("data/orders.json")
# data.to_json("output/orders.json", orient="records", indent=2)


# INTERMEDIATE PANDAS
# Make a fresh copy for cleaning exercises.
clean_orders = orders.copy()

# Handle missing values only after checking how much data is missing.
print(clean_orders.isna().sum())
rows_with_city = clean_orders.dropna(subset=["city"])  # Drop rows missing city
clean_orders["customer_filled"] = clean_orders["customer_name"].fillna("Unknown")
clean_orders["city_forward_filled"] = clean_orders["city"].ffill()  # Use previous non-missing value
clean_orders["city_backward_filled"] = clean_orders["city"].bfill()  # Use next non-missing value
clean_orders["unit_price_filled"] = clean_orders["unit_price"].fillna(clean_orders["unit_price"].median())
print(rows_with_city)
print(clean_orders[["customer_name", "customer_filled", "city", "city_forward_filled"]])

# Interpolate numeric gaps between known values.
linear_values = pd.Series([100.0, np.nan, np.nan, 400.0, 500.0])
print(linear_values.interpolate())  # Linear interpolation uses positions between values

# Time interpolation requires a DatetimeIndex.
time_values = pd.DataFrame(
    {"revenue": [100.0, np.nan, 50.0, np.nan, 400.0]},
    index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"])
)
print(time_values.interpolate(method="time"))

# Detect and remove duplicate records.
duplicate_orders = pd.DataFrame({
    "order_id": [1, 2, 2, 3],
    "customer": ["Asha", "Rahul", "Rahul", "Neha"]
})
print(duplicate_orders.duplicated())
print(duplicate_orders.drop_duplicates())
print(duplicate_orders.drop_duplicates(subset=["order_id"], keep="last"))

# Convert columns to suitable data types and handle invalid values.
clean_orders["order_date"] = pd.to_datetime(clean_orders["order_date"], errors="coerce")
clean_orders["quantity"] = pd.to_numeric(clean_orders["quantity"], errors="coerce")
print(clean_orders.dtypes)

# Validate basic assumptions before using the data.
assert clean_orders["order_id"].notna().all(), "Order ID cannot be missing"
assert clean_orders["quantity"].dropna().ge(0).all(), "Quantity must be non-negative"

# Append rows with concat. concat can combine DataFrames along rows or columns.
extra_order = pd.DataFrame([{
    "order_id": 109, "customer_name": "Isha", "city": "Pune", "product": "Keyboard",
    "quantity": 2, "unit_price": 2500, "revenue": 5000, "payment_method": "UPI",
    "status": "Delivered", "order_date": "2026-01-14"
}])
combined_orders = pd.concat([orders, extra_order], ignore_index=True)
print(combined_orders.tail(2))

# Group rows and calculate business summaries.
print(orders.groupby("city", dropna=False)["revenue"].sum())
city_summary = orders.groupby("city", dropna=False).agg(
    total_revenue=("revenue", "sum"),
    average_order_value=("revenue", "mean"),
    order_count=("order_id", "nunique"),
    total_units=("quantity", "sum")
).reset_index()
print(city_summary)

city_product_summary = orders.groupby(["city", "product"], dropna=False).agg(
    revenue=("revenue", "sum"),
    order_count=("order_id", "nunique")
).reset_index()
print(city_product_summary)

# Merge tables using a shared key. The join type controls which rows are retained.
customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "customer": ["Aman", "Priya", "Rahul"],
    "segment": ["Premium", "Regular", "Premium"]
})
customer_orders = pd.DataFrame({
    "order_id": [501, 502, 503, 504],
    "customer_id": [1, 2, 1, 4],
    "revenue": [5000, 3000, 2000, 4000]
})
print(pd.merge(customers, customer_orders, on="customer_id", how="inner"))  # Matching keys only
print(pd.merge(customers, customer_orders, on="customer_id", how="left"))   # Keep all left rows
print(pd.merge(customers, customer_orders, on="customer_id", how="right"))  # Keep all right rows
print(pd.merge(customers, customer_orders, on="customer_id", how="outer", indicator=True))
# validate checks whether the merge relationship matches expectations.
print(pd.merge(customer_orders, customers, on="customer_id", how="left", validate="many_to_one"))

# map replaces values using a dictionary; unmatched values become missing.
status_map = {"Delivered": "Successful", "Cancelled": "Unsuccessful", "Returned": "Unsuccessful"}
orders["status_group"] = orders["status"].map(status_map)

# apply runs a function against values. This example uses a simple revenue rule.
orders["order_class"] = orders["revenue"].apply(
    lambda revenue: "Unknown" if pd.isna(revenue) else
    "High Value" if revenue >= 50000 else
    "Medium Value" if revenue >= 10000 else "Low Value"
)

# transform returns a group-level result aligned to each original row.
orders["city_avg_revenue"] = orders.groupby("city", dropna=False)["revenue"].transform("mean")
orders["above_city_average"] = orders["revenue"] > orders["city_avg_revenue"]
print(orders[["status", "status_group", "revenue", "order_class", "city_avg_revenue", "above_city_average"]])

# Pivot tables summarize values across row and column categories.
print(pd.pivot_table(orders, index="city", columns="status", values="revenue",
                     aggfunc="sum", fill_value=0, margins=True))

# melt changes wide data into long format.
print(orders.melt(id_vars=["order_id", "city"], value_vars=["quantity", "unit_price"],
                  var_name="metric", value_name="value"))

# category dtype can be useful when a column repeats a small set of labels.
orders["status_category"] = orders["status"].astype("category")
print(orders["status_category"].dtype)


# STRING MANIPULATION AND REGULAR EXPRESSIONS
# The .str accessor applies string methods to an entire Series.
text_data = pd.DataFrame({
    "customer_name": ["  ASHA SHARMA ", "rahul kumar", "Neha   Singh", None],
    "email": [" ASHA@GMAIL.COM ", "rahul@yahoo.com", "neha.singh@gmail.com", "unknown"],
    "product_code": ["ELEC-1001", "FASH-2002", "ELEC-1003", "HOME-3004"]
})

# Strip outside spaces, lowercase text, and replace repeated spaces with one space.
text_data["customer_name_clean"] = (
    text_data["customer_name"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
)
text_data["email_clean"] = text_data["email"].str.strip().str.lower()

# Extract the letters before the hyphen and the digits after the hyphen.
text_data["product_category"] = text_data["product_code"].str.extract(r"^([A-Z]+)-")
text_data["product_id"] = text_data["product_code"].str.extract(r"-(\d+)$")

# Identify product codes and email addresses matching specific patterns.
text_data["is_electronics"] = text_data["product_code"].str.startswith("ELEC", na=False)
text_data["is_gmail"] = text_data["email_clean"].str.endswith("@gmail.com", na=False)
text_data["email_domain"] = text_data["email_clean"].str.extract(r"@([^@]+)$")
text_data["name_word_count"] = text_data["customer_name_clean"].str.split().str.len()
text_data["has_digit_in_name"] = text_data["customer_name_clean"].str.contains(r"\d", na=False)
print(text_data)
print("Gmail customer count:", int(text_data["is_gmail"].sum()))

# Useful string methods include strip, lower, upper, title, replace, contains,
# match, startswith, endswith, split, extract, and len.


# DATETIME AND TIME-SERIES OPERATIONS
# Convert date strings into Pandas datetime values before date calculations.
dates = orders.copy()
dates["order_date"] = pd.to_datetime(dates["order_date"], errors="coerce")

# The .dt accessor extracts parts of a datetime column.
dates["year"] = dates["order_date"].dt.year
dates["month"] = dates["order_date"].dt.month
dates["month_name"] = dates["order_date"].dt.month_name()
dates["day"] = dates["order_date"].dt.day
dates["weekday"] = dates["order_date"].dt.day_name()
dates["weekday_number"] = dates["order_date"].dt.dayofweek
dates["is_weekend"] = dates["weekday_number"].isin([5, 6])
dates["month_start"] = dates["order_date"].dt.to_period("M").dt.to_timestamp()
print(dates[["order_date", "year", "month_name", "weekday", "is_weekend", "month_start"]])

# Parse a non-standard date format; invalid entries become NaT with errors="coerce".
custom_dates = pd.Series(["17/10/2026", "01/11/2026", "bad-date"])
print(pd.to_datetime(custom_dates, format="%d/%m/%Y", errors="coerce"))

# Date arithmetic uses Timedelta values.
dates["delivery_date"] = dates["order_date"] + pd.to_timedelta([2, 3, 1, 4, 2, 2, 1, 3], unit="D")
dates["delivery_days"] = (dates["delivery_date"] - dates["order_date"]).dt.days
print(dates[["order_date", "delivery_date", "delivery_days"]])

# Resampling aggregates observations into regular time intervals.
dates["revenue"] = dates["quantity"] * dates["unit_price"]
daily_revenue = dates.set_index("order_date").resample("D")["revenue"].sum()
print(daily_revenue)

# A time-based rolling window uses elapsed calendar time, not just row count.
time_orders = dates.set_index("order_date").sort_index()
time_orders["rolling_7_day_revenue"] = time_orders["revenue"].rolling("7D", min_periods=1).sum()
print(time_orders[["revenue", "rolling_7_day_revenue"]])

# Monthly summaries can support month-over-month (MoM) comparisons.
monthly_revenue = dates.set_index("order_date")["revenue"].resample("MS").sum().to_frame("revenue")
monthly_revenue["previous_month_revenue"] = monthly_revenue["revenue"].shift(1)
monthly_revenue["mom_change"] = monthly_revenue["revenue"] - monthly_revenue["previous_month_revenue"]
monthly_revenue["mom_change_pct"] = monthly_revenue["mom_change"] / monthly_revenue["previous_month_revenue"] * 100
print(monthly_revenue)

# Time-zone examples: use only when the source timezone is known.
# timestamps = timestamps.dt.tz_localize("Asia/Kolkata")  # Assign a timezone to naive times
# timestamps = timestamps.dt.tz_convert("UTC")            # Convert timezone-aware times


# ADVANCED PANDAS: WINDOW FUNCTIONS AND TIME-SERIES FEATURES
# Generate a repeatable synthetic dataset for four cities over 30 days.
rng = np.random.default_rng(42)  # Fixed seed makes the random example reproducible
cities = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad"]
sales_dates = pd.date_range("2026-01-01", periods=30, freq="D")
base_revenue = {"Bengaluru": 50000, "Mumbai": 65000, "Delhi": 45000, "Hyderabad": 40000}
sales_rows = []

for city in cities:
    for day_number, date in enumerate(sales_dates, start=1):
        units_sold = int(rng.integers(20, 101))
        order_count = int(rng.integers(10, 51))
        trend = day_number * 250
        noise = rng.normal(0, 6000)
        revenue = max(1000, base_revenue[city] + trend + noise)
        sales_rows.append({
            "city": city,
            "date": date,
            "day_number": day_number,
            "orders": order_count,
            "units_sold": units_sold,
            "revenue": round(revenue, 2),
            "sales_target": base_revenue[city] + 3000
        })

sales = pd.DataFrame(sales_rows).sort_values(["city", "date"]).reset_index(drop=True)
print("Sales dataset shape:", sales.shape)
print(sales.head())

# Rank observations within each city. Tied values receive the same rank.
sales["rank"] = sales.groupby("city")["revenue"].rank(method="min", ascending=False)
sales["dense_rank"] = sales.groupby("city")["revenue"].rank(method="dense", ascending=False)
sales["row_number"] = sales.groupby("city").cumcount() + 1
print(sales[sales["dense_rank"] <= 3].sort_values(["city", "dense_rank"]))

# shift(1) gives the previous row; shift(-1) gives the next row in each group.
sales["previous_revenue"] = sales.groupby("city")["revenue"].shift(1)
sales["next_revenue"] = sales.groupby("city")["revenue"].shift(-1)
sales["revenue_change"] = sales["revenue"] - sales["previous_revenue"]
sales["revenue_change_pct"] = sales["revenue_change"] / sales["previous_revenue"] * 100

# Cumulative and whole-group calculations.
sales["cumulative_revenue"] = sales.groupby("city")["revenue"].cumsum()
sales["city_average_revenue"] = sales.groupby("city")["revenue"].transform("mean")
sales["city_total_revenue"] = sales.groupby("city")["revenue"].transform("sum")
sales["share_of_city_revenue_pct"] = sales["revenue"] / sales["city_total_revenue"] * 100

# Rolling windows based on the number of rows/observations.
sales["rolling_3_row_sum"] = (
    sales.groupby("city")["revenue"].rolling(window=3, min_periods=1)
    .sum().reset_index(level=0, drop=True)
)
sales["rolling_7_row_average"] = (
    sales.groupby("city")["revenue"].rolling(window=7, min_periods=1)
    .mean().reset_index(level=0, drop=True)
)

# Expanding average uses every observation from the start through the current row.
sales["expanding_average"] = (
    sales.groupby("city")["revenue"].expanding().mean().reset_index(level=0, drop=True)
)

# Compare actual revenue with the sales target.
sales["target_variance"] = sales["revenue"] - sales["sales_target"]
sales["target_attainment_pct"] = sales["revenue"] / sales["sales_target"] * 100

# Rolling over "7D" uses seven calendar days on a datetime index.
time_sales = sales.set_index("date").sort_values(["city", "date"])
time_sales["rolling_7_calendar_days"] = (
    time_sales.groupby("city")["revenue"].rolling("7D", min_periods=1)
    .sum().reset_index(level=0, drop=True)
)
print(time_sales[["city", "revenue", "rolling_7_calendar_days"]].head(10))

# Vectorized conditions can classify rows without a custom Python loop.
sales["performance"] = np.select(
    [sales["revenue"] >= sales["sales_target"] * 1.10,
     sales["revenue"] >= sales["sales_target"]],
    ["Above Target", "Met Target"],
    default="Below Target"
)
sales["revenue_capped"] = sales["revenue"].clip(lower=0, upper=100000)
sales["is_top_day"] = sales["rank"].eq(1)
print(sales[["city", "date", "revenue", "performance", "is_top_day"]].head())

# Pivot, stack, and unstack change the layout of tabular data.
revenue_matrix = sales.pivot(index="date", columns="city", values="revenue")
print(revenue_matrix.head())
stacked_revenue = revenue_matrix.stack()
print(stacked_revenue.head())
print(stacked_revenue.unstack().head())

# Inspect memory usage. This can help when datasets become large.
print(sales.memory_usage(deep=True))
print("Total memory in bytes:", sales.memory_usage(deep=True).sum())


# FEATURE ENGINEERING AND DATA QUALITY
# Feature engineering creates useful columns from existing information.
features = orders.copy()
features["order_date"] = pd.to_datetime(features["order_date"], errors="coerce")
features["revenue"] = features["quantity"] * features["unit_price"]
features["average_unit_revenue"] = features["revenue"] / features["quantity"].replace(0, np.nan)
features["is_high_value"] = features["revenue"] >= 50000
features["order_month"] = features["order_date"].dt.to_period("M").astype("string")
features["is_delivered"] = features["status"].eq("Delivered")
features["revenue_band"] = pd.cut(
    features["revenue"],
    bins=[-np.inf, 10000, 50000, np.inf],
    labels=["Low", "Medium", "High"]
)
features["city_missing_flag"] = features["city"].isna()
print(features)

# Assertions make assumptions explicit and stop execution when a check fails.
assert features["order_id"].is_unique, "Order IDs should be unique in this example"
assert features["quantity"].ge(0).all(), "Quantity must be non-negative"
assert features["unit_price"].ge(0).all(), "Unit price must be non-negative"
assert features["revenue"].notna().all(), "Revenue should be populated"
print("Data quality checks passed.")

# Avoid data leakage in predictive work: features should use only information
# that would actually be available at the time a prediction is made.
# Full-period averages or future values can leak information from the future.