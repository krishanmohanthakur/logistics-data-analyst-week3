# ============================================================
# Yuva Intern - Logistics Data Analyst Internship
# Week 3: Advanced Data Analysis and Visualization
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

DATA_PATH = "data"

OUTPUT_PATH = "week3_outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 70)
print("LOGISTICS DATA ANALYST - WEEK 3")
print("Advanced Data Analysis and Visualization")
print("=" * 70)


# ------------------------------------------------------------
# 2. Load Olist Dataset
# ------------------------------------------------------------

orders_file = os.path.join(DATA_PATH, "olist_orders_dataset.csv")
items_file = os.path.join(DATA_PATH, "olist_order_items_dataset.csv")

if not os.path.exists(orders_file):
    print("\nERROR: Orders dataset not found.")
    print("Please place olist_orders_dataset.csv inside the data folder.")
    raise FileNotFoundError(orders_file)

if not os.path.exists(items_file):
    print("\nERROR: Order items dataset not found.")
    print("Please place olist_order_items_dataset.csv inside the data folder.")
    raise FileNotFoundError(items_file)

orders = pd.read_csv(orders_file)
items = pd.read_csv(items_file)

print("\nOrders Dataset Shape:", orders.shape)
print("Order Items Dataset Shape:", items.shape)


# ------------------------------------------------------------
# 3. Convert Date Columns
# ------------------------------------------------------------

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    if column in orders.columns:
        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce"
        )


# ------------------------------------------------------------
# 4. Merge Orders and Items
# ------------------------------------------------------------

df = orders.merge(
    items,
    on="order_id",
    how="left"
)

print("\nMerged Dataset Shape:", df.shape)


# ------------------------------------------------------------
# 5. Create Logistics KPIs
# ------------------------------------------------------------

df["delivery_time_days"] = (
    df["order_delivered_customer_date"]
    - df["order_purchase_timestamp"]
).dt.total_seconds() / (60 * 60 * 24)

df["estimated_delivery_days"] = (
    df["order_estimated_delivery_date"]
    - df["order_purchase_timestamp"]
).dt.total_seconds() / (60 * 60 * 24)

df["delivery_delay_days"] = (
    df["order_delivered_customer_date"]
    - df["order_estimated_delivery_date"]
).dt.total_seconds() / (60 * 60 * 24)

df["is_delayed"] = np.where(
    df["delivery_delay_days"] > 0,
    1,
    0
)


# ------------------------------------------------------------
# 6. Basic Data Cleaning
# ------------------------------------------------------------

df = df.drop_duplicates()

df = df[
    df["delivery_time_days"].notna()
]

df = df[
    df["delivery_time_days"] >= 0
]

df = df[
    df["freight_value"].notna()
]

print("\nCleaned Dataset Shape:", df.shape)


# ------------------------------------------------------------
# 7. Descriptive Statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

statistics = df[
    [
        "delivery_time_days",
        "delivery_delay_days",
        "freight_value",
        "price"
    ]
].describe()

print(statistics)

statistics.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "week3_descriptive_statistics.csv"
    )
)


# ------------------------------------------------------------
# 8. Central Tendency
# ------------------------------------------------------------

central_tendency = pd.DataFrame({
    "Metric": [
        "Delivery Time",
        "Delivery Delay",
        "Freight Value",
        "Product Price"
    ],
    "Mean": [
        df["delivery_time_days"].mean(),
        df["delivery_delay_days"].mean(),
        df["freight_value"].mean(),
        df["price"].mean()
    ],
    "Median": [
        df["delivery_time_days"].median(),
        df["delivery_delay_days"].median(),
        df["freight_value"].median(),
        df["price"].median()
    ],
    "Mode": [
        df["delivery_time_days"].mode().iloc[0],
        df["delivery_delay_days"].mode().iloc[0],
        df["freight_value"].mode().iloc[0],
        df["price"].mode().iloc[0]
    ]
})

print("\nCentral Tendency:")
print(central_tendency)

central_tendency.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "week3_central_tendency.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 9. Logistics KPI Summary
# ------------------------------------------------------------

total_orders = df["order_id"].nunique()

average_delivery_time = df[
    "delivery_time_days"
].mean()

average_freight = df[
    "freight_value"
].mean()

delay_rate = (
    df["is_delayed"].mean() * 100
)

total_freight = df[
    "freight_value"
].sum()

kpi_summary = pd.DataFrame({
    "KPI": [
        "Total Orders",
        "Average Delivery Time (Days)",
        "Average Freight Cost",
        "Delivery Delay Rate (%)",
        "Total Freight Cost"
    ],
    "Value": [
        total_orders,
        average_delivery_time,
        average_freight,
        delay_rate,
        total_freight
    ]
})

print("\n" + "=" * 70)
print("LOGISTICS KPI SUMMARY")
print("=" * 70)

print(kpi_summary)

kpi_summary.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "week3_logistics_kpis.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 10. Distribution - Delivery Time
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["delivery_time_days"],
    bins=40,
    kde=True
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "01_delivery_time_distribution.png"
    ),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 11. Box Plot - Freight Cost
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=df["freight_value"]
)

plt.title("Distribution and Outliers of Freight Cost")
plt.xlabel("Freight Value")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "02_freight_cost_boxplot.png"
    ),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 12. Bar Chart - Order Status
# ------------------------------------------------------------

status_counts = orders[
    "order_status"
].value_counts()

plt.figure(figsize=(10, 6))

status_counts.plot(
    kind="bar"
)

plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "03_order_status_distribution.png"
    ),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 13. Scatter Plot - Freight vs Product Price
# ------------------------------------------------------------

sample_data = df.sample(
    min(5000, len(df)),
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_data,
    x="price",
    y="freight_value",
    alpha=0.5
)

plt.title(
    "Relationship Between Product Price and Freight Cost"
)

plt.xlabel("Product Price")
plt.ylabel("Freight Value")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "04_price_vs_freight_scatter.png"
    ),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 14. Correlation Heatmap
# ------------------------------------------------------------

correlation_columns = [
    "price",
    "freight_value",
    "delivery_time_days",
    "delivery_delay_days"
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

correlation_matrix.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "week3_correlation_matrix.csv"
    )
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Logistics Variables Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "05_logistics_correlation_heatmap.png"
    ),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 15. Monthly Order Trend
# ------------------------------------------------------------

df["order_month"] = (
    df["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

monthly_orders = (
    df.groupby("order_month")["order_id"]
    .nunique()
)

plt.figure(figsize=(12, 6))

monthly_orders.plot(
    marker="o"
)

plt.title("Monthly Order Volume Trend")
plt.xlabel("Month")
plt.ylabel("Number of Orders")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "06_monthly_order_trend.png"
    ),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 16. Delay Analysis
# ------------------------------------------------------------

delayed_orders = df[
    df["is_delayed"] == 1
]

on_time_orders = df[
    df["is_delayed"] == 0
]

delay_summary = pd.DataFrame({
    "Delivery Category": [
        "On Time",
        "Delayed"
    ],
    "Orders": [
        on_time_orders["order_id"].nunique(),
        delayed_orders["order_id"].nunique()
    ]
})

print("\nDelivery Performance:")
print(delay_summary)

delay_summary.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "week3_delivery_delay_summary.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 17. Automated Analytical Insights
# ------------------------------------------------------------

insights = []

insights.append(
    f"Total unique orders analyzed: {total_orders:,.0f}."
)

insights.append(
    f"Average delivery time was approximately "
    f"{average_delivery_time:.2f} days."
)

insights.append(
    f"Average freight cost was approximately "
    f"{average_freight:.2f}."
)

insights.append(
    f"The observed delivery delay rate was "
    f"{delay_rate:.2f}%."
)

highest_corr = correlation_matrix[
    "delivery_time_days"
].drop("delivery_time_days").abs().idxmax()

insights.append(
    f"The variable with the strongest absolute correlation "
    f"with delivery time among the selected variables was "
    f"{highest_corr}."
)

with open(
    os.path.join(
        OUTPUT_PATH,
        "week3_analytical_insights.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 3 - LOGISTICS ANALYTICAL INSIGHTS\n"
    )

    file.write("=" * 60 + "\n\n")

    for number, insight in enumerate(
        insights,
        start=1
    ):
        file.write(
            f"{number}. {insight}\n"
        )


# ------------------------------------------------------------
# 18. Final Message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WEEK 3 ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated outputs:")
print("- Descriptive statistics")
print("- Central tendency analysis")
print("- Logistics KPI summary")
print("- Delivery time distribution")
print("- Freight cost box plot")
print("- Order status visualization")
print("- Price vs freight scatter plot")
print("- Correlation heatmap")
print("- Monthly order trend")
print("- Delivery delay summary")
print("- Analytical insights")

print(
    f"\nAll output files are saved in: {OUTPUT_PATH}/"
)
