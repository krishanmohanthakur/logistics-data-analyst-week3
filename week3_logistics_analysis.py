"""
Yuva Intern - Logistics Data Analyst Internship
Week 3: Advanced Data Analysis and Visualization in Logistics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
OUTPUT = "week3_outputs"
os.makedirs(OUTPUT, exist_ok=True)

N = 1500
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

df = pd.DataFrame({
    "shipment_date": np.random.choice(dates, N),
    "region": np.random.choice(["North","South","East","West","Central"], N,
                               p=[.22,.20,.18,.22,.18]),
    "transport_mode": np.random.choice(["Road","Rail","Air","Sea"], N,
                                       p=[.58,.15,.12,.15]),
    "shipment_volume_kg": np.clip(np.random.gamma(3.2,35,N),5,400),
    "distance_km": np.clip(np.random.gamma(2.2,280,N),20,2500)
})

mode_base = df["transport_mode"].map({"Road":1.8,"Rail":3.2,"Air":0.8,"Sea":5.5})
warehouse = np.clip(np.random.normal(1.8,0.8,N),0.3,5)

df["warehouse_processing_days"] = warehouse
df["actual_delivery_days"] = np.clip(
    df["distance_km"]/450 + mode_base + warehouse + np.random.normal(0,1.3,N),
    1, 18
)
df["promised_delivery_days"] = np.clip(
    np.ceil(df["actual_delivery_days"] + np.random.normal(0.8,1.1,N)),2,20
)
df["delivery_delay_days"] = np.maximum(
    df["actual_delivery_days"] - df["promised_delivery_days"], 0
)
df["delayed"] = (df["delivery_delay_days"] > .2).astype(int)

df["transportation_cost"] = np.clip(
    35 + df["distance_km"]*.16 + df["shipment_volume_kg"]*.75
    + df["distance_km"]*df["shipment_volume_kg"]*.00008
    + np.where(df["transport_mode"]=="Air",180,
        np.where(df["transport_mode"]=="Sea",-20,
            np.where(df["transport_mode"]=="Rail",30,0)))
    + np.random.normal(0,35,N), 40, None
)

df["order_value"] = np.clip(
    df["shipment_volume_kg"] * np.random.uniform(7,18,N)
    + np.random.normal(500,180,N), 100, None
)

# EDA
print(df.describe())
print("\nDelay rate:", df["delayed"].mean()*100)
print("\nCorrelation matrix:")
print(df[["shipment_volume_kg","distance_km","actual_delivery_days",
          "transportation_cost","delivery_delay_days","order_value"]].corr())

# Six visualizations
plt.figure(figsize=(10,6))
plt.hist(df["actual_delivery_days"], bins=25)
plt.title("Distribution of Actual Delivery Time")
plt.xlabel("Actual Delivery Time (Days)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/01_delivery_time_distribution.png", dpi=180)
plt.close()

modes = ["Road","Rail","Air","Sea"]
plt.figure(figsize=(10,6))
plt.boxplot([df.loc[df["transport_mode"]==m,"transportation_cost"] for m in modes],
            labels=modes)
plt.title("Transportation Cost by Transport Mode")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/02_cost_by_mode_boxplot.png", dpi=180)
plt.close()

monthly = df.set_index("shipment_date").resample("ME").size()
plt.figure(figsize=(11,6))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Monthly Shipment Volume Trend")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/03_monthly_shipment_trend.png", dpi=180)
plt.close()

plt.figure(figsize=(10,6))
for m in modes:
    s = df[df["transport_mode"]==m].sample(
        min(220, len(df[df["transport_mode"]==m])), random_state=42)
    plt.scatter(s["distance_km"], s["transportation_cost"], alpha=.45, label=m)
plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/04_distance_vs_cost_scatter.png", dpi=180)
plt.close()

corr = df[["shipment_volume_kg","distance_km","actual_delivery_days",
           "transportation_cost","delivery_delay_days","order_value"]].corr()
plt.figure(figsize=(9,7))
plt.imshow(corr.values, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center")
plt.title("Correlation Heatmap of Logistics Variables")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/05_correlation_heatmap.png", dpi=180)
plt.close()

delay_mode = df.groupby("transport_mode")["delayed"].mean()*100
plt.figure(figsize=(9,6))
plt.bar(delay_mode.index, delay_mode.values)
plt.title("Delivery Delay Rate by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Delay Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/06_delay_rate_by_mode.png", dpi=180)
plt.close()

df.to_csv("week3_simulated_logistics_dataset.csv", index=False)

print("\nWeek 3 analysis completed successfully.")
