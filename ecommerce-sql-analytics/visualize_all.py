import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── 01 Cohort Retention Heatmap ──────────────────────────────
df1 = pd.read_csv("output/01_cohort_retention.csv", encoding="utf-8")

pivot = df1.pivot(
    index="cohort_month",
    columns="months_since_signup",
    values="retention_pct"
)
pivot.columns = [f"Month {c}" for c in pivot.columns]
pivot.index.name = "Cohort"

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8, color="black")

plt.colorbar(im, ax=ax, label="Retention %")
ax.set_title("Cohort Retention Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/01_cohort_retention.png", dpi=150)
plt.close()
print("Saved 01_cohort_retention.png")


# ── 02 Conversion Funnel ─────────────────────────────────────
df2 = pd.read_csv("output/02_conversion_funnel.csv", encoding="utf-8")

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(df2))
width = 0.25

bars1 = ax.bar([i - width for i in x], df2["view_to_cart_pct"],     width, label="View → Cart %",     color="#4C9BE8")
bars2 = ax.bar([i         for i in x], df2["cart_to_purchase_pct"], width, label="Cart → Purchase %", color="#F4A261")
bars3 = ax.bar([i + width for i in x], df2["overall_conversion_pct"],width, label="Overall %",         color="#2A9D8F")

ax.set_xticks(list(x))
ax.set_xticklabels(df2["category_name"], rotation=30, ha="right")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Funnel by Category")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
plt.tight_layout()
plt.savefig("output/02_conversion_funnel.png", dpi=150)
plt.close()
print("Saved 02_conversion_funnel.png")

# ── 03 RFM Segmentation ──────────────────────────────────────
df3 = pd.read_csv("output/03_rfm_segmentation.csv", encoding="utf-8")

segment_summary = df3.groupby("customer_segment").agg(
    user_count=("user_id", "count"),
    avg_spend=("monetary", "mean"),
    total_revenue=("monetary", "sum")
).reset_index().sort_values("total_revenue", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ["#2A9D8F","#4C9BE8","#F4A261","#E76F51","#264653","#E9C46A"]

axes[0].bar(segment_summary["customer_segment"], segment_summary["user_count"], color=colors)
axes[0].set_title("Users per Segment")
axes[0].set_ylabel("User Count")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(segment_summary["customer_segment"], segment_summary["total_revenue"], color=colors)
axes[1].set_title("Total Revenue per Segment")
axes[1].set_ylabel("Revenue ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].tick_params(axis="x", rotation=30)

plt.suptitle("RFM Customer Segmentation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/03_rfm_segmentation.png", dpi=150)
plt.close()
print("Saved 03_rfm_segmentation.png")

# ── 04 Rolling Revenue ───────────────────────────────────────
df4 = pd.read_csv("output/04_rolling_revenue.csv", encoding="utf-8")
df4["order_date"] = pd.to_datetime(df4["order_date"])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df4["order_date"], df4["daily_gmv"], alpha=0.3, color="#4C9BE8", linewidth=1,   label="Daily GMV")
ax.plot(df4["order_date"], df4["gmv_7d"],    alpha=0.7, color="#F4A261", linewidth=1.5, label="7-day Rolling")
ax.plot(df4["order_date"], df4["gmv_30d"],              color="#E76F51", linewidth=2,   label="30-day Rolling")

ax.set_title("Rolling Revenue Over Time")
ax.set_ylabel("GMV ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
plt.tight_layout()
plt.savefig("output/04_rolling_revenue.png", dpi=150)
plt.close()
print("Saved 04_rolling_revenue.png")

# ── 05 Market Basket ─────────────────────────────────────────
df5 = pd.read_csv("output/05_market_basket.csv", encoding="utf-8")
top15 = df5.head(15).copy()
top15["pair"] = top15["product_a"].str[:20] + " + " + top15["product_b"].str[:20]

fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#2A9D8F" if t == "cross-category" else "#4C9BE8" for t in top15["pair_type"]]
ax.barh(top15["pair"][::-1], top15["co_purchase_count"][::-1], color=colors[::-1])
ax.set_xlabel("Co-purchase Count")
ax.set_title("Top 15 Product Pairs Bought Together\n(green = cross-category, blue = same-category)")
plt.tight_layout()
plt.savefig("output/05_market_basket.png", dpi=150)
plt.close()
print("Saved 05_market_basket.png")

# ── 06 Geo Revenue Rollup ────────────────────────────────────
df6 = pd.read_csv("output/06_geo_revenue_rollup.csv", encoding="utf-8")
country_totals = df6[df6["row_type"] == "country_subtotal"].copy()
country_totals = country_totals.sort_values("gmv", ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(country_totals["country"], country_totals["gmv"], color="#4C9BE8")
axes[0].set_title("GMV by Country (Top 10)")
axes[0].set_ylabel("GMV ($)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(country_totals["country"], country_totals["aov"], color="#F4A261")
axes[1].set_title("Average Order Value by Country")
axes[1].set_ylabel("AOV ($)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].tick_params(axis="x", rotation=30)

plt.suptitle("Geographic Revenue Breakdown", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/06_geo_revenue_rollup.png", dpi=150)
plt.close()
print("Saved 06_geo_revenue_rollup.png")

print("\nAll visualizations saved to output/")