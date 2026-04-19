import pandas as pd

df = pd.read_csv("output/01_cohort_retention.csv", encoding="utf-16")

pivot = df.pivot(
    index="cohort_month",
    columns="months_since_signup",
    values="retention_pct"
)

# Rename columns to be readable
pivot.columns = [f"Month {c}" for c in pivot.columns]
pivot.index.name = "Cohort"

# Style it as a heatmap
styled = pivot.style\
    .format("{:.1f}%")\
    .background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=100)\
    .set_caption("Cohort Retention Heatmap")\
    .set_properties(**{"text-align": "center"})

# Export to HTML
styled.to_html("output/cohort_heatmap.html")
print("Saved to output/cohort_heatmap.html")