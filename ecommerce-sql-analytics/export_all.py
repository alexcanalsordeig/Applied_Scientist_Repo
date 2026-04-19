import psycopg2
import csv
import os

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "ecommerce_analytics",
    "user":     "postgres",
    "password": "sql_project",
}

queries = {
    "01_cohort_retention":  open("queries/01_cohort_retention.sql").read(),
    "02_conversion_funnel": open("queries/02_conversion_funnel.sql").read(),
    "03_rfm_segmentation":  open("queries/03_rfm_segmentation.sql").read(),
    "04_rolling_revenue":   open("queries/04_rolling_revenue.sql").read(),
    "05_market_basket":     open("queries/05_market_basket.sql").read(),
    "06_geo_revenue_rollup":open("queries/06_geo_revenue_rollup.sql").read(),
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

os.makedirs("output", exist_ok=True)

for name, sql in queries.items():
    print(f"Exporting {name}...")
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    with open(f"output/{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"  {len(rows)} rows saved")

cur.close()
conn.close()
print("\nAll exports done!")