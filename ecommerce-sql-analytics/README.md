# E-commerce SQL Analytics Platform

A self-contained analytics project built on a realistic synthetic e-commerce dataset.
Covers the core SQL patterns used daily in data and analytics engineering roles:
cohort analysis, funnel metrics, RFM segmentation, rolling revenue, market basket, and geo roll-ups.

---

## Project Structure

```
ecommerce-sql-analytics/
├── schema.sql          # PostgreSQL schema — tables, indexes, views
├── generate_data.py    # Synthetic data generator (500 users, 8k orders, 40k events)
├── queries/
│   ├── 01_cohort_retention.sql   # Monthly cohort retention matrix
│   ├── 02_conversion_funnel.sql  # View → cart → purchase funnel
│   ├── 03_rfm_segmentation.sql   # Recency / Frequency / Monetary scoring
│   ├── 04_rolling_revenue.sql    # 7-day and 30-day rolling GMV
│   ├── 05_market_basket.sql      # Product co-purchase (association rules)
│   └── 06_geo_revenue_rollup.sql # Revenue breakdown by country
├── visualize_all.py    # Matplotlib charts for all six analyses
├── visualize_cohorts.py # Standalone cohort heatmap
├── export_all.py       # Export results to CSV / Excel
└── output/             # Generated charts and exports
```

---

## Dataset

Fully synthetic data generated with `generate_data.py` using Faker and psycopg2.
Designed to mirror real e-commerce distributions:

| Table         | Rows      | Notes                                              |
|---------------|-----------|----------------------------------------------------|
| `users`       | 500       | Signup dates over 2 years, US-heavy country mix    |
| `categories`  | 8         | Amazon-style hierarchy                             |
| `products`    | 200       | Prices follow per-category realistic distributions |
| `orders`      | 8,000     | Power-law user distribution + seasonal weighting  |
| `order_items` | ~24,000   | 1–6 items per order                                |
| `events`      | ~40,000   | Clickstream: view → add_to_cart → purchase funnel  |
| `reviews`     | ~3,000    | Only from verified purchasers, skewed high         |

Key design choices:
- **Power-law users**: top buyers purchase ~100× more than median users, matching real e-commerce patterns.
- **Seasonal weighting**: Nov–Dec orders 3× baseline; Jan–Feb at 0.7× (post-holiday slump).
- **Funnel realism**: organic browse events with realistic drop-off rates at each stage.

---

## Analyses

### 1. Cohort Retention
Monthly cohort matrix showing what % of users who signed up in month M returned to buy in months M+1, M+2, etc.
Useful for measuring product stickiness and identifying drop-off points.

### 2. Conversion Funnel
Session-level funnel from `view` → `add_to_cart` → `checkout_start` → `purchase`.
Computes conversion rate at each step and overall funnel efficiency.

### 3. RFM Segmentation
Scores every user on Recency, Frequency, and Monetary value. Segments into Champions, Loyal, At Risk, and Lost customers using quintile scoring.

### 4. Rolling Revenue
7-day and 30-day rolling GMV using window functions. Smooths out day-of-week noise and surfaces true revenue trends.

### 5. Market Basket (Association Rules)
Identifies product pairs that frequently co-occur in the same order. Computes support, confidence, and lift — the core metrics for cross-sell recommendations.

### 6. Geo Revenue Roll-up
Revenue, AOV (average order value), and order count broken down by country. Ranks markets by GMV contribution.

---

## How to Run

**Requirements:** PostgreSQL 15+, Python 3.9+

```bash
# 1. Install Python dependencies
pip install psycopg2-binary faker matplotlib pandas openpyxl

# 2. Create the database
createdb ecommerce_analytics

# 3. Apply the schema
psql -d ecommerce_analytics -f schema.sql

# 4. Generate synthetic data (~30 seconds)
#    Update DB_CONFIG in generate_data.py with your credentials first
python generate_data.py

# 5. Run any query
psql -d ecommerce_analytics -f queries/01_cohort_retention.sql

# 6. Generate all charts
python visualize_all.py

# 7. Export results to CSV/Excel
python export_all.py
```

---

## Stack

- **Database**: PostgreSQL 15
- **Data generation**: Python · psycopg2 · Faker
- **Analysis**: SQL window functions, CTEs, lateral joins
- **Visualisation**: Python · Matplotlib · Pandas
- **Export**: openpyxl (Excel), csv

---

## Skills Demonstrated

- Dimensional data modelling (star schema with fact and dimension tables)
- Advanced SQL: window functions, CTEs, self-joins, JSONB
- Index design for analytics workloads
- Realistic synthetic data generation with controlled statistical distributions
- Core analytics patterns used in e-commerce and product analytics roles
