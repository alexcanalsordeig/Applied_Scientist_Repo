What to write in your Notion portfolio:

Business question: Which signup cohorts retain best over 6 months?
Method: Grouped users by signup month, then tracked what % placed at least one order in each of the following 6 months. Built using CTEs, window functions, and GENERATE_SERIES to ensure zero-retention months appear explicitly rather than as missing rows.
Key findings:

The 0.0% cells in recent cohorts (Aug-Dec 2024) are expected — those users haven't had 6 months yet, not a data quality issue.
November 2023 cohort stands out with 77.8% retention at Month 0 — likely holiday shoppers who bought immediately.
Most cohorts show non-linear retention, bouncing between months rather than steadily declining, which suggests repeat buyers are opportunistic (promotions, seasonality) rather than habitual.
No single cohort consistently outperforms others, which would suggest the re-engagement campaign hypothesis needs more segmentation — by country or acquisition channel — to find a real signal.



01 — Cohort Retention Heatmap

"I tracked what percentage of users from each signup month made at least one purchase in each of the following 6 months. The key technique is GENERATE_SERIES to force zero-retention months to appear explicitly rather than as missing rows — a common data quality trap. The dark red cells in recent cohorts aren't churn, they're users who simply haven't had 6 months yet."


02 — Conversion Funnel by Category

"I measured view → cart → purchase rates per category using COUNT DISTINCT with FILTER, which is cleaner than multiple subqueries. The interesting signal is the gap between view-to-cart and cart-to-purchase — a high view-to-cart but low cart-to-purchase suggests price anxiety, which would inform whether to add a price-match CTA or instalment payment option."


03 — RFM Segmentation

"I scored every customer on Recency, Frequency and Monetary value using NTILE(5) window functions, then combined the scores into named segments. This is the standard framework used in CRM and retention marketing. The output feeds directly into a re-engagement campaign — Gold customers get loyalty rewards, At-Risk VIPs get win-back offers."


04 — Rolling Revenue

"I built 7-day and 30-day rolling GMV using ROWS BETWEEN window frames, plus month-over-month growth using LAG. The 7-day line is what you'd put on a live dashboard to smooth weekly noise. The 30-day line is what leadership uses for reporting. The MoM column separates real growth from seasonal spikes."


05 — Market Basket

"I used a self-join on order_items where product_a_id < product_b_id to generate all product pairs within the same order without duplicates. I then calculated support percentage — the share of all orders containing both products — which is the foundation of Apriori association rule mining. The output feeds directly into a recommendation engine."


06 — Geographic Revenue Rollup

"I used GROUP BY ROLLUP to produce country-level subtotals and a grand total in a single query pass — equivalent to three separate GROUP BY queries combined with UNION ALL, but far more efficient. The GROUPING() function tells you which rows are subtotals vs detail rows, which is essential when consuming this output in Python or a BI tool."


The one sentence that ties it all together — say this at the start:

"This project covers the six analytical questions that come up most in e-commerce data roles: retention, funnel analysis, customer segmentation, revenue trends, product recommendations, and geographic breakdown. Each query is production-grade — I used window functions, CTEs, and PostgreSQL-specific features throughout rather than simple aggregations."