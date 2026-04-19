WITH
daily AS (
    SELECT
        order_date,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT user_id)  AS unique_buyers,
        SUM(total_amount)        AS daily_gmv,
        AVG(total_amount)        AS daily_aov
    FROM orders
    WHERE status NOT IN ('cancelled', 'refunded')
    GROUP BY order_date
),
rolling AS (
    SELECT
        order_date, orders, unique_buyers,
        ROUND(daily_gmv, 2) AS daily_gmv,
        ROUND(daily_aov, 2) AS daily_aov,
        ROUND(SUM(daily_gmv) OVER (ORDER BY order_date ROWS BETWEEN 6  PRECEDING AND CURRENT ROW), 2) AS gmv_7d,
        ROUND(SUM(daily_gmv) OVER (ORDER BY order_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW), 2) AS gmv_30d,
        SUM(unique_buyers)  OVER (ORDER BY order_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)      AS buyers_30d
    FROM daily
),
monthly AS (
    SELECT
        DATE_TRUNC('month', order_date)::date AS month,
        SUM(daily_gmv)       AS monthly_gmv,
        SUM(orders)          AS monthly_orders,
        ROUND(AVG(daily_aov), 2) AS avg_aov
    FROM daily
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT
    r.order_date, r.daily_gmv, r.gmv_7d, r.gmv_30d, r.buyers_30d,
    m.monthly_gmv,
    LAG(m.monthly_gmv) OVER (ORDER BY m.month) AS prev_month_gmv,
    ROUND(
        (m.monthly_gmv - LAG(m.monthly_gmv) OVER (ORDER BY m.month))
        / NULLIF(LAG(m.monthly_gmv) OVER (ORDER BY m.month), 0) * 100
    , 1) AS mom_growth_pct
FROM rolling r
JOIN monthly m ON DATE_TRUNC('month', r.order_date)::date = m.month
ORDER BY r.order_date;