WITH
order_facts AS (
    SELECT u.country, c.name AS category_name, o.order_id, o.total_amount
    FROM orders      o
    JOIN users       u  USING (user_id)
    JOIN order_items oi USING (order_id)
    JOIN products    p  USING (product_id)
    JOIN categories  c  USING (category_id)
    WHERE o.status NOT IN ('cancelled', 'refunded')
),
rollup_raw AS (
    SELECT
        country,
        category_name,
        COUNT(DISTINCT order_id)                                                AS orders,
        ROUND(SUM(total_amount), 2)                                             AS gmv,
        ROUND(AVG(total_amount), 2)                                             AS aov,
        COUNT(DISTINCT order_id) * 1.0
            / SUM(COUNT(DISTINCT order_id)) OVER ()                             AS order_share,
        CASE
            WHEN GROUPING(country) = 1       THEN 'grand_total'
            WHEN GROUPING(category_name) = 1 THEN 'country_subtotal'
            ELSE                                  'detail'
        END                                                                     AS row_type
    FROM order_facts
    GROUP BY ROLLUP (country, category_name)
)
SELECT
    COALESCE(country,       '── ALL COUNTRIES ──')  AS country,
    COALESCE(category_name, '  (all categories)')   AS category_name,
    orders,
    gmv,
    aov,
    ROUND(order_share * 100, 2)                     AS order_share_pct,
    ROUND(gmv / NULLIF(SUM(gmv) OVER (PARTITION BY country), 0) * 100, 1) AS gmv_share_in_country_pct,
    row_type
FROM rollup_raw
ORDER BY
    CASE row_type WHEN 'grand_total' THEN 2 WHEN 'country_subtotal' THEN 1 ELSE 0 END ASC,
    gmv DESC NULLS LAST,
    category_name;