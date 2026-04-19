WITH
category_funnel AS (
    SELECT
        c.name AS category_name,
        COUNT(DISTINCT e.user_id) FILTER (WHERE e.event_type = 'view')         AS viewers,
        COUNT(DISTINCT e.user_id) FILTER (WHERE e.event_type = 'add_to_cart')  AS add_to_carts,
        COUNT(DISTINCT e.user_id) FILTER (WHERE e.event_type = 'purchase')     AS purchasers,
        COUNT(DISTINCT p.product_id)                                            AS product_count
    FROM events e
    JOIN products   p USING (product_id)
    JOIN categories c USING (category_id)
    GROUP BY c.name
),
funnel_rates AS (
    SELECT
        category_name, product_count, viewers, add_to_carts, purchasers,
        viewers - purchasers AS lost_buyers,
        ROUND(add_to_carts::numeric / NULLIF(viewers,      0) * 100, 1) AS view_to_cart_pct,
        ROUND(purchasers::numeric   / NULLIF(add_to_carts, 0) * 100, 1) AS cart_to_purchase_pct,
        ROUND(purchasers::numeric   / NULLIF(viewers,      0) * 100, 1) AS overall_conversion_pct
    FROM category_funnel
)
SELECT
    category_name, product_count, viewers, add_to_carts, purchasers, lost_buyers,
    view_to_cart_pct, cart_to_purchase_pct, overall_conversion_pct,
    RANK() OVER (ORDER BY lost_buyers DESC) AS lost_buyer_rank,
    CASE WHEN viewers < 100 THEN 'low-traffic' ELSE 'sufficient volume' END AS data_quality_flag
FROM funnel_rates
ORDER BY overall_conversion_pct DESC;