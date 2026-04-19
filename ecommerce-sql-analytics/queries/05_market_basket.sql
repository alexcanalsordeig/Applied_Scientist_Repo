WITH
enriched_items AS (
    SELECT oi.order_id, oi.product_id, p.title AS product_title, c.name AS category_name
    FROM order_items oi
    JOIN products    p USING (product_id)
    JOIN categories  c USING (category_id)
),
product_pairs AS (
    SELECT
        a.product_id AS product_a_id, a.product_title AS product_a, a.category_name AS category_a,
        b.product_id AS product_b_id, b.product_title AS product_b, b.category_name AS category_b,
        COUNT(DISTINCT a.order_id) AS co_purchase_count
    FROM enriched_items a
    JOIN enriched_items b ON a.order_id = b.order_id AND a.product_id < b.product_id
    GROUP BY a.product_id, a.product_title, a.category_name, b.product_id, b.product_title, b.category_name
),
total_orders AS (SELECT COUNT(DISTINCT order_id) AS n FROM order_items),
pair_metrics AS (
    SELECT
        pp.*,
        t.n AS total_orders,
        ROUND(pp.co_purchase_count::numeric / t.n * 100, 3) AS support_pct,
        CASE WHEN pp.category_a <> pp.category_b THEN 'cross-category' ELSE 'same-category' END AS pair_type
    FROM product_pairs pp
    CROSS JOIN total_orders t
    WHERE pp.co_purchase_count >= 3
)
SELECT
    ROW_NUMBER() OVER (ORDER BY co_purchase_count DESC) AS rank,
    product_a, category_a, product_b, category_b, co_purchase_count, support_pct, pair_type
FROM pair_metrics
ORDER BY co_purchase_count DESC
LIMIT 50;