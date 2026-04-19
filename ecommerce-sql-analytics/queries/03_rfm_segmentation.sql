WITH
rfm_raw AS (
    SELECT
        user_id,
        CURRENT_DATE - MAX(order_date)::date AS recency_days,
        COUNT(DISTINCT order_id)             AS frequency,
        SUM(total_amount)                    AS monetary
    FROM orders
    WHERE status NOT IN ('cancelled', 'refunded')
    GROUP BY user_id
),
rfm_scores AS (
    SELECT
        user_id, recency_days, frequency, ROUND(monetary, 2) AS monetary,
        6 - NTILE(5) OVER (ORDER BY recency_days ASC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC)        AS f_score,
        NTILE(5) OVER (ORDER BY monetary   ASC)       AS m_score
    FROM rfm_raw
),
rfm_segments AS (
    SELECT
        user_id, recency_days, frequency, monetary, r_score, f_score, m_score,
        ROUND((r_score + f_score + m_score)::numeric / 3, 2) AS rfm_score,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Gold'
            WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Silver'
            WHEN r_score >= 4 AND f_score <= 2                  THEN 'New High-Value'
            WHEN r_score <= 2 AND f_score >= 4                  THEN 'At-Risk VIP'
            WHEN r_score <= 2 AND f_score <= 2                  THEN 'Churned'
            ELSE                                                     'Bronze'
        END AS customer_segment
    FROM rfm_scores
)
SELECT
    s.user_id, u.email, u.country,
    s.recency_days, s.frequency, s.monetary,
    s.r_score, s.f_score, s.m_score, s.rfm_score, s.customer_segment
FROM rfm_segments s
JOIN users u USING (user_id)
ORDER BY s.rfm_score DESC, s.monetary DESC;