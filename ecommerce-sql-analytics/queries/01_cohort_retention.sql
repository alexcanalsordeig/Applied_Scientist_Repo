-- ============================================================
-- 01_cohort_retention.sql  (enhanced)
-- ============================================================

WITH

cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', signup_date)::date AS cohort_month
    FROM users
),

cohort_size AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT user_id) AS total_users
    FROM cohorts
    GROUP BY cohort_month
),

user_activity AS (
    SELECT
        c.user_id,
        c.cohort_month,
        (DATE_PART('year',  DATE_TRUNC('month', o.ordered_at)) -
         DATE_PART('year',  c.cohort_month)) * 12
        + DATE_PART('month', DATE_TRUNC('month', o.ordered_at))
        - DATE_PART('month', c.cohort_month)  AS period_number
    FROM cohorts c
    JOIN orders  o ON o.user_id = c.user_id
    WHERE o.status NOT IN ('cancelled', 'refunded')
),

-- Generate every (cohort, period) combination so zeros appear explicitly
-- instead of missing rows. This is critical for a correct retention matrix.
cohort_periods AS (
    SELECT
        cs.cohort_month,
        p.period_number
    FROM cohort_size cs
    CROSS JOIN (SELECT generate_series(0, 6) AS period_number) AS p
),

retention_raw AS (
    SELECT
        cohort_month,
        period_number,
        COUNT(DISTINCT user_id) AS active_users
    FROM user_activity
    WHERE period_number BETWEEN 0 AND 6
    GROUP BY cohort_month, period_number
),

retention_filled AS (
    -- Left join ensures every period appears, even with 0 active users
    SELECT
        cp.cohort_month,
        cp.period_number,
        COALESCE(r.active_users, 0) AS active_users
    FROM cohort_periods cp
    LEFT JOIN retention_raw r USING (cohort_month, period_number)
)

SELECT
    r.cohort_month,
    s.total_users                                             AS cohort_size,
    r.period_number                                           AS months_since_signup,
    r.active_users,
    ROUND(r.active_users::numeric / NULLIF(s.total_users, 0) * 100, 1)
                                                              AS retention_pct,

    -- How much did retention drop vs the previous period?
    -- Useful for spotting which month has the steepest cliff.
    ROUND(
        r.active_users::numeric / NULLIF(s.total_users, 0) * 100
        - LAG(r.active_users::numeric / NULLIF(s.total_users, 0) * 100)
            OVER (PARTITION BY r.cohort_month ORDER BY r.period_number)
    , 1)                                                      AS retention_delta_ppt

FROM retention_filled r
JOIN cohort_size       s USING (cohort_month)
ORDER BY r.cohort_month, r.period_number;