-- =============================================================================
-- analysis.sql  —  MDM pipeline: analytical SQL over the golden layer
-- -----------------------------------------------------------------------------
-- Runs against the typed `golden_master` table built by schema.sql.
-- Each query is annotated with what it demonstrates and the one-line way to
-- describe it. Progression: aggregation -> filtered aggregation -> window
-- functions -> self-join hierarchy -> query tuning.
--
-- Active database: mdm
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Q1 — Revenue and headcount by industry (GROUP BY aggregation)
-- -----------------------------------------------------------------------------
-- Collapses rows into one summary row per industry.
-- SUM/AVG/ROUND only work because revenue is a real NUMERIC (not TEXT) — the
-- payoff of typing in the schema layer.
-- NOTE (data quality): this surfaces a standardisation gap — "Tech" and
-- "Technology" appear as separate industries, i.e. unmerged master-data values.
SELECT
    industry,
    COUNT(*)                AS company_count,
    SUM(revenue)            AS total_revenue,
    ROUND(AVG(revenue), 2)  AS avg_revenue,
    SUM(employees)          AS total_employees
FROM golden_master
GROUP BY industry
ORDER BY total_revenue DESC NULLS LAST;


-- -----------------------------------------------------------------------------
-- Q2 — Revenue by country, multi-company countries only (WHERE vs HAVING)
-- -----------------------------------------------------------------------------
-- Demonstrates the execution order FROM -> WHERE -> GROUP BY -> HAVING:
--   WHERE  filters individual rows BEFORE grouping (drops null-revenue rows)
--   HAVING filters groups AFTER aggregation (keeps countries with >1 company)
-- COUNT(*) can only go in HAVING, never WHERE — the count doesn't exist until
-- grouping has happened.
SELECT
    country,
    COUNT(*)      AS company_count,
    SUM(revenue)  AS total_revenue
FROM golden_master
WHERE revenue IS NOT NULL
GROUP BY country
HAVING COUNT(*) > 1
ORDER BY total_revenue DESC;


-- -----------------------------------------------------------------------------
-- Q3 — Rank companies by revenue within their industry (window function)
-- -----------------------------------------------------------------------------
-- RANK() OVER (PARTITION BY industry ORDER BY revenue DESC):
--   PARTITION BY = grouping (one window per industry) — but rows are NOT
--                  collapsed; every company is preserved.
--   ORDER BY     = ranking order within each window.
-- Contrast with Q1: GROUP BY returns one row per industry; a window function
-- returns EVERY row tagged with its within-group rank.
-- Tie behaviour to know: RANK() leaves gaps, DENSE_RANK() no gaps,
-- ROW_NUMBER() assigns a unique number even on ties.
SELECT
    company_name,
    industry,
    revenue,
    RANK() OVER (PARTITION BY industry ORDER BY revenue DESC) AS rank_in_industry
FROM golden_master
WHERE revenue IS NOT NULL
ORDER BY industry, rank_in_industry;


-- -----------------------------------------------------------------------------
-- Q4 — Each company's revenue as a % of its industry total (windowed aggregate)
-- -----------------------------------------------------------------------------
-- SUM(revenue) OVER (PARTITION BY industry) computes the industry total and
-- attaches it to EVERY company row (not collapsed), enabling a per-row ratio.
-- 100.0 (not 100) forces decimal division rather than integer division.
-- Sanity check: percentages sum to 100 within each industry.
-- This is impossible in a single pass with plain GROUP BY (would need a
-- subquery or self-join to get the total back alongside the detail rows).
SELECT
    company_name,
    industry,
    revenue,
    SUM(revenue) OVER (PARTITION BY industry)                        AS industry_total,
    ROUND(
        100.0 * revenue / SUM(revenue) OVER (PARTITION BY industry),
        1
    )                                                                AS pct_of_industry
FROM golden_master
WHERE revenue IS NOT NULL
ORDER BY industry, pct_of_industry DESC;


-- -----------------------------------------------------------------------------
-- Q5 — Parent/subsidiary hierarchy (SELF-JOIN)
-- -----------------------------------------------------------------------------
-- golden_master joined to itself with two aliases (child / parent).
-- Join condition child.parent_golden_id = parent.golden_id walks the hierarchy.
-- INNER JOIN returns only companies that HAVE a parent (the subsidiaries).
-- Swap to LEFT JOIN to list every company, with parent columns NULL for the
-- independents.
-- Ties directly back to MDM: the matching phase LINKED (not merged) these as
-- distinct golden records — e.g. Acme Cloud Services -> Acme Corp.
SELECT
    child.company_name   AS subsidiary,
    child.golden_id      AS subsidiary_id,
    parent.company_name  AS parent_company,
    parent.golden_id     AS parent_id,
    parent.industry      AS parent_industry
FROM golden_master AS child
JOIN golden_master AS parent
    ON child.parent_golden_id = parent.golden_id
ORDER BY parent.company_name;


-- =============================================================================
-- QUERY TUNING  —  indexing + EXPLAIN ANALYZE
-- -----------------------------------------------------------------------------
-- Honest framing: on 12 rows an index gives no real speedup — the cost-based
-- planner correctly prefers a sequential scan. The point is to demonstrate the
-- technique and the reasoning, not a speedup on toy data.
-- =============================================================================


-- Q6a — Baseline plan BEFORE (relying on) an index.
-- EXPLAIN shows the plan; ANALYZE actually runs it and reports real timings.
-- Expect: "Seq Scan" with "Filter: (industry = 'Technology')" and
-- "Rows Removed by Filter: 10" — it reads all 12 rows and keeps the matches.
-- (Do NOT run EXPLAIN ANALYZE on UPDATE/DELETE casually — ANALYZE executes it.)
EXPLAIN ANALYZE
SELECT * FROM golden_master WHERE industry = 'Technology';


-- Q6b — Create an index on the filtered column.
-- IF NOT EXISTS makes the script idempotent (safe to re-run).
-- A B-tree index keeps `industry` values sorted so lookups can jump straight to
-- matches instead of scanning every row. Trade-off: faster reads, but slower
-- writes and extra storage — so index columns you filter/join on, not everything.
CREATE INDEX IF NOT EXISTS idx_golden_industry ON golden_master (industry);


-- Q6c — Plan AFTER the index exists.
-- Expect: STILL "Seq Scan" — creating an index does not force its use. The
-- planner evaluates cost per query and, for 12 rows, correctly judges the scan
-- cheaper than an index lookup. This is the cost-based optimiser working, not a
-- failure. At scale (large table + selective filter) this flips to an Index Scan.
EXPLAIN ANALYZE
SELECT * FROM golden_master WHERE industry = 'Technology';


-- Q6d — Prove the index is valid and usable by forcing the planner's hand.
-- enable_seqscan = off is a DIAGNOSTIC toggle (never leave off in production).
-- With seq scans disabled the planner falls back to the index, so the plan
-- flips to "Index Scan using idx_golden_industry" (or a Bitmap Index Scan).
-- This confirms the seq scan earlier was a COST decision, not a broken index.
SET enable_seqscan = off;

EXPLAIN ANALYZE
SELECT * FROM golden_master WHERE industry = 'Technology';

SET enable_seqscan = on;   -- restore normal cost-based behaviour


-- Q6e — Inspect all indexes on the table.
-- Returns three: golden_master_pkey (auto, from PRIMARY KEY on golden_id),
-- golden_master_match_group_id_key (auto, from the UNIQUE constraint), and
-- idx_golden_industry (the one we added). Talking point: PKs and UNIQUE
-- constraints get their indexes created automatically.
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'golden_master';