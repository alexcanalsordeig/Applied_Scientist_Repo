-- =============================================================================
-- schema.sql  —  MDM pipeline: PostgreSQL landing + core layer
-- -----------------------------------------------------------------------------
-- Purpose:  Load the golden-master output of the MDM pipeline into PostgreSQL
--           using a two-layer pattern:
--             1. a STAGING layer (all TEXT) that ingests the CSV without any
--                risk of type-coercion failure, and
--             2. a typed CORE layer (golden_master) where the database enforces
--                types, keys, and referential integrity.
--
-- Design note (the one-line version to say out loud):
--   "I stage everything as TEXT so ingestion never fails, then cast into a typed
--    core layer so the database enforces validity, math/dates work, sorting is
--    correct, and queries can be indexed and optimised."
--
-- Run order: top to bottom. Assumes the active database is `mdm`.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- STEP 0 — Database
-- -----------------------------------------------------------------------------
-- Created once, from a session connected to the default `postgres` database.
-- A dedicated database keeps this project isolated from anything else on the
-- server (cleaner to reason about, cleaner to describe).
--
--   CREATE DATABASE mdm;
--
-- (Left commented because you cannot create the DB from inside itself; it is
--  recorded here so the build is fully documented.)


-- -----------------------------------------------------------------------------
-- STEP 1 — Staging table (immutable landing zone, everything TEXT)
-- -----------------------------------------------------------------------------
-- Every column is TEXT on purpose. TEXT makes no promise about content, so the
-- CSV import cannot fail on values like "450.0" or on empty cells. This table
-- is a faithful, untouched copy of exactly what arrived in the file — if a load
-- ever looks wrong, we can always inspect the raw original here.
DROP TABLE IF EXISTS stg_golden_master;

CREATE TABLE stg_golden_master (
    match_group_id    TEXT,
    company_name      TEXT,
    name_from         TEXT,
    industry          TEXT,
    country           TEXT,
    city              TEXT,
    employees         TEXT,
    employees_from    TEXT,
    revenue           TEXT,
    revenue_currency  TEXT,
    revenue_from      TEXT,
    vat_id            TEXT,
    email_domain      TEXT,
    last_activity     TEXT,
    sourced_from      TEXT,
    source_records    TEXT,
    golden_id         TEXT,
    parent_golden_id  TEXT
);


-- -----------------------------------------------------------------------------
-- STEP 2 — Load the CSV into staging
-- -----------------------------------------------------------------------------
-- Done via DBeaver's Import Data wizard (right-click stg_golden_master ->
-- Import Data -> CSV). The wizard reads the header row (Header position: top)
-- and maps the 18 columns by name. Because staging is all TEXT, every row loads.
--
-- Equivalent client-side command (psql), recorded for reproducibility:
--   \copy stg_golden_master FROM 'golden_master.csv' WITH (FORMAT csv, HEADER true);
--
-- Verify:
--   SELECT COUNT(*) FROM stg_golden_master;                               -- expect 12
--   SELECT * FROM stg_golden_master WHERE match_group_id = 'match_group_id'; -- expect 0 (no header leak)


-- -----------------------------------------------------------------------------
-- STEP 3 — Typed core table (created empty; shape only)
-- -----------------------------------------------------------------------------
-- A brand-new, SEPARATE table with correct types from birth. We do NOT alter the
-- staging columns in place — keeping raw and clean layers physically separate
-- means the transformation is an explicit, re-runnable step, not a destructive
-- one-time mutation.
--
-- Why each type matters:
--   employees        INTEGER        -> numeric sorting + AVG/SUM; rejects non-numbers
--   revenue          NUMERIC(18,2)  -> EXACT decimal money (no float rounding drift)
--   revenue_currency VARCHAR(3)     -> currency codes are 3 chars (EUR, USD)
--   last_activity    DATE           -> date arithmetic (intervals, "last 6 months")
--
-- Keys / constraints:
--   golden_id        PRIMARY KEY    -> the stable master identifier, unique + not null
--   match_group_id   NOT NULL UNIQUE-> each match group maps to exactly one golden record
--   company_name     NOT NULL       -> a golden record must have a name
--
-- The self-referencing hierarchy FK is added AFTER loading (Step 4) so we don't
-- have to worry about parent/child insert ordering.
DROP TABLE IF EXISTS golden_master;

CREATE TABLE golden_master (
    match_group_id    TEXT          NOT NULL UNIQUE,
    company_name      TEXT          NOT NULL,
    name_from         TEXT,
    industry          TEXT,
    country           TEXT,
    city              TEXT,
    employees         INTEGER,
    employees_from    TEXT,
    revenue           NUMERIC(18,2),
    revenue_currency  VARCHAR(3),
    revenue_from      TEXT,
    vat_id            TEXT,
    email_domain      TEXT,
    last_activity     DATE,
    sourced_from      TEXT,
    source_records    INTEGER,
    golden_id         TEXT          PRIMARY KEY,
    parent_golden_id  TEXT
);


-- -----------------------------------------------------------------------------
-- STEP 4 — Transform staging -> core (cast + clean)
-- -----------------------------------------------------------------------------
-- This is the transformation layer. Two techniques worth naming:
--
--   NULLIF(col, '')  turns empty strings into real NULLs. The CSV represents
--                    "missing" as an empty cell; in the typed table that should
--                    be a genuine NULL (e.g. Pied Piper has no city / employees /
--                    revenue), not an empty string.
--
--   ::NUMERIC::INTEGER  a deliberate double cast. The source values arrive as
--                    "450.0" (note the trailing .0). Casting TEXT straight to
--                    INTEGER errors on the decimal point, so we go via NUMERIC
--                    first, then down to INTEGER. This is the exact bug the
--                    staging layer is designed to let us handle in one place.
INSERT INTO golden_master
SELECT
    match_group_id,
    company_name,
    name_from,
    industry,
    NULLIF(country, ''),
    NULLIF(city, ''),
    NULLIF(employees, '')::NUMERIC::INTEGER,
    NULLIF(employees_from, ''),
    NULLIF(revenue, '')::NUMERIC,
    NULLIF(revenue_currency, ''),
    NULLIF(revenue_from, ''),
    NULLIF(vat_id, ''),
    email_domain,
    NULLIF(last_activity, '')::DATE,
    sourced_from,
    NULLIF(source_records, '')::NUMERIC::INTEGER,
    golden_id,
    NULLIF(parent_golden_id, '')
FROM stg_golden_master;


-- -----------------------------------------------------------------------------
-- STEP 4b — Referential integrity for the parent/subsidiary hierarchy
-- -----------------------------------------------------------------------------
-- A self-referencing foreign key: parent_golden_id must point at a real
-- golden_id in the same table. If this ALTER succeeds, every subsidiary link is
-- proven sound — a child can never reference a parent that doesn't exist. This
-- ties the relational model directly back to the MDM hierarchy-matching work.
ALTER TABLE golden_master
    ADD CONSTRAINT fk_parent
    FOREIGN KEY (parent_golden_id)
    REFERENCES golden_master (golden_id);


-- -----------------------------------------------------------------------------
-- STEP 4c — Verification checks
-- -----------------------------------------------------------------------------
-- Row count carried across:
SELECT COUNT(*) FROM golden_master;                       -- expect 12

-- Types are real (sorts by numeric magnitude, not alphabetically; dates render
-- as dates; NULLs sink to the bottom):
SELECT company_name, revenue, employees, last_activity
FROM golden_master ORDER BY revenue DESC NULLS LAST;
--
-- NULLIF worked (blanks became true NULLs):
SELECT company_name, city, employees, revenue
FROM golden_master WHERE employees IS NULL OR revenue IS NULL;
--
-- Hierarchy is visible and sound (FK already guarantees parents exist):
SELECT company_name, golden_id, parent_golden_id
FROM golden_master WHERE parent_golden_id IS NOT NULL;