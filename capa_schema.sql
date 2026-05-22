-- =============================================================
--   CAPA Recommendation System — PostgreSQL Schema
--   Database: capa_db
-- =============================================================

-- Step 1: Create the database (run as postgres superuser)
-- CREATE DATABASE capa_db;
-- \c capa_db

-- =============================================================
-- MAIN TABLE
-- =============================================================

CREATE TABLE IF NOT EXISTS historical_capa_records (

    -- Primary key
    id                   SERIAL          PRIMARY KEY,

    -- ── Identification ────────────────────────────────────────
    capa_id              VARCHAR(30)     NOT NULL UNIQUE,
    open_date            DATE            NOT NULL,
    close_date           DATE,
    days_to_close        INTEGER,

    -- ── Context ───────────────────────────────────────────────
    product              VARCHAR(100),
    department           VARCHAR(60),

    severity             VARCHAR(10)
                         CHECK (severity IN ('High', 'Medium', 'Low')),

    root_cause           VARCHAR(80),

    -- ── Core content (queried by recommendation engine) ───────
    incident_summary     TEXT            NOT NULL,
    corrective_action    TEXT,
    preventive_action    TEXT,
    effectiveness_check  TEXT,

    -- ── Outcome (used for ranking) ────────────────────────────
    effectiveness_rating VARCHAR(25)
                         CHECK (effectiveness_rating IN (
                             'Effective',
                             'Partially Effective',
                             'Not Effective'
                         )),

    recurrence           VARCHAR(5)
                         CHECK (recurrence IN ('Yes', 'No')),

    approved_by          VARCHAR(30),

    -- ── Audit timestamp ───────────────────────────────────────
    created_at           TIMESTAMP       DEFAULT NOW()
);

-- =============================================================
-- INDEXES  (speed up recommendation engine queries)
-- =============================================================

-- Filter by root cause (most common query from retrieval output)
CREATE INDEX IF NOT EXISTS idx_root_cause
    ON historical_capa_records (root_cause);

-- Filter by severity
CREATE INDEX IF NOT EXISTS idx_severity
    ON historical_capa_records (severity);

-- Filter and rank by effectiveness
CREATE INDEX IF NOT EXISTS idx_effectiveness
    ON historical_capa_records (effectiveness_rating);

-- Filter by department
CREATE INDEX IF NOT EXISTS idx_department
    ON historical_capa_records (department);

-- Filter by date range
CREATE INDEX IF NOT EXISTS idx_open_date
    ON historical_capa_records (open_date);

-- Filter non-recurring CAPAs only
CREATE INDEX IF NOT EXISTS idx_recurrence
    ON historical_capa_records (recurrence);

-- Composite index: most common recommendation query pattern
CREATE INDEX IF NOT EXISTS idx_rc_eff
    ON historical_capa_records (root_cause, effectiveness_rating);


-- =============================================================
-- VERIFICATION QUERIES
-- =============================================================

-- Check total records
SELECT COUNT(*) FROM historical_capa_records;

-- Distribution by root cause
SELECT root_cause, COUNT(*) as total
FROM historical_capa_records
GROUP BY root_cause
ORDER BY total DESC;

-- Distribution by severity
SELECT severity, COUNT(*) as total
FROM historical_capa_records
GROUP BY severity
ORDER BY total DESC;

-- Effectiveness breakdown
SELECT effectiveness_rating, COUNT(*) as total
FROM historical_capa_records
GROUP BY effectiveness_rating
ORDER BY total DESC;


-- =============================================================
-- RECOMMENDATION ENGINE QUERIES
-- =============================================================

-- 1. Fetch full CAPA details by capa_id
--    (called after FAISS/vector search returns top-k capa_ids)
SELECT
    capa_id,
    incident_summary,
    corrective_action,
    preventive_action,
    effectiveness_check,
    effectiveness_rating,
    recurrence,
    severity,
    days_to_close,
    department,
    product
FROM historical_capa_records
WHERE capa_id = ANY(ARRAY['CAPA-2021-0001','CAPA-2022-0045','CAPA-2023-0120']);


-- 2. Filter by root cause + effective only
--    (refine results after retrieval returns root cause)
SELECT
    capa_id,
    incident_summary,
    corrective_action,
    preventive_action,
    days_to_close
FROM historical_capa_records
WHERE root_cause          = 'Equipment Failure'
  AND effectiveness_rating = 'Effective'
ORDER BY days_to_close ASC
LIMIT 5;


-- 3. Full recommendation query
--    (combines root cause filter + effectiveness ranking)
SELECT
    capa_id,
    incident_summary,
    corrective_action,
    preventive_action,
    effectiveness_rating,
    recurrence,
    severity,
    days_to_close,
    CASE effectiveness_rating
        WHEN 'Effective'            THEN 1.0
        WHEN 'Partially Effective'  THEN 0.6
        WHEN 'Not Effective'        THEN 0.2
    END AS effectiveness_score
FROM historical_capa_records
WHERE root_cause = 'Environmental Control Failure'
  AND recurrence = 'No'
ORDER BY effectiveness_score DESC, days_to_close ASC
LIMIT 10;


-- 4. Effectiveness rate by root cause (analytics dashboard)
SELECT
    root_cause,
    COUNT(*)                                           AS total,
    SUM(CASE WHEN effectiveness_rating = 'Effective'
             THEN 1 ELSE 0 END)                        AS effective_count,
    ROUND(
        100.0 *
        SUM(CASE WHEN effectiveness_rating = 'Effective'
                 THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                  AS effectiveness_pct,
    ROUND(AVG(days_to_close))                          AS avg_days_to_close
FROM historical_capa_records
GROUP BY root_cause
ORDER BY effectiveness_pct DESC;
