import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "database": "capa_db",
    "user"    : "postgres",
    "password": "chachu@22",
}

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_capa_records.csv")

DDL_TABLE = """
CREATE TABLE IF NOT EXISTS historical_capa_records (
    id                      SERIAL          PRIMARY KEY,
    capa_id                 VARCHAR(30)     NOT NULL UNIQUE,
    open_date               DATE            NOT NULL,
    close_date              DATE,
    days_to_close           INTEGER,
    product                 VARCHAR(150),
    department              VARCHAR(70),
    severity                VARCHAR(10),
    root_cause              VARCHAR(100),
    regulatory_reference    VARCHAR(100),
    incident_summary        TEXT            NOT NULL,
    root_cause_detail       TEXT,
    corrective_action       TEXT,
    preventive_action       TEXT,
    effectiveness_check     TEXT,
    effectiveness_rating    VARCHAR(25),
    recurrence              VARCHAR(5),
    investigator_name       VARCHAR(60),
    investigator_role       VARCHAR(60),
    approver_name           VARCHAR(60),
    approver_role           VARCHAR(60),
    ca_completion_date      DATE,
    pa_completion_date      DATE,
    ec_completion_date      DATE,
    created_at              TIMESTAMP       DEFAULT NOW()
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_hcr_root_cause    ON historical_capa_records (root_cause);",
    "CREATE INDEX IF NOT EXISTS idx_hcr_severity       ON historical_capa_records (severity);",
    "CREATE INDEX IF NOT EXISTS idx_hcr_effectiveness  ON historical_capa_records (effectiveness_rating);",
    "CREATE INDEX IF NOT EXISTS idx_hcr_department     ON historical_capa_records (department);",
    "CREATE INDEX IF NOT EXISTS idx_hcr_recurrence     ON historical_capa_records (recurrence);",
    "CREATE INDEX IF NOT EXISTS idx_hcr_open_date      ON historical_capa_records (open_date);",
    "CREATE INDEX IF NOT EXISTS idx_hcr_rc_eff         ON historical_capa_records (root_cause, effectiveness_rating);",
    "CREATE INDEX IF NOT EXISTS idx_hcr_sev_eff        ON historical_capa_records (severity, effectiveness_rating);",
]

INSERT_SQL = """
INSERT INTO historical_capa_records (
    capa_id, open_date, close_date, days_to_close,
    product, department, severity, root_cause, regulatory_reference,
    incident_summary, root_cause_detail, corrective_action,
    preventive_action, effectiveness_check,
    effectiveness_rating, recurrence,
    investigator_name, investigator_role,
    approver_name, approver_role,
    ca_completion_date, pa_completion_date, ec_completion_date
)
VALUES (
    :capa_id, :open_date, :close_date, :days_to_close,
    :product, :department, :severity, :root_cause, :regulatory_reference,
    :incident_summary, :root_cause_detail, :corrective_action,
    :preventive_action, :effectiveness_check,
    :effectiveness_rating, :recurrence,
    :investigator_name, :investigator_role,
    :approver_name, :approver_role,
    :ca_completion_date, :pa_completion_date, :ec_completion_date
)
ON CONFLICT (capa_id) DO NOTHING;
"""

def connect():
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    try:
        engine = create_engine(url)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        print(f"  connected to {DB_CONFIG['database']}")
        return engine
    except Exception as e:
        print(f"  connection failed: {e}")
        sys.exit(1)

def create_schema(engine):
    print("  creating table and indexes...")
    with engine.begin() as c:
        c.execute(text(DDL_TABLE))
        for idx in DDL_INDEXES:
            c.execute(text(idx))
    print(f"  table and {len(DDL_INDEXES)} indexes created")

def load_and_insert(engine, csv_path):
    print(f"  loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows loaded")
    date_cols = ["open_date","close_date","ca_completion_date","pa_completion_date","ec_completion_date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col]).dt.date
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    inserted = 0
    with engine.begin() as c:
        for row in records:
            r = c.execute(text(INSERT_SQL), row)
            inserted += r.rowcount
    print(f"  inserted: {inserted}")

def verify(engine):
    with engine.connect() as c:
        total = c.execute(text("SELECT COUNT(*) FROM historical_capa_records")).scalar()
        print(f"  total records in db: {total}")

if __name__ == "__main__":
    print("=== Historical CAPA Records -> PostgreSQL ===")
    engine = connect()
    create_schema(engine)
    load_and_insert(engine, CSV_PATH)
    verify(engine)
    print("=== DONE ===")
