"""
CAPA Recommendation Service
Logic taken EXACTLY from:
  - Retrieval_System.ipynb      (clean_text, FAISS search, k=5)
  - Recommendation_layer.ipynb  (confidence formula, action ranking top-3, reason text)
"""

import re
import numpy as np
import pandas as pd
import faiss
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "models"
HISTORY_FILE = BASE_DIR / "historical_capa_records.csv"

# ── Globals ────────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_cleaned_df: pd.DataFrame | None = None
_capa_df: pd.DataFrame | None = None

_recommendation_history: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────
# clean_text()  — EXACT copy from Retrieval_System.ipynb
# ──────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace("temp", "temperature")
    text = text.replace("sop", "standard operating procedure")
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────
# Keyword extraction — domain vocabulary from cleaned_dataset categories
# ──────────────────────────────────────────────────────────────────────────
KEYWORD_VOCAB = {
    "Contamination":   ["contamination", "microbial", "particulate", "bioburden",
                        "endotoxin", "pyrogen", "sterility"],
    "Sterile Area":    ["sterile area", "sterile filling", "aseptic", "clean room",
                        "cleanroom", "grade a", "grade b", "iso 5", "iso 7", "laminar flow"],
    "HVAC":            ["hvac", "airflow", "ventilation", "air handling", "ahu",
                        "pressure differential", "humidity", "air quality"],
    "Equipment":       ["equipment", "machine", "instrument", "calibration",
                        "maintenance", "breakdown", "failure", "malfunction",
                        "sensor", "pump", "compressor"],
    "Temperature":     ["temperature", "excursion", "cold room", "thermal", "refriger"],
    "Documentation":   ["documentation", "batch record", "standard operating procedure",
                        "form", "signature", "missing", "incomplete"],
    "Process":         ["process deviation", "out of specification", "oos", "parameter",
                        "critical", "control"],
    "Supplier":        ["supplier", "vendor", "raw material", "component", "defect", "coa"],
    "Training":        ["training", "operator", "personnel", "human error", "competency"],
    "Software":        ["software", "system", "lims", "data integrity", "electronic",
                        "audit trail"],
    "Production":      ["production", "batch", "manufacturing", "yield", "blend"],
}

def extract_keywords(text: str) -> list[str]:
    """Extract domain keywords from raw incident text (before clean_text)."""
    text_lower = text.lower()
    found = []
    for label, terms in KEYWORD_VOCAB.items():
        for term in terms:
            if term in text_lower:
                if label not in found:
                    found.append(label)
                break
    # Also pick up uppercase acronyms from original text
    for ac in re.findall(r'\b[A-Z]{2,5}\b', text):
        if ac not in ("CAPA", "THE", "AND", "FOR", "WITH", "NOT") and ac not in found:
            found.append(ac)
    return found[:8]


# ──────────────────────────────────────────────────────────────────────────
# load_assets() — mirrors Recommendation_layer.ipynb cell 3
# ──────────────────────────────────────────────────────────────────────────
def load_assets():
    global _model, _index, _cleaned_df, _capa_df

    print("[CAPA-RS] Loading SentenceTransformer (all-MiniLM-L6-v2)…")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[CAPA-RS] Loading FAISS index…")
    _index = faiss.read_index(str(BASE_DIR / "capa_index.faiss"))

    print("[CAPA-RS] Loading cleaned_dataset.csv…")
    _cleaned_df = pd.read_csv(BASE_DIR / "cleaned_dataset.csv")

    print("[CAPA-RS] Loading historical_capa_records.csv…")
    _capa_df = pd.read_csv(HISTORY_FILE)

    print("[CAPA-RS] All assets ready.")


# ──────────────────────────────────────────────────────────────────────────
# recommend() — EXACT logic from both notebooks combined
# ──────────────────────────────────────────────────────────────────────────
def recommend(incident_description: str, k: int = 5) -> dict:
    """
    Pipeline (mirrors notebooks exactly):

    Retrieval_System.ipynb:
      1. clean_text(incident_description)
      2. model.encode([cleaned])
      3. index.search(query, k=5)
      4. retrieve incident / root_cause / severity from df

    Recommendation_layer.ipynb:
      5. Counter majority vote → root_cause, severity
      6. confidence = 100 * exp(-top_distance / (max_distance + 1e-10))
      7. filter capa_df by root_cause & severity
      8. Counter rank corrective_actions, preventive_actions → top 3
      9. Build reason string
    """
    if _model is None or _index is None:
        raise RuntimeError("Assets not loaded. Call load_assets() first.")

    # ── Extract keywords from RAW text (before cleaning) ──────────────────
    keywords = extract_keywords(incident_description)

    # ── Step 1: clean_text  (Retrieval_System.ipynb) ──────────────────────
    cleaned_query = clean_text(incident_description)

    # ── Step 2: encode  (Retrieval_System.ipynb) ──────────────────────────
    query_embedding = _model.encode([cleaned_query])

    # ── Step 3: FAISS search k=5  (Retrieval_System.ipynb) ───────────────
    distances, indices = _index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    # ── Step 4: build retrieved list  (Retrieval_System.ipynb) ───────────
    # Notebook fields: incident, root_cause, severity  (exactly these three)
    retrieved = []
    for i, dist in zip(indices[0], distances[0]):
        if i < 0 or i >= len(_cleaned_df):
            continue
        row = _cleaned_df.iloc[i]
        retrieved.append({
            "incident":   str(row["incident_description"]),
            "root_cause": str(row["root_cause"]),
            "severity":   str(row["severity_level"]),
            # distance kept for confidence calc; not shown to user directly
            "_distance":  float(dist),
        })

    if not retrieved:
        return _empty_result(keywords)

    retrieved_df = pd.DataFrame(retrieved)

    # ── Step 5: majority vote  (Recommendation_layer.ipynb) ──────────────
    root_cause = Counter(retrieved_df["root_cause"]).most_common(1)[0][0]
    severity   = Counter(retrieved_df["severity"]).most_common(1)[0][0]

    # ── Step 6: confidence formula  (Recommendation_layer.ipynb) ─────────
    # EXACT formula from notebook:
    # confidence = round(100 * np.exp(-top_distance / (max_distance + 1e-10)), 2)
    top_distance = float(distances[0][0])
    max_distance = float(np.max(distances))
    confidence = round(
        100 * float(np.exp(-top_distance / (max_distance + 1e-10))),
        2
    )

    # ── Step 7: filter historical CAPA records  (Recommendation_layer.ipynb)
    current_capa_df = pd.read_csv(HISTORY_FILE)
    recommendations = current_capa_df[
        (current_capa_df["root_cause"] == root_cause) &
        (current_capa_df["severity"]   == severity)
    ]
    # Fallback: root_cause only (same as original backend)
    if recommendations.empty:
        recommendations = current_capa_df[current_capa_df["root_cause"] == root_cause]

    historical_cases = len(recommendations)

    # ── Step 8: rank actions → top 3  (Recommendation_layer.ipynb) ───────
    corrective_actions: list[str] = []
    preventive_actions: list[str] = []

    if not recommendations.empty:
        corr_rank = Counter(recommendations["corrective_action"].dropna().tolist())
        prev_rank = Counter(recommendations["preventive_action"].dropna().tolist())

        # Notebook uses .most_common(3)
        corrective_actions = [a for a, _ in corr_rank.most_common(3)]
        preventive_actions = [a for a, _ in prev_rank.most_common(3)]

    # ── Step 9: reason string  (Recommendation_layer.ipynb) ──────────────
    # Mirrors the reason template in the notebook
    reason = (
        f"Generated using:\n"
        f"• Similar historical incidents\n"
        f"• Root Cause: {root_cause}\n"
        f"• Severity: {severity}\n"
        f"• Historical CAPAs: {historical_cases}"
    )

    # ── Similar incidents for UI display (top 3, notebook k=3 display) ───
    # Remove internal _distance key before sending to frontend
    similar_incidents = [
        {
            "incident":   r["incident"],
            "root_cause": r["root_cause"],
            "severity":   r["severity"],
            # Similarity % derived from distance for UI bar/display
            "similarity": round(
                100 * float(np.exp(-r["_distance"] / (max_distance + 1e-10))), 1
            ),
        }
        for r in retrieved[:3]
    ]

    # ── Save to session history ───────────────────────────────────────────
    _recommendation_history.append({
        "id":                   len(_recommendation_history) + 1,
        "timestamp":            datetime.now().isoformat(),
        "incident_description": incident_description,
        "root_cause":           root_cause,
        "severity":             severity,
        "confidence":           confidence,
        "keywords":             keywords,
    })

    return {
        "root_cause":           root_cause,
        "severity":             severity,
        "confidence":           confidence,
        "historical_cases":     historical_cases,
        "corrective_actions":   corrective_actions,
        "preventive_actions":   preventive_actions,
        "recommendation_reason": reason,
        "similar_incidents":    similar_incidents,
        "keywords":             keywords,
    }


# ──────────────────────────────────────────────────────────────────────────
# add_capa_record() — saves new record to historical_capa_records.csv
# ──────────────────────────────────────────────────────────────────────────
def add_capa_record(record: dict) -> dict:
    df = pd.read_csv(HISTORY_FILE)
    year = datetime.now().year
    nums = []
    for cid in df["capa_id"].tolist():
        try:
            nums.append(int(str(cid).split("-")[-1]))
        except Exception:
            pass
    next_num = (max(nums) + 1) if nums else 1
    capa_id  = f"CAPA-{year}-{next_num:04d}"

    new_row = {
        "capa_id":             capa_id,
        "open_date":           record.get("open_date", datetime.now().strftime("%Y-%m-%d")),
        "close_date":          record.get("close_date", ""),
        "days_to_close":       "",
        "product":             record.get("product", ""),
        "department":          record.get("department", ""),
        "severity":            record.get("severity", "Medium"),
        "root_cause":          record.get("root_cause", ""),
        "regulatory_reference":record.get("regulatory_reference", ""),
        "incident_summary":    record.get("incident_summary", ""),
        "root_cause_detail":   record.get("root_cause_detail", ""),
        "corrective_action":   record.get("corrective_action", ""),
        "preventive_action":   record.get("preventive_action", ""),
        "effectiveness_check": "",
        "effectiveness_rating":"Pending",
        "recurrence":          "No",
        "investigator_name":   record.get("investigator_name", ""),
        "investigator_role":   record.get("investigator_role", ""),
        "approver_name":       "",
        "approver_role":       "",
        "ca_completion_date":  record.get("ca_completion_date", ""),
        "pa_completion_date":  record.get("pa_completion_date", ""),
        "ec_completion_date":  "",
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)
    return {"capa_id": capa_id, "message": "CAPA record added successfully"}


# ──────────────────────────────────────────────────────────────────────────
# get_history / get_stats
# ──────────────────────────────────────────────────────────────────────────
def get_history() -> list[dict]:
    return list(reversed(_recommendation_history[-50:]))


def get_stats() -> dict:
    df = pd.read_csv(HISTORY_FILE)
    return {
        "total_records":            len(df),
        "root_cause_distribution":  df["root_cause"].value_counts().to_dict(),
        "severity_distribution":    df["severity"].value_counts().to_dict(),
        "department_distribution":  df["department"].value_counts().to_dict(),
        "effectiveness_distribution": df["effectiveness_rating"].value_counts().to_dict(),
    }


def _empty_result(keywords: list[str] = []) -> dict:
    return {
        "root_cause":            "Unknown",
        "severity":              "Unknown",
        "confidence":            0.0,
        "historical_cases":      0,
        "corrective_actions":    [],
        "preventive_actions":    [],
        "recommendation_reason": "No similar incidents found in the database.",
        "similar_incidents":     [],
        "keywords":              keywords,
    }
