"""
=============================================================
  CAPA Recommendation System — Step 1: Text Preprocessing
=============================================================

Pipeline:
  1. Lowercase
  2. Remove special characters
  3. Tokenise
  4. Remove stopwords  (built-in list — no NLTK download needed)
  5. Lemmatise         (rule-based suffix stripping)
  6. Extract pharma entities  (regex + keyword matching)
  7. Save capa_clean.csv

Run:
  python3 preprocess_capa.py

Output:
  capa_clean.csv   — original cols + clean_text + entity feature cols
"""

import re
import time
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# 1.  BUILT-IN STOPWORDS
#     Standard English stopwords — no NLTK required
# ─────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","for",
    "of","with","by","from","up","as","is","was","are","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","can","not","no","nor",
    "so","yet","both","either","neither","than","too","very","just",
    "that","this","these","those","it","its","itself","they","them",
    "their","there","here","when","where","which","who","whom","how",
    "all","each","every","any","some","such","into","through","during",
    "before","after","above","below","between","out","off","over",
    "then","once","more","also","about","found","showed","using",
    "per","due","prior","following","within","without","across",
    "against","along","among","around","near","upon","while",
    "further","only","own","same","other","however","therefore",
    "including","resulting","required","identified","confirmed",
    "noted","occurred","performed","conducted","completed","affected",
    "reported","detected","observed","initiated","placed","taken",
    "given","shown","used","made","set","put","went","came","left",
    "right","under","since","during","although","though","unless",
    "until","whereas","whether","because","since","being","having",
}


# ─────────────────────────────────────────────────────────────
# 2.  RULE-BASED LEMMATISER
#     Handles common English suffixes found in pharma text.
#     No external model needed.
# ─────────────────────────────────────────────────────────────

# Irregular word mapping  (common pharma-domain exceptions)
IRREGULAR = {
    "contaminated":  "contaminate",
    "contamination": "contaminate",
    "contaminating": "contaminate",
    "failed":        "fail",
    "failure":       "fail",
    "failing":       "fail",
    "detected":      "detect",
    "detection":     "detect",
    "exceeded":      "exceed",
    "exceedance":    "exceed",
    "exceeding":     "exceed",
    "exceeded":      "exceed",
    "filled":        "fill",
    "filling":       "fill",
    "fills":         "fill",
    "coated":        "coat",
    "coating":       "coat",
    "compressed":    "compress",
    "compression":   "compress",
    "processed":     "process",
    "processing":    "process",
    "processed":     "process",
    "granulated":    "granulate",
    "granulation":   "granulate",
    "manufactured":  "manufacture",
    "manufacturing": "manufacture",
    "sterilised":    "sterilise",
    "sterilization": "sterilise",
    "sterilising":   "sterilise",
    "quarantined":   "quarantine",
    "quarantine":    "quarantine",
    "validated":     "validate",
    "validation":    "validate",
    "retraining":    "retrain",
    "retrained":     "retrain",
    "documented":    "document",
    "documentation": "document",
    "investigated":  "investigate",
    "investigation": "investigate",
    "operator":      "operator",
    "operators":     "operator",
    "tablets":       "tablet",
    "batches":       "batch",
    "samples":       "sample",
    "results":       "result",
    "issues":        "issue",
    "errors":        "error",
    "deviations":    "deviation",
    "procedures":    "procedure",
    "systems":       "system",
    "requirements":  "requirement",
    "specifications":"specification",
    "materials":     "material",
    "products":      "product",
    "processes":     "process",
    "readings":      "reading",
    "findings":      "finding",
    "actions":       "action",
    "records":       "record",
    "tests":         "test",
    "limits":        "limit",
    "levels":        "level",
    "values":        "value",
    "units":         "unit",
    "checks":        "check",
    "steps":         "step",
    "stages":        "stage",
    "areas":         "area",
    "zones":         "zone",
    "lines":         "line",
    "components":    "component",
    "entries":       "entry",
    "versions":      "version",
    "revisions":     "revision",
}

# Suffix rules — ordered from longest to shortest
SUFFIX_RULES = [
    ("ications",  "icate"),
    ("isation",   "ise"),
    ("ization",   "ize"),
    ("nesses",    ""),
    ("fulness",   "ful"),
    ("iveness",   "ive"),
    ("nesses",    ""),
    ("ations",    "ate"),
    ("ments",     "ment"),
    ("ities",     "ity"),
    ("ation",     "ate"),
    ("ness",      ""),
    ("ment",      "ment"),
    ("ical",      "ic"),
    ("ings",      ""),
    ("ated",      "ate"),
    ("ting",      "te"),
    ("ding",      "de"),
    ("sing",      "se"),
    ("ring",      "re"),
    ("ning",      ""),
    ("ing",       ""),
    ("ised",      "ise"),
    ("ized",      "ize"),
    ("ied",       "y"),
    ("ves",       "f"),
    ("ies",       "y"),
    ("ued",       "ue"),
    ("red",       "re"),
    ("ned",       "ne"),
    ("sed",       "se"),
    ("ted",       "te"),
    ("ded",       "de"),
    ("ed",        ""),
    ("ers",       "er"),
    ("ors",       "or"),
    ("est",       ""),
    ("er",        ""),
    ("ly",        ""),
    ("es",        ""),
    ("'s",        ""),
    ("s",         ""),
]

def lemmatise(word: str) -> str:
    """Rule-based lemmatiser — checks irregular map then applies suffix rules."""
    if word in IRREGULAR:
        return IRREGULAR[word]
    # Don't lemmatise short words — creates nonsense
    if len(word) <= 4:
        return word
    for suffix, replacement in SUFFIX_RULES:
        if word.endswith(suffix):
            root = word[: len(word) - len(suffix)] + replacement
            # Only accept if root is at least 3 chars
            if len(root) >= 3:
                return root
    return word


# ─────────────────────────────────────────────────────────────
# 3.  ENTITY EXTRACTION
#     Regex + keyword patterns tuned for pharma incident text
# ─────────────────────────────────────────────────────────────

EQUIPMENT_KEYWORDS = [
    "coating pan", "autoclave", "hplc", "filling pump", "tablet press",
    "freeze dryer", "granulator", "centrifuge", "blender", "balance",
    "ph meter", "dissolution apparatus", "temperature logger", "hvac",
    "water purification", "compressed air", "lyophilizer", "bioreactor",
    "fermenter", "scada", "lims", "erm", "ebr",
]

ENV_CLASS_KEYWORDS = [
    "grade a", "grade b", "grade c", "grade d",
    "class 100", "class 1000", "class 10000", "class 100000",
    "iso 5", "iso 7", "iso 8",
]

DEFECT_KEYWORDS = [
    "contamination", "contaminate", "deviation", "oos", "out of specification",
    "exceedance", "failure", "fail", "mix-up", "mixup", "defect",
    "particulate", "foreign matter", "leakage", "breakage", "delamination",
    "discolouration", "discoloration",
]

ORGANISM_KEYWORDS = [
    "bacillus", "staphylococcus", "e. coli", "escherichia", "pseudomonas",
    "aspergillus", "candida", "mold", "mould", "yeast", "fungi", "fungal",
    "microbial", "bioburden", "endotoxin",
]

def extract_entities(text: str) -> dict:
    """
    Extract structured pharma features from raw incident text.
    Returns a dict of binary flags and category labels.
    """
    t = text.lower()

    # Equipment mention
    has_equipment = int(any(kw in t for kw in EQUIPMENT_KEYWORDS))

    # Equipment ID code  e.g. EQ-123, CP-456, FP-901
    equip_id_match = re.search(
        r'\b([A-Z]{2,4}-\d{2,4})\b', text
    )
    has_equip_id_code = int(equip_id_match is not None)

    # Batch / lot number  e.g.  A1234B,  LOT-12345,  Batch 9901A
    batch_match = re.search(
        r'\b(batch\s+[\w\d]+|lot[\s\-][\w\d]+|[A-Z]\d{3,5}[A-Z]?)\b',
        text, re.IGNORECASE
    )
    has_batch_id = int(batch_match is not None)

    # Environmental classification  Grade A / Class 100 etc.
    has_env_class = int(any(kw in t for kw in ENV_CLASS_KEYWORDS))

    # Defect type — first match wins
    defect_type = "none"
    for kw in DEFECT_KEYWORDS:
        if kw in t:
            defect_type = kw.replace(" ", "_")
            break

    # Microbial / organism reference
    has_organism = int(any(kw in t for kw in ORGANISM_KEYWORDS))

    # Numeric measurement  e.g. 12.3%  or  45 CFU  or  8°C
    has_measurement = int(
        bool(re.search(r'\d+[\.,]?\d*\s*(%|cfu|°c|degrees|ppm|µs|kcfu|ms|rpm|bar|pa)', t))
    )

    # Operator / personnel reference
    has_operator = int(
        bool(re.search(r'\bop-\d+\b|\boperator\b|\banalyst\b|\btechnician\b', t))
    )

    # SOP / document reference
    has_sop_ref = int(
        bool(re.search(r'\bsop[\s\-]\d+\b|\bsop\b|\bprocedure\b|\bprotocol\b', t))
    )

    return {
        "has_equipment":    has_equipment,
        "has_equip_id_code": has_equip_id_code,
        "has_batch_id":     has_batch_id,
        "has_env_class":    has_env_class,
        "defect_type":      defect_type,
        "has_organism":     has_organism,
        "has_measurement":  has_measurement,
        "has_operator":     has_operator,
        "has_sop_ref":      has_sop_ref,
    }


# ─────────────────────────────────────────────────────────────
# 4.  MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for one incident description.

    Steps:
      1. Lowercase
      2. Remove special characters (keep letters, digits, spaces)
      3. Tokenise by splitting on whitespace
      4. Remove stopwords
      5. Lemmatise each token
      6. Remove very short tokens (len < 3)
      7. Remove duplicate consecutive tokens
      8. Rejoin into clean string

    Parameters
    ----------
    text : str
        Raw incident description text.

    Returns
    -------
    str
        Cleaned, normalised text ready for TF-IDF vectorisation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # ── Step 1: Lowercase ────────────────────────────────────
    text = text.lower()

    # ── Step 2: Remove special characters ───────────────────
    # Keep letters, digits, and spaces only
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    # ── Step 3: Tokenise ────────────────────────────────────
    tokens = text.split()

    # ── Step 4: Remove stopwords ────────────────────────────
    tokens = [t for t in tokens if t not in STOPWORDS]

    # ── Step 5: Lemmatise ───────────────────────────────────
    tokens = [lemmatise(t) for t in tokens]

    # ── Step 6: Remove short tokens ─────────────────────────
    # Keep tokens with 3+ characters (removes "eq", "op", "hr" noise)
    tokens = [t for t in tokens if len(t) >= 3]

    # ── Step 7: Remove duplicate consecutive tokens ──────────
    deduped = []
    prev = None
    for t in tokens:
        if t != prev:
            deduped.append(t)
        prev = t

    # ── Step 8: Rejoin ───────────────────────────────────────
    return " ".join(deduped)


# ─────────────────────────────────────────────────────────────
# 5.  PROCESS THE FULL DATASET
# ─────────────────────────────────────────────────────────────

def process_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load capa_dataset.csv, run preprocessing on every row,
    extract entity features, and save capa_clean.csv.

    Parameters
    ----------
    input_path  : str  — path to raw capa_dataset.csv
    output_path : str  — path to save capa_clean.csv

    Returns
    -------
    pd.DataFrame — the enriched dataframe
    """
    print("\n" + "="*58)
    print("  CAPA Preprocessing Pipeline")
    print("="*58)

    # ── Load ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")

    # ── Null check ───────────────────────────────────────────
    nulls = df["incident_description"].isna().sum()
    if nulls > 0:
        print(f"      Warning: {nulls} null incident descriptions — filling with empty string")
        df["incident_description"] = df["incident_description"].fillna("")

    # ── Preprocess text ──────────────────────────────────────
    print(f"\n[2/5] Preprocessing incident_description text...")
    t0 = time.time()
    df["clean_text"] = df["incident_description"].apply(preprocess_text)
    elapsed = time.time() - t0
    print(f"      Done — {len(df)} rows in {elapsed:.2f}s "
          f"({len(df)/elapsed:.0f} rows/sec)")

    # ── Entity extraction ────────────────────────────────────
    print(f"\n[3/5] Extracting pharma entities...")
    t1 = time.time()
    entity_records = df["incident_description"].apply(extract_entities)
    entity_df = pd.DataFrame(list(entity_records))
    df = pd.concat([df, entity_df], axis=1)
    print(f"      Done — {len(entity_df.columns)} entity columns added in "
          f"{time.time()-t1:.2f}s")
    print(f"      Columns added: {list(entity_df.columns)}")

    # ── Token length stats ───────────────────────────────────
    print(f"\n[4/5] Text statistics after preprocessing:")
    token_counts = df["clean_text"].apply(lambda x: len(x.split()))
    print(f"      Avg tokens per incident : {token_counts.mean():.1f}")
    print(f"      Min tokens              : {token_counts.min()}")
    print(f"      Max tokens              : {token_counts.max()}")
    print(f"      Unique vocabulary size  : "
          f"{len(set(' '.join(df['clean_text']).split()))}")

    # ── Entity stats ─────────────────────────────────────────
    print(f"\n      Entity feature summary:")
    binary_cols = ["has_equipment","has_equip_id_code","has_batch_id",
                   "has_env_class","has_organism","has_measurement",
                   "has_operator","has_sop_ref"]
    for col in binary_cols:
        pct = df[col].mean() * 100
        print(f"        {col:25s}: {pct:5.1f}% of incidents")
    print(f"\n      Defect type breakdown:")
    print(df["defect_type"].value_counts().head(8).to_string())

    # ── Save ─────────────────────────────────────────────────
    print(f"\n[5/5] Saving clean dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"      Saved — {df.shape[0]} rows x {df.shape[1]} columns")

    return df


# ─────────────────────────────────────────────────────────────
# 6.  DEMO — show before/after for 5 sample rows
# ─────────────────────────────────────────────────────────────

def show_samples(df: pd.DataFrame, n: int = 5):
    """Print side-by-side before/after preprocessing for n random rows."""
    print("\n" + "="*58)
    print("  Sample preprocessing results")
    print("="*58)
    samples = df.sample(n, random_state=42)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n  [{i}] Root cause: {row['root_cause']}  |  "
              f"Severity: {row['severity_level']}")
        print(f"  BEFORE: {row['incident_description'][:120]}...")
        print(f"  AFTER:  {row['clean_text'][:120]}...")
        entity_flags = {
            k: v for k, v in row.items()
            if k.startswith("has_") or k == "defect_type"
        }
        active = {k: v for k, v in entity_flags.items() if v not in [0, "none"]}
        print(f"  ENTITIES: {active}")


# ─────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    INPUT  = "capa_dataset.csv"
    OUTPUT = "capa_clean.csv"

    df = process_dataset(INPUT, OUTPUT)
    show_samples(df, n=5)

    print("\n" + "="*58)
    print("  NEXT STEP: Step 2 — Feature Engineering")
    print("  Run:  python3 feature_engineering.py")
    print("  Input:  capa_clean.csv")
    print("  Output: X_features.npy  +  y_labels.npy")
    print("="*58 + "\n")
