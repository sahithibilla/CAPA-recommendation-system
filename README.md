# CAPA AI Recommendation System v2

> AI-powered Corrective and Preventive Action recommendations for pharmaceutical compliance incidents.

---

## Folder Structure

```
capa-webapp/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI application (all endpoints)
│   ├── services/
│   │   └── recommendation.py    # Core AI pipeline + CAPA add logic
│   └── requirements.txt
├── frontend/
│   └── index.html               # Complete single-file UI (no build step)
├── models/
│   ├── capa_index.faiss         # Pre-built FAISS index
│   ├── cleaned_dataset.csv      # 2,200 incident records (keyword source)
│   └── historical_capa_records.csv  # 1,000+ historical CAPA records
├── run_backend.sh               # One-click backend startup
└── README.md
```

---

## Quick Start

### Step 1 — Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~90MB) from HuggingFace automatically.

### Step 2 — Start Backend

```bash
bash run_backend.sh
```

Or manually:
```bash
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3 — Open Frontend

Simply open `frontend/index.html` in your browser. No build step needed.

---

## Features

### Home Page
- **Incident Description** — Enter free-text incident description
- **Severity & Department** filters
- **AI Analysis** with animated processing steps
- **Detected Keywords** — Extracted from `cleaned_dataset.csv` domain vocabulary
- **AI Confidence Gauge** — Visual donut chart showing similarity score
- **Similar Incidents** — Top 3 semantic matches from FAISS index (expandable)
- **Corrective Actions** — Ranked by frequency from historical records
- **Preventive Actions** — Ranked by frequency from historical records
- **Recommendation Reason** — Natural-language explanation

### History Page
- Table of all past analyses in current session (root cause, severity, confidence, keywords)

### Stats Page
- Total CAPA records, High severity count, Effective count
- Root Cause distribution bar chart
- Department distribution bar chart

### Add CAPA Record (➕ button in nav)
- Form to add new CAPA entry that is **saved to `historical_capa_records.csv`**
- Automatically generates the next CAPA ID
- Data persists across sessions

### Other
- 🌙 Dark / Light mode toggle
- 🟢 Live API status indicator
- Toast notifications

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/recommend` | Get AI CAPA recommendations |
| POST | `/capa/add` | Add a new CAPA record to dataset |
| GET | `/history` | Recent recommendation history |
| GET | `/stats` | Dataset statistics |

### POST /recommend — Request Body
```json
{
  "incident_description": "Contamination detected in sterile filling area..."
}
```

### POST /capa/add — Request Body
```json
{
  "product": "Amoxicillin 500mg",
  "department": "Manufacturing",
  "severity": "High",
  "root_cause": "Equipment Failure",
  "incident_summary": "...",
  "corrective_action": "...",
  "preventive_action": "..."
}
```

---

## AI Pipeline

```
Incident Description
       ↓
SentenceTransformer (all-MiniLM-L6-v2)
       ↓
FAISS Semantic Retrieval (k=5 nearest)
       ↓
Keyword Extraction (domain vocabulary from cleaned_dataset)
       ↓
Root Cause + Severity (majority vote)
       ↓
Filter historical_capa_records.csv
       ↓
Rank Corrective & Preventive Actions (by frequency)
       ↓
Confidence Score (mean similarity %)
       ↓
Return Recommendations + Explanation
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embedding | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector Search | `faiss-cpu` |
| Data | `pandas` · `numpy` |
| Backend | `FastAPI` · `uvicorn` |
| Frontend | HTML5 · CSS3 · Vanilla JS (no build step) |
| Fonts | DM Sans · DM Serif Display · JetBrains Mono |

---

## Troubleshooting

**"API Offline" in the browser** → Start the backend (`bash run_backend.sh`)

**`ModuleNotFoundError: faiss`** → Run `pip install faiss-cpu`

**`FileNotFoundError: capa_index.faiss`** → Ensure the `models/` folder is next to `backend/`

**Slow first request** → Normal — the SentenceTransformer model loads on first use. Subsequent requests are fast.

**CORS errors** → The backend allows all origins by default. If deploying remotely, update `allow_origins` in `backend/api/main.py`.
