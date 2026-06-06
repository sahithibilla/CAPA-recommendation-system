<div align="center">

<br/>

```
██╗███╗   ██╗████████╗███████╗██╗     ██╗ ██████╗ █████╗ ██████╗  █████╗
██║████╗  ██║╚══██╔══╝██╔════╝██║     ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗
██║██╔██╗ ██║   ██║   █████╗  ██║     ██║██║     ███████║██████╔╝███████║
██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██║██║     ██╔══██║██╔═══╝ ██╔══██║
██║██║ ╚████║   ██║   ███████╗███████╗██║╚██████╗██║  ██║██║     ██║  ██║
╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
```

**AI-Powered CAPA Recommendation System for Pharmaceutical Compliance**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-intellicapa.onrender.com-7c6fff?style=for-the-badge)](https://intellicapa.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatL2-0064AA?style=for-the-badge)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-5eead4?style=for-the-badge)](LICENSE)

<br/>

> *Describe your incident. Get proven CAPA recommendations in under a second.*  
> *Backed by 1,000 historical pharma CAPA records. No retraining required.*

<br/>

</div>

---

## ✦ What is intelliCAPA?

Quality deviations in pharmaceutical manufacturing demand fast, consistent, and explainable Corrective and Preventive Actions (CAPA) — yet most organisations still rely on manual lookups, institutional memory, and keyword search. **intelliCAPA** eliminates this bottleneck.

It encodes your incident description as a semantic vector, finds the 5 most similar historical incidents in milliseconds using FAISS, votes on root cause and severity, then surfaces the most frequently proven corrective and preventive actions — with a full evidence trail you can cite in your QMS.

<br/>

## ⚡ Performance at a Glance

| Metric | Score | What it means |
|--------|-------|---------------|
| **Root Cause Accuracy** | `100%` | Correct root cause predicted for all 300 test samples |
| **Precision @ 5** | `1.000` | Every retrieved incident shared the correct root cause |
| **MRR @ 5** | `1.000` | Correct match always ranked #1 |
| **Avg Top-1 Similarity** | `95.7%` | Top retrieved incident is semantically near-identical |
| **Response Time** | `< 1s` | After model warmup (FAISS search is sub-millisecond) |

> Evaluated via leave-one-out methodology on 300 randomly selected incidents (`random_state=42`).

<br/>

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDATION PIPELINE                     │
└─────────────────────────────────────────────────────────────────────┘

  User Input                                              Response
  ──────────                                              ────────
  "unexpected temperature                                 ✓ Root Cause: Equipment Failure
   excursion in cold room                                 ✓ Severity: High
   CR-08"                                                 ✓ Top-5 Corrective Actions
       │                                                  ✓ Top-5 Preventive Actions
       ▼                                                  ✓ Confidence: 94.2%
  ┌─────────────┐    384-dim      ┌──────────────┐        ✓ Evidence Trail (5 incidents)
  │ Sentence    │   float32 vec   │    FAISS     │
  │ Transformer │ ──────────────► │ IndexFlatL2  │
  │ all-MiniLM  │                 │ (2,200 recs) │
  └─────────────┘                 └──────┬───────┘
                                         │ top-5 nearest neighbours
                                         ▼
                                  ┌─────────────┐
                                  │ Majority    │  root_cause + severity
                                  │   Vote      │ ──────────────────────►
                                  └─────────────┘
                                         │
                                         ▼
                                  ┌─────────────────────┐
                                  │ Filter CAPA Records │  (1,000 historical records)
                                  │ Rank by Frequency   │  Counter.most_common(5)
                                  └─────────────────────┘
```

<br/>

## 🗂️ Project Structure

```
capa-rs/
├── backend/
│   ├── api/
│   │   └── main.py                 # FastAPI app, CORS, Pydantic validation
│   ├── services/
│   │   └── recommendation.py       # Core pipeline: encode → search → vote → rank
│   ├── evaluate.py                 # Leave-one-out evaluation script
│   └── requirements.txt
│
├── frontend/
│   └── index.html                  # Complete single-file UI (~4,500 lines)
│
├── models/
│   ├── capa_index.faiss            # Pre-built FAISS IndexFlatL2 (2,200 × 384)
│   ├── cleaned_dataset.csv         # 2,200 incident records with root cause labels
│   └── historical_capa_records.csv # 1,000 CAPA records for recommendation
│
├── run_backend.sh                  # One-click startup
└── README.md
```

<br/>

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended)
- 10 GB free disk space (for model cache)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/intellicapa.git
cd intellicapa

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r backend/requirements.txt
```

### Running Locally

```bash
# Start the backend (model loads once, stays warm)
bash run_backend.sh

# Or manually:
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `frontend/index.html` in your browser — or serve it via any static file server.

> **First run note:** `all-MiniLM-L6-v2` will be downloaded from HuggingFace (~90 MB) and cached locally. Subsequent starts are instant.

<br/>

## 🔌 API Reference

### `POST /recommend`

Returns CAPA recommendations for a given incident description.

**Request**
```json
{
  "incident_description": "Unexpected temperature excursion detected in cold room CR-08 during weekend monitoring"
}
```

**Response**
```json
{
  "root_cause": "Equipment Failure",
  "severity": "High",
  "confidence": 94.2,
  "corrective_actions": [
    "Calibrate temperature monitoring sensors",
    "Inspect and service refrigeration unit",
    "..."
  ],
  "preventive_actions": [
    "Implement redundant temperature alarm system",
    "Schedule quarterly preventive maintenance",
    "..."
  ],
  "similar_incidents": [
    {
      "incident": "Cold room temperature alarm triggered during night shift...",
      "similarity": 97.1,
      "root_cause": "Equipment Failure",
      "severity": "High"
    }
  ],
  "historical_cases": 47
}
```

### `GET /health`

```json
{ "status": "ok", "service": "intelliCAPA" }
```

<br/>

## 🖥️ Features

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Finds relevant incidents regardless of terminology — `"temperature excursion"` matches `"cold chain failure"` |
| **Evidence Trail** | Every recommendation is traceable to specific historical records |
| **Confidence Score** | Mean semantic similarity across top-5 retrieved incidents |
| **Analytics Dashboard** | 9 Chart.js charts built from real data — root cause distribution, severity trends, CAPA types, recurrence analysis |
| **Dark / Light Mode** | CSS custom properties, persisted via `localStorage` |
| **Add Record Modal** | Extend the dataset without restarting the backend |
| **Benchmark Panel** | System-level evaluation metrics available in-UI |
| **Responsive UI** | Works on desktop and mobile |

<br/>

## 🛠️ Tech Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — async REST API
- [sentence-transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` (384-dim embeddings)
- [FAISS](https://faiss.ai/) — `IndexFlatL2`, sub-millisecond nearest-neighbour search
- [NumPy](https://numpy.org/) + [Pandas](https://pandas.pydata.org/) — data processing

**Frontend**
- Vanilla HTML5 / CSS3 / JavaScript — zero build tools, single file
- [Chart.js 4.4.1](https://www.chartjs.org/) — analytics dashboard
- [DM Sans + DM Mono + Playfair Display](https://fonts.google.com/) — typography

**Regulatory Context**
- Designed for 21 CFR Part 211 and ICH Q10 compliance workflows

<br/>

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                        │
│          Browser · Single-page HTML/CSS/JS app           │
└────────────────────────┬─────────────────────────────────┘
                         │  HTTP / fetch()
┌────────────────────────▼─────────────────────────────────┐
│                       API LAYER                          │
│        FastAPI · uvicorn · Pydantic · CORS middleware    │
│            POST /recommend   ·   GET /health             │
└──────┬─────────────────────────────────────┬─────────────┘
       │                                     │
┌──────▼──────────┐                 ┌────────▼────────────┐
│    ML LAYER     │                 │     DATA LAYER      │
│                 │                 │                     │
│ SentenceTransf. │                 │ cleaned_dataset.csv │
│ all-MiniLM-L6   │                 │   (2,200 incidents) │
│                 │                 │                     │
│ FAISS FlatL2    │                 │ historical_capa_    │
│ Majority Vote   │                 │ records.csv         │
│                 │                 │   (1,000 CAPAs)     │
└─────────────────┘                 │                     │
                                    │ capa_index.faiss    │
                                    └─────────────────────┘
```

<br/>

## 🧪 Evaluation

Run the leave-one-out benchmark on 300 samples:

```bash
cd backend
python evaluate.py
```

This will output Root Cause Accuracy, Precision@5, MRR@5, and average Top-1 Similarity.

> **Note on Severity Accuracy (40%):** Severity is an administrative label assigned after the incident description is written — it is not encoded in the incident text itself. This is expected behaviour, not a model defect.

<br/>

## 🔮 Roadmap

- [ ] **PostgreSQL Persistence** — Replace CSV with a live database so `Add Record` persists across sessions
- [ ] **PDF Report Export** — One-click CAPA report generation via ReportLab for formal QMS documentation
- [ ] **Role-Based Access Control** — QA Engineer / QA Manager / Admin roles for 21 CFR Part 11 compliance
- [ ] **Distance-Weighted Voting** — Replace majority vote with distance-weighted voting for improved classification
- [ ] **Batch CSV Upload** — Process multiple incidents simultaneously for post-audit workflows
- [ ] **Hybrid Severity Filtering** — Use severity as an explicit pre-filter to push accuracy above 70%
- [ ] **SHAP Explainability** — Feature-level explanation of root cause classification
- [ ] **Multi-language Support** — Translation preprocessing for non-English incident descriptions

<br/>

## 📜 References

1. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* [sbert.net](https://www.sbert.net/)
2. Johnson, Douze & Jégou (2021). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. [faiss.ai](https://faiss.ai/)
3. [FastAPI Documentation](https://fastapi.tiangolo.com/)
4. [HuggingFace — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
5. ICH Q10: *Pharmaceutical Quality System* (2008).
6. U.S. FDA: *Guidance for Industry — Quality Systems Approach to Pharmaceutical CGMP Regulations.* 21 CFR Part 211.

<br/>

---

<div align="center">

Built by **Billa Sahithi**

[![Live](https://img.shields.io/badge/Try_it_live-intellicapa.onrender.com-7c6fff?style=flat-square)](https://intellicapa.onrender.com)

</div>
