"""
intelliCAPA — Retrieval System Evaluation
==========================================

Evaluates the quality of the FAISS semantic retrieval pipeline
using Leave-One-Out (LOO) methodology on the cleaned_dataset.

How it works:
  - 300 random incidents are selected from the dataset (seed=42)
  - Each incident's own stored embedding is used as the search query
  - The exact self-match is excluded from top-6 results (leave-one-out)
  - The remaining 5 results are used to compute evaluation metrics

Metrics computed:
  1. Root Cause Accuracy   — Did majority vote predict the correct root cause?
  2. Severity Accuracy     — Did majority vote predict the correct severity?
  3. Precision @ 5 (P@5)  — What fraction of top-5 share the correct root cause?
  4. MRR @ 5              — How high does the correct root cause appear in ranked list?
  5. Avg Top-1 Similarity  — How semantically close is the #1 retrieved incident?
  6. Per-class breakdown   — Accuracy split by each root cause category

Run:
  cd backend
  python evaluate.py

  Optional flags:
  python evaluate.py --samples 500   # evaluate on more samples
  python evaluate.py --k 3           # use top-3 instead of top-5
  python evaluate.py --verbose       # print per-sample results
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent / "models"
EMB_PATHS = [
    BASE_DIR / "capa_embeddings.npy",
    Path(__file__).resolve().parent.parent.parent /
    "CAPA-RS-CG" / "Retrieval_System" / "capa_embeddings.npy",
]


# ── METRIC HELPERS ─────────────────────────────────────────────────────────────

def precision_at_k(retrieved_labels: list, true_label: str) -> float:
    """
    Precision@k = (number of retrieved items with correct label) / k

    Example:
      retrieved = ['Env', 'Env', 'Env', 'Doc', 'Env']  true = 'Env'
      P@5 = 4/5 = 0.80
    """
    return sum(1 for r in retrieved_labels if r == true_label) / len(retrieved_labels)


def reciprocal_rank(retrieved_labels: list, true_label: str) -> float:
    """
    Reciprocal Rank = 1 / (rank of first correct result)

    Example:
      retrieved = ['Env', 'Env', 'Doc', 'Env', 'Env']  true = 'Env'
      RR = 1/1 = 1.0  (first result is correct)

      retrieved = ['Doc', 'Env', 'Env', 'Env', 'Env']  true = 'Env'
      RR = 1/2 = 0.50  (second result is the first correct one)
    """
    for rank, label in enumerate(retrieved_labels, start=1):
        if label == true_label:
            return 1.0 / rank
    return 0.0


def l2_to_similarity(distance: float) -> float:
    """
    Convert L2 (Euclidean) distance to a 0-100% similarity score.
    Formula: sim = max(0, (1 - distance/2) * 100)

    L2 distance of 0   → 100% similarity (identical vectors)
    L2 distance of 2   →   0% similarity (orthogonal vectors)
    L2 distance > 2    → clamped to 0%
    """
    return max(0.0, (1.0 - distance / 2.0) * 100.0)


# ── LOAD ASSETS ────────────────────────────────────────────────────────────────

def load_assets():
    print("Loading assets...")

    # Dataset
    df_path = BASE_DIR / "cleaned_dataset.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"cleaned_dataset.csv not found at {df_path}")
    df = pd.read_csv(df_path)
    print(f"  ✓ cleaned_dataset.csv loaded  — {len(df)} rows")

    # FAISS index
    index_path = BASE_DIR / "capa_index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"capa_index.faiss not found at {index_path}")
    index = faiss.read_index(str(index_path))
    print(f"  ✓ capa_index.faiss loaded     — {index.ntotal} vectors, dim={index.d}")

    # Pre-computed embeddings
    emb_path = None
    for p in EMB_PATHS:
        if p.exists():
            emb_path = p
            break
    if emb_path is None:
        raise FileNotFoundError(
            "capa_embeddings.npy not found. Expected in models/ or "
            "CAPA-RS-CG/Retrieval_System/"
        )
    embeddings = np.load(str(emb_path)).astype("float32")
    print(f"  ✓ capa_embeddings.npy loaded  — shape {embeddings.shape}")

    return df, index, embeddings


# ── MAIN EVALUATION ────────────────────────────────────────────────────────────

def evaluate(sample_size: int = 300, k: int = 5, verbose: bool = False):

    df, index, embeddings = load_assets()
    print()

    # Sample indices
    np.random.seed(42)
    sample_idx = np.random.choice(len(df), size=sample_size, replace=False)
    print(f"Evaluating on {sample_size} random samples (seed=42, k={k})...")
    print("─" * 55)

    # Accumulators
    rc_correct    = 0
    sev_correct   = 0
    p_at_k_total  = 0.0
    mrr_total     = 0.0
    sim_total     = 0.0

    # Per-class tracking
    rc_class_total   = Counter()
    rc_class_correct = Counter()

    start_time = time.time()

    for i, idx in enumerate(sample_idx):
        row      = df.iloc[idx]
        true_rc  = row["root_cause"]
        true_sev = row["severity_level"]

        # ── Query: use this incident's own stored embedding ──────────
        query_vec = embeddings[idx : idx + 1]          # shape (1, 384)

        # ── FAISS search: top k+1 to allow excluding self-match ──────
        distances, indices = index.search(query_vec, k + 1)

        # ── Leave-One-Out: remove the exact self-match ───────────────
        results = [
            (int(ind), float(dist))
            for ind, dist in zip(indices[0], distances[0])
            if int(ind) != idx and int(ind) < len(df)
        ][:k]   # keep exactly k

        if len(results) < k:
            # Edge case: dataset too small → skip
            continue

        # ── Retrieve labels and distances ────────────────────────────
        retrieved_rc  = [df.iloc[r[0]]["root_cause"]    for r in results]
        retrieved_sev = [df.iloc[r[0]]["severity_level"] for r in results]
        top_distance  = results[0][1]

        # ── 1. Root Cause Accuracy ────────────────────────────────────
        pred_rc = Counter(retrieved_rc).most_common(1)[0][0]
        if pred_rc == true_rc:
            rc_correct += 1
            rc_class_correct[true_rc] += 1
        rc_class_total[true_rc] += 1

        # ── 2. Severity Accuracy ──────────────────────────────────────
        pred_sev = Counter(retrieved_sev).most_common(1)[0][0]
        if pred_sev == true_sev:
            sev_correct += 1

        # ── 3. Precision @ k ─────────────────────────────────────────
        p_at_k_total += precision_at_k(retrieved_rc, true_rc)

        # ── 4. MRR @ k ───────────────────────────────────────────────
        mrr_total += reciprocal_rank(retrieved_rc, true_rc)

        # ── 5. Top-1 Similarity ───────────────────────────────────────
        sim_total += l2_to_similarity(top_distance)

        # ── Verbose output ────────────────────────────────────────────
        if verbose:
            correct = "✓" if pred_rc == true_rc else "✗"
            print(
                f"  [{correct}] Sample {i+1:3d} | "
                f"True: {true_rc:<30} | "
                f"Pred: {pred_rc:<30} | "
                f"Sim: {l2_to_similarity(top_distance):5.1f}%"
            )

    elapsed = time.time() - start_time
    n = sample_size

    # ── RESULTS ───────────────────────────────────────────────────────────────
    print()
    print("═" * 55)
    print("   intelliCAPA — Evaluation Results")
    print("═" * 55)
    print(f"  Samples Evaluated      : {n}")
    print(f"  k (neighbours)         : {k}")
    print(f"  Method                 : Leave-One-Out")
    print(f"  Evaluation Time        : {elapsed:.1f}s")
    print("─" * 55)
    print(f"  Root Cause Accuracy    : {rc_correct/n*100:6.1f}%")
    print(f"  Severity Accuracy      : {sev_correct/n*100:6.1f}%")
    print(f"  Precision @ {k}          : {p_at_k_total/n:6.3f}")
    print(f"  MRR @ {k}               : {mrr_total/n:6.3f}")
    print(f"  Avg Top-1 Similarity   : {sim_total/n:6.1f}%")
    print("═" * 55)

    # ── PER-CLASS BREAKDOWN ───────────────────────────────────────────────────
    print()
    print("  Root Cause Accuracy — Per Class")
    print("─" * 55)
    for rc in sorted(rc_class_total.keys()):
        total   = rc_class_total[rc]
        correct = rc_class_correct.get(rc, 0)
        pct     = correct / total * 100
        bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {rc:<35} {bar}  {pct:5.1f}%  ({correct}/{total})")
    print("─" * 55)

    # ── SEVERITY NOTE ─────────────────────────────────────────────────────────
    print()
    print("  Note on Severity Accuracy:")
    print("  ─────────────────────────────────────────────────")
    print("  Severity is an administrative label assigned AFTER")
    print("  the incident description is written. The embedding")
    print("  model captures semantic meaning — not severity")
    print("  labels. Therefore lower severity accuracy is")
    print("  expected and is NOT a defect in the system.")
    print()
    print("  To improve severity accuracy, a hybrid approach")
    print("  is recommended: semantic search + severity pre-filter.")
    print("═" * 55)
    print()

    # Return dict for programmatic use
    return {
        "samples_evaluated"   : n,
        "k"                   : k,
        "root_cause_accuracy" : round(rc_correct / n * 100, 1),
        "severity_accuracy"   : round(sev_correct / n * 100, 1),
        "precision_at_k"      : round(p_at_k_total / n, 3),
        "mrr_at_k"            : round(mrr_total / n, 3),
        "avg_top1_similarity" : round(sim_total / n, 1),
        "evaluation_time_sec" : round(elapsed, 1),
        "per_class_accuracy"  : {
            rc: round(rc_class_correct.get(rc, 0) / rc_class_total[rc] * 100, 1)
            for rc in sorted(rc_class_total.keys())
        },
    }


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate intelliCAPA retrieval system"
    )
    parser.add_argument(
        "--samples", type=int, default=300,
        help="Number of random samples to evaluate (default: 300)"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of nearest neighbours to retrieve (default: 5)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample results"
    )
    args = parser.parse_args()

    evaluate(sample_size=args.samples, k=args.k, verbose=args.verbose)
