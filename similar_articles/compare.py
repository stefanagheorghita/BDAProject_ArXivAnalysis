import os
import pandas as pd
from itertools import combinations

# -----------------------
# CONFIG
# -----------------------
BASE_DIRS = {
    "tfidf": "tfidf_query_results",
    "bm25": "bm25_results",
    "bm25_specter": "bm25_specter_results",
    "minilm": "minilm_query_results",
    "faiss_specter": "faiss_specter_results",
}

TOP_K = 20
OUT_FILE = "overlap_comparison.csv"


def load_ids(csv_path, k=20):
    df = pd.read_csv(csv_path)
    if "doc_id" in df.columns:
        return df["doc_id"].head(k).astype(str).tolist()
    elif "id" in df.columns:
        return df["id"].head(k).astype(str).tolist()
    else:
        raise ValueError(f"No id column in {csv_path}")

def overlap_at_k(a, b, k=20):
    return len(set(a) & set(b)) / k


example_dir = next(iter(BASE_DIRS.values()))
queries = sorted([
    f for f in os.listdir(example_dir)
    if f.endswith(".csv")
])

rows = []

for q in queries:
    results = {}
    for method, folder in BASE_DIRS.items():
        path = os.path.join(folder, q)
        if not os.path.exists(path):
            continue
        results[method] = load_ids(path, TOP_K)

    for (m1, ids1), (m2, ids2) in combinations(results.items(), 2):
        rows.append({
            "query": q,
            "method_1": m1,
            "method_2": m2,
            "overlap@20": overlap_at_k(ids1, ids2, TOP_K),
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_FILE, index=False)

print(f"Saved → {OUT_FILE}")
