import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter("ignore", InsecureRequestWarning)

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
import os
import faiss

ES_URL = "https://localhost:9200"
ES_USER = "elastic"
ES_PASS = "tOx20ZWJwDITugqKOiuq"
INDEX_NAME = "arxiv"
OUT_DIR = "faiss_specter_results"

OUT_DIR_IND = "faiss_indexes"

INDEX_PATH = f"{OUT_DIR_IND}/minilm_hnsw.index"
IDS_PATH = f"{OUT_DIR_IND}/minilm_ids.npy"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading ID mapping...")
ids = np.load(IDS_PATH)

print("Index size:", index.ntotal)
print("IDs loaded:", len(ids))

TOP_K_BM25 = 1000
TOP_K_FINAL = 20

QUERIES = {
    "gnn_citations": "graph neural networks for citation analysis",
    "quantum_field": "quantum field theory gauge symmetry",
    "optimization": "convex optimization gradient descent convergence",
    "language_models": "large language models transformers pretraining",
}

# ================================
# Elasticsearch
# ================================
es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASS),
    verify_certs=False,
    request_timeout=300,
)


spark = (
    SparkSession.builder
    .appName("ArXiv-Faiss-Specter")
    .config("spark.executor.cores", "2")
    .config("spark.executor.memory", "8g")
    .config("spark.executor.memoryOverhead", "2g")
    .config("spark.driver.memory", "6g")
    .config("spark.python.worker.reuse", "true")
    .getOrCreate()
)

model_minilm = SentenceTransformer("all-MiniLM-L6-v2")

ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"

meta_df = (
    spark.read.json(ARXIV_JSON)
    .select("id", "title", "abstract", "categories")
    .filter(col("abstract").isNotNull())
)

print("Loading SPECTER...")
model = SentenceTransformer("allenai/specter")

def fetch_metadata(doc_ids):
    rows = (
        meta_df
        .filter(col("id").isin(doc_ids))
        .select("id", "title", "abstract", "categories")
        .collect()
    )

    return {
        r["id"]: {
            "title": r["title"],
            "abstract": r["abstract"],
            "categories": r["categories"]
        }
        for r in rows
    }

TOP_K_FAISS = 1000

def faiss_candidates(query_text, k=TOP_K_FAISS):
    qvec = model_minilm.encode(
        [query_text],
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(qvec, k)

    return [str(ids[idx]) for idx in I[0]]


def query_by_text(query_text: str, top_k_final: int = TOP_K_FINAL):
    doc_ids = faiss_candidates(query_text)

    if not doc_ids:
        return pd.DataFrame()

    meta = fetch_metadata(doc_ids)

    texts = [
        meta[d]["abstract"]
        for d in doc_ids
        if d in meta
    ]

    valid_ids = [d for d in doc_ids if d in meta]

    q_emb = model.encode(query_text, normalize_embeddings=True)
    d_embs = model.encode(texts, batch_size=16, normalize_embeddings=True)

    scores = cosine_similarity(q_emb.reshape(1, -1), d_embs)[0]

    reranked = sorted(
        zip(valid_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k_final]

    rows = []
    for rank, (doc_id, score) in enumerate(reranked, start=1):
        m = meta[doc_id]
        rows.append({
            "query_type": "text",
            "query_value": query_text,
            "rank": rank,
            "doc_id": doc_id,
            "specter_score": float(score),
            "title": m["title"],
            "categories": m["categories"],
            "abstract": m["abstract"],
        })

    return pd.DataFrame(rows)

def query_by_id(doc_id: str, top_k_final: int = TOP_K_FINAL):
    row = (
        meta_df
        .filter(col("id") == doc_id)
        .select("title", "abstract")
        .first()
    )

    if row is None:
        raise ValueError(f"ID not found: {doc_id}")

    query_text = f"{row['title']}. {row['abstract']}"

    doc_ids = faiss_candidates(query_text)

    meta = fetch_metadata(doc_ids)

    texts = [
        meta[d]["abstract"]
        for d in doc_ids
        if d in meta
    ]

    valid_ids = [d for d in doc_ids if d in meta]

    q_emb = model.encode(query_text, normalize_embeddings=True)
    d_embs = model.encode(texts, batch_size=16, normalize_embeddings=True)

    scores = cosine_similarity(q_emb.reshape(1, -1), d_embs)[0]

    reranked = sorted(
        zip(valid_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    rows = []
    rank = 1
    for hit_id, score in reranked:
        if hit_id == doc_id:
            continue

        m = meta[hit_id]
        rows.append({
            "query_type": "id",
            "query_value": doc_id,
            "query_title": row["title"],
            "query_abstract": row["abstract"],
            "rank": rank,
            "doc_id": hit_id,
            "specter_score": float(score),
            "title": m["title"],
            "categories": m["categories"],
            "abstract": m["abstract"],
        })

        rank += 1
        if rank > top_k_final:
            break

    return pd.DataFrame(rows)


def save_results(df, name):
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


QUERIES = {
    "gnn_citations": "graph neural networks for citation analysis",
    "quantum_field": "quantum field theory gauge symmetry",
    "optimization": "convex optimization gradient descent convergence",
    "language_models": "large language models transformers pretraining",
}

    
if __name__ == "__main__":
    save_results(query_by_id("1810.04805"), "similar_to_1810.04805")
    save_results(query_by_text("Neutrino decays as a natural explanation of the neutrino mass tension"), "neutrino_title")
    save_results(query_by_text("This paper develops a unified framework for analyzing technology adoption in financial networks that incorporates spatial spillovers, network externalities, and their interaction. The framework characterizes adoption dynamics through a master equation whose solution admits a Feynman-Kac representation as expected cumulative adoption pressure along stochastic paths through spatial-network space. From this representation, I derive the Adoption Amplification Factor -- a structural measure of technology leadership that captures the ratio of total system-wide adoption to initial adoption following a localized shock. A Levy jump-diffusion extension with state-dependent jump intensity captures critical mass dynamics: below threshold, adoption evolves through gradual diffusion; above threshold, cascade dynamics accelerate adoption through discrete jumps. Applying the framework to SWIFT gpi adoption among 17 Global Systemically Important Banks, I find strong support for the two-regime characterization. Network-central banks adopt significantly earlier (\rho = -0.69, p = 0.002), and pre-threshold adopters have significantly higher amplification factors than post-threshold adopters (11.81 versus 7.83, p = 0.010). Founding members, representing 29 percent of banks, account for 39 percent of total system amplification -- sufficient to trigger cascade dynamics. Controlling for firm size and network position, CEO age delays adoption by 11-15 days per year."), "economy_abstract")
    for query_name, query in QUERIES.items():
        save_results(query_by_text(query), query_name)

