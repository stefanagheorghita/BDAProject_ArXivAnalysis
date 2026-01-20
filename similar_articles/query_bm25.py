from elasticsearch import Elasticsearch
import pandas as pd
import warnings
from urllib3.exceptions import InsecureRequestWarning
from pyspark.sql import SparkSession

warnings.simplefilter("ignore", InsecureRequestWarning)
import os
import pandas as pd

OUT_DIR = "bm25_results"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K = 20
INDEX_NAME = "arxiv"

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "tOx20ZWJwDITugqKOiuq"),
    verify_certs=False,
    request_timeout=300,
    max_retries=10,
    retry_on_timeout=True
)
from pyspark.sql.functions import col, concat_ws
ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"
spark = (
    SparkSession.builder
    .appName("ArXiv-BM25-MultiQuery-Search")
    .getOrCreate()
)

data = (
    spark.read.json(ARXIV_JSON)
    .select(
        col("id"),
        col("categories"),
        col("title"),
        col("abstract"),
    )
    .filter(col("abstract").isNotNull())
)

def fetch_metadata_spark(doc_ids):
    rows = (
        data
        .filter(col("id").isin(doc_ids))
        .select("id", "title", "abstract", "categories")
        .collect()
    )

    return {
        r["id"]: {
            "title": r["title"],
            "abstract": r["abstract"],
            "categories": r["categories"],
        }
        for r in rows
    }


def query_by_text(query_text: str, top_k: int = TOP_K):
    resp = es.search(
        index=INDEX_NAME,
        size=top_k,
        query={"match": {"text": query_text}}
    )

    hits = resp["hits"]["hits"]
    doc_ids = [h["_id"] for h in hits]

    meta = fetch_metadata_spark(doc_ids)

    rows = []
    for rank, hit in enumerate(hits, start=1):
        doc_id = hit["_id"]
        m = meta.get(doc_id, {})

        rows.append({
            "query_type": "text",
            "query_value": query_text,
            "rank": rank,
            "doc_id": doc_id,
            "score": float(hit["_score"]),
            "title": m.get("title"),
            "abstract": m.get("abstract"),
            "categories": m.get("categories"),
        })

    return pd.DataFrame(rows)

def query_by_id(doc_id: str, top_k: int = TOP_K):
    row = (
        data
        .filter(col("id") == doc_id)
        .select("title", "abstract")
        .first()
    )

    if row is None:
        raise ValueError(f"ID not found: {doc_id}")

    query_text = f"{row['title']}. {row['abstract']}"

    resp = es.search(
        index=INDEX_NAME,
        size=top_k + 1, 
        query={"match": {"text": query_text}}
    )

    hits = resp["hits"]["hits"]
    doc_ids = [h["_id"] for h in hits]

    meta = fetch_metadata_spark(doc_ids)

    rows = []
    rank = 1

    for hit in hits:
        hit_id = hit["_id"]
        if hit_id == doc_id:
            continue

        m = meta.get(hit_id, {})

        rows.append({
            "query_type": "id",
            "query_value": doc_id,
            "query_title": row["title"],
            "query_abstract": row["abstract"],
            "rank": rank,
            "doc_id": hit_id,
            "score": float(hit["_score"]),
            "title": m.get("title"),
            "abstract": m.get("abstract"),
            "categories": m.get("categories"),
        })

        rank += 1
        if rank > top_k:
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
