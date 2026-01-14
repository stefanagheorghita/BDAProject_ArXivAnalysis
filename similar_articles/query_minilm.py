

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from sentence_transformers import SentenceTransformer


EMB_PATH = "minilm.parquet"     
ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"
OUT_DIR = "minilm_query_results"

TOP_K = 20
ADD_BATCH = 20_000              
MAX_ROW_GROUPS = None         

QUERIES = {
    "gnn_citations": "graph neural networks for citation analysis",
    "quantum_field": "quantum field theory gauge symmetry",
    "optimization": "convex optimization gradient descent convergence",
    "language_models": "large language models transformers pretraining",
}

os.makedirs(OUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("ArXiv-MiniLM-HNSW-Query")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

meta = (
    spark.read.json(ARXIV_JSON)
    .select("id", "categories", "title", "abstract")
    .filter(col("abstract").isNotNull())
    .cache()
)

pf = pq.ParquetFile(EMB_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")

import faiss
import numpy as np
import os

OUT_DIR = "faiss_indexes"
OUT_DIR_RESULTS = "minilm_query_results"

INDEX_PATH = f"{OUT_DIR}/minilm_hnsw.index"
IDS_PATH = f"{OUT_DIR}/minilm_ids.npy"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading ID mapping...")
ids = np.load(IDS_PATH)

print("Index size:", index.ntotal)
print("IDs loaded:", len(ids))

def fetch_meta(ids_list):
    return (
        spark.createDataFrame([(i,) for i in ids_list], ["id"])
        .join(meta, on="id", how="left")
    )

def get_text_for_id(doc_id: str):
    row = (
        meta
        .filter(col("id") == doc_id)
        .select("title", "abstract")
        .first()
    )

    if row is None:
        raise ValueError(f"ID not found in metadata: {doc_id}")

    return row["title"], row["abstract"]

def get_embedding_for_id(doc_id: str):
    idx = np.where(ids == doc_id)[0]
    if len(idx) == 0:
        raise ValueError(f"ID not found in FAISS mapping: {doc_id}")

    return index.reconstruct(int(idx[0])).reshape(1, -1)

def query_by_text(query_text: str, top_k: int = 20):
    qvec = model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(qvec, top_k)

    rows = [
        (
            "text",
            str(query_text),   
            str(ids[idx]),            
            int(rank + 1),          
            float(score),             
        )
        for rank, (idx, score) in enumerate(zip(I[0][::-1], D[0][::-1]))
    ]


    df = spark.createDataFrame(
        rows,
        ["query_type", "query_value",  "id", "rank", "similarity"]
    )

    return (
        df.join(meta, "id", "left")
          .orderBy("rank")
          .select(
              "query_type",
              "query_value",
              "rank",
              "id",
              "categories",
              "similarity",
              "title",
              "abstract",
          )
    )

def query_by_id(doc_id: str, top_k: int = 20):
    qvec = get_embedding_for_id(doc_id)

    query_title, query_abstract = get_text_for_id(doc_id)

    D, I = index.search(qvec, top_k + 1)  

    rows = []
    rank = top_k

    for idx, score in zip(I[0], D[0]):
        hit_id = ids[idx]
        if hit_id == doc_id:
            continue

        rows.append(
            (
                "id",
                str(doc_id),              
                str(query_title),     
                str(query_abstract),  
                str(hit_id),                
                int(rank),                 
                float(score),               
            )
        )

        rank -= 1


    df = spark.createDataFrame(
        rows,
        [
            "query_type",
            "query_value",
            "query_title",
            "query_abstract",
            "id",
            "rank",
            "similarity",
        ]
    )

    return (
        df.join(meta, "id", "left")
          .orderBy("rank")
          .select(
            "query_type",
            "query_value",
            "query_title",
            "query_abstract",
            "id",
            "similarity",
              "rank",
              "categories",
              "title",
              "abstract",
          )
    )

def save_results(df, name):
    out_path = f"{OUT_DIR_RESULTS}/{name}.csv"
    df.toPandas().to_csv(out_path, index=False)
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
        
    