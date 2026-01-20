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
    .appName("ArXiv-MiniLM-HNSW")
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

ids = []
index = None
dim = None

print("Building HNSW index (memory-safe)...")

for rg in tqdm(range(pf.num_row_groups), desc="Row groups"):
    if MAX_ROW_GROUPS and rg >= MAX_ROW_GROUPS:
        break

    batch = pf.read_row_group(rg, columns=["id", "embedding"])
    emb = np.vstack(batch["embedding"].to_numpy()).astype("float32")

    if dim is None:
        dim = emb.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100

    faiss.normalize_L2(emb)

    for i in range(0, emb.shape[0], ADD_BATCH):
        index.add(emb[i:i + ADD_BATCH])

    ids.extend(batch["id"].to_numpy())

ids = np.array(ids)

print("HNSW index size:", index.ntotal)
import faiss
import numpy as np
import os

OUT_DIR = "faiss_indexes"
os.makedirs(OUT_DIR, exist_ok=True)

INDEX_PATH = f"{OUT_DIR}/minilm_hnsw.index"
IDS_PATH = f"{OUT_DIR}/minilm_ids.npy"

print("Saving FAISS index...")
faiss.write_index(index, INDEX_PATH)

print("Saving ID mapping...")
np.save(IDS_PATH, ids)

print("Saved:")
print(" ", INDEX_PATH)
print(" ", IDS_PATH)