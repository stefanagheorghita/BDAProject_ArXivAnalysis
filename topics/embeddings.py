import os
import sys

from pyspark.sql.functions import col, concat_ws, trim, regexp_replace, lower, split
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "specter": "allenai/specter",
    "aspire": "allenai/aspire",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "scibert": "sentence-transformers/scibert-scivocab-uncased"
}


_MODEL_CACHE = {}

def embed_partition(rows, model_name):
    rows = list(rows)
    if not rows:
        return

    global _MODEL_CACHE
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    model = _MODEL_CACHE[model_name]

    texts = [r.text_clean for r in rows]
    ids = [r.id for r in rows]

    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True
    )

    for doc_id, emb in zip(ids, embeddings):
        yield (doc_id, emb.tolist())


if __name__ == "__main__":
    path = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"

    spark = (
        SparkSession.builder
        .appName("Multi-Embedding Pipeline")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.driver.memory", "6g")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )

    data = spark.read.json(path)

    data = (
        data
        .select(
            col("id"),
            concat_ws(". ", col("title"), col("abstract")).alias("text"),
        )
        .filter(col("text").isNotNull())
        .withColumn(
            "text_clean",
            trim(regexp_replace(lower(col("text")), r"\s+", " "))
        )
        .select("id", "text_clean")
    )

    data.write.mode("overwrite").parquet(
        "hdfs:///user/ubuntu/docs/docs.parquet"
    )

    data = data.repartition(256)


    for name, model_name in EMBEDDING_MODELS.items():
        print(f"Running embeddings for: {name}")

        embeddings_rdd = (
            data
            .rdd
            .mapPartitions(
                lambda rows, m=model_name: embed_partition(rows, m)
            )
        )

        embeddings_df = embeddings_rdd.toDF(["id", "embedding"])

        out_path = f"hdfs:///user/ubuntu/embeddings/{name}"
        embeddings_df.write.mode("overwrite").parquet(out_path)

        print(f"Saved {name} embeddings to {out_path}")



   