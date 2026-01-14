import os
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch

# ======================
# CONFIG
# ======================
INPUT_PATH = "/kaggle/input/docs-parquet/docs.parquet"
OUT_DIR = "/kaggle/working/embeddings"
BATCH_SIZE = 256         # Safe on T4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2"
}

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_parquet(INPUT_PATH, columns=["id", "text_clean"])
ids = df["id"].tolist()
texts = df["text_clean"].tolist()

print(f"Loaded {len(texts):,} documents")

# Arrow schema (important for streaming)
schema = pa.schema([
    ("id", pa.string()),
    ("embedding", pa.list_(pa.float32()))
])

# ======================
# EMBEDDING LOOP
# ======================
for name, model_name in EMBEDDING_MODELS.items():
    print(f"\n=== Embedding with {name} ===")

    model = SentenceTransformer(model_name, device=DEVICE)
    model.eval()

    out_path = f"{OUT_DIR}/{name}.parquet"
    writer = None

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        with torch.no_grad():
            emb = model.encode(
                batch_texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")

        table = pa.Table.from_arrays(
            [
                pa.array(batch_ids),
                pa.array(emb.tolist())
            ],
            schema=schema
        )

        if writer is None:
            writer = pq.ParquetWriter(
                out_path,
                schema=schema,
                compression="snappy"
            )

        writer.write_table(table)

        if i % (10_000) == 0:
            print(f"Processed {i:,}/{len(texts):,}")

        del emb, table
        torch.cuda.empty_cache()

    writer.close()
    print(f"Saved embeddings to {out_path}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

print("All embeddings completed successfully.")