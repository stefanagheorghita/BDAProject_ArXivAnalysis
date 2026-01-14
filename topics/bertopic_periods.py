import json
import os
import sys

import hdbscan
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import umap
from bertopic import BERTopic
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.compute as pc


from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
import nltk

# nltk.download("punkt")
# nltk.download('punkt_tab')


import spacy
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

from tqdm import tqdm

from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def lemmatize_docs(
    docs,
    batch_size=2000,
    n_process=4,
    save_path=None,        # <- added
    save_batch_size=50_000 # <- added
):
    results = []
    writer = None

    for i, doc in enumerate(
        tqdm(
            nlp.pipe(
                docs,
                batch_size=batch_size,
                n_process=n_process
            ),
            total=len(docs),
            desc="Lemmatizing documents"
        ),
        1
    ):
        text = " ".join(
            tok.lemma_.lower()
            for tok in doc
            if tok.is_alpha and not tok.is_stop
        )

        results.append(text)

        # ---- incremental save ----
        if save_path and len(results) >= save_batch_size:
            df = pd.DataFrame({"text": results})
            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(save_path, table.schema)

            writer.write_table(table)
            results.clear()

    # ---- save remaining ----
    if save_path and results:
        df = pd.DataFrame({"text": results})
        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(save_path, table.schema)

        writer.write_table(table)
        writer.close()

    return results if not save_path else None






def compute_coherence(topic_model, docs, top_n=10):
    # tokenize documents
    tokenized_docs = [d.split() for d in docs]

    # extract topic words
    topic_words = []
    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        topic_words.append([w for w, _ in words[:top_n]])

    if len(topic_words) == 0:
        return np.nan

    dictionary = Dictionary(tokenized_docs)

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v"
    )

    return coherence_model.get_coherence()


def compute_topic_diversity(topic_model, top_n=10):
    all_words = []
    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        all_words.extend([w for w, _ in words[:top_n]])

    if len(all_words) == 0:
        return np.nan

    return len(set(all_words)) / len(all_words)



def get_topic_words(topic_model, top_n=10):
    topics = {}
    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        topics[topic_id] = [w for w, _ in words[:top_n]]
    return list(topics.values())



def embed_partition(rows):
    from sentence_transformers import SentenceTransformer

    rows = list(rows)
    if not rows:
        return

    model = SentenceTransformer("allenai/specter")

    texts = [r.text_clean for r in rows]
    ids = [r.id for r in rows]

    emb = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    for i, e in zip(ids, emb):
        yield (i, e.tolist())





def load_data(spark, path):
    df = spark.read.json(path)
    return df

def assign_topics(X_all, meta_centroids, threshold=0.3):
    sims = cosine_similarity(X_all, meta_centroids)
    best = sims.argmax(axis=1)
    scores = sims.max(axis=1)
    return np.where(scores > threshold, best, -1)

import matplotlib.pyplot as plt

def plot_topic_sizes(
    topics,
    title,
    out_path,
    show=False
):
    counts = pd.Series(topics).value_counts()
    counts = counts[counts.index != -1]

    plt.figure(figsize=(8, 4))
    counts.sort_values(ascending=False).plot(kind="bar")
    plt.title(title)
    plt.ylabel("Documents")
    plt.xlabel("Topic ID")
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def sample_data(docs, X, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(docs), size=n, replace=False)
    return [docs[i] for i in idx], X[idx]



def run_topic_model(docs, X, seed):
    umap_model = umap.UMAP(
        n_neighbors=50,
        n_components=5,
        min_dist=0.1,
        metric="cosine",
        low_memory=True,
        random_state=seed
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=150,
        min_samples=30,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=8
    )

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=False
    )



    topics, probs = topic_model.fit_transform(docs, embeddings=X)
    return topic_model, topics, probs

def get_topic_centroids(X, topics):
    centroids = {}
    for t in set(topics):
        if t == -1:
            continue
        idx = np.where(np.array(topics) == t)[0]
        centroids[t] = X[idx].mean(axis=0)
    return centroids


def make_periods(df):
    return {
        "1985_1994": df[(df.year >= 1985) & (df.year <= 1994)],
        "1995_2004": df[(df.year >= 1995) & (df.year <= 2004)],
        "2005_2014": df[(df.year >= 2005) & (df.year <= 2014)],

        "2015_2019": df[(df.year >= 2015) & (df.year <= 2019)],
        "2020_2024": df[(df.year >= 2020) & (df.year <= 2024)],
        "2025":      df[df.year == 2025],
    }


if __name__ == "__main__":

    # =========================
    # 1. Load document metadata
    # =========================
    print("Loading documents...")
    docs_df = pd.read_parquet("docs.parquet")

    # sanity check
    assert "id" in docs_df.columns
    assert "text_clean" in docs_df.columns
    assert "year" in docs_df.columns

    # =========================
    # 2. Load embeddings
    # =========================
    print("Loading embeddings...")
    table = pq.read_table(
        "minilm.parquet",
        columns=["id", "embedding"]
    )

    ids = table["id"].to_numpy()
    emb = table["embedding"].to_numpy()
    X = np.stack(emb).astype(np.float32)

    print("Embeddings shape:", X.shape)

    # build ID → embedding row mapping
    id_to_row = {doc_id: i for i, doc_id in enumerate(ids)}

    # =========================
    # 3. Load lemmatized text
    # =========================
    print("Loading lemmatized documents...")
    lemma_table = pq.read_table("arxiv_lemmas.parquet")
    lemma_df = lemma_table.to_pandas()

    assert "text" in lemma_df.columns

    # align lemma DF with docs DF (by index)
    docs_df["text_lemma"] = lemma_df["text"].values


    periods = make_periods(docs_df)

    os.makedirs("outputs-topics-berteley_periods", exist_ok=True)


    for period_name, df_p in periods.items():
        print(f"\n===== PERIOD {period_name} =====")

        doc_ids = df_p["id"].values

        rows = [id_to_row[i] for i in doc_ids if i in id_to_row]

        # aligned data
        docs_raw = df_p.loc[df_p["id"].isin(doc_ids), "text_clean"].tolist()
        docs_lemma_p = df_p.loc[df_p["id"].isin(doc_ids), "text_lemma"].tolist()
        X_p = X[rows]


        period_dir = f"outputs-topics-berteley_periods/period_{period_name}"
        os.makedirs(period_dir, exist_ok=True)


        topic_model, topics, _ = run_topic_model(
            docs_lemma_p,
            X_p,
            seed=42
        )

        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(f"{period_dir}/topic_info.csv", index=False)

      
        coherence = compute_coherence(
            topic_model,
            docs_lemma_p[:50_000]
        )
        diversity = compute_topic_diversity(topic_model)

        print(
            f"Topics: {len(topic_info) - 1}, "
            f"coherence={coherence:.3f}, "
            f"diversity={diversity:.3f}"
        )
