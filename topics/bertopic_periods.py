import json
import os
import sys

import hdbscan
import numpy as np

import umap
from bertopic import BERTopic

from sklearn.metrics.pairwise import cosine_similarity


from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


import spacy

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)


import pandas as pd
import pyarrow.parquet as pq



def compute_coherence(topic_model, docs, top_n=10):
    tokenized_docs = [d.split() for d in docs]

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





def load_data(spark, path):
    df = spark.read.json(path)
    return df


def assign_topics(X_all, meta_centroids, threshold=0.3):
    sims = cosine_similarity(X_all, meta_centroids)
    best = sims.argmax(axis=1)
    scores = sims.max(axis=1)
    return np.where(scores > threshold, best, -1)




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





def make_periods(df):
    return {
        # "1990_2000": df[(df.year >= 1990) & (df.year <= 2000)],
        # "2000_2010": df[(df.year >= 2000) & (df.year <= 2010)],
        # "2005_2014": df[(df.year >= 2005) & (df.year <= 2014)],
        #
        # "2015_2019": df[(df.year >= 2015) & (df.year <= 2019)],
        "2020_2022": df[(df.year >= 2020) & (df.year <= 2022)],
        "2023_2024": df[(df.year >= 2023) & (df.year <= 2024)],

        "2025": df[df.year == 2025],
    }


if __name__ == "__main__":


    print("Loading documents...")
    docs_df = pd.read_parquet("docs_with_year.parquet")


    print("Loading embeddings...")
    table = pq.read_table(
        "minilm.parquet",
        columns=["id", "embedding"]
    )

    ids = table["id"].to_numpy()
    emb = table["embedding"].to_numpy()
    X = np.stack(emb).astype(np.float32)

    print("Embeddings shape:", X.shape)

    id_to_row = {doc_id: i for i, doc_id in enumerate(ids)}


    print("Loading lemmatized documents...")
    lemma_table = pq.read_table("arxiv_lemmas.parquet")
    lemma_df = lemma_table.to_pandas()

    assert "text" in lemma_df.columns

    docs_df["text_lemma"] = lemma_df["text"].values

    periods = make_periods(docs_df)

    os.makedirs("outputs-topics-berteley_periods", exist_ok=True)

    for period_name, df_p in periods.items():
        print(f"\n===== PERIOD {period_name} =====")

        doc_ids = df_p["id"].values

        rows = [id_to_row[i] for i in doc_ids if i in id_to_row]

        docs_raw = df_p.loc[df_p["id"].isin(doc_ids), "text"].tolist()
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
