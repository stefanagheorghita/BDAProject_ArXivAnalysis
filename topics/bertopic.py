import json
import os
import sys

import hdbscan
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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


from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def lemmatize_docs(
    docs,
    batch_size=2000,
    n_process=4,
    save_path=None,
    save_batch_size=50_000
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

        if save_path and len(results) >= save_batch_size:
            df = pd.DataFrame({"text": results})
            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(save_path, table.schema)

            writer.write_table(table)
            results.clear()

    if save_path and results:
        df = pd.DataFrame({"text": results})
        table = pa.Table.from_pandas(df)

        if writer is None:
            writer = pq.ParquetWriter(save_path, table.schema)

        writer.write_table(table)
        writer.close()

    return results if not save_path else None






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

if __name__ == "__main__":

    # -------------------------
    # Load data
    # -------------------------
    print("Loading data...")
    docs_df = pd.read_parquet("docs.parquet")
    docs = docs_df["text_clean"].tolist()
    lengths = [(i, len(d)) for i, d in enumerate(docs)]
    lengths.sort(key=lambda x: x[1], reverse=True)

    print(lengths[:5])

    table = pq.read_table(
        "minilm.parquet",
        columns=["id", "embedding"]
    )
    print("read table")

    ids = table["id"].to_numpy()
    print("id")

    emb = table["embedding"].to_numpy()
    print("X")
    X = np.stack(emb).astype(np.float32)

    print("Embeddings:", X.shape)

    os.makedirs("outputs-topics-berteley", exist_ok=True)

    models = []
    samples = []
    metrics = []


    table = pq.read_table("arxiv_lemmas.parquet")
    df = table.to_pandas()

    docs_lemma = df["text"].tolist()

    for run_id, seed in enumerate(range(5)):
        rng = np.random.default_rng(seed)

        print(f"\n=== RUN {run_id} ===")

        run_dir = f"outputs-topics-berteley/{run_id}"
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(f"{run_dir}/topics_words", exist_ok=True)
        os.makedirs(f"{run_dir}/representative_docs", exist_ok=True)


        idx = rng.choice(len(docs), size=250_000, replace=False)

        docs_lemma_s = [docs_lemma[i] for i in idx]
        docs_raw_s = [docs[i] for i in idx]
        X_s = X[idx]


        topic_model, topics, _ = run_topic_model(docs_lemma_s, X_s, seed)

        models.append(topic_model)
        samples.append((idx, X_s, topics))  # FIX 2


        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(f"{run_dir}/topic_info.csv", index=False)


        rep_df = topic_model.get_representative_docs()
        rep_df = (
            pd.DataFrame.from_dict(rep_df, orient="index")
            .stack()
            .reset_index()
            .rename(columns={
                "level_0": "Topic",
                0: "Document"
            })
        )

        for tid in topic_info["Topic"]:
            if tid == -1:
                continue

            with open(f"{run_dir}/topics_words/topic_{tid}.json", "w") as f:
                json.dump(topic_model.get_topic(tid), f, indent=2)

            rep_indices = rep_df.loc[rep_df.Topic == tid]["Document"].values

            raw_reps = rep_indices.tolist()


            pd.DataFrame({"doc": raw_reps}).to_csv(
                f"{run_dir}/representative_docs/topic_{tid}.csv",
                index=False
            )

        counts = topic_info[topic_info["Topic"] != -1].set_index("Topic")["Count"]
        plt.figure(figsize=(8, 4))
        counts.sort_values(ascending=False).plot(kind="bar")
        plt.tight_layout()
        plt.savefig(f"{run_dir}/topic_sizes.png", dpi=150)
        plt.close()

        docs_for_coh = docs_lemma_s[:50_000]
        coherence = compute_coherence(topic_model, docs_for_coh)
        diversity = compute_topic_diversity(topic_model)

        metrics.append({
            "run": run_id,
            "n_topics": len(counts),
            "coherence_c_v": coherence,
            "topic_diversity": diversity,
            "noise_fraction": np.mean(np.array(topics) == -1)
        })

        print("coherence:", coherence)
        print("diversity:", diversity)

    pd.DataFrame(metrics).to_csv(
        "outputs-topics-berteley/metrics_across_runs.csv",
        index=False
    )
















