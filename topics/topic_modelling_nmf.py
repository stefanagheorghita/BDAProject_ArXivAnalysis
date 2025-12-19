import pandas as pd
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import numpy as np

from topics.metrics import coherence_all, topic_diversity
from topics.preprocessing import load_arxiv_sample, build_text_column


def train_tfidf(texts, max_df=0.9, min_df=20, max_features=5000, ngram_range=(1, 1),  dtype=np.float32, stop_words="english") :
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        dtype=dtype
    )

    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def train_nmf(
    X,
    n_topics=20,
    init="nndsvda",
    solver="cd",
    max_iter=500,
    random_state=42
):
    nmf = NMF(
        n_components=n_topics,
        init=init,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )

    W = nmf.fit_transform(X)
    H = nmf.components_
    return nmf, W, H

from typing import List, Dict

def extract_topics(  H,  feature_names: List[str],  top_n: int = 10):
    topics = {}
    for topic_idx, topic in enumerate(H):
        words = [
            feature_names[i]
            for i in topic.argsort()[-top_n:][::-1]
        ]
        topics[topic_idx] = words
    return topics


def print_topics(topics: Dict[int, List[str]]):
    for idx, words in topics.items():
        print(f"Topic {idx}: {', '.join(words)}")





import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_topic_matrix(
    matrix: np.ndarray,
    topics: dict,
    title: str,
    output_path: str,
    cmap="coolwarm"
):
    labels = [f"T{i}" for i in range(len(topics))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        center=0,
        square=True
    )
    plt.title(title)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()



from sklearn.metrics.pairwise import cosine_similarity

def topic_similarity_from_H(H):
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    return cosine_similarity(H_norm)


def analyze_topic_correlations(W, H, topics, output_dir = "topics"):

    sim_H = topic_similarity_from_H(H)
    plot_topic_matrix(
        sim_H,
        topics,
        title="Topic–Topic Similarity (Word Space)",
        output_path=f"{output_dir}/topic_similarity_words.png"
    )

    return sim_H


def plot_topic_words(
    H: np.ndarray,
    feature_names: list,
    topic_idx: int,
    top_n: int = 15,
    output_path: str = None
):
    topic = H[topic_idx]
    top_indices = topic.argsort()[-top_n:][::-1]

    words = [feature_names[i] for i in top_indices]
    weights = topic[top_indices]

    plt.figure(figsize=(8, 4))
    plt.barh(words[::-1], weights[::-1])
    plt.xlabel("Word importance (H value)")
    plt.title(f"Topic {topic_idx} – Top {top_n} words")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.show()



def compute_metrics(topics, df, n_topics, output_path, vectorizer):
    topic_words = [topics[i] for i in sorted(topics.keys())]


    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(text) for text in df["text"]]


    dictionary = Dictionary(tokens)

    corpus = [dictionary.doc2bow(t) for t in tokens]

    cv, cnpmi, umass = coherence_all(
        topic_words,
        tokens,
        dictionary,
        corpus
    )

    tdiv = topic_diversity(topic_words, topn=10)

    metrics = {
        "c_v": cv,
        "c_npmi": cnpmi,
        "u_mass": umass,
        "topic_div@10": tdiv,
        "n_docs": len(df),
        "n_topics": n_topics
    }

    print("\nNMF metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(f"{output_path}/metrics.json", "w") as f:
        json.dump(metrics, f)


def train_topic_model(
    data_path: str,
    sample_path: str,
    output_dir: str,
    chunk_size: int = 100_000,
    n_chunks: int = 5,
    n_topics: int = 5,
    tfidf_params: dict = None,
    nmf_params: dict = None
):
    tfidf_params = tfidf_params or {}
    nmf_params = nmf_params or {}

    df = pd.read_csv(sample_path)

    df = build_text_column(df)

    vectorizer, X = train_tfidf(df["text"], **tfidf_params)
    nmf, W, H = train_nmf(X, n_topics=n_topics, **nmf_params)

    feature_names = vectorizer.get_feature_names_out()
    topics = extract_topics(H, feature_names)

    print_topics(topics)

    output_path = Path(f"{output_dir}-2000000-{n_topics}")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "topics.json", "w") as f:
        json.dump(topics, f, indent=2)

    analyze_topic_correlations(W, H, topics, output_path)

    for idx in topics.keys():
        plot_topic_words(
            H,
            feature_names,
            topic_idx=idx,
            top_n=15,
            output_path=f"{output_path}/topic_{idx}_words.png"
        )

    compute_metrics(topics, df, n_topics, output_path, vectorizer)

    return {
        "vectorizer": vectorizer,
        "nmf": nmf,
        "topics": topics,
        "W": W
    }


if __name__ == "__main__":
    train_topic_model("../arxiv_data/arxiv-metadata-oai-snapshot.json", "samples/arxiv_sample_2000000.csv", "topics")


