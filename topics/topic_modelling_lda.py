from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from topics.metrics import coherence_all, topic_diversity
from topics.preprocessing import load_arxiv_sample, build_text_column
from topics.topic_modelling_nmf import plot_topic_matrix


def train_lda(
    texts,
    n_topics=10,
    no_below=20,
    no_above=0.9,
    keep_n=50_000,
    passes=1,
    iterations=50,
    random_state=42
):
    tokens = [
        [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
        for text in texts
    ]
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(
        no_below=no_below,
        no_above=no_above,
        keep_n=keep_n
    )

    corpus = [dictionary.doc2bow(t) for t in tokens]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=passes,
        iterations=iterations,
        alpha="auto",
        eta="auto",
        random_state=random_state
    )

    return lda, tokens, dictionary, corpus

def extract_lda_topics(lda, top_n=10):
    topic_words = []
    for k in range(lda.num_topics):
        words = [w for w, _ in lda.show_topic(k, topn=top_n)]
        topic_words.append(words)
    return topic_words

import json

def save_lda_artifacts(
    lda,
    dictionary,
    topic_words,
    metrics,
    output_path: Path
):
    output_path.mkdir(parents=True, exist_ok=True)
    lda.save(str(output_path / "lda_model"))

    dictionary.save(str(output_path / "dictionary.dict"))

    # save topics
    with open(output_path / "topics.json", "w") as f:
        json.dump(
            {i: words for i, words in enumerate(topic_words)},
            f,
            indent=2
        )

    # save metrics
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def plot_lda_topic_words(
    lda, dictionary, topic_idx, top_n=15, output_path=None
):
    topic = lda.get_topic_terms(topic_idx, topn=top_n)

    words = [dictionary[w_id] for w_id, _ in topic]
    probs = [p for _, p in topic]

    plt.figure(figsize=(8, 4))
    plt.barh(words[::-1], probs[::-1])
    plt.xlabel("P(word | topic)")
    plt.title(f"LDA Topic {topic_idx}")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.show()




def lda(data_path: str,
        sample_path: str,
    chunk_size: int = 500_000,
    n_chunks: int = 5,
    n_topics: int = 5):

    df = pd.read_csv(sample_path)

    df = build_text_column(df)


    lda, tokens, dictionary, corpus = train_lda(
        df["text"].tolist(),
        n_topics=n_topics
    )

    topic_words = extract_lda_topics(lda, top_n=10)

    cv, cnpmi, umass = coherence_all(
        topic_words,
        tokens,
        dictionary,
        corpus
    )

    print(topic_words)

    tdiv = topic_diversity(topic_words, topn=10)
    metrics = {
        "c_v": cv,
        "c_npmi": cnpmi,
        "u_mass": umass,
        "topic_div@10": tdiv,
        "n_docs": len(df),
        "n_topics": n_topics
    }

    print("LDA metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    value = 500000
    output_path = Path(f"topics_lda-{value}-{n_topics}")

    phi = lda.get_topics()
    topic_sim_words = cosine_similarity(phi)

    plot_topic_matrix(
        topic_sim_words,
        topics={i: None for i in range(phi.shape[0])},
        title="LDA Topic–Topic Similarity (Word Space)",
        output_path=f"topics_lda-{value}-{n_topics}/lda_topic_similarity_words.png"
    )


    for idx, topic in enumerate(topic_words):
        plot_lda_topic_words(
            lda,
            dictionary,
            topic_idx=idx,
            top_n=15,
            output_path=f"topics_lda-{value}-{n_topics}/lda_topic_{idx}_words.png"
        )


    save_lda_artifacts(
        lda,
        dictionary,
        topic_words,
        metrics,
        output_path
    )


if __name__ == '__main__':
    lda("../arxiv_data/arxiv-metadata-oai-snapshot.json", "samples/arxiv_sample_500000.csv")


