from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from metrics import coherence_all, topic_diversity
from preprocessing import load_arxiv_sample, build_text_column
from topic_modelling_nmf import plot_topic_matrix

from pyspark.sql.functions import col, concat_ws
from pyspark.sql import functions as F
from pyspark.sql.functions import year, to_timestamp, col, regexp_replace, element_at

def add_creation_date(data):
    return (
        data.withColumn(
            "created",
            F.col("versions").getItem(0).getItem("created")
        )
        .withColumn(
            "year",
            year(to_timestamp(col("created"), "EEE, dd MMM yyyy HH:mm:ss z"))
        )
    )
from gensim.utils import simple_preprocess

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
        [
            w for w in simple_preprocess(text, min_len=3)
            if w not in STOPWORDS
        ]
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

def train_lda_from_df(
    df_pd: pd.DataFrame,
    output_dir: str,
    n_topics: int = 5
):
    texts = df_pd["text"].tolist()

    lda_model, tokens, dictionary, corpus = train_lda(
        texts,
        n_topics=n_topics
    )

    topic_words = extract_lda_topics(lda_model, top_n=10)

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
        "n_docs": len(df_pd),
        "n_topics": n_topics
    }

    print("\nLDA metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    output_path = Path(output_dir)

    # topic similarity in word space
    phi = lda_model.get_topics()
    topic_sim_words = cosine_similarity(phi)

    plot_topic_matrix(
        topic_sim_words,
        topics={i: None for i in range(phi.shape[0])},
        title="LDA Topic–Topic Similarity (Word Space)",
        output_path=f"{output_dir}/lda_topic_similarity_words.png"
    )

    for idx in range(n_topics):
        plot_lda_topic_words(
            lda_model,
            dictionary,
            topic_idx=idx,
            top_n=15,
            output_path=f"{output_dir}/lda_topic_{idx}_words.png"
        )

    save_lda_artifacts(
        lda_model,
        dictionary,
        topic_words,
        metrics,
        output_path
    )

import os

if __name__ == '__main__':
    # lda("../arxiv_data/arxiv-metadata-oai-snapshot.json", "samples/arxiv_sample_500000.csv")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("Arxiv-LDA-Periods") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.driver.memory", "6g") \
        .config("spark.python.worker.reuse", "true") \
        .getOrCreate()

    ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"
    LEMMAS_PARQUET = "hdfs:///user/ubuntu/arxiv_data/arxiv_lemmas_with_id.parquet"
    OUT_BASE = "topics_lda_periods"
    
    os.makedirs(OUT_BASE, exist_ok=True)
    

    print("Loading arXiv metadata...")
    
    data = (
        spark.read.json(ARXIV_JSON)
        .select(
            col("id"),
            concat_ws(". ", col("title"), col("abstract")).alias("text"),
            col("versions")
        )
        .filter(col("text").isNotNull())
    )
    
 
    data = data.withColumn(
        "created_raw",
        element_at(col("versions.created"), 1)
    )
    
    data = data.withColumn(
        "created_ts",
        to_timestamp(
            regexp_replace(col("created_raw"), " GMT", ""),
            "EEE, dd MMM yyyy HH:mm:ss"
        )
    )
    
    data = data.withColumn("year", year(col("created_ts")))
    
    data = data.filter(col("year").isNotNull())
    

    print("Loading lemmatized text...")
    
    lemmas = (
        spark.read.parquet(LEMMAS_PARQUET)
        .select(
            col("id"),
            col("text").alias("text_lemma")
        )
    )
    
    # =========================
    # Join metadata + lemmas
    # =========================
    data = (
        data
        .join(lemmas, on="id", how="inner")
        .select("id", "year", "text_lemma")
        .cache()
    )
    
    print("Total documents after join:", data.count())
    YEAR_WINDOWS = [
    # (1985, 1995),
    # (1990, 2000),
    # (1995, 2005),

    # (2000, 2010),
    # (2005, 2010),

    # (2010, 2015),
    # (2010, 2020),
    # (2015, 2018),
    # (2018, 2020),
   # (1985, 1994),
   #  (1995, 2004),
    # (2005, 2010),
    # (2010, 2015),
    # (2015, 2019),
    # (2018, 2020),
    (2021, 2022),
    (2023, 2024),
    (2025, 2025)
    # (2010, 2020),
    # (2015, 2020),
    # (2018, 2022),
    # (2020, 2024),
    # (2024, 2025),
    # (2025, 2025)

    ]

    for start_year, end_year in YEAR_WINDOWS:
        print(f"\n=== Years {start_year}–{end_year} ===")
    
        df_slice = (
            data
            .filter((col("year") >= start_year) & (col("year") <= end_year))
            .select(col("text_lemma").alias("text"))
        )
    
        n_docs = df_slice.count()
        print("Documents:", n_docs)
    
        if n_docs < 5_000:
            print("Skipping (too small)")
            continue
    
        # Convert AFTER filtering
        df_slice_pd = df_slice.toPandas()
    
        # ---- LDA ----
        train_lda_from_df(
            df_slice_pd,
            output_dir=f"topics_lda_periods/{start_year}_{end_year}",
            n_topics=5
        )
    
      
    
    
    
    
