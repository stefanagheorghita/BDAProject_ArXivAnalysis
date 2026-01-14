import pandas as pd
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import numpy as np
from pyspark.sql.functions import col, concat_ws, split, regexp_replace

from metrics import coherence_all, topic_diversity
from preprocessing import load_arxiv_sample, build_text_column

from pyspark.sql.functions import col, concat_ws, element_at
from pyspark.sql import functions as F
from pyspark.sql.functions import year, to_timestamp, col

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def lemma_tokenizer(text):
    doc = nlp(text)
    return [
        t.lemma_.lower()
        for t in doc
        if t.is_alpha and not t.is_stop
    ]

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

def load_category_df(spark, data_path, category):
    df = spark.read.json(data_path)

    df = (
        df.filter(col("primary_category") == category)
          .select(
              concat_ws(". ", col("title"), col("abstract")).alias("text")
          )
          .dropna()
    )

    return df
    
def spark_to_pandas_text(df_spark):
    return df_spark.select("text").toPandas()


def train_tfidf(
    texts,
    max_df=0.9,
    min_df=0.001,
    max_features=5000,
    ngram_range=(1, 1),
    dtype=np.float32
):
    tokens = [t.split() for t in texts]

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,     
        preprocessor=None,
        lowercase=False,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        dtype=dtype
    )

    X = vectorizer.fit_transform(tokens)

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

def extract_topics(H,  feature_names: List[str],  top_n: int = 10):
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
    plt.close()




def compute_metrics(topics, df, n_topics, output_path, vectorizer):
    topic_words = [topics[i] for i in sorted(topics.keys())]


    tokens = [text.split() for text in df["text"]]

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
    df = spark.read.json(data_path)

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

def train_topic_model_from_df(
    df: pd.DataFrame,
    output_dir: str,
    n_topics: int = 5,
    tfidf_params: dict = None,
    nmf_params: dict = None
):
    tfidf_params = tfidf_params or {}
    nmf_params = nmf_params or {}

    vectorizer, X = train_tfidf(df["text"], **tfidf_params)
    nmf, W, H = train_nmf(X, n_topics=n_topics, **nmf_params)

    feature_names = vectorizer.get_feature_names_out()
    topics = extract_topics(H, feature_names)

    print_topics(topics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "topics.json", "w") as f:
        json.dump(topics, f, indent=2)

    analyze_topic_correlations(W, H, topics, output_path)

    for idx in topics:
        plot_topic_words(
            H,
            feature_names,
            topic_idx=idx,
            top_n=15,
            output_path=f"{output_path}/topic_{idx}_words.png"
        )

    # compute_metrics(topics, df, n_topics, output_path, vectorizer)

    return {
        "vectorizer": vectorizer,
        "nmf": nmf,
        "topics": topics,
        "W": W
    }

import os

if __name__ == "__main__":
    # train_topic_model("../arxiv_data/arxiv-metadata-oai-snapshot.json", "samples/arxiv_sample_2000000.csv", "topics")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("Arxiv-NMF-Periods")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.executor.cores", "2")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.driver.memory", "6g")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
    

    ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"
    LEMMAS_PARQUET = "hdfs:///user/ubuntu/arxiv_data/arxiv_lemmas_with_id.parquet"
    OUT_BASE = "topics_nmf_periods"
    
    os.makedirs(OUT_BASE, exist_ok=True)
    

    print("Loading arXiv metadata...")
    
    data = (
        spark.read.json(ARXIV_JSON)
        .select(
            col("id"),
            concat_ws(". ", col("title"), col("abstract")).alias("text"),
            col("versions"),
            col("categories")
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
        .select("id", "categories", "text_lemma")
        .cache()
    )
    
    print("Total documents after join:", data.count())
    YEAR_WINDOWS = [
    # (1985, 1994),
    # (1995, 2004),
    # (2005, 2010),
    # (2010, 2015),
    # (2015, 2019),
    (2018, 2021),
    (2022, 2024),

    (2025, 2025)
    # (2010, 2020),
    # (2015, 2020),
    # (2018, 2022),
    # (2020, 2024),
    # (2024, 2025),
    # (2025, 2025)
    ]

    # for start_year, end_year in YEAR_WINDOWS:
    #     print(f"\n=== Years {start_year}–{end_year} ===")
    
    #     df_slice = (
    #         data
    #         .filter((col("year") >= start_year) & (col("year") <= end_year))
    #         .select(col("text_lemma").alias("text"))
    #     )
    
    #     n_docs = df_slice.count()
    #     print("Documents:", n_docs)
    
    #     if n_docs < 5_000:
    #         print("Skipping (too few documents)")
    #         continue
    
    #     df_slice_pd = df_slice.toPandas()
    
    #     out_dir = f"{OUT_BASE}/{start_year}_{end_year}"
    #     os.makedirs(out_dir, exist_ok=True)

    
        # train_topic_model_from_df(
        #     df_slice_pd,
        #     output_dir=f"topics_nmf_periods/{start_year}_{end_year}",
        #     n_topics=5,
        #     tfidf_params={
        #         "min_df": 0.001,
        #         "max_df": 0.9,
        #         "max_features": 30_000
        #     },
        #     nmf_params={
        #         "max_iter": 400
        #     }
        # )



    data = data.select(
        col("id"),
        col("text_lemma").alias("text"),
        col("categories"),
    ).filter(col("text").isNotNull())

    data = data.withColumn(
        "primary_category",
        split(split(col("categories"), " ").getItem(0), "\\.").getItem(0)
    )

    categories = (
        data
        .select("primary_category")
        .distinct()
        .rdd
        .map(lambda r: r[0])
        .collect()
    )
    
    for cat in categories:
        print(f"\n=== Category: {cat} ===")
    
        df_cat = (
            data
            .filter(col("primary_category") == cat)
            .select("text")
        )
    
        n_docs = df_cat.count()
        print(f"Documents: {n_docs}")
    
        df_cat_pd = df_cat.toPandas()
    
        train_topic_model_from_df(
            df_cat_pd,
            output_dir=f"topics_nmf/{cat}",
            n_topics=5,
            tfidf_params={
                "min_df": 0.001,
                "max_df": 0.9,
                "max_features": 30_000
            },
            nmf_params={
                "max_iter": 400
            }
        )



