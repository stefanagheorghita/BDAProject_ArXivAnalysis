from pathlib import Path

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from matplotlib import pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, expr, regexp_replace, lower
from sklearn.metrics.pairwise import cosine_similarity

from topics.metrics import coherence_all, topic_diversity
from topics.preprocessing import load_arxiv_sample, build_text_column
from topics.topic_modelling_nmf import plot_topic_matrix
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA as SparkLDA


def train_lda(
    spark_df,
    n_topics=10,
    vocab_size=30_000,
    min_df=0.005,
    max_iter=20,
    optimizer="online",
    seed=42
):
    df = spark_df.withColumn(
        "text_clean",
        lower(regexp_replace(col("text"), "[^a-zA-Z\\s]", " "))
    )

    tokenizer = Tokenizer(
        inputCol="text_clean",
        outputCol="tokens"
    )
    df = tokenizer.transform(df)

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered"
    )

    df = remover.transform(df)

    df = df.withColumn(
        "filtered",
        expr("filter(filtered, x -> length(x) > 2)")
    )

    df = df.filter(F.size(col("filtered")) > 5)

    vectorizer = CountVectorizer(
        inputCol="filtered",
        outputCol="features",
        vocabSize=vocab_size,
        minDF=min_df
    )
    cv_model = vectorizer.fit(df)
    df = cv_model.transform(df)

    lda = SparkLDA(
        k=n_topics,
        maxIter=max_iter,
        optimizer=optimizer,
        seed=seed
    )
    lda_model = lda.fit(df)

    return lda_model, cv_model, df



def extract_lda_topics(lda_model, cv_model, top_n=10):
    vocab = np.array(cv_model.vocabulary)
    topics = lda_model.describeTopics(maxTermsPerTopic=top_n).collect()

    topic_words = [
        vocab[row.termIndices].tolist()
        for row in topics
    ]
    return topic_words


import json

def save_lda_artifacts(
    topic_words,
    output_path: Path
):
    output_path.mkdir(parents=True, exist_ok=True)


    with open(output_path / "topics.json", "w") as f:
        json.dump(
            {i: words for i, words in enumerate(topic_words)},
            f,
            indent=2
        )

def lda(
    data_path: str,
    n_topics: int = 5
):
    spark = SparkSession.builder \
        .appName("ArxivLDA") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    df = spark.read.json(data_path)
    df = df.select("title", "abstract").dropna()

    df = df.withColumn(
        "text",
        concat_ws(" ", col("title"), col("abstract"))
    )

    lda_model, cv_model, df = train_lda(
        df,
        n_topics=n_topics
    )

    topic_words = extract_lda_topics(
        lda_model,
        cv_model,
        top_n=10
    )

    print(topic_words)

    print("Log-likelihood", lda_model.logLikelihood(df))
    print("Log-perplexity", lda_model.logPerplexity(df))

    output_path = Path(f"topics_lda_spark-{n_topics}")

    save_lda_artifacts(topic_words, output_path)





if __name__ == '__main__':
    lda("../arxiv_data/arxiv-metadata-oai-snapshot.json")


