from pathlib import Path

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from matplotlib import pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, expr, regexp_replace, lower, split
from sklearn.metrics.pairwise import cosine_similarity

from metrics import coherence_all, topic_diversity
from preprocessing import load_arxiv_sample, build_text_column
from topic_modelling_nmf import plot_topic_matrix
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA as SparkLDA
import sparknlp

from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer
)
from pyspark.ml.clustering import LDA as SparkLDA


def train_lda(
    spark_df,
    n_topics=10,
    vocab_size=50_000,
    min_df=5_000,
    max_iter=20,
    optimizer="online",
    seed=42
):
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="\\W+",
        minTokenLength=3
    )

    stopwords = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered"
    )

    pipeline = Pipeline(stages=[tokenizer, stopwords])
    df = pipeline.fit(spark_df).transform(spark_df)

    df = df.filter(F.size(col("filtered")) > 5).persist()
    df.count()

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
    spark = (
        SparkSession.builder
        .appName("ArxivLDA")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.driver.memory", "6g")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )


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


def calculate_topic_diversity(topic_words_list):

    if not topic_words_list:
        return 0.0
        
    unique_words = set()
    total_words = 0
    for topic in topic_words_list:
        unique_words.update(topic)
        total_words += len(topic)
    
    if total_words == 0:
        return 0.0
        
    return len(unique_words) / total_words
    
def train_lda_for_category(
    spark_df,
    category_name,
    output_base_dir,
    n_topics=5,
    vocab_size=30_000,
    min_df=0.001,
    max_iter=20,
    max_df=0.9
):
    print(f"--- Processing Category: {category_name} ---")
    n_docs = spark_df.count()
    print(f"Number of documents: {n_docs}")
    

    tokenizer = RegexTokenizer(
        inputCol="text", 
        outputCol="tokens",
        pattern="\\s+",     
        minTokenLength=3
    )
    
    vectorizer = CountVectorizer(
        inputCol="tokens", 
        outputCol="features", 
        vocabSize=vocab_size, 
        minDF=min_df,
        maxDF=max_df
    )
    
    pipeline = Pipeline(stages=[tokenizer, vectorizer])
    model_pipeline = pipeline.fit(spark_df)
    
    transformed_df = model_pipeline.transform(spark_df)

   
    
    lda = SparkLDA(
        k=n_topics, 
        maxIter=max_iter, 
        optimizer="online", 
        seed=42 
    ) 
    
    lda_model = lda.fit(transformed_df)
    ll = lda_model.logLikelihood(transformed_df)
    lp = lda_model.logPerplexity(transformed_df)
    

    cv_model = model_pipeline.stages[-1] 
    vocab = np.array(cv_model.vocabulary)
    
    topics_desc = lda_model.describeTopics(maxTermsPerTopic=10).collect()
    topic_words_list = []
    
    formatted_topics = {}
    for row in topics_desc:
        topic_id = row.topic
        words = vocab[row.termIndices].tolist()
        topic_words_list.append(words)
        formatted_topics[topic_id] = words
        print(f"Topic {topic_id}: {words}")

    diversity = calculate_topic_diversity(topic_words_list)

    print(f"Metrics -> Docs: {n_docs}, Div: {diversity:.4f}, LL: {ll:.2f}, Perp: {lp:.2f}")

    output_path = Path(output_base_dir) / category_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "topics.json", "w") as f:
        json.dump(formatted_topics, f, indent=2)

    metrics_data = {
        "category": category_name,
        "n_documents": n_docs,
        "n_topics": n_topics,
        "topic_diversity": diversity,
        "log_likelihood": ll,
        "log_perplexity": lp,
        "parameters": {
            "vocab_size": vocab_size,
            "min_df": min_df,
            "max_df": max_df,
            "max_iter": max_iter
        }
    }
    
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    return lda_model

def run_per_category_pipeline(data_path):
    ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"
    LEMMAS_PARQUET = "hdfs:///user/ubuntu/arxiv_data/arxiv_lemmas_with_id.parquet"

    spark = (
        SparkSession.builder
        .appName("Arxiv-LDA-Periods")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.executor.cores", "2")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.driver.memory", "6g")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
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

    lemmas = (
        spark.read.parquet(LEMMAS_PARQUET)
        .select(
            col("id"),
            col("text").alias("text_lemma")
        )
    )
    
    
    data = (
        data
        .join(lemmas, on="id", how="inner")
        .select("id", "categories", "text_lemma")
        .cache()
    )
    
    print("Total documents after join:", data.count())
    
 
    data = data.select(
        col("id"),
        col("text_lemma").alias("text"),
        col("categories"),
    ).filter(col("text").isNotNull())

    df = data.withColumn(
        "primary_category",
        split(split(col("categories"), " ").getItem(0), "\\.").getItem(0)
    )

    categories = (
        df
        .select("primary_category")
        .distinct()
        .rdd
        .map(lambda r: r[0])
        .collect()
    )
    
    print(f"Found {len(categories)} categories.")

    for cat in categories:
        df_cat = df.filter(col("primary_category") == cat)
                 
        print(f"Processing {cat} with {df_cat.count()} documents...")
        
        df_cat.persist()
        
        try:
            train_lda_for_category(
                spark_df=df_cat,
                category_name=cat,
                output_base_dir="topics_lda_spark_by_cat",
                n_topics=5
            )
        except Exception as e:
            print(f"Error processing {cat}: {e}")
        finally:
            df_cat.unpersist()




if __name__ == '__main__':
    # lda("hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json")
    run_per_category_pipeline("hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json")


