
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import lit

import spacy
import os


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])




def lemmatize_query(text: str):
    return [
        t.lemma_.lower()
        for t in nlp(text)
        if t.is_alpha and not t.is_stop
    ]


spark = (
    SparkSession.builder
    .appName("ArXiv-TFIDF-MultiQuery-Search")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


TFIDF_PATH = "hdfs:///user/ubuntu/features/arxiv_tfidf_vectors"
MODEL_PATH = "hdfs:///user/ubuntu/models/arxiv_tfidf_v1"
ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"
OUT_DIR = "tfidf_query_results"

os.makedirs(OUT_DIR, exist_ok=True)


tfidf_df = (
    spark.read.parquet(TFIDF_PATH)
    .select("id", "tfidf")
)



normalizer = Normalizer(
    inputCol="tfidf",
    outputCol="tfidf_norm",
    p=2.0
)

tfidf_norm_df = (
    normalizer
    .transform(tfidf_df)
    .select("id", "tfidf_norm")
    .cache()
)

def get_doc_vector(doc_id: str):
    row = (
        tfidf_norm_df
        .filter(col("id") == doc_id)
        .select("tfidf_norm")
        .first()
    )

    if row is None:
        raise ValueError(f"ID not found: {doc_id}")

    return row["tfidf_norm"]




model = PipelineModel.load(MODEL_PATH)

data = (
    spark.read.json(ARXIV_JSON)
    .select(
        col("id"),
        col("categories"),
        col("title"), col("abstract"),
        concat_ws(". ", col("title"), col("abstract")).alias("text")
    )
    .filter(col("text").isNotNull())
)


QUERIES = {
    "gnn_citations": "graph neural networks for citation analysis",
    "quantum_field": "quantum field theory gauge symmetry",
    "optimization": "convex optimization gradient descent convergence",
    "language_models": "large language models transformers pretraining",
}

TOP_K = 20


def make_dot_udf(query_vec):
    def dot_sim(v):
        return float(v.dot(query_vec))
    return udf(dot_sim, DoubleType())

def query_by_text(query_text: str, top_k: int = 20):
    query_tokens = lemmatize_query(query_text)

    query_df = spark.createDataFrame(
        [("query", query_tokens)],
        ["id", "tokens"]
    )

    query_vec = (
        model
        .transform(query_df)
        .select("tfidf")
        .first()["tfidf"]
    )

    query_vec = query_vec / query_vec.norm(2)

    dot_udf = make_dot_udf(query_vec)

    scored = tfidf_norm_df.withColumn(
        "similarity",
        dot_udf(col("tfidf_norm"))
    )

    return (
        scored
        .orderBy(col("similarity").desc())
        .limit(top_k)
        .join(data, on="id", how="left")
        .withColumn("query_value", lit(query_text))
        .select("id", "categories", "similarity", "query_value", "title", "abstract")
    )

def get_query_entry(doc_id: str):
    row = (
        data
        .filter(col("id") == doc_id)
        .select("id", "categories","title", "abstract")
        .first()
    )

    if row is None:
        raise ValueError(f"ID not found in data: {doc_id}")

    return row

    
def query_by_id(doc_id: str, top_k: int = 20):
    query_vec = get_doc_vector(doc_id)
    entry = get_query_entry(doc_id)
    dot_udf = make_dot_udf(query_vec)

    scored = tfidf_norm_df.withColumn(
        "similarity",
        dot_udf(col("tfidf_norm"))
    )

    topk = (
        scored
         .filter(col("id") != doc_id)
        .filter(col("similarity").isNotNull())
        .orderBy(col("similarity").desc())
        .limit(top_k)
    )
    topk.count()


    return (
       topk
        .join(data, on="id", how="left")
        .orderBy(col("similarity").desc())
        .withColumn("query_value", lit(entry["id"]))
        .withColumn("query_title", lit(entry["title"]))
        .withColumn("query_abstract", lit(entry["abstract"]))
        .select("id", "categories", "similarity","query_value", "query_title", "query_abstract", "title", "abstract")
    )


def save_results(df, name):
    out_path = f"{OUT_DIR}/{name}.csv"
    df.toPandas().to_csv(out_path, index=False)
    print(f"Saved → {out_path}")



if __name__ == "__main__":
    save_results(query_by_id("1810.04805"), "similar_to_1810.04805")
    save_results(query_by_text("Neutrino decays as a natural explanation of the neutrino mass tension"), "neutrino_title")
    save_results(query_by_text("This paper develops a unified framework for analyzing technology adoption in financial networks that incorporates spatial spillovers, network externalities, and their interaction. The framework characterizes adoption dynamics through a master equation whose solution admits a Feynman-Kac representation as expected cumulative adoption pressure along stochastic paths through spatial-network space. From this representation, I derive the Adoption Amplification Factor -- a structural measure of technology leadership that captures the ratio of total system-wide adoption to initial adoption following a localized shock. A Levy jump-diffusion extension with state-dependent jump intensity captures critical mass dynamics: below threshold, adoption evolves through gradual diffusion; above threshold, cascade dynamics accelerate adoption through discrete jumps. Applying the framework to SWIFT gpi adoption among 17 Global Systemically Important Banks, I find strong support for the two-regime characterization. Network-central banks adopt significantly earlier (\rho = -0.69, p = 0.002), and pre-threshold adopters have significantly higher amplification factors than post-threshold adopters (11.81 versus 7.83, p = 0.010). Founding members, representing 29 percent of banks, account for 39 percent of total system amplification -- sufficient to trigger cascade dynamics. Controlling for firm size and network position, CEO age delays adoption by 11-15 days per year."), "economy_abstract")

