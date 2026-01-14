from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, split
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF

spark = (
    SparkSession.builder
    .appName("ArXiv-TFIDF-Lemmas")
    .getOrCreate()
)

ARXIV_JSON = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"
LEMMAS_PATH = "hdfs:///user/ubuntu/arxiv_data/arxiv_lemmas_with_id.parquet"

INPUT_PATH = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"

data = (
    spark.read.json(ARXIV_JSON)
    .select(
        col("id"),
        concat_ws(". ", col("title"), col("abstract")).alias("text")
    )
    .filter(col("text").isNotNull())
)

lemmas = (
    spark.read.parquet(LEMMAS_PATH)
    .select("id", col("text").alias("text_lemma"))
)

data = (
    data
    .join(lemmas, on="id", how="inner")
    .select("id", "text_lemma")
    .cache()
)

print("Documents after join:", data.count())
data = data.withColumn(
    "tokens",
    split(col("text_lemma"), " ")
)



tf = HashingTF(
    inputCol="tokens",
    outputCol="tf",
    numFeatures=1 << 20
)

idf = IDF(
    inputCol="tf",
    outputCol="tfidf",
    minDocFreq=5_000
)

tfidf_model = Pipeline(stages=[tf, idf]).fit(data)
df = tfidf_model.transform(data).select("id", "tfidf")



TFIDF_PATH = "hdfs:///user/ubuntu/features/arxiv_tfidf_vectors"

(
    df
    .repartition(200)       
    .write
    .mode("overwrite")
    .parquet(TFIDF_PATH)
)
MODEL_PATH = "hdfs:///user/ubuntu/models/arxiv_tfidf_v1"

tfidf_model.write().overwrite().save(MODEL_PATH)


