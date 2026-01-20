from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

spark = SparkSession.builder.appName("arxiv-classification").getOrCreate()

BASE_DIR = "../arxiv_data/arxiv-metadata-oai-snapshot.json"

df = spark.read.json(BASE_DIR)

df.printSchema()
print("Rows:", df.count())

from pyspark.sql import functions as F

df_work = df.select("id", "title", "abstract", "categories", "update_date")

key_cols = ["id", "title", "abstract", "categories", "update_date"]

missing = df_work.select([
    F.count("*").alias("rows"),
    *[F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in key_cols]
])
missing.show(truncate=False)

df_work.select(
    F.size(F.split(F.col("categories"), " ")).alias("n_cats")
).groupBy("n_cats").count().orderBy("n_cats").show(20, truncate=False)

df_clean = (df_work
            .na.drop(subset=["title", "abstract", "categories"])
            .withColumn("text", F.concat_ws(" ", F.col("title"), F.col("abstract")))
            .withColumn("label_str", F.split(F.col("categories"), " ").getItem(0))
            )

print("Clean rows:", df_clean.count())
df_clean.select("id", "label_str", "text").show(3, truncate=80)

# df_clean = df_clean.select("id", "text", "label_str", "update_date").cache()
# df_clean.count()

label_counts = (
    df_clean
    .groupBy("label_str")
    .count()
    .orderBy(F.desc("count"))
)

label_counts.show(50, truncate=False)
