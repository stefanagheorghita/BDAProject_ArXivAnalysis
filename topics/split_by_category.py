from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, split


spark = SparkSession.builder \
    .appName("SplitByCategory") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "512m") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()



data = spark.read.json("hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json")

data = data.select(
        col("id"),
        concat_ws(". ", col("title"), col("abstract")).alias("text"),
        col("categories"),
).filter(col("text").isNotNull())

data = data.withColumn(
        "primary_category",
        split(col("categories"), " ").getItem(0)
)

out_df = data.select(
    col("id"),
    col("text"),
    col("primary_category")
)
(
out_df
.write
.mode("overwrite")           
.option("header", True)
.partitionBy("primary_category")
.csv("hdfs:///user/ubuntu/arxiv_by_category")
)

