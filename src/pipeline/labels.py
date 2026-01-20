from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_primary_label(df_raw: DataFrame) -> DataFrame:
    return df_raw.withColumn("label_str", F.split(F.col("categories"), " ").getItem(0))


def compute_top_labels(df_with_label: DataFrame, top_n: int) -> list[str]:
    rows = (df_with_label
            .groupBy("label_str")
            .count()
            .orderBy(F.desc("count"))
            .limit(top_n)
            .collect())
    return [r["label_str"] for r in rows]


def filter_to_labels(df_with_label: DataFrame, labels: list[str]) -> DataFrame:
    return df_with_label.where(F.col("label_str").isin(labels))
