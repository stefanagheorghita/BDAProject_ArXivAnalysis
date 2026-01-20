import os
import time

from pipeline.spark import make_spark
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.sql import functions as F

from modeling.cluster_eval import (
    add_cluster_label_mapping,
    classification_report_from_predlabels,
    nmi_from_contingency,
    purity_from_contingency,
)

RAW_PARQUET = "../outputs/clean_top30/dataset.parquet"

OUT_DIR = "../outputs/kmeans_w2v_top30"
MODEL_DIR = f"{OUT_DIR}/model"
METRICS_PATH = f"{OUT_DIR}/metrics.txt"
CONTINGENCY_PATH = f"{OUT_DIR}/contingency.parquet"
CLUSTER_SUMMARY_PATH = f"{OUT_DIR}/cluster_summary.parquet"
PRED_LABELED_PATH = f"{OUT_DIR}/pred_labeled.parquet"
REPORT_PATH = f"{OUT_DIR}/classification_report.parquet"

SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    spark = make_spark("arxiv-kmeans-w2v-top30", driver_memory="12g")
    spark.conf.set("spark.sql.shuffle.partitions", "128")

    df = spark.read.parquet(RAW_PARQUET).select("id", "label_str", "text_clean")
    df = df.where(F.col("label_str").isNotNull())
    df = df.withColumn("text", F.coalesce(F.col("text_clean").cast("string"), F.lit("")))

    tok = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+", minTokenLength=2)
    sw = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")
    w2v = Word2Vec(inputCol="tokens_clean", outputCol="features", vectorSize=200, minCount=2)

    k = 30
    km = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=SEED, maxIter=50, initMode="k-means||")

    pipe = Pipeline(stages=[tok, sw, w2v, km])

    t0 = time.time()
    model = pipe.fit(df)
    fit_s = time.time() - t0

    pred = model.transform(df).select("label_str", "cluster", "features")

    pred.select("label_str", "cluster", "features") \
        .write.mode("overwrite") \
        .parquet(f"{OUT_DIR}/pred_with_features.parquet")

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    silhouette = evaluator.evaluate(pred)

    contingency = pred.groupBy("cluster", "label_str").count()
    contingency.write.mode("overwrite").parquet(CONTINGENCY_PATH)

    rows = contingency.collect()
    purity = purity_from_contingency(rows)
    nmi = nmi_from_contingency(rows)

    pred_labeled, cluster_summary = add_cluster_label_mapping(pred.select("label_str", "cluster"))
    pred_labeled.write.mode("overwrite").parquet(PRED_LABELED_PATH)
    cluster_summary.write.mode("overwrite").parquet(CLUSTER_SUMMARY_PATH)

    report, ext_acc, macro_f1, weighted_f1 = classification_report_from_predlabels(pred_labeled)
    report.write.mode("overwrite").parquet(REPORT_PATH)

    model.write().overwrite().save(MODEL_DIR)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"k: {k}\n")
        f.write(f"fit_seconds: {fit_s:.2f}\n")
        f.write(f"silhouette: {silhouette}\n")
        f.write(f"purity: {purity}\n")
        f.write(f"nmi: {nmi}\n")
        f.write(f"external_accuracy: {ext_acc}\n")
        f.write(f"external_macro_f1: {macro_f1}\n")
        f.write(f"external_weighted_f1: {weighted_f1}\n")

    print(
        {
            "k": k,
            "fit_seconds": fit_s,
            "silhouette": silhouette,
            "purity": purity,
            "nmi": nmi,
            "external_accuracy": ext_acc,
            "external_macro_f1": macro_f1,
            "external_weighted_f1": weighted_f1,
        }
    )

    spark.stop()


if __name__ == "__main__":
    main()
