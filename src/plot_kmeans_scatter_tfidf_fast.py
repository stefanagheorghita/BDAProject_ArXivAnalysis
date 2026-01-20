import argparse
import os

import matplotlib.pyplot as plt

from pipeline.spark import make_spark
from pyspark.ml.feature import ChiSqSelector, StringIndexer, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT


def densify(v):
    if v is None:
        return None
    return Vectors.dense(v.toArray())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="KMeans TF-IDF output folder, e.g. ../outputs/kmeans_tfidf_top30")
    ap.add_argument("--scatter_n", type=int, default=30000, help="Number of points to sample for scatter")
    ap.add_argument("--topk", type=int, default=3000, help="ChiSqSelector topK features to keep before PCA")
    ap.add_argument("--top_labels", type=int, default=12, help="How many true labels to keep, rest=OTHER")
    args = ap.parse_args()

    in_dir = args.input
    pred_with_features_path = os.path.join(in_dir, "pred_with_features.parquet")

    if not os.path.exists(pred_with_features_path):
        raise SystemExit(
            f"Missing {pred_with_features_path}. "
            f"Generate it by adding in your KMeans runner:\n"
            f"pred.select('label_str','cluster','features').write.mode('overwrite').parquet('{pred_with_features_path}')"
        )

    spark = make_spark("plot-kmeans-scatter-tfidf-fast", driver_memory="8g")
    spark.conf.set("spark.sql.shuffle.partitions", "32")

    df = spark.read.parquet(pred_with_features_path).select("label_str", "cluster", "features")
    df = df.where(F.col("label_str").isNotNull())

    n = int(args.scatter_n)
    df_s = df.orderBy(F.rand(seed=42)).limit(n).cache()
    _ = df_s.count()

    indexer = StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="skip")
    idx_model = indexer.fit(df_s)
    df_i = idx_model.transform(df_s)

    topk = int(args.topk)
    selector = ChiSqSelector(featuresCol="features", labelCol="label", outputCol="features_sel", numTopFeatures=topk)
    sel_model = selector.fit(df_i)
    df_sel = sel_model.transform(df_i).select("label_str", "cluster", "features_sel")

    densify_udf = F.udf(densify, VectorUDT())
    df_sel = df_sel.withColumn("features_dense", densify_udf(F.col("features_sel"))).select(
        "label_str", "cluster", F.col("features_dense").alias("features")
    )

    pca = PCA(k=2, inputCol="features", outputCol="pca2")
    pca_model = pca.fit(df_sel)
    p2 = pca_model.transform(df_sel).select("label_str", "cluster", "pca2")

    p2 = p2.withColumn("pca_arr", vector_to_array("pca2"))
    p2 = (
        p2.withColumn("x", F.col("pca_arr")[0])
        .withColumn("y", F.col("pca_arr")[1])
        .select("x", "y", "cluster", "label_str")
    )

    p2_pd = p2.toPandas()

    out_dir = os.path.join(in_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.scatter(p2_pd["x"], p2_pd["y"], s=3, c=p2_pd["cluster"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"TF-IDF KMeans (fast): ChiSq topK={topk} then PCA (n={len(p2_pd)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"clusters_pca_by_cluster_fast_topk{topk}.png"), dpi=220)
    plt.close()

    top_labels_n = int(args.top_labels)
    top_labels = (
        df.groupBy("label_str")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_labels_n)
        .select("label_str")
        .rdd.map(lambda r: r[0])
        .collect()
    )

    p2_pd["label_small"] = p2_pd["label_str"].where(p2_pd["label_str"].isin(top_labels), other="OTHER")
    labels = sorted(p2_pd["label_small"].unique())
    label_to_int = {lab: i for i, lab in enumerate(labels)}
    c_vals = p2_pd["label_small"].map(label_to_int)

    plt.figure()
    plt.scatter(p2_pd["x"], p2_pd["y"], s=3, c=c_vals)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"TF-IDF PCA view by true label (top {top_labels_n}, rest=OTHER)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"clusters_pca_by_true_label_fast_topk{topk}.png"), dpi=220)
    plt.close()

    print("Saved fast TF-IDF PCA scatter plots to:", out_dir)
    spark.stop()


if __name__ == "__main__":
    main()
