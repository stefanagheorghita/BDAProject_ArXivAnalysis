import argparse
import os

import matplotlib.pyplot as plt

from pipeline.spark import make_spark
from pyspark.ml.feature import PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT


def densify(v):
    if v is None:
        return None
    return Vectors.dense(v.toArray())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="KMeans output folder, e.g. ../outputs/kmeans_tfidf_top30")
    ap.add_argument("--scatter_n", type=int, default=30000, help="Number of points to sample for scatter")
    args = ap.parse_args()

    in_dir = args.input
    pred_with_features_path = os.path.join(in_dir, "pred_with_features.parquet")

    if not os.path.exists(pred_with_features_path):
        raise SystemExit(
            f"Missing {pred_with_features_path}. "
            f"Add this line in your KMeans runner after creating pred:\n"
            f"pred.select('label_str','cluster','features').write.mode('overwrite').parquet('{pred_with_features_path}')"
        )

    spark = make_spark("plot-kmeans-scatter", driver_memory="6g")
    spark.conf.set("spark.sql.shuffle.partitions", "16")

    pred = spark.read.parquet(pred_with_features_path).select("label_str", "cluster", "features")

    n = int(args.scatter_n)
    pred_s = pred.orderBy(F.rand(seed=42)).limit(n)

    first = pred_s.select("features").head()
    if first is not None and first["features"] is not None and "SparseVector" in str(type(first["features"])):
        densify_udf = F.udf(densify, VectorUDT())
        pred_s = pred_s.withColumn("features_dense", densify_udf(F.col("features"))).drop("features").withColumnRenamed("features_dense", "features")

    pca = PCA(k=2, inputCol="features", outputCol="pca2")
    pca_model = pca.fit(pred_s)
    p2 = pca_model.transform(pred_s).select("label_str", "cluster", "pca2")

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
    plt.title(f"KMeans clusters visualized with PCA (n={len(p2_pd)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clusters_pca_by_cluster.png"), dpi=220)
    plt.close()

    top_labels = (
        pred.groupBy("label_str")
        .count()
        .orderBy(F.desc("count"))
        .limit(12)
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
    plt.title("PCA view colored by true label (top 12 labels, rest=OTHER)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clusters_pca_by_true_label.png"), dpi=220)
    plt.close()

    print("Saved PCA scatter plots to:", out_dir)
    spark.stop()


if __name__ == "__main__":
    main()
