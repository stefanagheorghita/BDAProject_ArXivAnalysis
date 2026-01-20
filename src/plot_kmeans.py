import argparse
import os

import matplotlib.pyplot as plt

from pipeline.spark import make_spark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="KMeans output folder, e.g. ./outputs/kmeans_tfidf_top30")
    args = ap.parse_args()

    in_dir = args.input
    cluster_summary_path = os.path.join(in_dir, "cluster_summary.parquet")
    report_path = os.path.join(in_dir, "classification_report.parquet")

    spark = make_spark("plot-kmeans", driver_memory="4g")
    spark.conf.set("spark.sql.shuffle.partitions", "16")

    cs = spark.read.parquet(cluster_summary_path).orderBy("cluster")
    rep = spark.read.parquet(report_path)

    cs_pd = cs.toPandas()
    rep_pd = rep.toPandas()

    out_dir = os.path.join(in_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.bar(cs_pd["cluster"], cs_pd["cluster_size"])
    plt.xlabel("cluster")
    plt.ylabel("cluster_size")
    plt.title("KMeans cluster sizes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_sizes.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(cs_pd["cluster"], cs_pd["purity"])
    plt.xlabel("cluster")
    plt.ylabel("purity")
    plt.title("KMeans purity per cluster (majority label share)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_purity.png"), dpi=200)
    plt.close()

    rep_pd = rep_pd.sort_values("support", ascending=False).head(20)
    plt.figure()
    plt.bar(rep_pd["label"], rep_pd["f1"])
    plt.xlabel("label")
    plt.ylabel("f1")
    plt.title("Top-20 labels by support: F1 from cluster→label mapping")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_f1_top20.png"), dpi=200)
    plt.close()

    print("Saved plots to:", out_dir)

    spark.stop()


if __name__ == "__main__":
    main()
