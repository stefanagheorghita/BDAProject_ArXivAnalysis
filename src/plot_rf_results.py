from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot RandomForest results (saved outputs).")
    p.add_argument("--out_dir", default="../outputs/rf_top30", help="Same out_dir used by run_rf.py")
    p.add_argument("--top_k", type=int, default=10, help="Top-K items for error bars")
    p.add_argument("--confusion_max_classes", type=int, default=30, help="Max classes in confusion heatmap")
    p.add_argument("--dpi", type=int, default=160, help="Saved figure DPI")
    return p.parse_args()


def read_metrics_txt(path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or ":" not in s:
                continue
            k, v = s.split(":", 1)
            metrics[k.strip()] = float(v.strip())
    return metrics


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_metrics_bar(metrics: Dict[str, float], out_path: str, dpi: int) -> None:
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]

    plt.figure(figsize=(8, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Random Forest – Overall Metrics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_top_offdiag_bar(spark: SparkSession, decoded_path: str, out_path: str, top_k: int, dpi: int) -> None:
    df = spark.read.parquet(decoded_path).orderBy("count", ascending=False).limit(top_k)
    pdf = df.toPandas()
    if pdf.empty:
        raise RuntimeError("No rows found in decoded off-diagonal confusions parquet.")

    labels = (pdf["label_name"] + " → " + pdf["pred_name"]).tolist()
    counts = pdf["count"].astype(int).tolist()

    plt.figure(figsize=(10, max(3.5, 0.4 * len(labels))))
    plt.barh(labels, counts)
    plt.xlabel("Count")
    plt.title(f"Top {len(labels)} Off-Diagonal Confusions (RF)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _top_labels_from_confusion(pdf, n: int) -> Tuple[np.ndarray, np.ndarray]:
    row_sum = pdf.groupby("label")["count"].sum()
    col_sum = pdf.groupby("prediction")["count"].sum()
    all_ids = sorted(set(row_sum.index.tolist()) | set(col_sum.get(i, 0) for i in col_sum.index.tolist()) or set(
        col_sum.index.tolist()))
    all_ids = sorted(set(row_sum.index.tolist()) | set(col_sum.index.tolist()))

    totals = {}
    for i in all_ids:
        totals[i] = float(row_sum.get(i, 0.0)) + float(col_sum.get(i, 0.0))
    top = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:n]
    ids = np.array([k for k, _ in top], dtype=float)
    return ids, np.array([v for _, v in top], dtype=float)


def plot_confusion_heatmap(
        spark: SparkSession,
        confusion_path: str,
        out_path: str,
        max_classes: int,
        dpi: int,
) -> None:
    conf = spark.read.parquet(confusion_path)
    pdf = conf.toPandas()
    if pdf.empty:
        raise RuntimeError("No rows found in confusion parquet.")

    ids, _ = _top_labels_from_confusion(pdf, max_classes)

    sub = pdf[pdf["label"].isin(ids) & pdf["prediction"].isin(ids)].copy()
    if sub.empty:
        raise RuntimeError("Confusion subset is empty; check label IDs and input parquet.")

    labels_sorted = np.sort(ids)
    idx = {v: i for i, v in enumerate(labels_sorted)}
    mat = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=float)

    for _, r in sub.iterrows():
        i = idx[float(r["label"])]
        j = idx[float(r["prediction"])]
        mat[i, j] = float(r["count"])

    plt.figure(figsize=(8, 7))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xlabel("Predicted label id")
    plt.ylabel("True label id")
    plt.title(f"Confusion Matrix (Top {len(labels_sorted)} Labels by Frequency) – RF")
    plt.xticks(range(len(labels_sorted)), [int(x) for x in labels_sorted], rotation=90)
    plt.yticks(range(len(labels_sorted)), [int(x) for x in labels_sorted])
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    metrics_path = os.path.join(out_dir, "metrics.txt")
    decoded_offdiag_path = os.path.join(out_dir, "top_confusions_offdiag_decoded.parquet")
    confusion_path = os.path.join(out_dir, "confusion_top50.parquet")

    metrics = read_metrics_txt(metrics_path)
    plot_metrics_bar(metrics, os.path.join(fig_dir, "metrics_bar.png"), dpi=args.dpi)

    spark = SparkSession.builder.appName("plot-rf-results").getOrCreate()

    plot_top_offdiag_bar(
        spark,
        decoded_path=decoded_offdiag_path,
        out_path=os.path.join(fig_dir, f"top_offdiag_bar_top{args.top_k}.png"),
        top_k=args.top_k,
        dpi=args.dpi,
    )

    plot_confusion_heatmap(
        spark,
        confusion_path=confusion_path,
        out_path=os.path.join(fig_dir, f"confusion_heatmap_top{args.confusion_max_classes}.png"),
        max_classes=args.confusion_max_classes,
        dpi=args.dpi,
    )

    spark.stop()
    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
