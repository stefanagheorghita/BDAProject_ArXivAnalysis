from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from pyspark.sql import functions as F

from pipeline.spark import make_spark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze label distribution, top-K coverage, and export plots."
    )
    p.add_argument(
        "--input_parquet",
        required=True,
        help="Parquet path containing at least the label column (e.g. label_str).",
    )
    p.add_argument(
        "--label_col",
        default="label_str",
        help="Name of the label column (default: label_str).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="How many most-frequent labels to list/plot (default: 30).",
    )
    p.add_argument(
        "--coverage_k_max",
        type=int,
        default=200,
        help="Max K for coverage curve (default: 200).",
    )
    p.add_argument(
        "--out_dir",
        default="../outputs/label_analysis",
        help="Directory to write outputs (default: ../outputs/label_analysis).",
    )
    p.add_argument(
        "--driver_memory",
        default="6g",
        help='Spark driver memory (e.g. "6g").',
    )
    return p.parse_args()


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def collect_label_counts_ordered(counts_df, label_col: str, limit: int) -> List[Tuple[str, int]]:
    rows = (
        counts_df.orderBy(F.desc("count"))
        .limit(limit)
        .select(label_col, "count")
        .collect()
    )
    return [(r[label_col], int(r["count"])) for r in rows]


def plot_topk_bar(topk: List[Tuple[str, int]], out_png: str) -> None:
    labels = [x[0] for x in topk]
    counts = [x[1] for x in topk]

    plt.figure(figsize=(14, 7))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel("count")
    plt.title(f"Top-{len(labels)} labels by frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_coverage_curve(ordered_counts: List[int], total: int, out_png: str, k_max: int) -> None:
    k_max = min(k_max, len(ordered_counts))
    cum = 0
    xs = []
    ys = []
    for k in range(1, k_max + 1):
        cum += ordered_counts[k - 1]
        xs.append(k)
        ys.append(cum / total if total else 0.0)

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o", markersize=3)
    plt.ylim(0, 1.0)
    plt.xlabel("k (top-k labels)")
    plt.ylabel("coverage (fraction of dataset)")
    plt.title("Cumulative coverage of dataset by top-k labels")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_long_tail_distribution(ordered_counts: List[int], out_png: str) -> None:
    ranks = list(range(1, len(ordered_counts) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, ordered_counts)
    plt.yscale("log")
    plt.xlabel("label rank (1 = most frequent)")
    plt.ylabel("count (log scale)")
    plt.title("Label frequency long-tail distribution")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    spark = make_spark("arxiv-label-analysis", driver_memory=args.driver_memory)

    df = spark.read.parquet(args.input_parquet).select(args.label_col).dropna()
    total_rows = df.count()
    distinct_labels = df.select(args.label_col).distinct().count()

    counts = df.groupBy(args.label_col).count().cache()

    ordered_all = (
        counts.orderBy(F.desc("count"))
        .select(args.label_col, "count")
        .collect()
    )
    ordered_labels = [r[args.label_col] for r in ordered_all]
    ordered_counts = [int(r["count"]) for r in ordered_all]

    topk_labels = ordered_labels[: args.top_k]
    topk_rows = sum(ordered_counts[: args.top_k])
    coverage = (topk_rows / total_rows) if total_rows else 0.0


    print("\n================ Label Analysis ================")
    print(f"Input parquet     : {args.input_parquet}")
    print(f"Label column      : {args.label_col}")
    print(f"Total rows        : {total_rows}")
    print(f"Distinct labels   : {distinct_labels}")
    print(f"Top-{args.top_k} rows     : {topk_rows}")
    print(f"Top-{args.top_k} coverage : {coverage:.4f} ({coverage*100:.2f}%)\n")

    print(f"Top-{args.top_k} labels (descending frequency):")
    for i, lab in enumerate(topk_labels):
        print(f"{i:02d}  {lab}")


    labels_txt = os.path.join(args.out_dir, f"top_{args.top_k}_labels.txt")
    with open(labels_txt, "w", encoding="utf-8") as f:
        for i, lab in enumerate(topk_labels):
            f.write(f"{i}\t{lab}\n")
    print("Saved:", labels_txt)

    summary_txt = os.path.join(args.out_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Label analysis summary\n")
        f.write(f"input_parquet: {args.input_parquet}\n")
        f.write(f"label_col: {args.label_col}\n")
        f.write(f"total_rows: {total_rows}\n")
        f.write(f"distinct_labels: {distinct_labels}\n")
        f.write(f"top_k: {args.top_k}\n")
        f.write(f"top_k_rows: {topk_rows}\n")
        f.write(f"top_k_coverage: {coverage:.6f}\n")
        f.write("\nTop-K labels (descending frequency):\n")
        for i, lab in enumerate(topk_labels):
            f.write(f"{i}\t{lab}\n")
    print("Saved:", summary_txt)

    counts_out = os.path.join(args.out_dir, "label_counts.parquet")
    counts.write.mode("overwrite").parquet(counts_out)
    print("Saved:", counts_out)

    topk_pairs = list(zip(topk_labels, ordered_counts[: args.top_k]))

    plot1 = os.path.join(args.out_dir, f"plot_top_{args.top_k}_labels.png")
    plot_topk_bar(topk_pairs, plot1)
    print("Saved:", plot1)

    plot2 = os.path.join(args.out_dir, "plot_coverage_curve.png")
    plot_coverage_curve(ordered_counts, total_rows, plot2, k_max=args.coverage_k_max)
    print("Saved:", plot2)

    plot3 = os.path.join(args.out_dir, "plot_label_long_tail.png")
    plot_long_tail_distribution(ordered_counts, plot3)
    print("Saved:", plot3)

    spark.stop()


if __name__ == "__main__":
    main()
