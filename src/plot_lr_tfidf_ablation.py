import argparse
import os

import matplotlib.pyplot as plt

from pipeline.spark import make_spark


def fmt_minutes(seconds: float) -> float:
    return float(seconds) / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../outputs/ablation_lr_tfidf/results.parquet", help="Path to results.parquet")
    args = ap.parse_args()

    spark = make_spark("plot-lr-tfidf-ablation", driver_memory="4g")
    df = spark.read.parquet(args.input)
    pdf = df.toPandas()
    spark.stop()

    out_dir = os.path.join(os.path.dirname(args.input), "plots")
    os.makedirs(out_dir, exist_ok=True)

    def make_label(row):
        v = int(row["vocab_size"]) // 1000
        if bool(row["use_bigrams"]):
            return f"uni+bi\n{v}k"
        return f"uni\n{v}k"

    pdf["label"] = pdf.apply(make_label, axis=1)

    pdf["fit_min"] = pdf["fit_seconds"].apply(fmt_minutes)
    if "prep_seconds" in pdf.columns:
        pdf["prep_min"] = pdf["prep_seconds"].apply(fmt_minutes)
    else:
        pdf["prep_min"] = 0.0
    if "transform_seconds" in pdf.columns:
        pdf["transform_min"] = pdf["transform_seconds"].apply(fmt_minutes)
    else:
        pdf["transform_min"] = 0.0
    if "total_seconds" in pdf.columns:
        pdf["total_min"] = pdf["total_seconds"].apply(fmt_minutes)
    else:
        pdf["total_min"] = (pdf["prep_min"] + pdf["fit_min"] + pdf["transform_min"])

    pdf_sorted = pdf.sort_values(["use_bigrams", "vocab_size"])

    plt.figure()
    plt.bar(pdf_sorted["label"], pdf_sorted["accuracy"])
    plt.ylabel("accuracy")
    plt.title("TF-IDF + Logistic Regression: Accuracy by setting")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_by_setting.png"), dpi=220)
    plt.close()

    plt.figure()
    plt.bar(pdf_sorted["label"], pdf_sorted["f1_macro"])
    plt.ylabel("macro F1")
    plt.title("TF-IDF + Logistic Regression: Macro-F1 by setting")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "macro_f1_by_setting.png"), dpi=220)
    plt.close()

    plt.figure()
    plt.bar(pdf_sorted["label"], pdf_sorted["fit_min"])
    plt.ylabel("training time (minutes)")
    plt.title("TF-IDF + Logistic Regression: Fit time by setting")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fit_time_minutes_by_setting.png"), dpi=220)
    plt.close()

    plt.figure()
    plt.scatter(pdf["total_min"], pdf["accuracy"])
    for _, r in pdf.iterrows():
        plt.annotate(r["label"].replace("\n", " "), (r["total_min"], r["accuracy"]))
    plt.xlabel("total time (minutes)")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs total runtime (tradeoff)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_total_time.png"), dpi=220)
    plt.close()

    plt.figure()
    plt.scatter(pdf["total_min"], pdf["f1_macro"])
    for _, r in pdf.iterrows():
        plt.annotate(r["label"].replace("\n", " "), (r["total_min"], r["f1_macro"]))
    plt.xlabel("total time (minutes)")
    plt.ylabel("macro F1")
    plt.title("Macro-F1 vs total runtime (tradeoff)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "macro_f1_vs_total_time.png"), dpi=220)
    plt.close()

    if all(c in pdf.columns for c in ["prep_min", "fit_min", "transform_min"]):
        labels = list(pdf_sorted["label"])
        prep = list(pdf_sorted["prep_min"])
        fit = list(pdf_sorted["fit_min"])
        trans = list(pdf_sorted["transform_min"])

        plt.figure()
        plt.bar(labels, prep, label="prep")
        plt.bar(labels, fit, bottom=prep, label="fit")
        bottom2 = [prep[i] + fit[i] for i in range(len(prep))]
        plt.bar(labels, trans, bottom=bottom2, label="transform")
        plt.ylabel("minutes")
        plt.title("Runtime breakdown by setting")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "runtime_breakdown_stacked.png"), dpi=220)
        plt.close()

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
