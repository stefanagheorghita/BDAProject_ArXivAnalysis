import os
from pipeline.spark import make_spark
from pipeline.labels import add_primary_label, compute_top_labels, filter_to_labels
from pipeline.cleaning import build_text_raw, clean_text, drop_too_short

INPUT_PATH = "../arxiv_data/arxiv-metadata-oai-snapshot.json"
OUT_DIR = "../outputs/clean_top30"
OUT_PARQUET = f"{OUT_DIR}/dataset.parquet"
TOP_N = 30


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_PARQUET):
        print("Cleaned dataset already exists:", OUT_PARQUET)
        print("Delete the folder if you want to rebuild.")
        return

    spark = make_spark("arxiv-clean-top30", driver_memory="6g")

    df_raw = spark.read.json(INPUT_PATH).select("id", "title", "abstract", "categories", "update_date")
    df_raw = df_raw.na.drop(subset=["title", "abstract", "categories"])

    df_labeled = add_primary_label(df_raw)

    top_labels = compute_top_labels(df_labeled, TOP_N)
    with open(f"{OUT_DIR}/labels_top30.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(top_labels))

    df_top = filter_to_labels(df_labeled, top_labels)

    df_top = build_text_raw(df_top)
    df_top = clean_text(df_top)
    df_top = drop_too_short(df_top)

    df_out = df_top.select("id", "label_str", "text_clean", "update_date")
    df_out.write.mode("overwrite").parquet(OUT_PARQUET)

    print("Saved cleaned dataset to:", OUT_PARQUET)
    print("Saved top-30 labels to:", f"{OUT_DIR}/labels_top30.txt")

    spark.stop()


if __name__ == "__main__":
    main()
