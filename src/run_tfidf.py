import os
from pipeline.spark import make_spark
from pipeline.tfidf import build_tfidf_pipeline, fit_transform_save

CLEAN_PARQUET = "../outputs/clean_top30/dataset.parquet"
OUT_DIR = "../outputs/tfidf_top30"
OUT_PARQUET = f"{OUT_DIR}/dataset.parquet"
PIPELINE_MODEL_DIR = f"{OUT_DIR}/feature_pipeline_model"

NUM_FEATURES = 1 << 16


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_PARQUET) and os.path.exists(PIPELINE_MODEL_DIR):
        print("TF-IDF dataset already exists:", OUT_PARQUET)
        print("Pipeline model already exists:", PIPELINE_MODEL_DIR)
        print("Delete the folder if you want to rebuild.")
        return

    spark = make_spark("arxiv-tfidf-top30", driver_memory="6g")

    df_clean = spark.read.parquet(CLEAN_PARQUET)

    pipe = build_tfidf_pipeline(num_features=NUM_FEATURES)
    df_feat, model = fit_transform_save(df_clean, pipe)

    df_feat.write.mode("overwrite").parquet(OUT_PARQUET)
    model.write().overwrite().save(PIPELINE_MODEL_DIR)

    print("Saved TF-IDF dataset to:", OUT_PARQUET)
    print("Saved TF-IDF pipeline model to:", PIPELINE_MODEL_DIR)

    spark.stop()


if __name__ == "__main__":
    main()
