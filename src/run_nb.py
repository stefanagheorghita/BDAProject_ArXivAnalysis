from __future__ import annotations

import argparse
import os
from typing import List

from pipeline.spark import make_spark
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql import functions as F

from modeling.train_nb import build_nb_pipeline
from modeling.eval import eval_metrics, confusion_counts

SEED = 42


def load_label_list_from_feature_pipeline(feature_pipeline_dir: str) -> List[str]:
    """
    Reads the fitted TF-IDF feature PipelineModel and extracts the label order
    from the StringIndexerModel (label_str -> label).
    """
    pm = PipelineModel.load(feature_pipeline_dir)

    sim = None
    for st in pm.stages:
        if isinstance(st, StringIndexerModel) and st.getOutputCol() == "label":
            sim = st
            break

    if sim is None:
        raise RuntimeError("StringIndexerModel (outputCol='label') not found in feature pipeline model.")

    return list(sim.labels)


def save_label_map_txt(labels: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, lab in enumerate(labels):
            f.write(f"{i} -> {lab}\n")


def decode_confusions(conf_df, labels: List[str]):
    """
    conf_df: columns [label, prediction, count] (label/prediction are doubles)
    returns: columns [label_name, pred_name, count]
    """
    spark = conf_df.sparkSession

    mapping = spark.createDataFrame(
        [(float(i), lab) for i, lab in enumerate(labels)],
        ["idx", "name"],
    )

    decoded = (
        conf_df.join(
            mapping.withColumnRenamed("idx", "label").withColumnRenamed("name", "label_name"),
            on="label",
            how="left",
        )
        .join(
            mapping.withColumnRenamed("idx", "prediction").withColumnRenamed("name", "pred_name"),
            on="prediction",
            how="left",
        )
        .select("label_name", "pred_name", "count")
        .orderBy(F.desc("count"))
    )
    return decoded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Naive Bayes on TF-IDF features.")
    p.add_argument("--tfidf_parquet", default="../outputs/tfidf_top30/dataset.parquet")
    p.add_argument("--feature_pipeline_dir", default="../outputs/tfidf_top30/feature_pipeline_model")
    p.add_argument("--out_dir", default="../outputs/nb_top30")

    p.add_argument("--driver_memory", default="6g")
    p.add_argument("--repartition", type=int, default=16)

    p.add_argument("--train_frac", type=float, default=0.8)

    p.add_argument("--smoothing", type=float, default=1.0)
    p.add_argument("--model_type", default="multinomial", choices=["multinomial", "bernoulli"])

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    model_dir = f"{out_dir}/model"
    metrics_path = f"{out_dir}/metrics.txt"
    confusion_path = f"{out_dir}/confusion_top50.parquet"

    top_conf_off_raw_path = f"{out_dir}/top_confusions_offdiag_raw.parquet"
    top_conf_off_decoded_path = f"{out_dir}/top_confusions_offdiag_decoded.parquet"
    label_map_path = f"{out_dir}/label_map.txt"

    spark = make_spark("arxiv-nb-top30", driver_memory=args.driver_memory)

    df = spark.read.parquet(args.tfidf_parquet).select(
        "id", "label_str", "label", "features", "update_date"
    )

    train_df, test_df = df.randomSplit([args.train_frac, 1.0 - args.train_frac], seed=SEED)
    train_df = train_df.repartition(args.repartition)

    pipe = build_nb_pipeline(smoothing=args.smoothing, model_type=args.model_type)
    model = pipe.fit(train_df)

    pred = model.transform(test_df)

    metrics = eval_metrics(pred)
    print("Metrics:", metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    conf = confusion_counts(pred, limit=50)
    conf.write.mode("overwrite").parquet(confusion_path)
    conf.show(20, truncate=False)

    model.write().overwrite().save(model_dir)
    print("Saved NB model to:", model_dir)
    print("Saved metrics to:", metrics_path)
    print("Saved confusion to:", confusion_path)

    conf_off = (
        pred.where(F.col("label") != F.col("prediction"))
        .groupBy("label", "prediction")
        .count()
        .orderBy(F.desc("count"))
        .limit(50)
    )
    conf_off.write.mode("overwrite").parquet(top_conf_off_raw_path)
    print("Saved top off-diagonal confusions (raw) to:", top_conf_off_raw_path)
    conf_off.show(30, truncate=False)

    labels = load_label_list_from_feature_pipeline(args.feature_pipeline_dir)
    save_label_map_txt(labels, label_map_path)
    print("Saved label map to:", label_map_path)

    decoded = decode_confusions(conf_off, labels)
    decoded.write.mode("overwrite").parquet(top_conf_off_decoded_path)
    print("Saved decoded off-diagonal confusions to:", top_conf_off_decoded_path)
    decoded.show(50, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
