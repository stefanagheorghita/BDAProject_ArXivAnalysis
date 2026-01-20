import os
import time

from pipeline.spark import make_spark
from pyspark.sql import functions as F

from modeling.eval import eval_metrics, confusion_counts
from modeling.train_w2v_lr import build_w2v_lr_pipeline

RAW_PARQUET = "../outputs/clean_top30/dataset.parquet"

OUT_DIR = "../outputs/w2v_lr_top30_full"
MODEL_DIR = f"{OUT_DIR}/model"
METRICS_PATH = f"{OUT_DIR}/metrics.txt"
CONFUSION_PATH = f"{OUT_DIR}/confusion_top50.parquet"
TOP_CONF_OFF_RAW_PATH = f"{OUT_DIR}/top_confusions_offdiag_raw.parquet"
TIMES_PATH = f"{OUT_DIR}/timings.txt"

SEED = 42


def fmt(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def log_line(lines: list[str], msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    lines = []

    spark = make_spark("arxiv-w2v-lr-top30-full", driver_memory="12g")
    spark.conf.set("spark.sql.shuffle.partitions", "128")

    t_read0 = time.time()
    df = spark.read.parquet(RAW_PARQUET).select("id", "label_str", "text_clean", "update_date")
    df = df.withColumn("text", F.coalesce(F.col("text_clean").cast("string"), F.lit("")))
    log_line(lines, f"[time] load+select: {fmt(time.time() - t_read0)}")

    t_split0 = time.time()
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)
    train_n = train_df.count()
    test_n = test_df.count()
    log_line(lines, f"Train size: {train_n}")
    log_line(lines, f"Test size:  {test_n}")
    log_line(lines, f"[time] split+count: {fmt(time.time() - t_split0)}")

    pipe = build_w2v_lr_pipeline(
        text_col="text",
        label_str_col="label_str",
        vector_size=200,
        min_count=2,
        max_iter=60,
        reg_param=0.1,
        elastic_net=0.0,
    )

    t_fit0 = time.time()
    model = pipe.fit(train_df)
    log_line(lines, f"[time] fit: {fmt(time.time() - t_fit0)}")

    t_pred0 = time.time()
    pred = model.transform(test_df)
    pred.cache()
    _ = pred.count()
    log_line(lines, f"[time] transform+cache: {fmt(time.time() - t_pred0)}")

    t_eval0 = time.time()
    metrics = eval_metrics(pred)
    log_line(lines, f"Metrics: {metrics}")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    log_line(lines, f"[time] eval+write-metrics: {fmt(time.time() - t_eval0)}")

    t_conf0 = time.time()
    conf = confusion_counts(pred, limit=50)
    conf.write.mode("overwrite").parquet(CONFUSION_PATH)
    conf.show(20, truncate=False)
    log_line(lines, f"[time] confusion: {fmt(time.time() - t_conf0)}")

    t_save0 = time.time()
    model.write().overwrite().save(MODEL_DIR)
    log_line(lines, f"Saved Word2Vec+LR model to: {MODEL_DIR}")
    log_line(lines, f"[time] save-model: {fmt(time.time() - t_save0)}")

    t_off0 = time.time()
    conf_off = (
        pred.where(F.col("label") != F.col("prediction"))
        .groupBy("label", "prediction")
        .count()
        .orderBy(F.desc("count"))
        .limit(50)
    )
    conf_off.write.mode("overwrite").parquet(TOP_CONF_OFF_RAW_PATH)
    conf_off.show(50, truncate=False)
    log_line(lines, f"[time] offdiag: {fmt(time.time() - t_off0)}")

    total = time.time() - t0
    log_line(lines, f"[time] TOTAL: {fmt(total)}")

    with open(TIMES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
