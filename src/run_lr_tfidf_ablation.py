import os
import time

from pipeline.spark import make_spark
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    NGram,
    StringIndexer,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


RAW_PARQUET = "../outputs/clean_top30/dataset.parquet"

OUT_DIR = "../outputs/ablation_lr_tfidf"
RESULTS_PARQUET = f"{OUT_DIR}/results.parquet"
RESULTS_CSV_DIR = f"{OUT_DIR}/results_csv"
MODEL_ROOT = f"{OUT_DIR}/models"

SEED = 42


def build_lr_tail_pipeline(vocab_size: int, terms_col: str):
    cv = CountVectorizer(inputCol=terms_col, outputCol="tf", vocabSize=int(vocab_size), minDF=2.0)
    idf = IDF(inputCol="tf", outputCol="features")
    label_indexer = StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="skip")

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.0,
        elasticNetParam=0.0,
        family="multinomial",
    )

    return Pipeline(stages=[cv, idf, label_indexer, lr])


def eval_metrics(pred_df):
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    wprec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    wrec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    acc = acc_eval.evaluate(pred_df)
    f1 = f1_eval.evaluate(pred_df)
    wprec = wprec_eval.evaluate(pred_df)
    wrec = wrec_eval.evaluate(pred_df)
    return acc, f1, wprec, wrec


def preprocess(df, use_bigrams: bool):
    tok = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+", minTokenLength=2)
    sw = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")

    df2 = tok.transform(df)
    df2 = sw.transform(df2)

    if not use_bigrams:
        return df2, "tokens_clean"

    ng = NGram(n=2, inputCol="tokens_clean", outputCol="bigrams")
    df2 = ng.transform(df2)

    df2 = df2.withColumn("all_terms", F.concat(F.col("tokens_clean"), F.col("bigrams")))
    return df2, "all_terms"


def load_existing_results(spark):
    if os.path.exists(RESULTS_PARQUET):
        try:
            return spark.read.parquet(RESULTS_PARQUET)
        except Exception:
            return None
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)

    spark = make_spark("arxiv-lr-tfidf-ablation", driver_memory="12g")
    spark.conf.set("spark.sql.shuffle.partitions", "128")

    t_global0 = time.time()

    base = spark.read.parquet(RAW_PARQUET).select("label_str", "text_clean")
    base = base.where(F.col("label_str").isNotNull())
    base = base.withColumn("text", F.coalesce(F.col("text_clean").cast("string"), F.lit(""))).select("label_str", "text")

    train_df, test_df = base.randomSplit([0.8, 0.2], seed=SEED)
    train_df = train_df.repartition(8).cache()
    test_df = test_df.repartition(8).cache()
    _ = train_df.count()
    _ = test_df.count()

    settings = [
        (10_000, False),
        (50_000, False),
        (100_000, False),
        (10_000, True),
    ]

    results_new = []

    for vocab_size, use_bigrams in settings:
        cfg_name = f"v{vocab_size}_{'uni+bi' if use_bigrams else 'uni'}"
        model_path = os.path.join(MODEL_ROOT, cfg_name)

        if os.path.exists(model_path):
            print(f"\n=== Skipping (already exists): {cfg_name} ===")
            continue

        print(f"\n=== Running: {cfg_name} ===")
        t_cfg0 = time.time()

        tprep0 = time.time()
        train_p, terms_col = preprocess(train_df, use_bigrams=use_bigrams)
        test_p, _ = preprocess(test_df, use_bigrams=use_bigrams)
        train_p = train_p.select("label_str", terms_col).repartition(8).cache()
        test_p = test_p.select("label_str", terms_col).repartition(8).cache()
        _ = train_p.count()
        _ = test_p.count()
        prep_s = time.time() - tprep0

        pipe = build_lr_tail_pipeline(vocab_size=vocab_size, terms_col=terms_col)

        tfit0 = time.time()
        model = pipe.fit(train_p)
        fit_s = time.time() - tfit0

        ttr0 = time.time()
        pred = model.transform(test_p).select("label", "prediction").cache()
        _ = pred.count()
        transform_s = time.time() - ttr0

        teval0 = time.time()
        acc, f1, wprec, wrec = eval_metrics(pred)
        eval_s = time.time() - teval0

        tsave0 = time.time()
        model.write().overwrite().save(model_path)
        save_s = time.time() - tsave0

        row = {
            "config": cfg_name,
            "vocab_size": int(vocab_size),
            "use_bigrams": bool(use_bigrams),
            "prep_seconds": float(prep_s),
            "fit_seconds": float(fit_s),
            "transform_seconds": float(transform_s),
            "eval_seconds": float(eval_s),
            "save_seconds": float(save_s),
            "total_seconds": float(time.time() - t_cfg0),
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "weighted_precision": float(wprec),
            "weighted_recall": float(wrec),
            "model_path": model_path,
        }

        print(row)
        results_new.append(row)

        pred.unpersist()
        train_p.unpersist()
        test_p.unpersist()

    if results_new:
        res_new_df = spark.createDataFrame(results_new)
        res_old_df = load_existing_results(spark)

        if res_old_df is not None:
            res_all = res_old_df.unionByName(res_new_df, allowMissingColumns=True)
        else:
            res_all = res_new_df

        res_all = res_all.dropDuplicates(["config"]).orderBy(F.desc("accuracy"))
        res_all.show(truncate=False)

        res_all.write.mode("overwrite").parquet(RESULTS_PARQUET)
        res_all.coalesce(1).write.mode("overwrite").option("header", True).csv(RESULTS_CSV_DIR)
    else:
        if os.path.exists(RESULTS_PARQUET):
            print("No new configs trained. Existing results:")
            spark.read.parquet(RESULTS_PARQUET).orderBy(F.desc("accuracy")).show(truncate=False)
        else:
            print("No new configs were trained and no results file exists yet.")

    total_s = time.time() - t_global0
    print(f"\n[time] TOTAL SCRIPT: {int(total_s // 60)}m {int(total_s % 60)}s")

    spark.stop()


if __name__ == "__main__":
    main()
