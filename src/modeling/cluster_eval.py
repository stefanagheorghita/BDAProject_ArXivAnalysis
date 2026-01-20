import math
from collections import defaultdict

from pyspark.sql import functions as F
from pyspark.sql.window import Window


def purity_from_contingency(rows):
    best_by_cluster = defaultdict(int)
    total = 0
    for r in rows:
        k = int(r["cluster"])
        cnt = int(r["count"])
        total += cnt
        if cnt > best_by_cluster[k]:
            best_by_cluster[k] = cnt
    if total == 0:
        return 0.0
    return sum(best_by_cluster.values()) / total


def nmi_from_contingency(rows):
    n_ck = defaultdict(int)
    n_c = defaultdict(int)
    n_k = defaultdict(int)
    total = 0

    for r in rows:
        c = r["label_str"]
        k = int(r["cluster"])
        cnt = int(r["count"])
        if cnt <= 0:
            continue
        total += cnt
        n_ck[(c, k)] += cnt
        n_c[c] += cnt
        n_k[k] += cnt

    if total == 0:
        return 0.0

    mi = 0.0
    for (c, k), n in n_ck.items():
        mi += (n / total) * math.log((total * n) / (n_c[c] * n_k[k]))

    h_c = 0.0
    for c, n in n_c.items():
        p = n / total
        h_c -= p * math.log(p)

    h_k = 0.0
    for k, n in n_k.items():
        p = n / total
        h_k -= p * math.log(p)

    denom = math.sqrt(h_c * h_k)
    if denom == 0.0:
        return 0.0
    return mi / denom


def add_cluster_label_mapping(pred_df):
    w = Window.partitionBy("cluster").orderBy(F.desc("count"))
    mapping = (
        pred_df.groupBy("cluster", "label_str")
        .count()
        .withColumn("rn", F.row_number().over(w))
        .where(F.col("rn") == 1)
        .select("cluster", F.col("label_str").alias("pred_label_str"), F.col("count").alias("top_count"))
    )

    sizes = pred_df.groupBy("cluster").count().withColumnRenamed("count", "cluster_size")

    cluster_summary = (
        sizes.join(mapping, on="cluster", how="left")
        .withColumn("purity", F.col("top_count") / F.col("cluster_size"))
        .select("cluster", "cluster_size", "purity", "pred_label_str")
        .orderBy("cluster")
    )

    pred_labeled = (
        pred_df.join(mapping.select("cluster", "pred_label_str"), on="cluster", how="left")
        .select("label_str", "pred_label_str", "cluster")
    )

    return pred_labeled, cluster_summary


def classification_report_from_predlabels(df_pred_labeled):
    tp = (
        df_pred_labeled.groupBy("label_str", "pred_label_str")
        .count()
        .where(F.col("label_str") == F.col("pred_label_str"))
        .select(F.col("label_str").alias("label"), F.col("count").alias("tp"))
    )

    actual = (
        df_pred_labeled.groupBy("label_str")
        .count()
        .select(F.col("label_str").alias("label"), F.col("count").alias("support"))
    )

    predicted = (
        df_pred_labeled.groupBy("pred_label_str")
        .count()
        .select(F.col("pred_label_str").alias("label"), F.col("count").alias("pred_count"))
    )

    rep = (
        actual.join(predicted, on="label", how="left")
        .join(tp, on="label", how="left")
        .fillna(0, subset=["tp", "pred_count"])
        .withColumn("precision", F.when(F.col("pred_count") > 0, F.col("tp") / F.col("pred_count")).otherwise(F.lit(0.0)))
        .withColumn("recall", F.when(F.col("support") > 0, F.col("tp") / F.col("support")).otherwise(F.lit(0.0)))
        .withColumn(
            "f1",
            F.when((F.col("precision") + F.col("recall")) > 0, 2 * F.col("precision") * F.col("recall") / (F.col("precision") + F.col("recall")))
            .otherwise(F.lit(0.0)),
        )
        .select("label", "precision", "recall", "f1", "support")
        .orderBy(F.desc("support"))
    )

    totals = df_pred_labeled.count()
    correct = df_pred_labeled.where(F.col("label_str") == F.col("pred_label_str")).count()
    accuracy = correct / totals if totals else 0.0

    weighted_f1 = rep.select((F.sum(F.col("f1") * F.col("support")) / F.sum("support")).alias("weighted_f1")).collect()[0]["weighted_f1"]
    macro_f1 = rep.select(F.avg("f1").alias("macro_f1")).collect()[0]["macro_f1"]

    return rep, accuracy, macro_f1, weighted_f1
