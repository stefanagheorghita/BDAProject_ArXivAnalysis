from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def eval_metrics(pred: DataFrame) -> dict:
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    return {
        "accuracy": acc_eval.evaluate(pred),
        "f1": f1_eval.evaluate(pred),
    }

def confusion_counts(pred: DataFrame, limit: int = 50) -> DataFrame:
    return (
        pred.groupBy("label", "prediction")
            .count()
            .orderBy(F.desc("count"))
            .limit(limit)
    )
