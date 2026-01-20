from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes


def build_nb_pipeline(
    smoothing: float = 1.0,
    model_type: str = "multinomial",
) -> Pipeline:
    nb = NaiveBayes(
        featuresCol="features",
        labelCol="label",
        smoothing=smoothing,
        modelType=model_type,
    )
    return Pipeline(stages=[nb])
