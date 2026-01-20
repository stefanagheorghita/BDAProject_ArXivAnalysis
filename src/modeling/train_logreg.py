from __future__ import annotations

from typing import List

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import IndexToString
from pyspark.sql import DataFrame


def build_logreg_pipeline(
    max_iter: int = 50,
    reg_param: float = 0.1,
    elastic_net: float = 0.0,
) -> Pipeline:
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net,
        family="multinomial",
    )
    return Pipeline(stages=[lr])


def add_label_decoder(pred_df: DataFrame, labels: List[str]) -> DataFrame:
    """
    Convert numeric prediction back to label string using the StringIndexer label order.
    """
    converter = IndexToString(
        inputCol="prediction",
        outputCol="predicted_label",
        labels=labels,
    )
    return converter.transform(pred_df)
