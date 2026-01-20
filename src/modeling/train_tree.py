from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier


def build_tree_pipeline(
        max_depth: int = 12,
        max_bins: int = 64,
        min_instances_per_node: int = 1,
        seed: int = 42,
) -> Pipeline:
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=max_depth,
        maxBins=max_bins,
        minInstancesPerNode=min_instances_per_node,
        seed=seed,
    )
    return Pipeline(stages=[dt])
