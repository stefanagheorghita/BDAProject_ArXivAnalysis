from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier


def build_rf_pipeline(
    num_trees: int = 200,
    max_depth: int = 8,
    max_bins: int = 32,
    min_instances_per_node: int = 10,
    feature_subset_strategy: str = "sqrt",
    seed: int = 42,
) -> Pipeline:

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=num_trees,
        maxDepth=max_depth,
        maxBins=max_bins,
        minInstancesPerNode=min_instances_per_node,
        featureSubsetStrategy=feature_subset_strategy,
        seed=seed,
    )
    return Pipeline(stages=[rf])
