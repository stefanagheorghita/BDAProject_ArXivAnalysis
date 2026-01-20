from __future__ import annotations
from typing import Dict
from pyspark.sql import DataFrame


def stratified_subset_by_per_class(
    df: DataFrame,
    label_col: str = "label",
    per_class: int = 1000,
    seed: int = 42,
) -> DataFrame:
    counts = df.groupBy(label_col).count().collect()

    fractions: Dict[float, float] = {}
    for r in counts:
        lab = r[label_col]
        cnt = r["count"]
        frac = min(1.0, float(per_class) / float(cnt))
        fractions[lab] = frac

    return df.sampleBy(label_col, fractions=fractions, seed=seed)
