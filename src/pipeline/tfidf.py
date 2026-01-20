from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer

def build_tfidf_pipeline(num_features: int = 1 << 18) -> Pipeline:
    tokenizer = RegexTokenizer(inputCol="text_clean", outputCol="tokens", pattern="\\W+")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    tf = HashingTF(inputCol="filtered_tokens", outputCol="tf", numFeatures=num_features)
    idf = IDF(inputCol="tf", outputCol="features")
    label_indexer = StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="skip")

    return Pipeline(stages=[tokenizer, remover, tf, idf, label_indexer])

def fit_transform_save(
    df_clean: DataFrame,
    pipeline: Pipeline,
) -> tuple[DataFrame, PipelineModel]:
    model = pipeline.fit(df_clean)
    df_feat = model.transform(df_clean).select("id", "label_str", "label", "features", "update_date")
    return df_feat, model
