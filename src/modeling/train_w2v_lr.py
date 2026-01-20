from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, StringIndexer, Word2Vec


def build_w2v_lr_pipeline(
    text_col: str = "text",
    label_str_col: str = "label_str",
    vector_size: int = 200,
    min_count: int = 2,
    max_iter: int = 60,
    reg_param: float = 0.1,
    elastic_net: float = 0.0,
) -> Pipeline:
    idx = StringIndexer(inputCol=label_str_col, outputCol="label", handleInvalid="skip")
    tok = RegexTokenizer(inputCol=text_col, outputCol="tokens", pattern="\\W+", minTokenLength=2)
    sw = StopWordsRemover(inputCol="tokens", outputCol="tokens_clean")
    w2v = Word2Vec(inputCol="tokens_clean", outputCol="features", vectorSize=vector_size, minCount=min_count)
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        family="multinomial",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net,
    )
    return Pipeline(stages=[idx, tok, sw, w2v, lr])
