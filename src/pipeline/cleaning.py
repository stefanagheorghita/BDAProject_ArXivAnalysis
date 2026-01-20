from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def build_text_raw(df: DataFrame) -> DataFrame:
    return df.withColumn("text_raw", F.concat_ws(" ", F.col("title"), F.col("abstract")))

def clean_text(df: DataFrame, input_col: str = "text_raw", output_col: str = "text_clean") -> DataFrame:
    c = F.lower(F.col(input_col))

    c = F.regexp_replace(c, r"http\S+|www\.\S+", " ")

    c = F.regexp_replace(c, r"\$.*?\$", " ")                 # inline math
    c = F.regexp_replace(c, r"\\[a-zA-Z]+(\{.*?\})?", " ")   # commands like \alpha or \textbf{...}

    c = F.regexp_replace(c, r"[^a-z0-9]+", " ")

    c = F.regexp_replace(c, r"\s+", " ")
    c = F.trim(c)

    return df.withColumn(output_col, c)

def drop_too_short(df: DataFrame, col: str = "text_clean", min_chars: int = 50) -> DataFrame:
    return df.where(F.length(F.col(col)) >= F.lit(min_chars))
