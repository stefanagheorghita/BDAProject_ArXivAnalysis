from colors import *
from download_data import *
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from author_analysis import *

import os

os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-25"
os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + os.pathsep + os.environ["PATH"]


def load_data(spark, path):
    df = spark.read.json(path)
    return df


def analyze_categories(data):
    print("\n=== CATEGORY ANALYSIS ===")

    data_cat = data.withColumn("cat_array", F.split(F.col("categories"), " "))

    distinct_categories = (
        data_cat
        .withColumn("cat", F.explode("cat_array"))
        .select("cat")
        .distinct()
        .count()
    )
    print("Unique categories:", distinct_categories)

    print("\nTop categories:")
    category_freq = (
        data_cat
        .withColumn("cat", F.explode("cat_array"))
        .groupBy("cat")
        .count()
        .orderBy(F.desc("count"))
    )
    category_freq.show(20, truncate=False)

    data_cat = data_cat.withColumn("num_categories", F.size("cat_array"))
    mean_cats = data_cat.select(F.mean("num_categories")).first()[0]
    print("\nMean number of categories per paper:", mean_cats)
    print("Median number of categories per paper:", data_cat.select(F.median("num_categories")).first()[0])

    print("\nDistribution of category counts per paper:")
    dist_num_cats = (
        data_cat
        .groupBy("num_categories")
        .count()
        .orderBy("num_categories")
    )
    dist_num_cats.show(truncate=False)


def analyze_single_valued_column(data, col, top_n=20):
    print(f"\n=== Analysis for column: {col} ===")

    non_null = data.filter(F.col(col).isNotNull() & (F.col(col) != "")).count()
    print(f"Non-null entries: {non_null}")

    distinct_vals = (
        data.filter(F.col(col).isNotNull() & (F.col(col) != ""))
        .select(col)
        .distinct()
        .count()
    )
    print(f"Distinct non-null values: {distinct_vals}")

    print(f"\nTop {top_n} most common values for {col}:")
    freq = (
        data.filter(F.col(col).isNotNull() & (F.col(col) != ""))
        .groupBy(col)
        .count()
        .orderBy(F.desc("count"))
    )
    freq.show(top_n, truncate=False)


def show_nulls(data):
    nested_columns = ["authors_parsed", "versions"]

    non_nested = [c for c in data.columns if c not in nested_columns]

    non_nested_nulls = data.select([
        F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c)
        for c in non_nested
    ])

    print("Null counts for non-nested: ")
    non_nested_nulls.show(truncate=False)

    authors_nulls = (
        data
        .withColumn("author", F.explode_outer("authors_parsed"))
        .withColumn("surname", F.col("author")[0])
        .withColumn("first_names", F.col("author")[1])
        .withColumn("suffix", F.col("author")[2])
        .select(
            F.count(F.when(F.col("authors_parsed").isNull(), 1)).alias("authors_parsed_NULL"),

            F.count(F.when(F.col("author").isNull(), 1)).alias("authors_inner_array_NULL"),

            F.count(
                F.when(
                    F.col("surname").isNull() | (F.col("surname") == ""),
                    1
                )
            ).alias("surname_NULL_or_empty"),

            F.count(
                F.when(
                    F.col("first_names").isNull() | (F.col("first_names") == ""),
                    1
                )
            ).alias("first_names_NULL_or_empty"),

            F.count(
                F.when(
                    F.col("suffix").isNull() | (F.col("suffix") == ""),
                    1
                )
            ).alias("suffix_NULL_or_empty"),
        )
    )

    max_len = (
        data
        .withColumn("author", F.explode_outer("authors_parsed"))
        .select(F.size("author").alias("len"))
        .agg(F.max("len"))
        .first()[0]
    )
    print("Maximum number of elements in authors_parsed entries:", max_len)

    print("\nNull counts for nested column 'authors_parsed':")
    authors_nulls.show(truncate=False)

    versions_nulls = (
        data
        .withColumn("v", F.explode_outer("versions"))
        .select(
            F.count(F.when(F.col("versions").isNull(), 1)).alias("versions_NULL"),
            F.count(F.when(F.col("v").isNull(), 1)).alias("versions_element_NULL"),
            F.count(F.when(F.col("v.created").isNull() | (F.col("v.created") == ""), 1)).alias(
                "versions_created_NULL_or_empty"),
            F.count(F.when(F.col("v.version").isNull() | (F.col("v.version") == ""), 1)).alias(
                "versions_version_NULL_or_empty")
        )
    )

    print("\nNull counts for nested column 'versions':")
    versions_nulls.show(truncate=False)


def verify_duplicated_ids(data, duplicates):
    dup_ids = [row["id"] for row in duplicates.select("id").collect()]
    dups = data.filter(F.col("id").isin(dup_ids))
    dup_data = dups.groupBy("id").agg(F.collect_list(F.struct(data.columns)).alias("rows"))
    dup_reports = dup_data.collect()
    for row in dup_reports:
        id_val = row["id"]
        rows = [r.asDict() for r in row["rows"]]

        diff = {}
        cols = rows[0].keys()

        for c in cols:
            vals = [str(r[c]) for r in rows]
            if len(set(vals)) > 1:
                diff[c] = vals

        if diff:
            print(f"\n=== Differences for ID {id_val} ===")
            for k, v in diff.items():
                print(f"{k}:")
                for idx, val in enumerate(v):
                    print(f"  row {idx}: {val}")
            print()
        else:
            print(f"ID {id_val} is duplicated but all rows identical.")


def remove_duplicated_ids(data):
    w = Window.partitionBy("id").orderBy(F.col("update_date").desc())

    data_latest = (
        data
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") == 1)
        .drop("rank")
    )

    return data_latest


def print_uniqueness_info(data):
    total = data.count()
    distinct_ids = data.select("id").distinct().count()
    duplicates = (
        data.groupBy("id")
        .count()
        .filter(F.col("count") > 1)
    )

    print("\nTotal rows :", total)
    print("ID distinct:", distinct_ids)
    print("Duplicates :")
    duplicates.show()
    # verify_duplicated_ids(data, duplicates)

    data = remove_duplicated_ids(data)

    print("Removed duplicate IDs:")
    total = data.count()
    distinct_ids = data.select("id").distinct().count()
    print("\nTotal rows :", total)
    print("ID distinct:", distinct_ids)

    doi_non_null = data.filter(F.col("doi").isNotNull()).count()
    doi_distinct = data.filter(F.col("doi").isNotNull()).select("doi").distinct().count()

    print("\nTotal rows:", total)
    print("DOI non-null rows:", doi_non_null)
    print("Distinct DOI:", doi_distinct)

    distinct_titles = data.select("title").distinct().count()
    print("\nTotal rows:", total)
    print("Number of distinct titles:", distinct_titles)

    return data


if __name__ == '__main__':
    path = "../arxiv_data/arxiv-metadata-oai-snapshot.json"
    if not os.path.exists(path):
        download_arxiv_metadata()

    spark = SparkSession.builder \
        .appName("arXiv Analysis") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    data = load_data(spark, path)
    total = data.count()

    print(f"{YELLOW}Number of rows: {RESET}{total}")

    print(f"\n{YELLOW}Schema of the data:{RESET}")
    data.printSchema()

    print(f"\n{YELLOW}Info: {RESET}")
    data = print_uniqueness_info(data)

    print(f"\n{RED}Number of missing values: {RESET}:")
    show_nulls(data)

    analyze_categories(data)

    cols = ["journal-ref", "license", "report-no", "submitter"]

    for c in cols:
        analyze_single_valued_column(data, c, top_n=20)

    author_analysis(data)
