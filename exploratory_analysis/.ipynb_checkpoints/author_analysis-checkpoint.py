from pyspark.sql import functions as F
from pyspark.sql.functions import year, to_timestamp, col


def extract_authors_basic(data):
        return (
            data
            .withColumn("author", F.explode_outer("authors_parsed"))
            .withColumn("last_name", F.col("author")[0])
            .withColumn("first_names", F.col("author")[1])
            .withColumn("full_name", F.concat_ws(" ", "last_name", "first_names"))
            .select("id", "full_name")
        )

def extract_authors_full(data):
    return (
        data
        .withColumn("author", F.explode_outer("authors_parsed"))
        .withColumn("full_name", F.concat_ws(" ", F.col("author")))
        .select("id", "full_name")
    )


def add_creation_date(data):
    return (
        data.withColumn(
            "created",
            F.col("versions").getItem(0).getItem("created")
        )
        .withColumn(
            "year",
            year(to_timestamp(col("created"), "EEE, dd MMM yyyy HH:mm:ss z"))
        )
    )



def author_analysis(data, top_n=20):

    print("\n====================")
    print(" Author Analysis")
    print("====================\n")

    authors_basic = extract_authors_basic(data)
    authors_full = extract_authors_full(data)


    print("\nNumber of authors per paper")
    num_auth = data.withColumn("num_authors", F.size("authors_parsed"))

    num_auth.groupBy("num_authors") \
        .count() \
        .orderBy(F.col("count").desc()) \
        .show(30, truncate=False)

    stats = num_auth.agg(
        F.mean("num_authors").alias("mean"),
        F.median("num_authors").alias("median"),
        F.min("num_authors").alias("min"),
        F.max("num_authors").alias("max")
    )
    print("\nStatistics:")
    stats.show(truncate=False)


    print("\n Most prolific authors (top N) – Basic extraction")
    authors_basic.groupBy("full_name").count().orderBy(F.desc("count")).show(top_n, truncate=False)

    print("\n Most prolific authors (top N) – Full extraction")
    authors_full.groupBy("full_name").count().orderBy(F.desc("count")).show(top_n, truncate=False)


    data = add_creation_date(data)

    print("\n Authors per year")
    authors_year = (
        authors_basic
        .join(data.select("id", "year"), on="id", how="left")
    )

    authors_per_year = (
        authors_year.groupBy("year").count().orderBy("year")
    )

    authors_per_year.show(50)

    print("\n Top authors over time (sample of top 10)")
    top_authors = (
        authors_basic.groupBy("full_name").count()
        .orderBy(F.desc("count"))
        .limit(5)
    )

    top_author_list = [r["full_name"] for r in top_authors.collect()]

    authors_year.filter(F.col("full_name").isin(top_author_list)) \
        .groupBy("full_name", "year") \
        .count() \
        .orderBy("full_name", "year") \
        .show(200, truncate=False)


    print("\n Top co-author pairs")

    author_lists = (
        authors_basic.groupBy("id")
        .agg(F.collect_list("full_name").alias("authors_list"))
    )

    pairs = (
        author_lists
        .withColumn("a1", F.explode("authors_list"))
        .withColumn("a2", F.explode("authors_list"))
        .filter(F.col("a1") < F.col("a2"))
        .groupBy("a1", "a2")
        .count()
        .orderBy(F.desc("count"))
    )

    pairs.show(top_n, truncate=False)

