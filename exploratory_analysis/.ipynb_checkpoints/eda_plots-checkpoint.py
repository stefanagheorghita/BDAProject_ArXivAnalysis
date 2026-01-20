import os
import re
import json
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(REPO_ROOT, "arxiv_data", "arxiv-metadata-oai-snapshot.json")
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")
OUT_DIR = os.path.join(REPO_ROOT, "eda_outputs")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name: str):
    out = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

def created_to_year_col():
    return F.year(F.to_timestamp(F.col("versions").getItem(0).getItem("created"), "EEE, d MMM yyyy HH:mm:ss z"))

def author_name_from_parsed(author_col):
    return F.concat_ws(" ", author_col.getItem(0), author_col.getItem(1))

def safe_title(s: str, max_len=80):
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def load_df(spark: SparkSession, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return spark.read.json(path)

def get_citation_col(df):
    for c in ["citation_count", "citations", "num_citations", "n_citation", "cited_by_count"]:
        if c in df.columns:
            return c
    return None

def write_summary(summary_rows, json_path, csv_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("key,value\n")
        for r in summary_rows:
            k = str(r.get("key", "")).replace("\n", " ").replace("\r", " ")
            v = str(r.get("value", "")).replace("\n", " ").replace("\r", " ")
            f.write(f"{k},{json.dumps(v, ensure_ascii=False)}\n")
    print("Saved:", json_path)
    print("Saved:", csv_path)

def plot_papers_per_year(df):
    pdf = (df.withColumn("year", created_to_year_col())
             .filter(F.col("year").isNotNull())
             .groupBy("year").count().orderBy("year").toPandas())
    plt.figure()
    plt.plot(pdf["year"], pdf["count"])
    plt.title("Papers per year")
    plt.xlabel("Year")
    plt.ylabel("Number of papers")
    savefig("01_papers_per_year.png")

def plot_authors_per_paper_distribution(df, cap=20):
    pdf = (df.withColumn("num_authors", F.size("authors_parsed"))
             .groupBy("num_authors").count().orderBy("num_authors")
             .filter(F.col("num_authors").between(0, cap)).toPandas())
    plt.figure()
    plt.bar(pdf["num_authors"], pdf["count"])
    plt.title(f"Authors per paper distribution (<= {cap})")
    plt.xlabel("# authors")
    plt.ylabel("# papers")
    savefig("02_authors_per_paper_dist.png")

def plot_mean_authors_per_year(df):
    pdf = (df.withColumn("year", created_to_year_col())
             .withColumn("num_authors", F.size("authors_parsed"))
             .filter(F.col("year").isNotNull())
             .groupBy("year").agg(F.mean("num_authors").alias("mean_authors"))
             .orderBy("year").toPandas())
    plt.figure()
    plt.plot(pdf["year"], pdf["mean_authors"])
    plt.title("Mean authors per paper over time")
    plt.xlabel("Year")
    plt.ylabel("Mean # authors")
    savefig("03_mean_authors_per_year.png")

def plot_categories_per_paper_distribution(df, cap=10):
    pdf = (df.withColumn("num_categories", F.size(F.split(F.col("categories"), " ")))
             .groupBy("num_categories").count().orderBy("num_categories")
             .filter(F.col("num_categories").between(0, cap)).toPandas())
    plt.figure()
    plt.bar(pdf["num_categories"], pdf["count"])
    plt.title(f"Categories per paper distribution (<= {cap})")
    plt.xlabel("# categories")
    plt.ylabel("# papers")
    savefig("04_categories_per_paper_dist.png")

def plot_mean_categories_per_year(df):
    pdf = (df.withColumn("year", created_to_year_col())
             .withColumn("num_categories", F.size(F.split(F.col("categories"), " ")))
             .filter(F.col("year").isNotNull())
             .groupBy("year").agg(F.mean("num_categories").alias("mean_categories"))
             .orderBy("year").toPandas())
    plt.figure()
    plt.plot(pdf["year"], pdf["mean_categories"])
    plt.title("Mean categories per paper over time")
    plt.xlabel("Year")
    plt.ylabel("Mean # categories")
    savefig("05_mean_categories_per_year.png")

def plot_top_categories(df, top_n=20):
    pdf = (df.withColumn("cat", F.explode(F.split(F.col("categories"), " ")))
             .groupBy("cat").count().orderBy(F.desc("count"))
             .limit(top_n).toPandas().sort_values("count"))
    plt.figure()
    plt.barh(pdf["cat"], pdf["count"])
    plt.title(f"Top {top_n} categories")
    plt.xlabel("# papers")
    plt.ylabel("Category")
    savefig("06_top_categories.png")

def plot_top_authors(df, top_n=20):
    authors = (df.withColumn("author", F.explode_outer("authors_parsed"))
                 .withColumn("full_name", author_name_from_parsed(F.col("author")))
                 .filter(F.col("full_name").isNotNull() & (F.col("full_name") != "")))
    pdf = (authors.groupBy("full_name").count()
             .orderBy(F.desc("count")).limit(top_n).toPandas().sort_values("count"))
    plt.figure()
    plt.barh(pdf["full_name"], pdf["count"])
    plt.title(f"Top {top_n} most prolific authors (name strings)")
    plt.xlabel("# papers")
    plt.ylabel("Author")
    savefig("07_top_authors.png")

def plot_author_productivity_over_time(df, top_k=5):
    authors = (df.withColumn("author", F.explode_outer("authors_parsed"))
                 .withColumn("full_name", author_name_from_parsed(F.col("author")))
                 .filter(F.col("full_name").isNotNull() & (F.col("full_name") != ""))
                 .select("id", "full_name"))
    top_authors = (authors.groupBy("full_name").count()
                     .orderBy(F.desc("count")).limit(top_k)
                     .select("full_name").toPandas()["full_name"].tolist())
    df_year = df.withColumn("year", created_to_year_col()).select("id", "year")
    ay = (authors.join(df_year, on="id", how="inner")
            .filter(F.col("year").isNotNull())
            .filter(F.col("full_name").isin(top_authors))
            .groupBy("full_name", "year").count().orderBy("year").toPandas())
    plt.figure()
    for name in top_authors:
        sub = ay[ay["full_name"] == name].sort_values("year")
        plt.plot(sub["year"], sub["count"], label=name)
    plt.title(f"Top {top_k} authors: papers per year")
    plt.xlabel("Year")
    plt.ylabel("# papers")
    plt.legend(fontsize=8)
    savefig("08_top_authors_over_time.png")

def plot_top_categories_over_time(df, top_k=5):
    cats = df.withColumn("cat", F.explode(F.split(F.col("categories"), " "))).select("id", "cat")
    top_cats = (cats.groupBy("cat").count()
                  .orderBy(F.desc("count")).limit(top_k)
                  .select("cat").toPandas()["cat"].tolist())
    df_year = df.withColumn("year", created_to_year_col()).select("id", "year")
    cy = (cats.join(df_year, on="id", how="inner")
            .filter(F.col("year").isNotNull())
            .filter(F.col("cat").isin(top_cats))
            .groupBy("cat", "year").count().orderBy("year").toPandas())
    plt.figure()
    for c in top_cats:
        sub = cy[cy["cat"] == c].sort_values("year")
        plt.plot(sub["year"], sub["count"], label=c)
    plt.title(f"Top {top_k} categories over time")
    plt.xlabel("Year")
    plt.ylabel("# papers")
    plt.legend(fontsize=8)
    savefig("09_top_categories_over_time.png")

def plot_top_coauthor_pairs(df, top_n=15, max_authors_per_paper=50, sample_frac=0.25):
    authors = (df.withColumn("author", F.explode_outer("authors_parsed"))
                 .withColumn("full_name", author_name_from_parsed(F.col("author")))
                 .filter(F.col("full_name").isNotNull() & (F.col("full_name") != ""))
                 .select("id", "full_name"))
    author_lists = authors.groupBy("id").agg(F.collect_set("full_name").alias("alist"))
    author_lists = (author_lists.withColumn("n", F.size("alist"))
                               .filter(F.col("n") >= 2)
                               .filter(F.col("n") <= max_authors_per_paper))
    if sample_frac < 1.0:
        author_lists = author_lists.sample(withReplacement=False, fraction=sample_frac, seed=42)
    pairs = (author_lists.withColumn("a1", F.explode("alist"))
                        .withColumn("a2", F.explode("alist"))
                        .filter(F.col("a1") < F.col("a2"))
                        .groupBy("a1", "a2").count()
                        .orderBy(F.desc("count")).limit(top_n)
                        .toPandas().sort_values("count"))
    labels = [f"{r.a1} — {r.a2}" for r in pairs.itertuples(index=False)]
    plt.figure()
    plt.barh(labels, pairs["count"])
    plt.title(f"Top {top_n} co-author pairs (n<={max_authors_per_paper}, sample={sample_frac})")
    plt.xlabel("# coauthored papers")
    plt.ylabel("Pair")
    savefig("10_top_coauthor_pairs.png")

def plot_abstract_length_over_time(df):
    pdf = (df.withColumn("year", created_to_year_col())
             .withColumn("abs_len", F.length(F.col("abstract")))
             .filter(F.col("year").isNotNull())
             .filter(F.col("abs_len").isNotNull())
             .groupBy("year").agg(F.mean("abs_len").alias("mean_abs_len"))
             .orderBy("year").toPandas())
    plt.figure()
    plt.plot(pdf["year"], pdf["mean_abs_len"])
    plt.title("Mean abstract length over time")
    plt.xlabel("Year")
    plt.ylabel("Mean abstract length (chars)")
    savefig("11_mean_abstract_length_over_time.png")

def plot_title_length_vs_num_authors_scatter(df, sample_n=200000):
    sampled = (df.select(F.size("authors_parsed").alias("num_authors"), F.length("title").alias("title_len"))
                 .filter(F.col("num_authors").isNotNull() & F.col("title_len").isNotNull())
                 .sample(withReplacement=False, fraction=1.0, seed=42)
                 .limit(sample_n).toPandas())
    plt.figure()
    plt.scatter(sampled["num_authors"], sampled["title_len"], s=5, alpha=0.4)
    plt.title("Title length vs number of authors (sample)")
    plt.xlabel("# authors")
    plt.ylabel("Title length (chars)")
    savefig("12_title_len_vs_num_authors_scatter.png")

def plot_author_category_heatmap(df, top_authors_n=10, top_cats_n=12, sample_frac=0.35):
    if sample_frac < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_frac, seed=42)
    authors = (df.withColumn("author", F.explode_outer("authors_parsed"))
                 .withColumn("full_name", author_name_from_parsed(F.col("author")))
                 .filter(F.col("full_name").isNotNull() & (F.col("full_name") != ""))
                 .select("id", "full_name"))
    top_authors = (authors.groupBy("full_name").count()
                     .orderBy(F.desc("count")).limit(top_authors_n)
                     .select("full_name").toPandas()["full_name"].tolist())
    cats = df.withColumn("cat", F.explode(F.split(F.col("categories"), " "))).select("id", "cat")
    top_cats = (cats.groupBy("cat").count()
                  .orderBy(F.desc("count")).limit(top_cats_n)
                  .select("cat").toPandas()["cat"].tolist())
    ac = (authors.join(cats, on="id", how="inner")
            .filter(F.col("full_name").isin(top_authors))
            .filter(F.col("cat").isin(top_cats))
            .groupBy("full_name", "cat").count())
    pdf = ac.toPandas()
    author_idx = {a: i for i, a in enumerate(top_authors)}
    cat_idx = {c: j for j, c in enumerate(top_cats)}
    mat = [[0 for _ in top_cats] for _ in top_authors]
    for row in pdf.itertuples(index=False):
        mat[author_idx[row.full_name]][cat_idx[row.cat]] = row.count
    plt.figure(figsize=(10, 5))
    plt.imshow(mat, aspect="auto")
    plt.title("Top authors × top categories (counts)")
    plt.xlabel("Category")
    plt.ylabel("Author")
    plt.xticks(range(len(top_cats)), top_cats, rotation=45, ha="right")
    plt.yticks(range(len(top_authors)), [safe_title(a, 25) for a in top_authors])
    savefig("13_author_category_heatmap.png")

def plot_citations_distribution(df, cit_col):
    pdf = (df.select(F.col(cit_col).cast("long").alias("c"))
             .filter(F.col("c").isNotNull() & (F.col("c") >= 0))
             .sample(withReplacement=False, fraction=0.2, seed=42)
             .select("c").toPandas())
    if len(pdf) == 0:
        return
    vals = pdf["c"].clip(lower=0)
    maxv = int(vals.quantile(0.995)) if len(vals) > 1000 else int(vals.max())
    maxv = max(maxv, 1)
    vals = vals[vals <= maxv]
    plt.figure()
    plt.hist(vals, bins=50)
    plt.title(f"Citation count distribution (sampled, clipped at p99.5={maxv})")
    plt.xlabel("Citations")
    plt.ylabel("Papers")
    savefig("14_citations_distribution.png")

def build_interesting_outputs(df):
    summary = []
    rows = df.count()
    summary.append({"key": "rows_total", "value": int(rows)})

    by_year = (df.withColumn("year", created_to_year_col())
                 .filter(F.col("year").isNotNull())
                 .groupBy("year").count()
                 .orderBy(F.desc("count")).limit(1).collect())
    if by_year:
        summary.append({"key": "year_with_most_papers", "value": {"year": int(by_year[0]["year"]), "papers": int(by_year[0]["count"])}})

    most_authors = (df.withColumn("num_authors", F.size("authors_parsed"))
                      .orderBy(F.desc("num_authors"))
                      .select("id", "title", "num_authors")
                      .limit(1).collect())
    if most_authors:
        r = most_authors[0]
        summary.append({"key": "paper_with_most_authors", "value": {"id": r["id"], "title": safe_title(r["title"], 200), "num_authors": int(r["num_authors"])}})

    most_cats = (df.withColumn("num_categories", F.size(F.split(F.col("categories"), " ")))
                   .orderBy(F.desc("num_categories"))
                   .select("id", "title", "categories", "num_categories")
                   .limit(1).collect())
    if most_cats:
        r = most_cats[0]
        summary.append({"key": "paper_with_most_categories", "value": {"id": r["id"], "title": safe_title(r["title"], 200), "categories": r["categories"], "num_categories": int(r["num_categories"])}})

    top_category = (df.withColumn("cat", F.explode(F.split(F.col("categories"), " ")))
                      .groupBy("cat").count().orderBy(F.desc("count"))
                      .limit(1).collect())
    if top_category:
        summary.append({"key": "most_common_category", "value": {"category": top_category[0]["cat"], "papers": int(top_category[0]["count"])}})

    authors = (df.withColumn("author", F.explode_outer("authors_parsed"))
                 .withColumn("full_name", author_name_from_parsed(F.col("author")))
                 .filter(F.col("full_name").isNotNull() & (F.col("full_name") != ""))
                 .select("id", "full_name"))

    top_author = (authors.groupBy("full_name").count()
                    .orderBy(F.desc("count")).limit(1).collect())
    if top_author:
        summary.append({"key": "most_prolific_author", "value": {"author": top_author[0]["full_name"], "papers": int(top_author[0]["count"])}})

    cit_col = get_citation_col(df)
    if cit_col:
        top_paper = (df.select("id", "title", F.col(cit_col).cast("long").alias("cit"))
                       .filter(F.col("cit").isNotNull())
                       .orderBy(F.desc("cit"))
                       .limit(1).collect())
        if top_paper:
            r = top_paper[0]
            summary.append({"key": "most_cited_paper", "value": {"id": r["id"], "title": safe_title(r["title"], 200), "citations": int(r["cit"])}})

        author_cit = (authors.join(df.select("id", F.col(cit_col).cast("long").alias("cit")), on="id", how="inner")
                            .filter(F.col("cit").isNotNull())
                            .groupBy("full_name")
                            .agg(F.sum("cit").alias("cit_sum"), F.countDistinct("id").alias("papers"))
                            .orderBy(F.desc("cit_sum"))
                            .limit(1).collect())
        if author_cit:
            r = author_cit[0]
            summary.append({"key": "most_cited_author_by_sum", "value": {"author": r["full_name"], "citations_sum": int(r["cit_sum"]), "papers": int(r["papers"])}})

        avg_cit_by_year = (df.withColumn("year", created_to_year_col())
                             .select(F.col("year"), F.col(cit_col).cast("long").alias("cit"))
                             .filter(F.col("year").isNotNull() & F.col("cit").isNotNull())
                             .groupBy("year").agg(F.mean("cit").alias("mean_cit"))
                             .orderBy("year").toPandas())
        if len(avg_cit_by_year) > 0:
            plt.figure()
            plt.plot(avg_cit_by_year["year"], avg_cit_by_year["mean_cit"])
            plt.title("Mean citations per paper over time")
            plt.xlabel("Year")
            plt.ylabel("Mean citations")
            savefig("15_mean_citations_over_time.png")

        top_cited_categories = (df.withColumn("cat", F.explode(F.split(F.col("categories"), " ")))
                                  .select("cat", F.col(cit_col).cast("long").alias("cit"))
                                  .filter(F.col("cit").isNotNull())
                                  .groupBy("cat")
                                  .agg(F.mean("cit").alias("mean_cit"), F.count("*").alias("papers"))
                                  .filter(F.col("papers") >= 500)
                                  .orderBy(F.desc("mean_cit"))
                                  .limit(10).toPandas())
        top_cited_categories.to_csv(os.path.join(OUT_DIR, "top_cited_categories_mean_citations.csv"), index=False)
        print("Saved:", os.path.join(OUT_DIR, "top_cited_categories_mean_citations.csv"))

    else:
        summary.append({"key": "citations_available", "value": False})

    return summary, cit_col

def export_top_lists(df, cit_col):
    authors = (df.withColumn("author", F.explode_outer("authors_parsed"))
                 .withColumn("full_name", author_name_from_parsed(F.col("author")))
                 .filter(F.col("full_name").isNotNull() & (F.col("full_name") != ""))
                 .select("id", "full_name"))

    top_auth = (authors.groupBy("full_name").count()
                  .orderBy(F.desc("count")).limit(50).toPandas())
    top_auth.to_csv(os.path.join(OUT_DIR, "top_50_authors_by_papers.csv"), index=False)
    print("Saved:", os.path.join(OUT_DIR, "top_50_authors_by_papers.csv"))

    top_cats = (df.withColumn("cat", F.explode(F.split(F.col("categories"), " ")))
                  .groupBy("cat").count().orderBy(F.desc("count")).limit(50).toPandas())
    top_cats.to_csv(os.path.join(OUT_DIR, "top_50_categories_by_papers.csv"), index=False)
    print("Saved:", os.path.join(OUT_DIR, "top_50_categories_by_papers.csv"))

    top_papers_by_categories = (df.withColumn("num_categories", F.size(F.split(F.col("categories"), " ")))
                                  .orderBy(F.desc("num_categories"))
                                  .select("id", "title", "categories", "num_categories")
                                  .limit(50).toPandas())
    top_papers_by_categories.to_csv(os.path.join(OUT_DIR, "top_50_papers_by_num_categories.csv"), index=False)
    print("Saved:", os.path.join(OUT_DIR, "top_50_papers_by_num_categories.csv"))

    top_papers_by_authors = (df.withColumn("num_authors", F.size("authors_parsed"))
                               .orderBy(F.desc("num_authors"))
                               .select("id", "title", "num_authors")
                               .limit(50).toPandas())
    top_papers_by_authors.to_csv(os.path.join(OUT_DIR, "top_50_papers_by_num_authors.csv"), index=False)
    print("Saved:", os.path.join(OUT_DIR, "top_50_papers_by_num_authors.csv"))

    if cit_col:
        top_cited_papers = (df.select("id", "title", F.col(cit_col).cast("long").alias("cit"))
                              .filter(F.col("cit").isNotNull())
                              .orderBy(F.desc("cit"))
                              .limit(50).toPandas())
        top_cited_papers.to_csv(os.path.join(OUT_DIR, "top_50_papers_by_citations.csv"), index=False)
        print("Saved:", os.path.join(OUT_DIR, "top_50_papers_by_citations.csv"))

        author_cit = (authors.join(df.select("id", F.col(cit_col).cast("long").alias("cit")), on="id", how="inner")
                            .filter(F.col("cit").isNotNull())
                            .groupBy("full_name")
                            .agg(F.sum("cit").alias("cit_sum"), F.mean("cit").alias("cit_mean"), F.countDistinct("id").alias("papers"))
                            .orderBy(F.desc("cit_sum"))
                            .limit(50).toPandas())
        author_cit.to_csv(os.path.join(OUT_DIR, "top_50_authors_by_citations_sum.csv"), index=False)
        print("Saved:", os.path.join(OUT_DIR, "top_50_authors_by_citations_sum.csv"))

def print_extremes(df, cit_col):
    top_cats = (df.withColumn("num_categories", F.size(F.split(F.col("categories"), " ")))
                  .orderBy(F.desc("num_categories"))
                  .select("id", "title", "categories", "num_categories")
                  .limit(10).toPandas())
    print("\nTop 10 papers by #categories:")
    for r in top_cats.itertuples(index=False):
        print(f"- {int(r.num_categories):2d} | {r.id} | {safe_title(r.title, 90)} | {r.categories}")

    top_auth = (df.withColumn("num_authors", F.size("authors_parsed"))
                  .orderBy(F.desc("num_authors"))
                  .select("id", "title", "num_authors")
                  .limit(10).toPandas())
    print("\nTop 10 papers by #authors:")
    for r in top_auth.itertuples(index=False):
        print(f"- {int(r.num_authors):4d} | {r.id} | {safe_title(r.title, 90)}")

    if cit_col:
        top_cited = (df.select("id", "title", F.col(cit_col).cast("long").alias("cit"))
                       .filter(F.col("cit").isNotNull())
                       .orderBy(F.desc("cit"))
                       .limit(10).toPandas())
        print("\nTop 10 papers by citations:")
        for r in top_cited.itertuples(index=False):
            print(f"- {int(r.cit):6d} | {r.id} | {safe_title(r.title, 90)}")

def main():
    spark = (SparkSession.builder
             .appName("arXiv EDA Plots")
             .config("spark.driver.memory", "8g")
             .config("spark.executor.memory", "8g")
             .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
             .config("spark.sql.shuffle.partitions", "200")
             .getOrCreate())

    print("Loading data from:", DATA_PATH)
    df = load_df(spark, DATA_PATH)

    cit_col = get_citation_col(df)
    print("Citation column:", cit_col if cit_col else "None")

    summary, cit_col = build_interesting_outputs(df)
    write_summary(summary, os.path.join(OUT_DIR, "summary.json"), os.path.join(OUT_DIR, "summary.csv"))
    export_top_lists(df, cit_col)

    print_extremes(df, cit_col)

    plot_papers_per_year(df)
    plot_authors_per_paper_distribution(df, cap=20)
    plot_mean_authors_per_year(df)
    plot_categories_per_paper_distribution(df, cap=10)
    plot_mean_categories_per_year(df)
    plot_top_categories(df, top_n=20)
    plot_top_authors(df, top_n=20)
    plot_author_productivity_over_time(df, top_k=5)
    plot_top_categories_over_time(df, top_k=5)
    plot_top_coauthor_pairs(df, top_n=15, max_authors_per_paper=50, sample_frac=0.25)
    plot_abstract_length_over_time(df)
    plot_title_length_vs_num_authors_scatter(df, sample_n=200000)
    plot_author_category_heatmap(df, top_authors_n=10, top_cats_n=12, sample_frac=0.35)

    if cit_col:
        plot_citations_distribution(df, cit_col)

    print("\nAll plots saved to:", PLOTS_DIR)
    print("All outputs saved to:", OUT_DIR)
    spark.stop()

if __name__ == "__main__":
    main()
