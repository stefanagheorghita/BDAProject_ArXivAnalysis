import os
import pandas as pd


BASE_DIRS = {
    "tfidf": "tfidf_query_results",
    "bm25": "bm25_results",
    "bm25_specter": "bm25_specter_results",
    "minilm": "minilm_query_results",
    "faiss_specter": "faiss_specter_results",
}

TOP_K = 20
OUT_FILE = "category_diversity_at_20.csv"

def parse_subcats(cat_str):
    if pd.isna(cat_str):
        return []
    return cat_str.split()

def primary_subcat(cat_str):
    subs = parse_subcats(cat_str)
    return subs[0] if subs else None

def primary_cat(subcat):
    return subcat.split(".")[0]


rows = []

for method, folder in BASE_DIRS.items():
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(folder, fname)
        df = pd.read_csv(path)

        if "categories" not in df.columns:
            continue

        topk = df.head(TOP_K)

        all_subcats = set()
        all_primaries = set()
        primary_subcats = set()
        primary_primaries = set()

        for c in topk["categories"]:
            subs = parse_subcats(c)

            for sub in subs:
                all_subcats.add(sub)
                all_primaries.add(primary_cat(sub))

            if subs:
                ps = subs[0]
                primary_subcats.add(ps)
                primary_primaries.add(primary_cat(ps))

        rows.append({
            "query": fname,
            "method": method,
            "unique_primary_categories_all@20": len(all_primaries),
            "unique_subcategories_all@20": len(all_subcats),
            "unique_primary_categories_primaryOnly@20": len(primary_primaries),
            "unique_subcategories_primaryOnly@20": len(primary_subcats),
        })

out = pd.DataFrame(rows)

out = out.sort_values(by=["query", "method"]).reset_index(drop=True)

out.to_csv(OUT_FILE, index=False)
print(f"Saved → {OUT_FILE}")
