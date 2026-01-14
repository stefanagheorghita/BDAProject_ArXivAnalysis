import re

import numpy as np
import pandas as pd


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_arxiv_sample(
    path: str,
    chunk_size: int = 100_000,
    n_chunks: int = 5,
    random_state: int = 42
):
    rng = np.random.default_rng(random_state)

    chunks = pd.read_json(
        path,
        lines=True,
        chunksize=chunk_size
    )

    all_chunks = list(chunks)

    selected = rng.choice(
        len(all_chunks),
        size=n_chunks,
        replace=False
    )

    df = pd.concat(
        [all_chunks[i] for i in selected],
        ignore_index=True
    )

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df



# for n_chunks in [1, 5, 10, 20]:
#     df = load_arxiv_sample(
#         "../arxiv_data/arxiv-metadata-oai-snapshot.json",
#         chunk_size=100_000,
#         n_chunks=n_chunks,
#         random_state=42)
#     df.to_csv(f"samples/arxiv_sample_{100000*n_chunks}.csv", index=False)
#     print("1 done")

# import pandas as pd

# df = pd.read_json(
#     "../exploratory_analysis/arxiv_data/arxiv-metadata-oai-snapshot.json",
#     lines=True
# )

# df.to_csv("arxiv_full.csv", index=False)


def build_text_column( df: pd.DataFrame, title_col: str = "title", abstract_col: str = "abstract", output_col: str = "text"
) :
    df = df.copy()
    df[output_col] = (
        df[title_col].astype(str) + ". " +
        df[abstract_col].astype(str)
    ).map(clean_text)
    return df