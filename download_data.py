import os

os.environ["KAGGLE_USERNAME"] = "stefanagheorghita"
os.environ["KAGGLE_KEY"] = "KGAT_02d916c66bef489c3115c34ba1d6b3bb"

from kaggle.api.kaggle_api_extended import KaggleApi

def download_arxiv_metadata(download_dir="arxiv_data"):
    os.makedirs(download_dir, exist_ok=True)

    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset...")
    api.dataset_download_files(
        dataset="Cornell-University/arxiv",
        path=download_dir,
        unzip=True
    )

    print(f"Download complete. Files saved in: {download_dir}")


