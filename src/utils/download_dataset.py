import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset():
    os.makedirs("data/raw", exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset...")
    api.dataset_download_files(
        "akashshingha850/mrl-eye-dataset",
        path="data/raw",
        unzip=True
    )

    print("Dataset ready at data/raw/")


if __name__ == "__main__":
    download_kaggle_dataset()