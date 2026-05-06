import os

if not os.path.exists("data/raw"):
    raise RuntimeError(
        "Dataset not found. Run: python src/utils/download_dataset.py"
    )