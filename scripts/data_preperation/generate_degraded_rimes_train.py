"""
generate_degraded_rimes_train.py

Loads the RIMES train and val set from HuggingFace, applies each degradation method
with one intensity level, and uploads the results as parquet files to HF.

Each degraded set is saved under:
    rimes/<method>/train/<method>_<intensity>_train.parquet
    rimes/<method>/train/<method>_<intensity>_val.parquet
"""

import os
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi, file_exists
from datasets import load_dataset
from PIL import Image
import io
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import sys

sys.path.append("src")
from degradation import (
    apply_awgn,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_downscaling,
)

# Configuration

HF_REPO_ID = "LeMerta/bachelor-thesis-datasets"

DEGRADATIONS = [
    ("awgn", apply_awgn, 150),
    ("blur", apply_gaussian_blur, 3.0),
    ("jpeg_compression", apply_jpeg_compression, 1),
    ("downscale", apply_downscaling, 0.25),
]

api = HfApi()

# Load raw train and val set

print("Loading RIMES train and val sets...")
raw_train = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{HF_REPO_ID}/rimes/raw/train.parquet",
    split="train",
)
raw_val = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{HF_REPO_ID}/rimes/raw/validation.parquet",
    split="train",
)
print(f"  {len(raw_train)} samples in train set")
print(f"  {len(raw_val)} samples in val set")

# Generate degraded versions

for method_name, method_fn, intensity in DEGRADATIONS:
    print(f"\n{method_name} | intensity={intensity}")

    # Degrade train set
    target_path = f"rimes/{method_name}/train/{method_name}_{intensity}_train.parquet"
    if file_exists(repo_id=HF_REPO_ID, filename=target_path, repo_type="dataset"):
        print(f"  Train set already exists, skipping")
        continue
    else:
        print(f"  Degrading train set...")

    samples = []
    for sample in raw_train:
        image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
        degraded = method_fn(image, intensity)
        samples.append(
            {
                "image": degraded,
                "text": sample["text"],
            }
        )

    print(f"  Building parquet...")
    dataset = Dataset.from_list(
        samples,
        features=Features(
            {
                "image": HFImage(),
                "text": Value("string"),
            }
        ),
    )

    local_path = Path(f"/data/{os.environ['USER']}/temp_parquet/{method_name}_{intensity}_train.parquet")
    local_path.parent.mkdir(exist_ok=True)
    dataset.to_parquet(local_path)

    print(f"  Uploading to HF...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=target_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    local_path.unlink()
    print(f"  Done")

    # Degrade val set
    target_path = f"rimes/{method_name}/train/{method_name}_{intensity}_val.parquet"
    if file_exists(repo_id=HF_REPO_ID, filename=target_path, repo_type="dataset"):
        print(f"  Val set already exists, skipping")
        continue
    else:
        print(f"  Degrading val set...")

    samples = []
    for sample in raw_val:
        image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
        degraded = method_fn(image, intensity)
        samples.append(
            {
                "image": degraded,
                "text": sample["text"],
            }
        )

    print(f"  Building parquet...")
    dataset = Dataset.from_list(
        samples,
        features=Features(
            {
                "image": HFImage(),
                "text": Value("string"),
            }
        ),
    )

    local_path = Path(f"/data/{os.environ['USER']}/temp_parquet/{method_name}_{intensity}_val.parquet")
    local_path.parent.mkdir(exist_ok=True)
    dataset.to_parquet(local_path)

    print(f"  Uploading to HF...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=target_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    local_path.unlink()
    print(f"  Done")

print("\nAll degraded sets uploaded.")