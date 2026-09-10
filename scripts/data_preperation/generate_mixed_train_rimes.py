"""
generate_mixed_train_rimes.py

Loads the RIMES train and val set from HuggingFace, applies multiple degradation methods
one at a time and uploads the results as parquet files to HF.

Each degraded set is saved under:
    rimes/combined_degradations/train/mixed_train.parquet
    rimes/combined_degradations/train/mixed_val.parquet
"""

import os
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi
from datasets import load_dataset
from PIL import Image
import io

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
    ( apply_awgn, 150),
    ( apply_gaussian_blur, 3.0),
    ( apply_jpeg_compression, 1),
    ( apply_downscaling, 25),
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


# Degrade train set
target_path = f"rimes/combined_degradations/train/mixed_train.parquet"

samples = []
for i, sample in enumerate(raw_train):
    image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")

    degradation_fn, parameter = DEGRADATIONS[i % len(DEGRADATIONS)]
    degraded = degradation_fn(image, parameter)

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

local_path = Path(f"/data/{os.environ['USER']}/parquets/mixed_train.parquet")
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
target_path = f"rimes/combined_degradations/train/mixed_val.parquet"

samples = []
for i, sample in enumerate(raw_val):
    image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")

    degradation_fn, parameter = DEGRADATIONS[i % len(DEGRADATIONS)]
    degraded = degradation_fn(image, parameter)

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

local_path = Path(f"/data/{os.environ['USER']}/parquets/mixed_val.parquet")
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