"""
generate_degraded_combo_rimes.py

Loads the RIMES train and val set from HuggingFace, applies multiple degradation methods
and uploads the results as parquet files to HF.

Each degraded set is saved under:
    rimes/combined_degradations/train/combo_train.parquet
    rimes/combined_degradations/train/combo_val.parquet
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
    apply_downscaling,
)

# Configuration

HF_REPO_ID = "LeMerta/bachelor-thesis-datasets"

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
target_path = f"rimes/combined_degradations/train/combo_train.parquet"

samples = []
for sample in raw_train:
    image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
    degraded = apply_downscaling(apply_gaussian_blur(apply_awgn(image, 100), 2.0), 30)
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

local_path = Path(f"/data/{os.environ['USER']}/parquets/combo_train.parquet")
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
target_path = f"rimes/combined_degradations/train/combo_val.parquet"

samples = []
for sample in raw_val:
    image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
    degraded = apply_downscaling(apply_gaussian_blur(apply_awgn(image, 100), 2.0), 30)
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

local_path = Path(f"/data/{os.environ['USER']}/parquets/combo_val.parquet")
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