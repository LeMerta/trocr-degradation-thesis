"""
generate_degraded_combo_test_rimes.py

Loads the RIMES test set from HuggingFace, applies multiple degradation methods combined
in different intensity levels, and uploads the results as parquet files to HF.

Each degraded set is saved under:
    rimes/combined_degradations/test/combo_<1,2,.. ,5>.parquet
"""

import os
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi, file_exists
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

DEGRADATIONS = [
    [100, 0.0, 30],
    [0, 2.0, 30],
    [50, 1.0, 30],
    [100, 2.0, 30],
    [150, 3.0, 30],
]

api = HfApi()

# Load raw test set
print("Loading Rimes test set...")
raw_dataset = load_dataset(HF_REPO_ID, data_dir="rimes/raw", split="test")
print(f"  {len(raw_dataset)} samples")

# Generate degraded versions

for i, intensities in enumerate(DEGRADATIONS):
    print(f"\n creating for intensities={intensities}")

    target_path = f"rimes/combined_degradations/test/combo_{i + 1}.parquet"
    if file_exists(repo_id=HF_REPO_ID, filename=target_path, repo_type="dataset"):
        print(f"  Already exists, skipping")
        continue

    samples = []
    for sample in raw_dataset:
        image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
        degraded = apply_downscaling(apply_gaussian_blur(apply_awgn(image, intensities[0]), intensities[1]), intensities[2])
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

    local_path = Path(f"/data/{os.environ['USER']}/parquets/combo_{i + 1}.parquet")
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

print("\nAll degraded test sets uploaded.")