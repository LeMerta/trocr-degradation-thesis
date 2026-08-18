"""
generate_degraded_iam_test.py

Loads the IAM test set from HuggingFace, applies each degradation method
at each intensity level, and uploads the results as parquet files to HF.

Each degraded set is saved under:
    iam/<method>/test/<method>_<intensity>.parquet
"""

from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi, file_exists
from datasets import load_dataset

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
    ("awgn", apply_awgn, [50, 100, 150, 200, 250]),
    ("blur", apply_gaussian_blur, [1.0, 2.0, 3.0, 4.0, 5.0]),
    ("jpeg_compression", apply_jpeg_compression, [10, 6, 3, 1]),
    ("downscale", apply_downscaling, [35, 30, 25, 20, 15]),
]

api = HfApi()

# Load raw test set

print("Loading IAM test set...")
raw_dataset = load_dataset(HF_REPO_ID, data_dir="iam/raw", split="test")
print(f"  {len(raw_dataset)} samples")

# Generate degraded versions

for method_name, method_fn, intensities in DEGRADATIONS:
    for intensity in intensities:
        print(f"\n{method_name} | intensity={intensity}")

        target_path = f"iam/{method_name}/test/{method_name}_{intensity}.parquet"
        if file_exists(repo_id=HF_REPO_ID, filename=target_path, repo_type="dataset"):
            print(f"  Already exists, skipping")
            continue

        samples = []
        for sample in raw_dataset:
            image = sample["image"].convert("RGB")
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

        local_path = Path(f"{method_name}_{intensity}.parquet")
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
