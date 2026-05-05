"""
create_iam_parquet.py

Packages the IAM line images from local train/val/test folders into
HuggingFace parquet dataset files and uploads them to HF.

Images are stored as lossless PNG bytes — no degradation is applied.
Ground truth labels are read from lines.txt and paired with each image.

Requires:
    - temp_iam/ folder created by copy_iam_splits.py
    - IAM lines.txt downloaded locally from:
        https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

"""

from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi

# Configuration

HF_REPO_ID = "LeMerta/bachelor-thesis-datasets"
LINES_TXT = Path(r"path/to/iam/lines.txt")  # update to local IAM path
TEMP_DIR = Path("temp_iam")
SPLITS = ["train", "val", "test"]

api = HfApi()

# Load ground truth labels from lines.txt


def load_labels(lines_txt: Path) -> dict[str, str]:
    """
    Parse lines.txt and return a dict mapping image ID to transcription.

    Word seperators "|" are replaced by " ".

    Only includes lines marked as 'ok' — err lines are excluded.
    """
    labels = {}
    with open(lines_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(" ")
            img_id = parts[0]
            # transcript is the ninth field, with | used as word separator, " " may appear
            transcript = " ".join(parts[8:]).replace("|", " ")
            labels[img_id] = transcript
    return labels


# Build and upload dataset

print("Loading labels from lines.txt...")
labels = load_labels(LINES_TXT)
print(f"  Loaded {len(labels)} labels")

for split_name in SPLITS:
    split_dir = TEMP_DIR / split_name
    images = list(split_dir.glob("*.png"))
    print(f"\nBuilding {split_name} split ({len(images)} images)...")

    samples = []
    skipped = 0
    for img_path in images:
        img_id = img_path.stem
        if img_id not in labels:
            skipped += 1
            continue
        samples.append(
            {
                "image": open(img_path, "rb").read(),
                "text": labels[img_id],
            }
        )

    if skipped:
        print(f"  Skipped {skipped} images with no label")

    dataset = Dataset.from_list(
        samples,
        features=Features(
            {
                "image": HFImage(),
                "text": Value("string"),
            }
        ),
    )
    print(f"  Built {len(samples)} samples")

    local_path = Path(f"{split_name}.parquet")
    dataset.to_parquet(local_path)

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=f"iam/raw/{split_name}.parquet",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    local_path.unlink()
    print(f"  Uploaded {split_name}.parquet")

print("Done.")
