"""
copy_iam_splits.py

Copies IAM line images into local train/val/test folders based on the
Aachen partition, which is the split used in the TrOCR paper (Li et al., 2021).

Requires the IAM Handwriting Database downloaded locally from:
    https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

Split files (tr.lst, va.lst, te.lst) are from:
    https://github.com/jpuigcerver/Laia/tree/master/egs/iam/data/part/lines/aachen

Expected split sizes (from TrOCR paper):
    train: 6161 lines
    val:   966 lines
    test:  2915 lines

Reference:
    Li et al. (2021). TrOCR: Transformer-based Optical Character Recognition
    with Pre-trained Models. https://arxiv.org/abs/2109.10282

"""

import shutil
from pathlib import Path

IAM_LINES = Path(r"path/to/iam/lines")  # update to local IAM path
SPLITS_DIR = Path("iam_splits")
TEMP_DIR = Path("temp_iam")

SPLITS = {
    "train": SPLITS_DIR / "tr.lst",
    "val": SPLITS_DIR / "va.lst",
    "test": SPLITS_DIR / "te.lst",
}


def load_ids(lst_path: Path) -> list[str]:
    """Read a .lst file and return list of line IDs e.g. ['c04-156-01', ...]"""
    with open(lst_path) as f:
        return [line.strip() for line in f if line.strip()]


def id_to_image_path(line_id: str) -> Path:
    """
    Convert a line ID like 'c04-156-01' to its image path.

    Assumes folder structure from IAM lines dataset, e.g. 'lines/c04/c04-156/c04-156-01.png'.
    """
    parts = line_id.split("-")
    prefix = parts[0]  # c04
    folder = f"{parts[0]}-{parts[1]}"  # c04-156
    return IAM_LINES / prefix / folder / f"{line_id}.png"


for split_name, lst_path in SPLITS.items():
    dest_dir = TEMP_DIR / split_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    line_ids = load_ids(lst_path)
    copied, missing = 0, 0

    for line_id in line_ids:
        src = id_to_image_path(line_id)
        if not src.exists():
            print(f"  Warning: not found: {src}")
            missing += 1
            continue
        shutil.copy(src, dest_dir / src.name)
        copied += 1

    print(f"{split_name}: {copied} copied, {missing} missing")

print("\nDone.")
