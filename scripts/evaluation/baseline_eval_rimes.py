"""
baseline_eval_rimes.py

Evaluates the pretrained TrOCR base handwritten model on the raw RIMES test set
and computes CER and WER.
Stores results in results/results.csv .
"""

import torch
import csv
import os
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
from jiwer import cer, wer

# Configuration

HF_REPO_ID = "LeMerta/bachelor-thesis-datasets"
MODEL_ID = "microsoft/trocr-base-handwritten"
BATCH_SIZE = 8
CSV_PATH = "results/results.csv"
FIELDNAMES = [
    "dataset",
    "method",
    "intensity",
    "model",
    "cer",
    "wer",
    "cer_lower",
    "wer_lower",
]

# Load dataset

print("Loading RIMES test set...")
dataset = load_dataset(HF_REPO_ID, data_dir="rimes/raw", split="test")
print(f"  {len(dataset)} samples")

# Load model

print(f"\nLoading model: {MODEL_ID}...")
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"  Running on: {device}")

# Run inference

print("\nRunning inference...")
predictions = []
references = []

for i in range(0, len(dataset), BATCH_SIZE):
    batch = dataset[i : i + BATCH_SIZE]

    images = [
        Image.open(io.BytesIO(img["bytes"])).convert("RGB") for img in batch["image"]
    ]
    labels = batch["text"]

    pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64, num_beams=10)

    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    predictions.extend(preds)
    references.extend(labels)

    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Processed {min(i + BATCH_SIZE, len(dataset))}/{len(dataset)} samples")

# Compute metrics

predictions = [p.strip() for p in predictions]
references = [r.strip() for r in references]

predictions_lowered = [p.lower() for p in predictions]
references_lowered = [r.lower() for r in references]

print("\nComputing metrics...")
cer_score = cer(references, predictions)
wer_score = wer(references, predictions)

cer_score_lowered = cer(references_lowered, predictions_lowered)
wer_score_lowered = wer(references_lowered, predictions_lowered)

print(f"\nResults:")
print(f"  CER: {cer_score * 100:.2f}%")
print(f"  WER: {wer_score * 100:.2f}%")
print(f"  CER (lowered): {cer_score_lowered * 100:.2f}%")
print(f"  WER (lowered): {wer_score_lowered * 100:.2f}%")

# Push results to csv file

write_header = not os.path.exists(CSV_PATH)

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
    writer.writerow(
        {
            "dataset": "rimes",
            "method": "none",
            "intensity": "none",
            "model": MODEL_ID,
            "cer": round(cer_score),
            "wer": round(wer_score),
            "cer_lower": round(cer_score_lowered),
            "wer_lower": round(wer_score_lowered),
        }
    )

print("Results saved to results/results.csv")
