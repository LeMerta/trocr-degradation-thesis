"""
iam_degradation_ft_eval.py

Evaluates the finetuned TrOCR models on all IAM degraded test sets and saves
CER and WER (lowered and not lowered) results to the shared results CSV.

Results are appended to "results/results.csv" — evaluate_baseline.py needs to be run first to
ensure the CSV exists before running this script.

"""

import csv
import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
import os

# Configuration

HF_REPO_ID = "LeMerta/bachelor-thesis-datasets"
BATCH_SIZE = 16
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

DEGRADATIONS = [
    ("awgn", [50, 100, 150, 200, 250]),
    ("blur", [1.0, 2.0, 3.0, 4.0, 5.0]),
    ("jpeg_compression", [10, 6, 3, 1]),
    ("downscale", [35, 30, 25, 20, 15]),
]

MODEL_IDS = [
    "iam_awgn",
    "iam_blur",
    "iam_jpeg_compression",
    "iam_downscale",
]

# Check if CSV exists

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Results CSV not found at {CSV_PATH} — run evaluate_baseline.py first"
    )

# Help functions

def already_evaluated(method: str, intensity, Model_ID: str) -> bool:
    """Check if this method+intensity+model combo already exists in the results CSV"""
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        return any(
            row["method"] == method and row["intensity"] == str(intensity) and row["model"] == Model_ID
            for row in reader
        )


def run_inference(dataset, model, processor) -> tuple[list[str], list[str]]:
    """Run inference on a dataset and return predictions and references."""
    predictions = []
    references = []

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        images = [img.convert("RGB") for img in batch["image"]]
        labels = batch["text"]

        pixel_values = processor(images, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=10,
            )

        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(labels)

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    {min(i + BATCH_SIZE, len(dataset))}/{len(dataset)} samples")

    return predictions, references


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute CER and WER with and without lowercasing."""
    predictions_stripped = [p.strip() for p in predictions]
    references_stripped = [r.strip() for r in references]

    predictions_lowered = [p.lower() for p in predictions_stripped]
    references_lowered = [r.lower() for r in references_stripped]

    return {
        "cer": cer(references_stripped, predictions_stripped),
        "wer": wer(references_stripped, predictions_stripped),
        "cer_lower": cer(references_lowered, predictions_lowered),
        "wer_lower": wer(references_lowered, predictions_lowered),
    }

def eval_and_write(method_name, intensity, Model_ID, model, processor, writer, f):
    """Evaluate a single method+intensity+model combo and write results to CSV if not already done."""
    print(f"\nMethod={method_name} | intensity={intensity}")
    if not already_evaluated(method_name, intensity, Model_ID):
        if method_name == "none" and intensity=="none":
            path = f"iam/raw/test.parquet"
            dataset = load_dataset(
                "parquet",
                data_files=f"hf://datasets/{HF_REPO_ID}/{path}",
                split="train",
            )
            predictions, references = run_inference(dataset, model, processor)
        else:
            path = f"iam/{method_name}/test/{method_name}_{intensity}.parquet"
            dataset = load_dataset(
                "parquet",
                data_files=f"hf://datasets/{HF_REPO_ID}/{path}",
                split="train",
            )
            predictions, references = run_inference(dataset, model, processor)
        
        metrics = compute_metrics(predictions, references)

        writer.writerow(
            {
                "dataset": "iam",
                "method": method_name,
                "intensity": intensity,
                "model": Model_ID,
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "cer_lower": metrics["cer_lower"],
                "wer_lower": metrics["wer_lower"],
            }
        )
        f.flush()

        print(f"  CER: {metrics['cer']*100:.2f}% | WER: {metrics['wer']*100:.2f}%")
        print(
            f"  CER (lower): {metrics['cer_lower']*100:.2f}% | WER (lower): {metrics['wer_lower']*100:.2f}%"
        )

        dataset.cleanup_cache_files()
    else:
        print(f" Already done, skipping")

# Main evaluation loop

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

    for Model_ID in MODEL_IDS:
        # Load finetuned model
        print(f"Loading model: {Model_ID}...")
        model = VisionEncoderDecoderModel.from_pretrained(
            "LeMerta/finetuned-trocr-models",
            subfolder=f"iam_models/{Model_ID}",
        )
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        print(f"  Running on: {device}")

        eval_and_write(
            method_name="none", 
            intensity="none", 
            Model_ID=Model_ID, 
            model=model, 
            processor=processor, 
            writer=writer,
            f=f
        )     

        for method_name, intensities in DEGRADATIONS:
            for intensity in intensities:
                eval_and_write(
                    method_name=method_name, 
                    intensity=intensity, 
                    Model_ID=Model_ID, 
                    model=model, 
                    processor=processor, 
                    writer=writer,
                    f=f
                )
            

print("\nAll evaluations complete.")