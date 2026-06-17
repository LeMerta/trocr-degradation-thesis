"""
iam_finetuning.py

Finetunes the base TrOCR handwritten model on degraded IAM training sets stored in Hugging Face.
At the end the checkpoint with the lowest validation CER is uploaded to Hugging Face
under LeMerta/finetuned-trocr-models in the iam_models directory.

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from jiwer import cer
import shutil
from huggingface_hub import HfApi

# Configuration

HF_REPO_ID = "LeMerta/bachelor-thesis-datasets"
MODEL_ID = "microsoft/trocr-base-handwritten"
TRAINING_ARGS = Seq2SeqTrainingArguments(
    output_dir=f"/data/{os.environ['USER']}/checkpoints/finetune_iam",
    num_train_epochs=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-06,
    logging_strategy="epoch",
    disable_tqdm=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
)

DEGRADATIONS = [
    ("awgn", 150),
    ("blur", 3.0),
    ("jpeg_compression", 1),
    ("downscale", 0.25),
]

api = HfApi()

# Load processor

print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_ID)


# Preprocessing function


def preprocess(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    pixel_values = processor(images, return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        batch["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    ).input_ids

    # Replace padding token id with -100 so it's ignored in loss
    labels = [
        [
            -100 if token == processor.tokenizer.pad_token_id else token
            for token in label
        ]
        for label in labels
    ]

    return {"pixel_values": pixel_values, "labels": labels}


# Function to compute CER while training


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100 in predictions and labels with pad token id
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    labels_ids = np.where(
        labels_ids != -100, labels_ids, processor.tokenizer.pad_token_id
    )

    pred_str = processor.batch_decode(
        pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    label_str = processor.batch_decode(
        labels_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Normalize
    pred_str = [p.strip() for p in pred_str]
    label_str = [l.strip() for l in label_str]

    cer_score = cer(label_str, pred_str)
    return {"cer": cer_score}


# Model init function


def model_init():
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.generation_config.max_length = 128
    model.generation_config.num_beams = 4

    return model


# Training for each degradation method

for method_name, intensity in DEGRADATIONS:
    print(f"Training for: method={method_name} | intensity={intensity} ")

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{HF_REPO_ID}/iam/{method_name}/train/{method_name}_{intensity}_train.parquet",
        split="train",
    )
    val_dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{HF_REPO_ID}/iam/{method_name}/train/{method_name}_{intensity}_val.parquet",
        split="train",
    )
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Preprocessing

    print("Preprocessing datasets...")
    train_processed = train_dataset.map(
        preprocess, batched=True, remove_columns=["image", "text"]
    )
    val_processed = val_dataset.map(
        preprocess, batched=True, remove_columns=["image", "text"]
    )
    train_processed.set_format("torch")
    val_processed.set_format("torch")

    # Set up trainer

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=TRAINING_ARGS,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        compute_metrics=compute_metrics,
    )

    # Baseline evaluation
    baseline_metrics = trainer.evaluate()
    print(baseline_metrics)

    # Train
    trainer.train()
    print(
        f"Best val CER for {method_name}_{intensity}: {trainer.state.best_metric:.4f} at epoch {trainer.state.best_model_checkpoint}"
    )

    # Save model locally
    save_path = f"/data/{os.environ['USER']}/temp_model/iam_{method_name}"
    trainer.model.save_pretrained(save_path)

    # Upload model to HF
    api.upload_folder(
        folder_path=save_path,
        repo_id="LeMerta/finetuned-trocr-models",
        repo_type="model",
        path_in_repo=f"iam_models/iam_{method_name}",
    )

    shutil.rmtree(save_path)

    # Clear checkpoints
    shutil.rmtree(
        f"/data/{os.environ['USER']}/checkpoints/finetune_iam", ignore_errors=True
    )

    # Clear HF datasets cache
    shutil.rmtree(f"/data/{os.environ['USER']}/hf_cache/hub", ignore_errors=True)

print("Finished Training for all degradation methods.")
