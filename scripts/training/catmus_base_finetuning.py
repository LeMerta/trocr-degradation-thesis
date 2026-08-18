"""
catmus_base_finetuning.py

Finetunes the base TrOCR handwritten model on the normal CATMuS training set stored in Hugging Face.
At the end the checkpoint with the lowest validation CER is uploaded to Hugging Face
under LeMerta/finetuned-trocr-models in the CATMuS_models directory as CATMuS_base.
This is done because trocr-base-handwritten is not trained on CATMuS and doesnt perform well.
The final model is later used to further finetune with degraded CATMuS sets. 

"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from jiwer import cer
import shutil
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
from datasets import disable_progress_bar
disable_progress_bar()

# Configuration

MODEL_ID = "microsoft/trocr-base-handwritten"
TRAINING_ARGS = Seq2SeqTrainingArguments(
    output_dir=f"/data/{os.environ['USER']}/checkpoints/finetune_catmus",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
    fp16=False,
    save_total_limit=1,
    remove_unused_columns=False,
)

api = HfApi()

# Load processor

print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_ID)

# transform function

def transform(batch):
    images = [img.convert("RGB") for img in batch["im"]]
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


# Training
print("Training for CATMuS base model...")

# Load datasets
print("Loading datasets...")
catmus = load_dataset("CATMuS/medieval")
train_dataset = catmus["train"]
val_dataset = catmus["validation"]
print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

train_dataset = train_dataset.with_transform(transform)
val_dataset = val_dataset.with_transform(transform)

# Set up trainer

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=TRAINING_ARGS,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Baseline evaluation
baseline_metrics = trainer.evaluate()
print(baseline_metrics)

# Train
trainer.train()
print(
    f"Best val CER for CATMuS base model: {trainer.state.best_metric:.4f} "
)

# Save model locally
save_path = f"/data/{os.environ['USER']}/temp_model/catmus_base"
trainer.model.save_pretrained(save_path)

# Upload model to HF
api.upload_folder(
    folder_path=save_path,
    repo_id="LeMerta/finetuned-trocr-models",
    repo_type="model",
    path_in_repo=f"CATMuS_models/CATMuS_base",
)

shutil.rmtree(save_path)

# Clear checkpoints
shutil.rmtree(
    f"/data/{os.environ['USER']}/checkpoints/finetune_catmus", ignore_errors=True
)

# Clear HF datasets cache
shutil.rmtree(f"/data/{os.environ['USER']}/hf_cache/datasets", ignore_errors=True)
shutil.rmtree(f"/data/{os.environ['USER']}/hf_cache/hub", ignore_errors=True)

print("Finished Training for CATMuS base model.")