# TrOCR Degradation Thesis

Bachelor thesis investigating TrOCR performance under image degradation.

## Project Structure


- `notebooks/` 
  - `results/` — displays results from CSV file in results folder
  - `tests/` — tests
- `scripts/`
  - `data_preparation/` — one-time scripts for downloading and uploading degraded datasets to HuggingFace
  - `evaluation/` — scripts for calculating metrics
  - `training/` — finetuning scripts
- `results/` — CSV file containing calculated metrics
- `src/` — core modules (degradation functions)
- `iam_splits/` — split files for IAM dataset

## Setup
`pip install -r requirements.txt`

## Datasets
Raw and degraded datasets are stored on HuggingFace:
[LeMerta/bachelor-thesis-datasets](https://huggingface.co/datasets/LeMerta/bachelor-thesis-datasets)

Original IAM dataset was downloaded from:
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

Original RIMES dataset was downloaded from:
https://huggingface.co/datasets/Teklia/RIMES-2011-line

## Data Preparation
Split files for IAM use the Aachen partition from the TrOCR paper:
https://github.com/jpuigcerver/Laia/tree/master/egs/iam/data/part/lines/aachen

## Models
Base TrOCR model used from Microsoft's HuggingFace repository:
[microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten)

Finetuned Models stored under:
[LeMerta/finetuned-trocr-models](https://huggingface.co/LeMerta/finetuned-trocr-models)