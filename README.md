# TrOCR Degradation Thesis

Bachelor thesis investigating TrOCR performance under image degradation.

## Project Structure
- `scripts/` — one-time data preparation scripts

## Setup
`pip install -r requirements.txt`

## Datasets
Raw and degraded datasets are stored on HuggingFace:
[LeMerta/bachelor-thesis-datasets](https://huggingface.co/datasets/LeMerta/bachelor-thesis-datasets)

Original IAM dataset was downloaded from:
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

## Data Preparation
Split files for IAM use the Aachen partition from the TrOCR paper:
https://github.com/jpuigcerver/Laia/tree/master/egs/iam/data/part/lines/aachen