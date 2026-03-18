# FarmGuard AI - Detect crop diseases instantly using artificial intelligence

This document describes the trained crop-disease classifier model, how to reproduce training/evaluation, and how to publish model artifacts to Hugging Face.

## Model Summary
- **Task**: Multi-class image classification of crop leaf diseases.
- **Architecture**: `efficientnet_b0` with a replaced classifier head.
- **Framework**: PyTorch + torchvision.
- **Classes**: 18 total classes across cassava, maize, pepper, and tomato labels.

## Training Pipeline
Training is implemented in `src/train.py`:
- Stage 1: train classifier head with frozen backbone.
- Stage 2: unfreeze top feature blocks and fine-tune.
- Best model selected by validation macro-F1 and saved as checkpoint.

Default output artifacts:
- `outputs/checkpoints/best_model.pth`
- `outputs/checkpoints/class_to_idx.json`

## Dataset Choice and Access Rationale
- **PlantVillage (Kaggle dataset: `abdallahalidev/plantvillage-dataset`)** is used for tomato/maize/pepper/corn classes.
- **Cassava Disease (Kaggle competition data, manually downloaded)** is used for cassava classes.

### Why cassava was downloaded manually
Automated cassava download in notebook/runtime can fail when competition access is restricted by authentication/rules acceptance in the execution environment. To keep training reproducible, cassava files were downloaded manually from Kaggle and then uploaded to the runtime before preparation.

### Expected manual cassava layout
After upload/extraction, cassava data should be placed under `data/raw` in one of these supported formats:
- `train.csv` + `train_images/`
- class folders (`cbb`, `cbsd`, `cgm`, `cmd`, `healthy`) or equivalent verbose names handled by alias mapping in `prepare_dataset.py`.

## Dataset Preparation
Dataset preparation is implemented in `src/prepare_dataset.py`:
- PlantVillage source via Kaggle dataset download (`abdallahalidev/plantvillage-dataset`).
- Cassava source is **manually downloaded Kaggle competition data** uploaded into `data/raw` (primary workflow).
- For custom folder datasets, preferred class folder names are: `cbb`, `cbsd`, `cgm`, `cmd`, `healthy`.
- Optional: set `CASSAVA_KAGGLEHUB_DATASET` only if you want non-competition cassava auto-download.
- Robust folder-name normalization/mapping for variant class names.
- Processed split output to:
  - `data/processed/train`
  - `data/processed/val`
  - `data/processed/test`

## Evaluate the Model
Evaluation CLI is implemented in `src/evaluate.py`.

Run:

```bash
python -m src.evaluate \
  --checkpoint outputs/checkpoints/best_model.pth \
  --class-map outputs/checkpoints/class_to_idx.json
```

Generated evaluation artifacts:
- `outputs/checkpoints/test_classification_report.txt`
- `outputs/checkpoints/test_confusion_matrix.json`

## Run Inference on One Image
Inference CLI is implemented in `src/infer.py`.

```bash
python -m src.infer \
  --image /path/to/leaf.jpg \
  --checkpoint outputs/checkpoints/best_model.pth \
  --class-map outputs/checkpoints/class_to_idx.json \
  --top-k 3
```

## End-to-End Colab Commands
From `ml/` directory:

```bash
pip install -r requirements.txt
# optional ONLY if you want non-competition cassava auto-download
# export CASSAVA_KAGGLEHUB_DATASET="<owner/dataset-slug>"
python -m src.prepare_dataset
python -m src.train
python -m src.evaluate
```

## Publish to Hugging Face
Required package is already listed in `requirements.txt` (`huggingface_hub`).

Example upload script:

```python
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = "your-hf-username/farmguard-ai-crop-disease-model"
token = os.environ["HF_TOKEN"]

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

for artifact in [
    "outputs/checkpoints/best_model.pth",
    "outputs/checkpoints/class_to_idx.json",
    "outputs/checkpoints/test_classification_report.txt",
    "outputs/checkpoints/test_confusion_matrix.json",
]:
    p = Path(artifact)
    if p.exists():
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
```

## Notes
- `train_config.yaml` exists, but current training code uses hardcoded defaults in `src/train.py`.
- For reproducibility, consider pinning dependency versions in `requirements.txt`.
