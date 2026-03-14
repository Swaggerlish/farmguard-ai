# FarmGuard AI: Google Colab Readiness Review

## Verdict
The repository is **close to Colab-ready**, but not fully plug-and-play yet.

## What already works well for Colab
- Uses PyTorch + torchvision with a compact EfficientNet-B0 model (`ml/src/model.py`).
- Has a 2-stage transfer-learning flow (head training + fine-tune), which is practical for Colab GPU sessions (`ml/src/train.py`).
- Keeps per-class sampling capped (`MAX_IMAGES_PER_CLASS = 600`) to shorten training (`ml/src/prepare_dataset.py`).
- Uses standard Kaggle datasets and automatic download script (`ml/src/prepare_dataset.py`).

## Blocking / high-impact issues found
1. **Import path bug in dataloader module**
   - `ml/src/dataset.py` imported transforms as `from transforms import ...`.
   - In Colab (running from repo root), this often raises `ModuleNotFoundError`.
   - Fixed to `from src.transforms import ...`.

2. **No environment setup runbook**
   - Colab users need explicit steps for:
     - installing dependencies
     - adding `KAGGLE_USERNAME` and `KAGGLE_KEY`
     - preparing data
     - starting training
   - Added a concrete setup section below.

3. **`train_config.yaml` is currently not wired into `train.py`**
   - Config exists but training script still uses hardcoded values.
   - This is not a blocker, but it reduces reproducibility and tuning convenience in Colab.

## Colab quickstart (recommended)
Run this in a Colab notebook (GPU runtime):

```bash
!git clone <your-repo-url>
%cd farmguard-ai/ml
!pip install -r requirements.txt
```

Set Kaggle credentials (needed for PlantVillage Kaggle dataset download `abdallahalidev/plantvillage-dataset`):

```python
import os
os.environ["KAGGLE_USERNAME"] = "<your-kaggle-username>"
os.environ["KAGGLE_KEY"] = "<your-kaggle-api-key>"
```

Prepare dataset and train:

```bash
!python -m src.prepare_dataset
!python -m src.train --img-size 300 --label-smoothing 0.05
!python -m src.evaluate
```


## Cassava source (manual Kaggle competition upload)
- Primary workflow: manually download cassava competition data from Kaggle website and upload/extract to `data/raw`.
- The preparation script first tries to discover cassava assets already in `data/raw`.
- Supported layouts:
  - `train.csv` + `train_images`, or
  - class-folder formatted cassava images (`cbb`, `cbsd`, `cgm`, `cmd`, `healthy`).
- Zip files under `data/raw` are now auto-extracted before cassava asset discovery.

If your folders are verbose names (for example `Cassava CB (Cassava Blight)`), rename them to `cbb`, `cmd`, `healthy` for best compatibility.

Optional (advanced): set `CASSAVA_KAGGLEHUB_DATASET` only if you want non-competition cassava auto-download.


## Nice-to-have improvements for best Colab UX
- Pin versions in `requirements.txt` for reproducibility.
- Add checkpoint resume support (for interrupted Colab sessions).
- Wire `configs/train_config.yaml` into `src/train.py`.
- Save outputs to Google Drive mount (`/content/drive/MyDrive/...`).
- Keep `src/evaluate.py` and `src/infer.py` as the standard evaluation/inference entrypoints.

## Bottom line
- **Suitable for training on Colab after small setup steps.**
- With the import fix in this commit, the core train pipeline should run more reliably from a standard Colab working directory.



## Cross-source robustness tip
For web/in-the-wild images, train with a larger input size and mild label smoothing:

```bash
!python -m src.train --img-size 300 --label-smoothing 0.05
```
