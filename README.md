# CALF: Cross-Attentive Multimodal Latent Fusion

Modular PyTorch implementation for multimodal skin lesion classification combining tabular clinical metadata (via a task-aware VAE + Transformer encoder) and lesion images (via EfficientNet-B4) with bi-directional cross-attention and Conv1D-BiLSTM fusion.

## Folder Structure

```text
calf_skin_lesion/
├── data/               # Keep empty in repo; place dataset files here before running
├── outputs/            # Generated checkpoints (*.pth) and plots (*.png)
├── src/
│   ├── config.py       # Ablation switches, paths, hyperparameters, seed/device setup
│   ├── losses.py       # FocalLoss, SRCE_Loss, and criterion builder
│   ├── data/
│   │   ├── transforms.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── vae.py
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   └── calf.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md

```

## Setup

1. Create and activate a virtual environment, then install dependencies from the **project root**:
```bash
pip install -r requirements.txt

```


2. **Prepare the `data/` folder:**
The `data/` directory is intentionally kept empty. Before running, place your dataset files inside `data/`:
* `data/metadata.csv` (containing `img_id`, `diagnostic`, clinical categorical/numeric columns)
* `data/processed_images/` (folder containing lesion image files referenced by `img_id`)
* `data/pad-ufes-20_parsed_test.csv` *(optional)* — if present, rows whose `img_id` appears here are held out as the external test set, and the remaining rows are split 80/20 into train/validation. If omitted, a stratified 64/16/20 train/val/test split is used automatically.


If your filenames or directory layout differ, update `METADATA_PATH`, `IMAGE_DIR`, or `TEST_CSV_PATH` in `src/config.py`.

## Running the Code (From Project Root)

Always run commands from the **project root** directory (`calf_skin_lesion/`) using `-m` so package imports resolve properly:

### 1. Train (Pretrains VAE + Trains CALF + Evaluates on Test Set)

```bash
python -m src.train

```

### 2. Evaluate an Existing Checkpoint on the Test Set

```bash
python -m src.evaluate

```

## Ablation Switches

Edit the flags at the top of `src/config.py` before running:

* `USE_CROSS_ATT` (`True` / `False`): Use bi-directional cross-attention (`CrossAttentionMultiply`) vs. element-wise product `z_i * z_m`.
* `USE_SRCE_LOSS` (`True` / `False`): Use Self-Reweighted Cross Entropy (`SRCE_Loss`) vs. `FocalLoss` or `CrossEntropyLoss`.
* `USE_FOCAL_LOSS` (`True` / `False`): Used when `USE_SRCE_LOSS = False`.
* `USE_CONCAT_FUSION` (`True` / `False`): Concatenate `[z_i, z_m, z_c]` into an MLP classifier vs. stacking into `Conv1DLSTMFusion`.

