import os
import random
import numpy as np
import torch


# ===================== ABLATION SWITCHES =====================
USE_CROSS_ATT = True         # False -> use z_i * z_m
USE_SRCE_LOSS = True         # False -> use CE or Focal
USE_FOCAL_LOSS = False       # True -> use Focal instead of CE when SRCE is False
USE_CONCAT_FUSION = False    # True -> concatenate [z_i, z_m, z_c] instead of stack
# =============================================================


class Config:
    # Paths (place files inside data/ before running)
    DATA_DIR = "data"
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
    TEST_CSV_PATH = os.path.join(DATA_DIR, "pad-ufes-20_parsed_test.csv")
    IMAGE_DIR = os.path.join(DATA_DIR, "processed_images")
    OUTPUT_DIR = "outputs"

    # Model architecture
    IMG_SIZE = 224
    LATENT_DIM = 768          # Aligned latent dimension for both modalities
    VAE_LATENT = 128          # VAE latent dimension
    NUM_CLASSES = 6
    EFFICIENTNET_VERSION = "b4"

    # Training
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 200
    LR = 1e-4
    VAE_PRETRAIN_EPOCHS = 50
    VAE_VAL_SPLIT = 0.1
    VAE_BATCH_SIZE = 64
    VAE_LR = 1e-3
    NUM_WORKERS = 2

    # Loss weights
    VAE_KL_WEIGHT = 0.01
    VAE_TASK_WEIGHT = 0.02

    # Optimization & gradual unfreezing
    EARLY_STOPPING_PATIENCE = 50
    UNFREEZE_BLOCK7_EPOCH = 15
    UNFREEZE_BLOCK6_EPOCH = 30
    LR_SCHEDULE_TMAX = 20

    @classmethod
    def print_config(cls):
        loss_name = "SRCE" if USE_SRCE_LOSS else ("Focal" if USE_FOCAL_LOSS else "CE")
        fusion_name = "Concat" if USE_CONCAT_FUSION else "Stack"
        print(
            f"Configuration:\n"
            f"  Cross-Attn={USE_CROSS_ATT}\n"
            f"  Loss={loss_name}\n"
            f"  Fusion={fusion_name}"
        )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ Random seed set to {seed}")


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    return device

