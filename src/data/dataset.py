import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.data.transforms import get_transforms


CATEGORICAL_COLS = [
    "background_father",
    "background_mother",
    "gender",
    "region",
    "smoke",
    "drink",
    "pesticide",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
    "biopsed",
]

NUMERIC_COLS = ["age", "fitspatrick", "diameter_1", "diameter_2"]


def prepare_metadata(df: pd.DataFrame, categorical_cols=CATEGORICAL_COLS, numeric_cols=NUMERIC_COLS):
    """Prepare metadata with mask for missing values."""
    meta_df = df[categorical_cols + numeric_cols].copy()
    mask = meta_df.notna().astype(int)
    meta_df_filled = meta_df.fillna(0)

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols),
        ]
    )

    meta_encoded = preprocessor.fit_transform(meta_df_filled)

    cat_encoder = preprocessor.named_transformers_["cat"]
    num_scaler = preprocessor.named_transformers_["num"]

    mask_cat = mask[categorical_cols].astype(str)
    mask_num = mask[numeric_cols].astype(float)

    mask_cat_encoded = cat_encoder.transform(mask_cat)
    mask_num_encoded = num_scaler.transform(mask_num)

    mask_encoded = np.concatenate([mask_cat_encoded, mask_num_encoded], axis=1)
    meta_input = np.concatenate([meta_encoded, mask_encoded], axis=1)

    print(f"  → meta_encoded shape: {meta_encoded.shape}")
    print(f"  → mask_encoded shape: {mask_encoded.shape}")
    print(f"  → meta_input shape: {meta_input.shape}")

    return meta_input, mask_encoded, preprocessor


class SkinLesionDataset(Dataset):
    def __init__(self, indices, df, meta_input, mask, transform=None):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.meta_input = meta_input[indices]
        self.mask = mask[indices]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        meta = torch.FloatTensor(self.meta_input[idx])
        mask_t = torch.FloatTensor(self.mask[idx])
        label = torch.LongTensor([row["diagnostic_encoded"]])[0]

        return image, meta, mask_t, label


def load_and_prepare_data(config: Config):
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    if not os.path.exists(config.METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata CSV not found at '{config.METADATA_PATH}'. "
            f"Place your dataset files inside '{config.DATA_DIR}/' before running."
        )

    df = pd.read_csv(config.METADATA_PATH)

    for col in df.columns:
        if len(df[col]) > 0 and isinstance(df[col].iloc[0], bool):
            df[col] = df[col].map(lambda x: "True" if x else "False")

    float_cols = ["fitspatrick", "diameter_1", "diameter_2"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: int(x) if pd.notna(x) else pd.NA
            ).astype("Int64")

    le = LabelEncoder()
    df["diagnostic_encoded"] = le.fit_transform(df["diagnostic"])
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    print(f"✓ Classes: {list(label_mapping.values())}")

    df["image_path"] = df["img_id"].apply(lambda x: os.path.join(config.IMAGE_DIR, str(x)))

    print("\n✓ Converting categorical columns to strings...")
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype(str).fillna("nan")

    print("\n✓ Preparing metadata...")
    meta_input, mask, preprocessor = prepare_metadata(df, CATEGORICAL_COLS, NUMERIC_COLS)
    metadata_dim = meta_input.shape[1]
    print(f"✓ Metadata dimension: {metadata_dim}")

    valid_idx = df["image_path"].notna() & df["diagnostic_encoded"].notna()
    df = df[valid_idx].reset_index(drop=True)
    meta_input = meta_input[valid_idx]
    mask = mask[valid_idx]

    if os.path.exists(config.TEST_CSV_PATH):
        test_data = pd.read_csv(config.TEST_CSV_PATH)
        test_ids = set(test_data["img_id"])
        test_idx = df[df["img_id"].isin(test_ids)].index.values
        remaining_idx = df[~df["img_id"].isin(test_ids)].index.values

        train_idx, val_idx = train_test_split(
            remaining_idx,
            test_size=0.2,
            random_state=config.SEED,
            stratify=df.loc[remaining_idx, "diagnostic_encoded"],
        )
        print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    else:
        train_val_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=config.SEED,
            stratify=df["diagnostic_encoded"],
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.2,
            random_state=config.SEED,
            stratify=df.loc[train_val_idx, "diagnostic_encoded"],
        )
        print(
            f"⚠ External test CSV not found at '{config.TEST_CSV_PATH}'; "
            f"using stratified 64/16/20 split -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
        )

    train_transforms, val_transforms = get_transforms(config.IMG_SIZE)

    train_dataset = SkinLesionDataset(train_idx, df, meta_input, mask, train_transforms)
    val_dataset = SkinLesionDataset(val_idx, df, meta_input, mask, val_transforms)
    test_dataset = SkinLesionDataset(test_idx, df, meta_input, mask, val_transforms)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_mem,
    )

    print("✓ DataLoaders ready")

    return {
        "df": df,
        "meta_input": meta_input,
        "mask": mask,
        "metadata_dim": metadata_dim,
        "label_mapping": label_mapping,
        "preprocessor": preprocessor,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }

