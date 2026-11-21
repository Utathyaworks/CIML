import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import torch

from src.config import Config, get_device, set_seed
from src.data.dataset import load_and_prepare_data
from src.models.calf import CALF


def evaluate_model(
    model,
    test_loader,
    label_mapping,
    device,
    output_dir="outputs",
):
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (ON TEST SET)")
    print("=" * 80)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x_img, x_meta, x_mask, y in test_loader:
            x_img = x_img.to(device)
            x_meta = x_meta.to(device)
            x_mask = x_mask.to(device)

            logits = model(x_img, x_meta, x_mask)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("=" * 80)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    test_acc = 100.0 * accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average="macro")
    try:
        test_roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        test_roc_auc = float("nan")

    print(f"✓ Test Accuracy: {test_acc:.2f}%")
    print(f"✓ Test F1 (macro): {test_f1_macro:.4f}")
    print(f"✓ Test ROC-AUC: {test_roc_auc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CALF Test Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_calf_test.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix saved as {cm_path}")

    return {
        "accuracy": test_acc,
        "f1_macro": test_f1_macro,
        "roc_auc": test_roc_auc,
    }


def evaluate_from_checkpoint(checkpoint_path: str = None, config: Config = Config):
    set_seed(config.SEED)
    device = get_device()
    data = load_and_prepare_data(config)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.OUTPUT_DIR, "best_calf_model.pth")

    model = CALF(
        metadata_dim=data["metadata_dim"],
        vae_latent=config.VAE_LATENT,
        latent_dim=config.LATENT_DIM,
        num_classes=config.NUM_CLASSES,
        effnet_version=config.EFFICIENTNET_VERSION,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded checkpoint from {checkpoint_path}")

    return evaluate_model(
        model=model,
        test_loader=data["test_loader"],
        label_mapping=data["label_mapping"],
        device=device,
        output_dir=config.OUTPUT_DIR,
    )


if __name__ == "__main__":
    evaluate_from_checkpoint()

