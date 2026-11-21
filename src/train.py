import os
import warnings
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import Config, get_device, set_seed
from src.data.dataset import load_and_prepare_data
from src.evaluate import evaluate_model
from src.losses import build_criterion
from src.models.calf import CALF
from src.models.vae import pretrain_vae

warnings.filterwarnings("ignore")


def _rebuild_optimizer_with_unfrozen_blocks(model: CALF, config: Config):
    effnet_params = [
        p for _, p in model.image_encoder.features.named_parameters() if p.requires_grad
    ]
    other_params = [
        p for n, p in model.named_parameters() if "image_encoder.features" not in n
    ]
    optimizer = optim.Adam(
        [
            {"params": effnet_params, "lr": config.LR * 0.1},
            {"params": other_params, "lr": config.LR},
        ],
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.LR_SCHEDULE_TMAX, eta_min=1e-6
    )
    return optimizer, scheduler


def train(config: Config = Config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    Config.print_config()
    set_seed(config.SEED)
    device = get_device()

    data = load_and_prepare_data(config)
    train_idx = data["train_idx"]
    df = data["df"]
    meta_input = data["meta_input"]
    mask = data["mask"]
    metadata_dim = data["metadata_dim"]
    label_mapping = data["label_mapping"]
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    _, vae_weights_path = pretrain_vae(
        meta_input[train_idx],
        df.iloc[train_idx]["diagnostic_encoded"].values,
        mask[train_idx],
        metadata_dim=metadata_dim,
        config=config,
        device=device,
        epochs=config.VAE_PRETRAIN_EPOCHS,
    )

    print("\n" + "=" * 80)
    print("TRAINING CALF MODEL (OPTIMIZED)")
    print("=" * 80)

    model = CALF(
        metadata_dim=metadata_dim,
        vae_latent=config.VAE_LATENT,
        latent_dim=config.LATENT_DIM,
        num_classes=config.NUM_CLASSES,
        effnet_version=config.EFFICIENTNET_VERSION,
    ).to(device)

    model.vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
    print(f"✓ Loaded pretrained VAE from {vae_weights_path}")

    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.LR_SCHEDULE_TMAX, eta_min=1e-6
    )
    criterion = build_criterion(config.NUM_CLASSES)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 0
    best_model_path = os.path.join(config.OUTPUT_DIR, "best_calf_model.pth")
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    block6_unfrozen = False
    block7_unfrozen = False

    for epoch in range(config.EPOCHS):
        if epoch == config.UNFREEZE_BLOCK7_EPOCH and not block7_unfrozen:
            print("\n🔓 Unfreezing EfficientNet block 7")
            for name, param in model.image_encoder.features.named_parameters():
                if "7." in name:
                    param.requires_grad = True
            block7_unfrozen = True
            optimizer, scheduler = _rebuild_optimizer_with_unfrozen_blocks(model, config)

        if epoch == config.UNFREEZE_BLOCK6_EPOCH and not block6_unfrozen:
            print("\n🔓 Unfreezing EfficientNet block 6")
            for name, param in model.image_encoder.features.named_parameters():
                if "6." in name:
                    param.requires_grad = True
            block6_unfrozen = True
            optimizer, scheduler = _rebuild_optimizer_with_unfrozen_blocks(model, config)

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for x_img, x_meta, x_mask, y in pbar:
            x_img = x_img.to(device)
            x_meta = x_meta.to(device)
            x_mask = x_mask.to(device)
            y = y.to(device)

            logits = model(x_img, x_meta, x_mask)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
            )

        train_acc = 100.0 * correct / total

        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x_img, x_meta, x_mask, y in val_loader:
                x_img = x_img.to(device)
                x_meta = x_meta.to(device)
                x_mask = x_mask.to(device)

                logits = model(x_img, x_meta, x_mask)
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(y.numpy())

        val_acc = 100.0 * accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        scheduler.step()

        print(
            f"\nEpoch {epoch + 1}: Train Acc={train_acc:.2f}%, "
            f"Val Acc={val_acc:.2f}%, Val F1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "config": {
                        k: v
                        for k, v in config.__dict__.items()
                        if not k.startswith("__") and not callable(v)
                    },
                    "label_mapping": label_mapping,
                },
                best_model_path,
            )
            print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f})")
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if patience >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch + 1}")
            break

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(
        model=model,
        test_loader=test_loader,
        label_mapping=label_mapping,
        device=device,
        output_dir=config.OUTPUT_DIR,
    )


if __name__ == "__main__":
    train()

