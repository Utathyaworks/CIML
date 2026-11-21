import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import Config


class MetadataVAE(nn.Module):
    """VAE with task-aware auxiliary loss for metadata encoding."""

    def __init__(self, input_dim, latent_dim=128, num_classes=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask=None):
        feature_dim = x.shape[1] // 2
        x_features = x[:, :feature_dim]

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        x_recon_features = x_recon[:, :feature_dim]

        if mask is not None:
            recon_loss = ((x_recon_features - x_features) ** 2 * mask).sum() / (
                mask.sum() + 1e-8
            )
        else:
            recon_loss = F.mse_loss(x_recon_features, x_features)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return z, mu, recon_loss, kl_loss


def pretrain_vae(
    meta_input,
    labels,
    mask,
    metadata_dim: int,
    config: Config,
    device: torch.device,
    epochs: int = None,
):
    """Pretrain MetadataVAE with train/validation split and save best weights."""
    if epochs is None:
        epochs = config.VAE_PRETRAIN_EPOCHS

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    vae_weights_path = os.path.join(config.OUTPUT_DIR, "vae_pretrained.pth")
    vae_plot_path = os.path.join(config.OUTPUT_DIR, "vae_training.png")

    print("\n" + "=" * 80)
    print("PRETRAINING VAE (with validation)")
    print("=" * 80)

    train_x, val_x, train_y, val_y, train_m, val_m = train_test_split(
        meta_input,
        labels,
        mask,
        test_size=config.VAE_VAL_SPLIT,
        random_state=config.SEED,
        stratify=labels,
    )
    print(f"✓ VAE Train: {len(train_x)}, VAE Val: {len(val_x)}")

    vae = MetadataVAE(metadata_dim, config.VAE_LATENT, config.NUM_CLASSES).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=config.VAE_LR)

    train_dataset = TensorDataset(
        torch.FloatTensor(train_x),
        torch.LongTensor(train_y),
        torch.FloatTensor(train_m),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_x),
        torch.LongTensor(val_y),
        torch.FloatTensor(val_m),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.VAE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.VAE_BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    vae_history = {"train_loss": [], "val_loss": []}

    for ep in range(epochs):
        vae.train()
        train_total_loss = 0.0

        for x, y, m in tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}"):
            x, y, m = x.to(device), y.to(device), m.to(device)

            _, mu, recon_loss, kl_loss = vae(x, m)
            logits = vae.classifier(mu)
            aux_loss = F.cross_entropy(logits, y)

            loss = (
                recon_loss
                + config.VAE_KL_WEIGHT * kl_loss
                + config.VAE_TASK_WEIGHT * aux_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            train_total_loss += loss.item()

        avg_train_loss = train_total_loss / len(train_loader)

        vae.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                _, mu, recon_loss, kl_loss = vae(x, m)
                logits = vae.classifier(mu)
                aux_loss = F.cross_entropy(logits, y)
                loss = (
                    recon_loss
                    + config.VAE_KL_WEIGHT * kl_loss
                    + config.VAE_TASK_WEIGHT * aux_loss
                )
                val_total_loss += loss.item()

        avg_val_loss = val_total_loss / len(val_loader)

        vae_history["train_loss"].append(avg_train_loss)
        vae_history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {ep + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
            f"Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vae.state_dict(), vae_weights_path)
            print(f"  ✓ Best VAE saved (Val Loss: {avg_val_loss:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(vae_history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(vae_history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Pretraining")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(vae_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ VAE training plot saved to {vae_plot_path}")
    print("✓ VAE pretraining complete\n")

    return vae, vae_weights_path

