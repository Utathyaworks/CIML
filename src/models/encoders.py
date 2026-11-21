import torch
import torch.nn as nn
import torchvision.models as models


class MetadataBERTEncoder(nn.Module):
    """Transformer/BERT-inspired encoder for contextual metadata embeddings."""

    def __init__(self, vae_latent=128, output_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(vae_latent, output_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, z_vae):
        x = self.embedding(z_vae).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.norm(x)


class ImageEncoder(nn.Module):
    """EfficientNet-based image encoder."""

    def __init__(self, output_dim=768, version="b4"):
        super().__init__()
        if version == "b4":
            effnet = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
            )
            in_features = 1792
        else:
            raise ValueError(f"Unsupported version: {version}")

        self.features = effnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Freeze backbone initially
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)

