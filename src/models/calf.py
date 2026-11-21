import torch
import torch.nn as nn

from src.config import USE_CONCAT_FUSION, USE_CROSS_ATT
from src.models.encoders import ImageEncoder, MetadataBERTEncoder
from src.models.fusion import Conv1DLSTMFusion, CrossAttentionMultiply
from src.models.vae import MetadataVAE


class CALF(nn.Module):
    """Cross-Attentive Multimodal Latent Fusion."""

    def __init__(
        self,
        metadata_dim,
        vae_latent=128,
        latent_dim=768,
        num_classes=6,
        effnet_version="b4",
    ):
        super().__init__()
        self.vae = MetadataVAE(metadata_dim, vae_latent, num_classes)
        self.bert_encoder = MetadataBERTEncoder(vae_latent, latent_dim)
        self.image_encoder = ImageEncoder(latent_dim, version=effnet_version)
        self.cross_att = CrossAttentionMultiply(
            dim=latent_dim, num_heads=8, attn_dropout=0.1
        )
        self.fusion = Conv1DLSTMFusion(latent_dim, 256, num_classes)
        self.concat_classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_img, x_meta, x_mask):
        _, mu, _, _ = self.vae(x_meta, x_mask)
        z_m = self.bert_encoder(mu)
        z_i = self.image_encoder(x_img)

        if USE_CROSS_ATT:
            z_c = self.cross_att(z_i, z_m)
        else:
            z_c = z_i * z_m

        if USE_CONCAT_FUSION:
            z_stack = torch.cat([z_i, z_m, z_c], dim=1)
            logits = self.concat_classifier(z_stack)
        else:
            z_stack = torch.stack([z_i, z_m, z_c], dim=1)
            logits = self.fusion(z_stack)

        return logits

