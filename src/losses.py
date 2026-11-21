import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import USE_FOCAL_LOSS, USE_SRCE_LOSS


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        p = torch.exp(logp)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1
        loss = (
            -alpha_t
            * (1 - p.gather(1, targets.unsqueeze(1))) ** self.gamma
            * logp.gather(1, targets.unsqueeze(1))
        )
        loss = loss.squeeze(1)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class SRCE_Loss(nn.Module):
    """
    Self-Reweighted Cross Entropy (SRCE)
    Dynamically reweights CE based on per-class loss + loss velocity.
    """

    def __init__(self, num_classes, momentum=0.9, alpha=2.0, beta=1.0, smoothing=0.1):
        super().__init__()
        self.C = num_classes
        self.momentum = momentum
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing

        self.register_buffer("ema_loss", torch.zeros(num_classes))
        self.register_buffer("ema_vel", torch.zeros(num_classes))

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.smoothing
        )

        device = logits.device
        if self.ema_loss.device != device:
            self.ema_loss = self.ema_loss.to(device)
            self.ema_vel = self.ema_vel.to(device)

        with torch.no_grad():
            batch_loss = torch.zeros(self.C, device=device)
            for c in range(self.C):
                mask = targets == c
                if mask.sum() > 0:
                    batch_loss[c] = ce[mask].mean()

            prev = self.ema_loss.clone()
            self.ema_loss = self.momentum * self.ema_loss + (1 - self.momentum) * batch_loss
            self.ema_vel = self.momentum * self.ema_vel + (1 - self.momentum) * (
                self.ema_loss - prev
            )

        weights = torch.softmax(self.alpha * self.ema_loss + self.beta * self.ema_vel, dim=0)
        weighted_ce = weights[targets] * ce
        return weighted_ce.mean()


def build_criterion(num_classes: int) -> nn.Module:
    if USE_SRCE_LOSS:
        return SRCE_Loss(num_classes=num_classes, smoothing=0.1)
    if USE_FOCAL_LOSS:
        return FocalLoss(gamma=2.0)
    return nn.CrossEntropyLoss(label_smoothing=0.1)

