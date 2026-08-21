"""ArcFace additive angular margin loss head.

Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
Recognition" (2019), https://arxiv.org/abs/1801.07698

This module is a *training-time classification head* on top of the face
embedding. It is not part of the inference-time embedding model: at
inference, only the backbone (e.g. MobileFaceNet) is used, and identities
are compared via cosine similarity between embeddings.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ArcMarginHead(nn.Module):
    """Computes ArcFace logits from L2-normalized embeddings and class labels.

    Args:
        embedding_dim: dimensionality of the input face embeddings.
        num_classes: number of identities in the training set.
        margin: additive angular margin `m` (radians), typically 0.5.
        scale: logit scale `s`, typically 64.
    """

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold beyond which cos(theta + m) would increase instead of decrease
        # (i.e. theta + m > pi); used for the numerically-stable "easy margin" fallback.
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, normalized_weight)  # (B, num_classes), already in [-1, 1]

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # Where theta + m would exceed pi, fall back to a linear penalty so the
        # loss stays monotonically decreasing (Deng et al., Sec. 3.3).
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    embed_dim, num_classes, batch = 256, 1000, 8
    head = ArcMarginHead(embed_dim, num_classes)
    embeddings = F.normalize(torch.randn(batch, embed_dim), p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch,))
    logits = head(embeddings, labels)
    loss = F.cross_entropy(logits, labels)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())
