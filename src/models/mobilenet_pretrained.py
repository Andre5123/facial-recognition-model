"""Face embedding model built on a torchvision MobileNetV2 backbone,
pretrained on ImageNet, fine-tuned for face verification with ArcFace.

Unlike models/mobilefacenet.py, this model starts from ImageNet-pretrained
weights rather than random initialization -- an experiment to see whether
that head start helps past the plateau the from-scratch baseline hit
(see docs/RESULTS_baseline.md).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetV2Embedding(nn.Module):
    """MobileNetV2 backbone (ImageNet-pretrained) + a face-embedding head.

    Input: 3x112x112 face crop.
    Output: L2-normalized embedding of dimension `embedding_dim`.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features  # conv layers only, no classifier
        backbone_out_channels = backbone.last_channel  # 1280 for MobileNetV2

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(backbone_out_channels, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

        if not pretrained:
            self._init_backbone_weights()
        self._init_head_weights()

    def _init_backbone_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_head_weights(self):
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.bn(x)
        embedding = nn.functional.normalize(x, p=2, dim=1)
        return embedding


if __name__ == "__main__":
    model = MobileNetV2Embedding(embedding_dim=256, pretrained=True)
    dummy = torch.randn(2, 3, 112, 112)
    out = model(dummy)
    print("output shape:", out.shape)
    print("output norm (should be ~1.0 per row):", out.norm(dim=1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
