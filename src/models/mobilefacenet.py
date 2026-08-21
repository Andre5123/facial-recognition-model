"""MobileFaceNet architecture, recreated from the original paper:

Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-Time Face
Verification on Mobile Devices" (2018), https://arxiv.org/abs/1804.07573

Weights are randomly initialized; no pretrained checkpoints are loaded here.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseBlock(nn.Module):
    """Depthwise conv followed by pointwise conv (standard MobileNet separable conv)."""

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.depthwise = ConvBlock(in_c, in_c, kernel, stride, padding, groups=in_c)
        self.pointwise = ConvBlock(in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Bottleneck(nn.Module):
    """Inverted residual block (expand -> depthwise -> project), as used in MobileNetV2/MobileFaceNet."""

    def __init__(self, in_c, out_c, stride, expansion):
        super().__init__()
        self.use_residual = stride == 1 and in_c == out_c
        hidden_dim = in_c * expansion

        self.expand = ConvBlock(in_c, hidden_dim, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.depthwise = ConvBlock(
            hidden_dim, hidden_dim, kernel=(3, 3), stride=(stride, stride), padding=(1, 1), groups=hidden_dim
        )
        self.project = ConvBlock(hidden_dim, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), use_act=False)

    def forward(self, x):
        out = self.project(self.depthwise(self.expand(x)))
        if self.use_residual:
            out = out + x
        return out


class BottleneckStage(nn.Module):
    """A stack of `n` bottleneck blocks; only the first block uses `stride`."""

    def __init__(self, in_c, out_c, stride, expansion, n):
        super().__init__()
        layers = [Bottleneck(in_c, out_c, stride, expansion)]
        for _ in range(n - 1):
            layers.append(Bottleneck(out_c, out_c, 1, expansion))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(nn.Module):
    """MobileFaceNet backbone producing a fixed-size face embedding.

    Input: 3x112x112 face crop.
    Output: L2-normalized embedding of dimension `embedding_dim`.

    Architecture (channels/strides/expansion/repeats) follows Table 1 of the
    MobileFaceNets paper, sized for 112x112 input.
    """

    # (out_channels, stride, expansion, num_blocks)
    _STAGE_CFG = [
        (64, 2, 2, 5),
        (128, 2, 4, 1),
        (128, 1, 2, 6),
        (128, 2, 4, 1),
        (128, 1, 2, 2),
    ]

    def __init__(self, embedding_dim: int = 256, input_size: int = 112):
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError("input_size must be divisible by 16")

        self.stem = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.dw_stem = DepthwiseBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        stages = []
        in_c = 64
        for out_c, stride, expansion, n in self._STAGE_CFG:
            stages.append(BottleneckStage(in_c, out_c, stride, expansion, n))
            in_c = out_c
        self.stages = nn.Sequential(*stages)

        self.conv_1x1 = ConvBlock(in_c, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

        # Global depthwise conv (GDConv) replaces global average pooling, per the paper,
        # since faces are spatially aligned and different regions carry unequal information.
        feat_map_size = input_size // 16
        self.gdconv = nn.Conv2d(512, 512, kernel_size=feat_map_size, groups=512, bias=False)
        self.gdconv_bn = nn.BatchNorm2d(512)

        self.linear = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear_bn = nn.BatchNorm1d(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dw_stem(x)
        x = self.stages(x)
        x = self.conv_1x1(x)
        x = self.gdconv_bn(self.gdconv(x))
        x = self.linear(x)
        x = x.flatten(1)
        x = self.linear_bn(x)
        embedding = nn.functional.normalize(x, p=2, dim=1)
        return embedding


if __name__ == "__main__":
    model = MobileFaceNet(embedding_dim=256)
    dummy = torch.randn(2, 3, 112, 112)
    out = model(dummy)
    print("output shape:", out.shape)
    print("output norm (should be ~1.0 per row):", out.norm(dim=1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
