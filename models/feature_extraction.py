"""
Multi-Resolution Feature Extraction Module (Sec. 3.3, Fig. 2).

Three parallel branches operating on the stem features:
    - Fine-scale branch   (Sec 3.3.1): shallow, small receptive field, captures
      edges / corners / microstructures / boundaries.
    - Mid-scale branch    (Sec 3.3.2): dilated convolutions, larger receptive
      field without heavy extra computation; captures object contours and
      regional illumination variation.
    - Global-context branch (Sec 3.3.3): global average pooling + attention
      guided contextual encoding; captures long-range illumination / scene
      context.

The three branches are concatenated and fused with a 1x1 conv (Table 3,
"Feature Fusion Module": 192 -> 128 channels), followed by an edge-aware
fusion step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DepthwiseSeparableConv, EdgeAwareFusion


class FineScaleBranch(nn.Module):
    """Shallow depthwise-separable convolutions with a small receptive field."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(DepthwiseSeparableConv(c_in, out_channels, kernel_size=3))
            c_in = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MidScaleBranch(nn.Module):
    """Dilated convolutions to enlarge the receptive field cheaply."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3):
        super().__init__()
        layers = []
        c_in = in_channels
        dilations = [2, 2, 4][:num_layers] or [2] * num_layers
        for i in range(num_layers):
            d = dilations[i] if i < len(dilations) else dilations[-1]
            layers.append(
                DepthwiseSeparableConv(c_in, out_channels, kernel_size=5, dilation=d)
            )
            c_in = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlobalContextBranch(nn.Module):
    """Global average pooling + attention-guided contextual encoding."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(c_in, out_channels, kernel_size=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            c_in = out_channels
        self.net = nn.Sequential(*layers)
        self.attn = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)          # B x C x 1 x 1  (global illumination summary)
        ctx = self.net(pooled)
        ctx = ctx * self.attn(ctx)
        # broadcast back to spatial resolution of x
        ctx = F.interpolate(ctx, size=x.shape[-2:], mode="nearest")
        return ctx


class MultiResolutionFeatureExtraction(nn.Module):
    """Combines fine / mid / global branches and fuses them (Sec. 3.3).

    Matches Table 3:
        Input Layer            : 3   -> 32   (3x3 conv)
        Fine-Scale Branch      : 32  -> 64   (depthwise separable, x3)
        Mid-Scale Branch       : 32  -> 64   (dilated conv, x3)
        Global Context Branch  : 32  -> 64   (global pool + conv, x2)
        Feature Fusion Module  : 192 -> 128  (concat + 1x1 conv, + edge-aware attn)
    """

    def __init__(self, in_channels: int = 3, stem_channels: int = 32,
                 branch_channels: int = 64, fused_channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fine_branch = FineScaleBranch(stem_channels, branch_channels, num_layers=3)
        self.mid_branch = MidScaleBranch(stem_channels, branch_channels, num_layers=3)
        self.global_branch = GlobalContextBranch(stem_channels, branch_channels, num_layers=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 3, fused_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.edge_aware_fusion = EdgeAwareFusion(fused_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        fine = self.fine_branch(stem)
        mid = self.mid_branch(stem)
        glob = self.global_branch(stem)
        fused = torch.cat([fine, mid, glob], dim=1)
        fused = self.fusion(fused)
        fused = self.edge_aware_fusion(fused)
        return fused
