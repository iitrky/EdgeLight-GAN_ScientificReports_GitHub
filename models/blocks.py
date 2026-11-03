"""
Core building blocks for EdgeLight-GAN.

Implements the lightweight operators described in Section 3 of the paper:
    - Depthwise Separable Convolution (Sec. 3.4.1)
    - Residual Attention Block: channel + spatial attention (Sec. 3.4.2)
    - Edge-Aware Feature Fusion (Sec. 3.4.3 / 3.7, Sobel-based)
    - Learnable Channel Gating / Adaptive Feature Resource Control (Sec. 3.5, Eq. 21-24)

None of these operators is claimed as individually novel by the paper; the
contribution of EdgeLight-GAN is their coordinated integration (see Sec. 1.6).
This module is a faithful, from-scratch PyTorch re-implementation written for
this repository -- it is not code released by the original authors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Depthwise Separable Convolution  (Sec. 3.4.1, Eq. 26-27)
# --------------------------------------------------------------------------- #
class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv followed by pointwise (1x1) conv.

    Parameter count relative to a standard k x k convolution is reduced by
    approximately 1/C_out + 1/k^2, matching Eq. (27) of the paper.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, bias: bool = False,
                 norm: bool = True, activation: bool = True):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# --------------------------------------------------------------------------- #
# Channel + Spatial Attention  (used inside Residual Attention Block, Sec 3.4.2)
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention computed from channel-pooled statistics."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block (RAB) -- Sec. 3.4.2.

    residual-learning unit + channel-attention mechanism + spatial-attention
    mechanism, combined with a skip connection so the attention mechanism can
    "automatically find the lighting deficient regions and distribute more
    resources to the improvement in those regions".
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, 3)
        self.conv2 = DepthwiseSeparableConv(channels, channels, 3, activation=False)
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        return self.act(out + identity)


# --------------------------------------------------------------------------- #
# Learnable Channel Gating / Adaptive Feature Resource Control (Sec. 3.5, Eq. 21-24)
# --------------------------------------------------------------------------- #
class ChannelGating(nn.Module):
    """Implements Eq. (21)-(24) of the paper.

        z_c   = (1/HW) * sum_{h,w} F_{c,h,w}                     (21)  global descriptor
        g     = sigma( W2 * delta(W1 * z) )                      (22)  gating function
        F_hat = g (*) F                                          (23)  channel modulation
        F_out = F + F_hat                                        (24)  residual pathway

    This performs *feature-response modulation*, not physical parameter
    pruning or dynamic kernel replacement -- consistent with Sec. 3.5.1/3.5.2
    and the clarification in Sec. 4.9 of the paper.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.global_descriptor = nn.AdaptiveAvgPool2d(1)
        self.w1 = nn.Linear(channels, hidden, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.w2 = nn.Linear(hidden, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = self.global_descriptor(x).view(b, c)          # Eq. 21
        g = self.sigmoid(self.w2(self.act(self.w1(z))))   # Eq. 22
        g = g.view(b, c, 1, 1)
        f_hat = x * g                                      # Eq. 23
        return x + f_hat                                   # Eq. 24 (residual pathway)

    def gate_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw per-channel importance weights (for analysis/ablation)."""
        b, c, _, _ = x.shape
        z = self.global_descriptor(x).view(b, c)
        return self.sigmoid(self.w2(self.act(self.w1(z))))


# Backwards-compatible alias matching the paper's "Parameter Pruning Layer"
# naming in Table 3 -- functionally this is ChannelGating (see Sec 3.5, 4.9,
# 6.5.2 and 7 which clarify the mechanism is adaptive gating, not hard pruning).
AdaptiveFeatureResourceControl = ChannelGating


# --------------------------------------------------------------------------- #
# Sobel Edge Extraction (Sec. 3.7, used by EdgeAwareFusion and the edge loss)
# --------------------------------------------------------------------------- #
class SobelEdge(nn.Module):
    """Fixed (non-trainable) Sobel gradient operator producing an edge map."""

    def __init__(self, channels: int = 3):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("weight_x", kx)
        self.register_buffer("weight_y", ky)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.weight_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.weight_y, padding=1, groups=self.channels)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return edge


class EdgeAwareFusion(nn.Module):
    """Edge-Aware Feature Fusion (Sec. 3.4.3 / 3.7).

    Computes a Sobel edge map from an auxiliary (image-space or feature-space)
    signal and uses it to re-weight the fused feature map, giving "higher
    priority ... in the regions with more number of edges".
    """

    def __init__(self, channels: int):
        super().__init__()
        self.edge_extractor = SobelEdge(channels=channels)
        self.edge_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        edge_map = self.edge_extractor(features)
        attn = self.edge_attn(edge_map)
        return features * attn + features


# --------------------------------------------------------------------------- #
# Illumination Reconstruction Module (Sec. 3.9)
# --------------------------------------------------------------------------- #
class IlluminationReconstruction(nn.Module):
    """Estimates a learnable illumination map and applies it to reflectance
    features to recover a visually realistic, well-illuminated output.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.illum_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, reflectance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        illum_map = self.illum_estimator(reflectance)
        enhanced = reflectance * (1.0 + illum_map)
        return enhanced, illum_map
