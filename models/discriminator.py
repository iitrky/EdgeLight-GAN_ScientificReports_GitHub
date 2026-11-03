"""
Multi-Scale Discriminator Network (Sec. 3.6, Fig. 5).

Three complementary branches, used only during adversarial training
(Sec. 3.6, 5.5): the discriminator is excluded from the inference path.

    - Local Discriminator   (Sec. 3.6.1): PatchGAN-style, fine texture / edge
      consistency at the patch level.
    - Global Discriminator  (Sec. 3.6.2): whole-image illumination consistency,
      color fidelity, scene realism.
    - Texture Preservation Unit (Sec. 3.6.3): checks texture continuity /
      structural similarity between generated and reference images.

Outputs of the three branches are fused in a "Multi-Scale Score Fusion"
module (Table 3: fully-connected, 192 -> 1) to produce the final adversarial
decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_c, out_c, kernel_size=4, stride=2, padding=1, norm=True):
    layers = [nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class LocalDiscriminator(nn.Module):
    """PatchGAN-style discriminator operating on local image patches."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        layers = [conv_block(in_channels, base_channels, norm=False)]
        c = base_channels
        for _ in range(num_layers - 1):
            layers.append(conv_block(c, c, stride=1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.pool(feat).flatten(1)  # B x base_channels


class GlobalDiscriminator(nn.Module):
    """Whole-image discriminator for global illumination / color consistency."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        layers = [conv_block(in_channels, base_channels, norm=False)]
        c = base_channels
        for _ in range(num_layers - 1):
            layers.append(conv_block(c, c, stride=2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.pool(feat).flatten(1)  # B x base_channels


class TexturePreservationUnit(nn.Module):
    """Checks texture continuity / structural consistency of the image."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_channels, base_channels, stride=2, norm=False),
            conv_block(base_channels, base_channels, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return self.pool(feat).flatten(1)  # B x base_channels


class MultiScaleDiscriminator(nn.Module):
    """Fuses local, global and texture-preservation branch scores (Fig. 5).

    Training-only module: see Sec. 3.6 ("the multi-scale discriminator is used
    only during training ... during inference, the discriminator is removed
    from the execution path").
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 74):
        super().__init__()
        self.local_disc = LocalDiscriminator(in_channels, base_channels)
        self.global_disc = GlobalDiscriminator(in_channels, base_channels)
        self.texture_unit = TexturePreservationUnit(in_channels, base_channels)

        # Multi-Scale Score Fusion: fully connected, 192 -> 1 (Table 3)
        self.score_fusion = nn.Linear(base_channels * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_disc(x)
        global_feat = self.global_disc(x)
        texture_feat = self.texture_unit(x)
        fused = torch.cat([local_feat, global_feat, texture_feat], dim=1)
        score = self.score_fusion(fused)
        return score

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    disc = MultiScaleDiscriminator()
    dummy = torch.randn(2, 3, 256, 256)
    out = disc(dummy)
    print("Discriminator output shape:", out.shape)
    print(f"Discriminator parameters: {disc.count_parameters() / 1e6:.3f} M")
