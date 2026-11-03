"""
Intelligent Lightweight Generator Network (Sec. 3.4) and the full EdgeLight-GAN
generator pipeline (multi-resolution extraction -> generator -> illumination
reconstruction), following the audited configuration in Table 2/Table 3/Table 4
of the paper (0.748 M generator parameters at 256x256 input).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import (
    DepthwiseSeparableConv,
    ResidualAttentionBlock,
    EdgeAwareFusion,
    ChannelGating,
    IlluminationReconstruction,
)
from .feature_extraction import MultiResolutionFeatureExtraction


class EdgeLightGenerator(nn.Module):
    """Full generator: G(I_low) -> I_enhanced.

    Pipeline (Sec. 3.2, Table 3):
        1. Multi-Resolution Feature Extraction Module      (Sec. 3.3)
        2. Encoder stage (depthwise separable, downsample)  Table 3
        3. 3x Residual Attention Blocks                     (Sec. 3.4.2)
        4. Edge-Aware Fusion Layer                           (Sec. 3.4.3)
        5. Feature Compression Unit (128 -> 96)              Table 3
        6. Learnable Channel Gating ("Parameter Pruning Layer" in Table 3,
           formally the adaptive feature resource-control mechanism of
           Sec. 3.5 / Eq. 21-24)
        7. Decoder stage (transposed conv, upsample)         Table 3
        8. Reconstruction Block (64 -> 3)                    Table 3
        9. Illumination Reconstruction Module                (Sec. 3.9)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        stem_channels: int = 32,
        branch_channels: int = 64,
        fused_channels: int = 128,
        num_residual_blocks: int = 3,
        compressed_channels: int = 96,
        decoder_channels: int = 64,
    ):
        super().__init__()

        self.feature_extractor = MultiResolutionFeatureExtraction(
            in_channels=in_channels,
            stem_channels=stem_channels,
            branch_channels=branch_channels,
            fused_channels=fused_channels,
        )

        # Encoder stage: 4 depthwise-separable layers, first one downsamples 2x
        encoder_layers = [DepthwiseSeparableConv(fused_channels, fused_channels, 3, stride=2)]
        for _ in range(3):
            encoder_layers.append(DepthwiseSeparableConv(fused_channels, fused_channels, 3))
        self.encoder = nn.Sequential(*encoder_layers)

        # 3 Residual Attention Blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualAttentionBlock(fused_channels) for _ in range(num_residual_blocks)]
        )

        # Edge-Aware Fusion Layer (applied after residual attention stack)
        self.edge_aware_fusion = EdgeAwareFusion(fused_channels)

        # Feature Compression Unit: 128 -> 96 (pointwise)
        self.compress = nn.Sequential(
            nn.Conv2d(fused_channels, compressed_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Learnable channel gating / adaptive feature resource control
        self.channel_gating = ChannelGating(compressed_channels)

        # Decoder stage: transposed conv upsample back to input resolution,
        # 96 -> 64 across 3 layers (first layer upsamples 2x)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(compressed_channels, decoder_channels, kernel_size=4,
                                stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Reconstruction Block: 64 -> 3
        self.reconstruct = nn.Conv2d(decoder_channels, out_channels, kernel_size=3, padding=1)

        # Illumination Reconstruction Module (Sec. 3.9)
        self.illum_reconstruction = IlluminationReconstruction(out_channels)

        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor, return_illum: bool = False):
        feat = self.feature_extractor(x)
        feat = self.encoder(feat)

        for block in self.residual_blocks:
            feat = block(feat)

        feat = self.edge_aware_fusion(feat)
        feat = self.compress(feat)
        feat = self.channel_gating(feat)

        feat = self.decoder(feat)
        reflectance = self.reconstruct(feat)

        enhanced, illum_map = self.illum_reconstruction(reflectance)
        enhanced = self.output_activation(enhanced)

        # Residual global skip connection: recover fine input detail and
        # stabilize optimization (standard practice for enhancement GANs).
        if enhanced.shape[-2:] != x.shape[-2:]:
            enhanced = nn.functional.interpolate(
                enhanced, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if return_illum:
            return enhanced, illum_map
        return enhanced

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EdgeLightGenerator()
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)
    print(f"Generator parameters: {model.count_parameters() / 1e6:.3f} M")
