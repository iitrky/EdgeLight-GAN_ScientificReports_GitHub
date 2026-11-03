"""
Basic shape / sanity tests for EdgeLight-GAN components.
Run with:  python -m pytest tests/ -v      (or)      python tests/test_models.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.blocks import (
    DepthwiseSeparableConv, ResidualAttentionBlock, ChannelGating,
    EdgeAwareFusion, SobelEdge, IlluminationReconstruction,
)
from models.feature_extraction import MultiResolutionFeatureExtraction
from models.generator import EdgeLightGenerator
from models.discriminator import MultiScaleDiscriminator
from models.losses import EdgeLightGANLoss


def test_depthwise_separable_conv():
    x = torch.randn(2, 16, 32, 32)
    block = DepthwiseSeparableConv(16, 32)
    out = block(x)
    assert out.shape == (2, 32, 32, 32)


def test_residual_attention_block():
    x = torch.randn(2, 64, 16, 16)
    block = ResidualAttentionBlock(64)
    out = block(x)
    assert out.shape == x.shape


def test_channel_gating_residual_property():
    x = torch.randn(2, 32, 8, 8)
    gate = ChannelGating(32)
    out = gate(x)
    assert out.shape == x.shape
    # gate values must lie in (0, 1)
    g = gate.gate_values(x)
    assert torch.all(g >= 0) and torch.all(g <= 1)


def test_sobel_edge():
    x = torch.randn(1, 3, 32, 32)
    sobel = SobelEdge(channels=3)
    edge = sobel(x)
    assert edge.shape == x.shape
    assert torch.all(edge >= 0)


def test_edge_aware_fusion():
    x = torch.randn(2, 64, 16, 16)
    fusion = EdgeAwareFusion(64)
    out = fusion(x)
    assert out.shape == x.shape


def test_illumination_reconstruction():
    x = torch.randn(2, 3, 32, 32)
    module = IlluminationReconstruction(3)
    enhanced, illum = module(x)
    assert enhanced.shape == x.shape
    assert illum.shape == (2, 1, 32, 32)


def test_multi_resolution_feature_extraction():
    x = torch.randn(1, 3, 64, 64)
    module = MultiResolutionFeatureExtraction()
    out = module(x)
    assert out.shape == (1, 128, 64, 64)


def test_generator_forward_and_param_budget():
    x = torch.randn(1, 3, 256, 256)
    generator = EdgeLightGenerator()
    out = generator(x)
    assert out.shape == x.shape
    params_m = sum(p.numel() for p in generator.parameters()) / 1e6
    # paper reports 0.748 M generator parameters (Table 4) -- allow tolerance
    # since exact per-layer widths beyond Table 3 were not all specified.
    assert 0.4 < params_m < 1.2


def test_discriminator_forward_and_param_budget():
    x = torch.randn(2, 3, 256, 256)
    discriminator = MultiScaleDiscriminator()
    out = discriminator(x)
    assert out.shape == (2, 1)
    params_m = sum(p.numel() for p in discriminator.parameters()) / 1e6
    # paper reports 0.616 M discriminator parameters (Table 4)
    assert 0.3 < params_m < 1.0


def test_generator_discriminator_total_matches_paper_order_of_magnitude():
    generator = EdgeLightGenerator()
    discriminator = MultiScaleDiscriminator()
    total = (sum(p.numel() for p in generator.parameters()) +
             sum(p.numel() for p in discriminator.parameters())) / 1e6
    # paper reports 1.364 M total training parameters (Table 4)
    assert 0.8 < total < 2.2


def test_loss_pipeline_runs():
    generator = EdgeLightGenerator()
    discriminator = MultiScaleDiscriminator()
    loss_fn = EdgeLightGANLoss(use_perceptual=False)

    low = torch.randn(2, 3, 64, 64)
    high = torch.randn(2, 3, 64, 64)
    pred, illum = generator(low, return_illum=True)
    fake_score = discriminator(pred)
    total, parts = loss_fn.generator_loss(pred, high, fake_score, illum)
    assert total.dim() == 0
    assert set(parts.keys()) == {"rec", "adv", "ssim", "edge", "illum", "perc", "total"}

    real_score = discriminator(high)
    d_loss = loss_fn.discriminator_loss(real_score, fake_score.detach())
    assert d_loss.dim() == 0


if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ...", end=" ")
        t()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
