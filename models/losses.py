"""
Hybrid optimization objective for EdgeLight-GAN (Sec. 4.3-4.8).

    L_total = lambda_rec   * L_rec                    (reconstruction, L1)
            + lambda_adv   * L_adv_G                   (adversarial, Sec 4.4)
            + lambda_perc  * L_perceptual               (VGG feature loss, Sec 4.5)
            + lambda_ssim  * L_ssim                      (structural similarity, Sec 4.6)
            + lambda_edge  * L_edge                      (Sobel edge preservation, Sec 4.7)
            + lambda_illum * L_illum                     (illumination consistency, Sec 4.8)

The paper states these six coefficients were "empirically selected" (Sec 5.4)
but the numeric values are not recoverable from the manuscript text supplied
to this repository (they appear to have been embedded as inline equation
objects that did not extract as plain text). The defaults below are
reasonable, commonly used starting points for GAN-based image-enhancement
losses -- they are placeholders and should be re-tuned against the LOL
validation split; see README "Reproducibility Notes" for details.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    _TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCHVISION_AVAILABLE = False

from .blocks import SobelEdge


# --------------------------------------------------------------------------- #
# Reconstruction loss (Sec. 4.3)
# --------------------------------------------------------------------------- #
class ReconstructionLoss(nn.Module):
    def __init__(self, norm: str = "l1"):
        super().__init__()
        self.norm = norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.norm == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)


# --------------------------------------------------------------------------- #
# Adversarial loss (Sec. 4.4, Eq. 11-12) -- 'vanilla' is the manuscript's formulation;
# 'lsgan' is kept as an option.
# --------------------------------------------------------------------------- #
class AdversarialLoss(nn.Module):
    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        self.mode = mode
        if mode == "lsgan":
            self.criterion = nn.MSELoss()
        elif mode == "vanilla":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported adversarial loss mode: {mode}")

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        value = 1.0 if is_real else 0.0
        return torch.full_like(pred, value)

    def discriminator_loss(self, real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
        real_loss = self.criterion(real_score, self._target(real_score, True))
        fake_loss = self.criterion(fake_score, self._target(fake_score, False))
        return 0.5 * (real_loss + fake_loss)

    def generator_loss(self, fake_score: torch.Tensor) -> torch.Tensor:
        return self.criterion(fake_score, self._target(fake_score, True))


# --------------------------------------------------------------------------- #
# Perceptual loss (Sec. 4.5) -- VGG19 deep feature distance
# --------------------------------------------------------------------------- #
class PerceptualLoss(nn.Module):
    def __init__(self, layer_indices=(3, 8, 17, 26), device: str | None = None):
        super().__init__()
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for PerceptualLoss")
        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
        ).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layer_indices = set(layer_indices)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # inputs are assumed in [-1, 1] (tanh output) -> map to [0, 1] -> ImageNet norm
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor):
        feats = []
        h = x
        for idx, layer in enumerate(self.vgg):
            h = layer(h)
            if idx in self.layer_indices:
                feats.append(h)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)
        pred_feats = self._extract(pred_n)
        with torch.no_grad():
            target_feats = self._extract(target_n)
        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf)
        return loss / len(pred_feats)


# --------------------------------------------------------------------------- #
# Structural Similarity loss (Sec. 4.6)
# --------------------------------------------------------------------------- #
class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a Gaussian window (standard differentiable SSIM)."""

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._create_window(window_size, channels))

    @staticmethod
    def _gaussian(window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, channels):
        g1d = self._gaussian(window_size, 1.5).unsqueeze(1)
        g2d = g1d @ g1d.t()
        window = g2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        window = self.window.to(pred.device, pred.dtype)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=self.channels)
        mu_target = F.conv2d(target, window, padding=pad, groups=self.channels)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=self.channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=self.channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=self.channels) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        )
        return 1.0 - ssim_map.mean()


# --------------------------------------------------------------------------- #
# Edge preservation loss (Sec. 4.7) -- Sobel operator
# --------------------------------------------------------------------------- #
class EdgePreservationLoss(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.sobel = SobelEdge(channels=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge_pred = self.sobel(pred)
        edge_target = self.sobel(target)
        return F.l1_loss(edge_pred, edge_target)


# --------------------------------------------------------------------------- #
# Illumination consistency loss (Sec. 4.8)
# --------------------------------------------------------------------------- #
class IlluminationConsistencyLoss(nn.Module):
    """Penalizes deviation between the estimated illumination map and the
    illumination implied by the reference image, discouraging over-exposure.
    """

    def forward(self, illum_map: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Illumination proxy: per-pixel max over RGB channels (common Retinex approximation)
        target_illum = torch.max(target, dim=1, keepdim=True)[0]
        target_illum = (target_illum + 1) / 2  # map to [0, 1] to match illum_map range
        return F.l1_loss(illum_map, target_illum)


# --------------------------------------------------------------------------- #
# Aggregated hybrid loss
# --------------------------------------------------------------------------- #
class EdgeLightGANLoss(nn.Module):
    """Combines all six objective terms with configurable weights.

    Default weights (placeholders -- see module docstring):
        lambda_rec=100.0, lambda_adv=1.0, lambda_perc=10.0,
        lambda_ssim=5.0, lambda_edge=5.0, lambda_illum=2.0
    """

    def __init__(
        self,
        lambda_rec: float = 100.0,
        lambda_adv: float = 1.0,
        lambda_perc: float = 10.0,
        lambda_ssim: float = 5.0,
        lambda_edge: float = 5.0,
        lambda_illum: float = 2.0,
        adversarial_mode: str = "vanilla",
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.weights = dict(
            rec=lambda_rec, adv=lambda_adv, perc=lambda_perc,
            ssim=lambda_ssim, edge=lambda_edge, illum=lambda_illum,
        )
        self.rec_loss = ReconstructionLoss()
        self.adv_loss = AdversarialLoss(adversarial_mode)
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservationLoss()
        self.illum_loss = IlluminationConsistencyLoss()
        self.use_perceptual = use_perceptual and _TORCHVISION_AVAILABLE
        if self.use_perceptual:
            self.perc_loss = PerceptualLoss()
        else:
            self.perc_loss = None

    def generator_loss(self, pred, target, fake_score, illum_map):
        losses = {}
        losses["rec"] = self.rec_loss(pred, target)
        losses["adv"] = self.adv_loss.generator_loss(fake_score)
        losses["ssim"] = self.ssim_loss(pred, target)
        losses["edge"] = self.edge_loss(pred, target)
        losses["illum"] = self.illum_loss(illum_map, pred, target)
        if self.use_perceptual:
            losses["perc"] = self.perc_loss(pred, target)
        else:
            losses["perc"] = torch.zeros((), device=pred.device)

        total = sum(self.weights[k] * losses[k] for k in losses)
        losses["total"] = total
        return total, losses

    def discriminator_loss(self, real_score, fake_score):
        return self.adv_loss.discriminator_loss(real_score, fake_score)
