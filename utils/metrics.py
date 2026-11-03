"""
Evaluation metrics used in Sec. 5.7 of the paper: PSNR, SSIM (full-reference),
LPIPS (perceptual similarity), NIQE and BRISQUE (no-reference naturalness).

PSNR/SSIM are implemented from scratch (numpy) so the repository has no hard
dependency for the core reconstruction metrics. LPIPS, NIQE and BRISQUE use
well-established third-party packages (`lpips`, `pyiqa`) when available, with
a clear error if they are not installed -- we do not reimplement these
non-trivial statistical/perceptual models from scratch.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor in [-1, 1] to an HWC uint8-range float numpy array in [0, 1]."""
    if t.dim() == 4:
        t = t[0]
    img = (t.clamp(-1, 1) + 1) / 2
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    return img


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (Sec. 5.7). Higher is better."""
    pred_np = _to_numpy_image(pred)
    target_np = _to_numpy_image(target)
    mse = np.mean((pred_np - target_np) ** 2)
    if mse <= 1e-10:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def _ssim_single_channel(pred: np.ndarray, target: np.ndarray, window_size: int = 11) -> float:
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    # simple uniform window (box filter) implementation to avoid a scipy dependency
    def box_filter(img, k):
        pad = k // 2
        img_p = np.pad(img, pad, mode="reflect")
        cumsum = np.cumsum(np.cumsum(img_p, axis=0), axis=1)
        cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")
        h, w = img.shape
        s = (
            cumsum[k:k + h, k:k + w]
            - cumsum[0:h, k:k + w]
            - cumsum[k:k + h, 0:w]
            + cumsum[0:h, 0:w]
        )
        return s / (k * k)

    mu_p = box_filter(pred, window_size)
    mu_t = box_filter(target, window_size)
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p_sq = box_filter(pred * pred, window_size) - mu_p_sq
    sigma_t_sq = box_filter(target * target, window_size) - mu_t_sq
    sigma_pt = box_filter(pred * target, window_size) - mu_pt

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    )
    return float(ssim_map.mean())


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Structural Similarity Index (Sec. 5.7). Higher is better."""
    pred_np = _to_numpy_image(pred)
    target_np = _to_numpy_image(target)
    scores = [
        _ssim_single_channel(pred_np[..., c], target_np[..., c])
        for c in range(pred_np.shape[-1])
    ]
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Perceptual / no-reference metrics -- optional third-party dependencies
# --------------------------------------------------------------------------- #
_lpips_model = None


def lpips_distance(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> float:
    """Learned Perceptual Image Patch Similarity (Sec. 5.7). Lower is better.

    Requires `pip install lpips`.
    """
    global _lpips_model
    try:
        import lpips
    except ImportError as e:
        raise ImportError(
            "LPIPS metric requires the `lpips` package: pip install lpips"
        ) from e
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net)
        _lpips_model.eval()
    with torch.no_grad():
        p = pred if pred.dim() == 4 else pred.unsqueeze(0)
        t = target if target.dim() == 4 else target.unsqueeze(0)
        dist = _lpips_model(p.clamp(-1, 1), t.clamp(-1, 1))
    return float(dist.mean().item())


def niqe_score(pred: torch.Tensor) -> float:
    """Natural Image Quality Evaluator (Sec. 5.7). Lower is better, no reference needed.

    Requires `pip install pyiqa`.
    """
    try:
        import pyiqa
    except ImportError as e:
        raise ImportError(
            "NIQE metric requires the `pyiqa` package: pip install pyiqa"
        ) from e
    metric = pyiqa.create_metric("niqe")
    img = (pred.clamp(-1, 1) + 1) / 2
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return float(metric(img).item())


def brisque_score(pred: torch.Tensor) -> float:
    """Blind/Referenceless Image Spatial Quality Evaluator (Sec. 5.7). Lower is better.

    Requires `pip install pyiqa`.
    """
    try:
        import pyiqa
    except ImportError as e:
        raise ImportError(
            "BRISQUE metric requires the `pyiqa` package: pip install pyiqa"
        ) from e
    metric = pyiqa.create_metric("brisque")
    img = (pred.clamp(-1, 1) + 1) / 2
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return float(metric(img).item())


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor | None = None,
                         include_no_reference: bool = True) -> dict:
    """Convenience wrapper matching Table 5-11 of the paper.

    If `target` is None (unpaired evaluation, e.g. ExDark, Sec. 6.1.3), only
    no-reference metrics (NIQE, BRISQUE) are computed.
    """
    results = {}
    if target is not None:
        results["PSNR"] = psnr(pred, target)
        results["SSIM"] = ssim(pred, target)
        try:
            results["LPIPS"] = lpips_distance(pred, target)
        except ImportError:
            results["LPIPS"] = None
    if include_no_reference:
        try:
            results["NIQE"] = niqe_score(pred)
        except ImportError:
            results["NIQE"] = None
        try:
            results["BRISQUE"] = brisque_score(pred)
        except ImportError:
            results["BRISQUE"] = None
    return results
