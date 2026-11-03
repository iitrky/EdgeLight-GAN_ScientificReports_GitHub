"""
Computational-complexity profiling utilities matching Sec. 6.7 / Table 23-26:

    - Parameter counting (generator-only and discriminator-only, Table 23)
    - FLOPs via torch.profiler at a given input resolution (generator-only,
      256x256, Sec 3.10 / Table 4: ~89.32 GFLOPs)
    - Inference latency / FPS benchmarking (Table 22, 24)
    - Peak memory utilization during inference (Table 27)

Notes
-----
The paper is explicit (Sec. 6.7.3, 6.7.4, 7.1) that these figures are
hardware-, software- and profiler-specific measurements and not universal
model properties. This module reproduces the *measurement procedure*
described in the paper; absolute numbers will vary with your hardware.
"""

from __future__ import annotations

import time
from typing import Tuple

import torch


def count_parameters(model: torch.nn.Module) -> float:
    """Returns trainable parameter count in millions (M)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def estimate_flops(model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 256, 256),
                    device: str = "cpu") -> float:
    """Generator-only FLOPs at the given resolution using torch.profiler
    (Sec 3.10: "A generator-only PyTorch profiler measurement at 256x256
    resolution gives approximately 89.32 GFLOPs").

    Returns GFLOPs (approximated as 2x the profiler-reported MACs, following
    the standard FLOPs = 2 * MACs convention).
    """
    model = model.to(device).eval()
    dummy = torch.randn(1, *input_size, device=device)

    with torch.profiler.profile(with_flops=True) as prof:
        with torch.no_grad():
            model(dummy)

    total_flops = sum(
        evt.flops for evt in prof.key_averages() if evt.flops is not None and evt.flops > 0
    )
    return total_flops / 1e9  # GFLOPs


@torch.no_grad()
def benchmark_latency(model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 256, 256),
                       device: str = "cpu", warmup: int = 10, iterations: int = 50) -> dict:
    """Measures per-image inference latency and derived FPS (Sec 6.7.1/6.7.2,
    Table 21-22). Mirrors the paper's inference-only measurement protocol:
    the discriminator is excluded (generator-only forward pass).
    """
    model = model.to(device).eval()
    dummy = torch.randn(1, *input_size, device=device)

    for _ in range(warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_image_ms = (elapsed / iterations) * 1000
    fps = 1000.0 / per_image_ms
    return {"latency_ms": per_image_ms, "fps": fps, "iterations": iterations, "device": device}


@torch.no_grad()
def measure_memory(model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 256, 256),
                    device: str = "cuda") -> dict:
    """Peak GPU memory utilization during a single forward pass (Table 27).

    Falls back to reporting `None` on CPU-only environments, since
    torch.cuda memory stats are not available there.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {"peak_memory_mb": None, "note": "CUDA not available; memory profiling requires a GPU."}

    model = model.to(device).eval()
    dummy = torch.randn(1, *input_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    model(dummy)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return {"peak_memory_mb": peak_bytes / (1024 ** 2)}


def full_report(model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 256, 256),
                 device: str | None = None) -> dict:
    """Runs the full computational-complexity report for a generator model,
    matching the structure of Table 23 in the paper.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "parameters_M": count_parameters(model),
        "device": device,
    }
    try:
        report["flops_G"] = estimate_flops(model, input_size, device)
    except Exception as e:  # pragma: no cover
        report["flops_G"] = f"unavailable ({e})"

    report.update(benchmark_latency(model, input_size, device))
    report.update(measure_memory(model, input_size, device))
    return report
