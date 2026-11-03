"""Common utilities: reproducibility, checkpointing, simple logging."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fixed random seed used throughout the paper's experimental protocol
    (Sec. 5.1 / 5.6: seed = 42, three repeated runs averaged)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "edgelight_gan", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    # weights_only=False: our checkpoints bundle the run config (a plain
    # dict) alongside the tensors, which PyTorch's default safe-unpickler
    # (weights_only=True since torch 2.6) rejects. Only load checkpoints
    # produced by this repository or another source you trust.
    return torch.load(path, map_location=map_location, weights_only=False)


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    """Tracks a running average (used for per-epoch loss/metric logging)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
