"""
Dataset loaders for the benchmark and application-oriented collections used
in the paper (Sec. 5.2, Table 4):

    - LOL              (paired low/normal light, https://daooshee.github.io/BMVC2018website/)
    - LOL-v2           (real + synthetic, https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
    - ExDark           (unpaired, no-reference eval only, https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
    - SID              (raw sensor, https://github.com/cchen156/Learning-to-See-in-the-Dark)
    - Generic paired folders for the application-oriented collections
      (medical / remote-sensing / surveillance, Sec. 5.2.5-5.2.7) -- these are
      not redistributed by this repository; point --data_root at your own copy.

All loaders expect a directory layout of the form:

    <root>/
        low/    *.png|*.jpg   (degraded input images)
        high/   *.png|*.jpg   (paired ground-truth images; omit for unpaired sets)

and return tensors normalized to [-1, 1] (matching the generator's Tanh
output), resized/cropped to a square `image_size` (default 256, Sec. 5.1/5.3).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    )


class PairedLowLightDataset(Dataset):
    """Generic paired low-light / normal-light dataset.

    Works for LOL, LOL-v2, and any application-oriented collection organized
    as `<root>/low/*` and `<root>/high/*` with matching filenames (Sec 5.2,
    5.3 -- preprocessing: resize/crop to 256x256, normalize to [0,1],
    augmentation applied only on the training split).
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        split: str = "train",
        augment: Optional[bool] = None,
    ):
        self.root = Path(root)
        self.low_dir = self.root / "low"
        self.high_dir = self.root / "high"
        if not self.low_dir.exists():
            raise FileNotFoundError(f"Expected low-light folder at {self.low_dir}")

        self.low_paths = _list_images(self.low_dir)
        self.paired = self.high_dir.exists()
        if self.paired:
            self.high_paths = _list_images(self.high_dir)
            if len(self.high_paths) != len(self.low_paths):
                raise ValueError(
                    f"Mismatched pair counts: {len(self.low_paths)} low vs "
                    f"{len(self.high_paths)} high images in {root}"
                )
        else:
            self.high_paths = None

        self.image_size = image_size
        self.split = split
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self):
        return len(self.low_paths)

    def _load(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _sync_augment(self, low: Image.Image, high: Optional[Image.Image]):
        # Random crop
        w, h = low.size
        size = self.image_size
        if w < size or h < size:
            low = low.resize((max(w, size), max(h, size)), Image.BICUBIC)
            if high is not None:
                high = high.resize((max(w, size), max(h, size)), Image.BICUBIC)
            w, h = low.size

        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        low = low.crop((x, y, x + size, y + size))
        if high is not None:
            high = high.crop((x, y, x + size, y + size))

        # Horizontal / vertical flips
        if random.random() < 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            if high is not None:
                high = high.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            if high is not None:
                high = high.transpose(Image.FLIP_TOP_BOTTOM)

        # Random rotation (multiples of 90 degrees to avoid border artifacts)
        if random.random() < 0.5:
            k = random.choice([90, 180, 270])
            low = low.rotate(k)
            if high is not None:
                high = high.rotate(k)

        return low, high

    def _center_resize(self, img: Image.Image) -> Image.Image:
        return img.resize((self.image_size, self.image_size), Image.BICUBIC)

    def __getitem__(self, idx):
        low_img = self._load(self.low_paths[idx])
        high_img = self._load(self.high_paths[idx]) if self.paired else None

        if self.augment:
            low_img, high_img = self._sync_augment(low_img, high_img)
        else:
            low_img = self._center_resize(low_img)
            if high_img is not None:
                high_img = self._center_resize(high_img)

        low_t = TF.to_tensor(low_img) * 2 - 1  # [0,1] -> [-1,1]
        sample = {"low": low_t, "path": str(self.low_paths[idx])}

        if high_img is not None:
            high_t = TF.to_tensor(high_img) * 2 - 1
            sample["high"] = high_t
        return sample


class UnpairedLowLightDataset(Dataset):
    """For no-reference evaluation collections such as ExDark (Sec. 5.2.3)."""

    def __init__(self, root: str, image_size: int = 256):
        self.root = Path(root)
        self.paths = _list_images(self.root)
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        tensor = TF.to_tensor(img) * 2 - 1
        return {"low": tensor, "path": str(self.paths[idx])}


def build_dataset(name: str, root: str, image_size: int = 256, split: str = "train"):
    """Factory used by scripts/train.py and scripts/evaluate.py.

    name: one of {lol, lolv2, exdark, sid, medical, remote_sensing, surveillance, generic}
    """
    name = name.lower()
    if name == "exdark":
        return UnpairedLowLightDataset(root, image_size=image_size)
    # LOL, LOL-v2, SID and the application-oriented collections all follow the
    # same low/high paired-folder convention once pre-processed (Sec. 5.2, 5.3).
    return PairedLowLightDataset(root, image_size=image_size, split=split)
