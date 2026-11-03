#!/usr/bin/env python3
"""
Single-image / folder inference with a trained EdgeLight-GAN generator.
Note: only the generator is used at inference time (Sec. 3.6, 5.5) -- the
discriminator plays no role in deployment.

Usage:
    python scripts/inference.py --checkpoint runs/edgelight_gan_lol/best.pt \
        --input path/to/image_or_folder --output_dir ./outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import EdgeLightGenerator
from utils import device_auto
from data.datasets import IMG_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Run EdgeLight-GAN inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="image file or folder")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--image_size", type=int, default=None,
                         help="resize before inference; omit to keep original resolution")
    return parser.parse_args()


def load_generator(checkpoint_path: str, device: torch.device) -> EdgeLightGenerator:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model", {})
    keys = ["in_channels", "out_channels", "stem_channels", "branch_channels",
            "fused_channels", "num_residual_blocks", "compressed_channels", "decoder_channels"]
    generator = EdgeLightGenerator(**{k: v for k, v in cfg.items() if k in keys}).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    return generator


@torch.no_grad()
def enhance_image(generator, img_path: Path, out_path: Path, device, image_size=None):
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size
    if image_size:
        img = img.resize((image_size, image_size), Image.BICUBIC)

    tensor = (TF.to_tensor(img) * 2 - 1).unsqueeze(0).to(device)
    pred = generator(tensor)
    pred = ((pred.clamp(-1, 1) + 1) / 2).squeeze(0).cpu()
    out_img = TF.to_pil_image(pred)
    if image_size:
        out_img = out_img.resize(orig_size, Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)


def main():
    args = parse_args()
    device = device_auto()
    generator = load_generator(args.checkpoint, device)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    else:
        images = [input_path]

    for img_path in images:
        out_path = output_dir / img_path.name
        enhance_image(generator, img_path, out_path, device, args.image_size)
        print(f"Enhanced {img_path} -> {out_path}")


if __name__ == "__main__":
    main()
