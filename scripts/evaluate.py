#!/usr/bin/env python3
"""
Evaluation script reproducing the quantitative tables of Sec. 6.1
(Tables 5-11): PSNR, SSIM, LPIPS on paired benchmarks, plus NIQE/BRISQUE
on no-reference collections such as ExDark.

Usage:
    python scripts/evaluate.py --checkpoint runs/edgelight_gan_lol/best.pt \
        --data_root ./datasets/LOL/test --dataset lol
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_dataset
from models import EdgeLightGenerator
from utils import device_auto, AverageMeter
from utils.metrics import psnr, ssim, lpips_distance, niqe_score, brisque_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EdgeLight-GAN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="lol")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--no_reference_only", action="store_true",
                         help="Skip PSNR/SSIM/LPIPS (use for unpaired sets such as ExDark)")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = device_auto()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model", {})
    generator = EdgeLightGenerator(**{k: v for k, v in cfg.items() if k in
        ["in_channels", "out_channels", "stem_channels", "branch_channels",
         "fused_channels", "num_residual_blocks", "compressed_channels", "decoder_channels"]}
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    dataset = build_dataset(args.dataset, args.data_root, image_size=args.image_size, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    meters = {k: AverageMeter() for k in ["PSNR", "SSIM", "LPIPS", "NIQE", "BRISQUE"]}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {args.dataset}"):
            low = batch["low"].to(device)
            pred = generator(low)

            if not args.no_reference_only and "high" in batch:
                high = batch["high"].to(device)
                meters["PSNR"].update(psnr(pred, high))
                meters["SSIM"].update(ssim(pred, high))
                try:
                    meters["LPIPS"].update(lpips_distance(pred, high))
                except ImportError:
                    pass

            try:
                meters["NIQE"].update(niqe_score(pred))
            except ImportError:
                pass
            try:
                meters["BRISQUE"].update(brisque_score(pred))
            except ImportError:
                pass

    results = {k: (float(m.avg) if m.count > 0 else None) for k, m in meters.items()}
    print(json.dumps(results, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
