#!/usr/bin/env python3
"""
Reproduces the computational-efficiency measurements of Sec. 6.7
(Tables 21-27): parameter counts, generator-only FLOPs, inference latency,
FPS, and peak memory utilization.

Usage:
    python scripts/benchmark.py --checkpoint runs/edgelight_gan_lol/best.pt --device cuda
    python scripts/benchmark.py --device cpu   # benchmark an untrained model architecture

Reported numbers are implementation- and hardware-specific (Sec. 6.7,
7.1): re-running this script on different hardware will not reproduce the
paper's exact figures (Table 4: 0.748 M / 89.32 GFLOPs / 17.5 ms on an
RTX 4090; Table 28: Raspberry Pi 5 / Jetson Xavier NX / Jetson Orin Nano
figures were collected on the respective physical boards).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import EdgeLightGenerator, MultiScaleDiscriminator
from utils import device_auto
from utils.profiling import full_report, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark EdgeLight-GAN")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else device_auto()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_cfg = ckpt.get("config", {}).get("model", {})
        gen_keys = ["in_channels", "out_channels", "stem_channels", "branch_channels",
                    "fused_channels", "num_residual_blocks", "compressed_channels", "decoder_channels"]
        generator = EdgeLightGenerator(**{k: v for k, v in model_cfg.items() if k in gen_keys})
        discriminator = MultiScaleDiscriminator(
            in_channels=model_cfg.get("in_channels", 3),
            base_channels=model_cfg.get("discriminator_base_channels", 74),
        )
        generator.load_state_dict(ckpt["generator"])
        if "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
    else:
        generator = EdgeLightGenerator()
        discriminator = MultiScaleDiscriminator()

    report = full_report(generator, input_size=(3, args.image_size, args.image_size),
                          device=str(device))
    report["discriminator_parameters_M"] = count_parameters(discriminator)
    report["total_training_parameters_M"] = (
        report["parameters_M"] + report["discriminator_parameters_M"]
    )

    print(json.dumps(report, indent=2, default=str))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
