#!/usr/bin/env python3
"""
Training script for EdgeLight-GAN (Sec. 5.4-5.6 of the paper).

Usage:
    python scripts/train.py --config configs/default.yaml

Implements the alternating generator/discriminator adversarial optimization
described in Sec. 5.5, using Adam (Sec. 5.4), a cosine-annealing learning
rate schedule, and the hybrid loss of Sec. 4.3-4.8 / models/losses.py.
Reproducibility follows Sec. 5.6 (fixed seed, periodic checkpointing, best
model selected on the validation split).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_dataset
from models import EdgeLightGenerator, MultiScaleDiscriminator, EdgeLightGANLoss
from utils import set_seed, get_logger, save_checkpoint, device_auto, AverageMeter
from utils.metrics import psnr, ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Train EdgeLight-GAN")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_models(cfg: dict, device: torch.device):
    m = cfg["model"]
    generator = EdgeLightGenerator(
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        stem_channels=m["stem_channels"],
        branch_channels=m["branch_channels"],
        fused_channels=m["fused_channels"],
        num_residual_blocks=m["num_residual_blocks"],
        compressed_channels=m["compressed_channels"],
        decoder_channels=m["decoder_channels"],
    ).to(device)

    discriminator = MultiScaleDiscriminator(
        in_channels=m["in_channels"],
        base_channels=m["discriminator_base_channels"],
    ).to(device)

    return generator, discriminator


@torch.no_grad()
def validate(generator, val_loader, device, logger, max_batches: int = 50):
    generator.eval()
    psnr_meter, ssim_meter = AverageMeter(), AverageMeter()
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        low = batch["low"].to(device)
        high = batch.get("high")
        pred = generator(low)
        if high is not None:
            high = high.to(device)
            for b in range(pred.shape[0]):
                psnr_meter.update(psnr(pred[b], high[b]))
                ssim_meter.update(ssim(pred[b], high[b]))
    generator.train()
    if psnr_meter.count > 0:
        logger.info(f"[val] PSNR={psnr_meter.avg:.2f} dB  SSIM={ssim_meter.avg:.4f}")
    return psnr_meter.avg, ssim_meter.avg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["train"]["seed"])
    device = device_auto()

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=str(output_dir / "train.log"))
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {cfg}")

    # --- Data ---
    train_set = build_dataset(cfg["data"]["dataset"], cfg["data"]["train_root"],
                               image_size=cfg["data"]["image_size"], split="train")
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], drop_last=True,
    )

    val_loader = None
    if cfg["data"].get("val_root") and Path(cfg["data"]["val_root"]).exists():
        val_set = build_dataset(cfg["data"]["dataset"], cfg["data"]["val_root"],
                                 image_size=cfg["data"]["image_size"], split="val")
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                 num_workers=cfg["data"]["num_workers"])

    # --- Models ---
    generator, discriminator = build_models(cfg, device)
    logger.info(f"Generator parameters: {sum(p.numel() for p in generator.parameters())/1e6:.3f} M")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())/1e6:.3f} M")

    loss_fn = EdgeLightGANLoss(**cfg["loss"]).to(device)

    # --- Optimizers (Sec. 5.4: Adam, alternating updates) ---
    t = cfg["train"]
    opt_g = torch.optim.Adam(generator.parameters(), lr=t["lr"], betas=(t["beta1"], t["beta2"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=t["lr"], betas=(t["beta1"], t["beta2"]))

    if t.get("lr_schedule") == "cosine":
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=t["epochs"])
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=t["epochs"])
    else:
        sched_g = sched_d = None

    start_epoch = 0
    best_psnr = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -1.0)
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    # --- Training loop (Sec. 5.5) ---
    for epoch in range(start_epoch, t["epochs"]):
        g_loss_meter, d_loss_meter = AverageMeter(), AverageMeter()

        for step, batch in enumerate(train_loader):
            low = batch["low"].to(device)
            high = batch["high"].to(device)

            # --- Discriminator step ---
            with torch.no_grad():
                fake = generator(low)
            real_score = discriminator(high)
            fake_score = discriminator(fake.detach())
            d_loss = loss_fn.discriminator_loss(real_score, fake_score)

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # --- Generator step ---
            fake, illum_map = generator(low, return_illum=True)
            fake_score = discriminator(fake)
            g_loss, parts = loss_fn.generator_loss(fake, high, fake_score, illum_map)

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_loss_meter.update(g_loss.item())
            d_loss_meter.update(d_loss.item())

            if step % t["log_every"] == 0:
                logger.info(
                    f"Epoch {epoch}/{t['epochs']} Step {step}/{len(train_loader)} "
                    f"G={g_loss.item():.4f} D={d_loss.item():.4f} "
                    f"(rec={parts['rec'].item():.3f} adv={parts['adv'].item():.3f} "
                    f"perc={parts['perc'].item():.3f} ssim={parts['ssim'].item():.3f} "
                    f"edge={parts['edge'].item():.3f} illum={parts['illum'].item():.3f})"
                )

        if sched_g is not None:
            sched_g.step()
            sched_d.step()

        logger.info(f"Epoch {epoch} done. mean G={g_loss_meter.avg:.4f} mean D={d_loss_meter.avg:.4f}")

        if val_loader is not None and (epoch + 1) % t["val_every"] == 0:
            val_psnr, val_ssim = validate(generator, val_loader, device, logger)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint({
                    "epoch": epoch, "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "best_psnr": best_psnr, "config": cfg,
                }, str(output_dir / "best.pt"))
                logger.info(f"New best checkpoint saved (PSNR={best_psnr:.2f} dB)")

        if (epoch + 1) % t["checkpoint_every"] == 0:
            save_checkpoint({
                "epoch": epoch, "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "best_psnr": best_psnr, "config": cfg,
            }, str(output_dir / f"checkpoint_epoch{epoch+1}.pt"))

    save_checkpoint({
        "epoch": t["epochs"] - 1, "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        "best_psnr": best_psnr, "config": cfg,
    }, str(output_dir / "final.pt"))
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
