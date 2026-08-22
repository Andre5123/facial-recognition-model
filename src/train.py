"""Baseline training loop: MobileFaceNet + ArcFace on the CASIA-WebFace
identity-disjoint train split (see docs/DATASET.md, scripts/make_identity_split.py).

Usage:
    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/baseline.yaml --epochs 1 --max-steps 20  # smoke test
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.dataset import FaceDataset, build_transform
from losses.arcface import ArcMarginHead
from models.mobilefacenet import MobileFaceNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(cfg: dict, params):
    train_cfg = cfg["train"]
    if train_cfg["optimizer"] != "sgd":
        raise ValueError(f"unsupported optimizer: {train_cfg['optimizer']}")
    return torch.optim.SGD(
        params,
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )


def build_scheduler(cfg: dict, optimizer):
    train_cfg = cfg["train"]
    if train_cfg["lr_schedule"] != "multistep":
        raise ValueError(f"unsupported lr_schedule: {train_cfg['lr_schedule']}")
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=train_cfg["lr_milestones"], gamma=train_cfg["lr_gamma"]
    )


def save_checkpoint(path: Path, epoch: int, model, head, optimizer, scheduler, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": cfg,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--data-root", type=str, default=None, help="override data.root")
    parser.add_argument("--splits-dir", type=str, default=None, help="override data.splits_dir")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="override checkpointing.dir")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps per epoch (smoke test)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, default=None, help="resume from a checkpoint saved by this script")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.splits_dir:
        cfg["data"]["splits_dir"] = args.splits_dir
    if args.checkpoint_dir:
        cfg["checkpointing"]["dir"] = args.checkpoint_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["seed"])
    device = args.device
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    splits_dir = Path(cfg["data"]["splits_dir"])
    transform = build_transform(
        cfg["data"]["image_size"], train=True, horizontal_flip=cfg["augmentation"]["horizontal_flip"]
    )
    train_ds = FaceDataset(splits_dir / "train.csv", transform)
    num_classes = train_ds.num_classes
    print(f"train images: {len(train_ds)}, identities: {num_classes}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    model = MobileFaceNet(embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    head = ArcMarginHead(
        cfg["model"]["embedding_dim"], num_classes, margin=cfg["arcface"]["margin"], scale=cfg["arcface"]["scale"]
    ).to(device)

    optimizer = build_optimizer(cfg, list(model.parameters()) + list(head.parameters()))
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = cfg["train"]["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if args.resume:
        print(f"resuming from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        head.load_state_dict(ckpt["head_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed at epoch {start_epoch}", flush=True)

    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        head.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch0 = time.time()

        for step, (images, labels) in enumerate(train_loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

            if step % 50 == 0:
                elapsed = time.time() - t_epoch0
                imgs_per_sec = (step + 1) * cfg["train"]["batch_size"] / max(elapsed, 1e-6)
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {loss.item():.4f} ({imgs_per_sec:.1f} img/s)",
                    flush=True,
                )

        scheduler.step()
        mean_loss = epoch_loss / max(n_batches, 1)
        print(f"epoch {epoch} done, mean_loss={mean_loss:.4f}, time={time.time()-t_epoch0:.1f}s", flush=True)

        if (epoch + 1) % cfg["checkpointing"]["save_every_epochs"] == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch}.pt", epoch, model, head, optimizer, scheduler, cfg)

        if cfg["checkpointing"]["keep_best"] and mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(ckpt_dir / "best.pt", epoch, model, head, optimizer, scheduler, cfg)

    save_checkpoint(ckpt_dir / "final.pt", cfg["train"]["epochs"] - 1, model, head, optimizer, scheduler, cfg)
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
