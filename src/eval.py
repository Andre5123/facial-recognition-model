"""Verification evaluation: seen-identity/unseen-image and unseen-identity.

Loads a checkpoint saved by train.py, embeds the images referenced by
data/splits/val_seen.csv and/or test_unseen.csv, samples genuine/impostor
verification pairs, and reports:

  - k-fold cross-validated verification accuracy (LFW-style protocol: pick
    the best threshold on 9 folds, evaluate on the held-out fold, average)
  - ROC-AUC
  - TAR (true accept rate) at the FAR (false accept rate) targets in the config

Usage:
    python src/eval.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/best.pt
    python src/eval.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/best.pt --split unseen
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from data.dataset import build_transform
from models.mobilefacenet import MobileFaceNet


def read_manifest(csv_path: Path, label_col: str) -> dict[str, list[str]]:
    """Groups image paths by identity (label_col is 'label' or 'identity_name')."""
    by_identity: dict[str, list[str]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_identity[row[label_col]].append(row["image_path"])
    return dict(by_identity)


def sample_pairs(by_identity: dict[str, list[str]], n_pairs: int, seed: int):
    """Samples n_pairs/2 genuine + n_pairs/2 impostor pairs."""
    rng = random.Random(seed)
    identities = [name for name, imgs in by_identity.items() if len(imgs) >= 2]
    all_identities = list(by_identity.keys())
    if not identities:
        raise ValueError("no identity has >=2 images; cannot form genuine pairs")

    n_each = n_pairs // 2
    pairs = []

    for _ in range(n_each):
        name = rng.choice(identities)
        a, b = rng.sample(by_identity[name], 2)
        pairs.append((a, b, 1))

    for _ in range(n_each):
        name_a, name_b = rng.sample(all_identities, 2)
        a = rng.choice(by_identity[name_a])
        b = rng.choice(by_identity[name_b])
        pairs.append((a, b, 0))

    rng.shuffle(pairs)
    return pairs


@torch.no_grad()
def extract_embeddings(model, image_paths: list[str], transform, device: str, batch_size: int = 256):
    model.eval()
    embeddings = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as img:
                imgs.append(transform(img.convert("RGB")))
        batch = torch.stack(imgs).to(device)
        emb = model(batch).cpu().numpy()
        for p, e in zip(batch_paths, emb):
            embeddings[p] = e
    return embeddings


def kfold_accuracy(similarities: np.ndarray, labels: np.ndarray, n_folds: int = 10):
    """LFW-style protocol: per fold, pick the threshold that maximizes accuracy on
    the other folds, then evaluate on the held-out fold."""
    n = len(similarities)
    indices = np.arange(n)
    fold_size = n // n_folds
    thresholds = np.linspace(-1.0, 1.0, 400)
    fold_accuracies = []

    for fold in range(n_folds):
        test_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        best_acc, best_thresh = -1.0, 0.0
        for t in thresholds:
            preds = (similarities[train_idx] > t).astype(int)
            acc = (preds == labels[train_idx]).mean()
            if acc > best_acc:
                best_acc, best_thresh = acc, t

        test_preds = (similarities[test_idx] > best_thresh).astype(int)
        fold_accuracies.append((test_preds == labels[test_idx]).mean())

    return float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))


def tar_at_far(similarities: np.ndarray, labels: np.ndarray, far_targets: list[float]):
    fpr, tpr, _ = roc_curve(labels, similarities)
    results = {}
    for target in far_targets:
        idx = np.searchsorted(fpr, target, side="right") - 1
        idx = max(idx, 0)
        results[target] = float(tpr[idx])
    return results


def evaluate_split(name: str, manifest_csv: Path, label_col: str, n_pairs: int, seed: int, model, transform, device):
    by_identity = read_manifest(manifest_csv, label_col)
    pairs = sample_pairs(by_identity, n_pairs, seed)

    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    embeddings = extract_embeddings(model, unique_paths, transform, device)

    sims = np.array([float(np.dot(embeddings[a], embeddings[b])) for a, b, _ in pairs])
    labels = np.array([label for _, _, label in pairs])

    acc_mean, acc_std = kfold_accuracy(sims, labels)
    auc = roc_auc_score(labels, sims)

    print(f"\n=== {name} ===")
    print(f"pairs: {len(pairs)} ({labels.sum()} genuine, {len(labels) - labels.sum()} impostor)")
    print(f"verification accuracy: {acc_mean:.4f} +/- {acc_std:.4f} ({10}-fold)")
    print(f"ROC-AUC: {auc:.4f}")
    return sims, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=str, default=None, help="override data.splits_dir")
    parser.add_argument("--split", choices=["seen", "unseen", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.splits_dir:
        cfg["data"]["splits_dir"] = args.splits_dir
    splits_dir = Path(cfg["data"]["splits_dir"])

    device = args.device
    model = MobileFaceNet(embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded checkpoint from epoch {ckpt['epoch']}: {args.checkpoint}")

    transform = build_transform(cfg["data"]["image_size"], train=False, horizontal_flip=False)
    far_targets = cfg["eval"]["far_targets"]

    if args.split in ("seen", "both"):
        sims, labels = evaluate_split(
            "seen identity / unseen image",
            splits_dir / "val_seen.csv",
            "label",
            cfg["eval"]["verification_pairs_seen"],
            cfg["seed"],
            model,
            transform,
            device,
        )
        for far, tar in tar_at_far(sims, labels, far_targets).items():
            print(f"TAR@FAR={far}: {tar:.4f}")

    if args.split in ("unseen", "both"):
        sims, labels = evaluate_split(
            "unseen identity",
            splits_dir / "test_unseen.csv",
            "identity_name",
            cfg["eval"]["verification_pairs_unseen"],
            cfg["seed"],
            model,
            transform,
            device,
        )
        for far, tar in tar_at_far(sims, labels, far_targets).items():
            print(f"TAR@FAR={far}: {tar:.4f}")


if __name__ == "__main__":
    main()
