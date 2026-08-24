"""Visualize the learned embedding space via PCA and t-SNE.

Samples a handful of identities from both the seen (val_seen.csv) and
unseen (test_unseen.csv) splits, embeds several images per identity, and
plots the resulting embeddings in 2D to show whether identities form
natural clusters -- and whether unseen identities (never trained on)
cluster just as cleanly as seen ones, which is the real test of whether
the model learned a generalizable face representation.

Usage:
    python src/visualize_embeddings.py --config configs/baseline.yaml \
        --checkpoint ../checkpoints/baseline/epoch_29.pt \
        --output ../docs/embedding_viz.png
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data.dataset import build_transform
from eval import build_model, extract_embeddings


def sample_identities(manifest_csv: Path, label_col: str, n_identities: int, min_images: int, seed: int):
    by_identity: dict[str, list[str]] = defaultdict(list)
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_identity[row[label_col]].append(row["image_path"])

    eligible = [name for name, imgs in by_identity.items() if len(imgs) >= min_images]
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(n_identities, len(eligible)))
    return {name: by_identity[name][:min_images] for name in chosen}


def plot_2d(coords: np.ndarray, identity_ids: list[int], is_unseen: list[bool], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    unique_ids = sorted(set(identity_ids))
    cmap = plt.get_cmap("tab20", len(unique_ids))
    color_of = {uid: cmap(i) for i, uid in enumerate(unique_ids)}

    for i in range(len(coords)):
        marker = "^" if is_unseen[i] else "o"
        ax.scatter(
            coords[i, 0], coords[i, 1], color=color_of[identity_ids[i]], marker=marker, s=40, alpha=0.8,
            edgecolors="black", linewidths=0.3,
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="gray", linestyle="", label="seen identity"),
            plt.Line2D([], [], marker="^", color="gray", linestyle="", label="unseen identity"),
        ],
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--n-seen-identities", type=int, default=10)
    parser.add_argument("--n-unseen-identities", type=int, default=10)
    parser.add_argument("--images-per-identity", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("../docs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.splits_dir:
        cfg["data"]["splits_dir"] = args.splits_dir
    splits_dir = Path(cfg["data"]["splits_dir"])

    device = args.device
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded checkpoint from epoch {ckpt['epoch']}: {args.checkpoint}")

    transform = build_transform(cfg["data"]["image_size"], train=False, horizontal_flip=False)

    seen = sample_identities(
        splits_dir / "val_seen.csv", "label", args.n_seen_identities, args.images_per_identity, args.seed
    )
    unseen = sample_identities(
        splits_dir / "test_unseen.csv",
        "identity_name",
        args.n_unseen_identities,
        args.images_per_identity,
        args.seed,
    )
    print(f"sampled {len(seen)} seen identities, {len(unseen)} unseen identities")

    all_paths = []
    identity_ids: list[int] = []
    is_unseen: list[bool] = []
    id_counter = 0
    for name, paths in seen.items():
        for p in paths:
            all_paths.append(p)
            identity_ids.append(id_counter)
            is_unseen.append(False)
        id_counter += 1
    for name, paths in unseen.items():
        for p in paths:
            all_paths.append(p)
            identity_ids.append(id_counter)
            is_unseen.append(True)
        id_counter += 1

    embeddings_by_path = extract_embeddings(model, all_paths, transform, device)
    embeddings = np.stack([embeddings_by_path[p] for p in all_paths])
    print(f"embedded {len(all_paths)} images, shape {embeddings.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pca_coords = PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)
    plot_2d(pca_coords, identity_ids, is_unseen, "PCA of face embeddings", args.output_dir / "embeddings_pca.png")

    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(embeddings) // 4))
    tsne_coords = tsne.fit_transform(embeddings)
    plot_2d(tsne_coords, identity_ids, is_unseen, "t-SNE of face embeddings", args.output_dir / "embeddings_tsne.png")


if __name__ == "__main__":
    main()
