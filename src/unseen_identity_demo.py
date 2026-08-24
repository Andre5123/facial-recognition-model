"""Unseen-identity demonstration (context.md's "Future Work" scenario).

Picks a handful of identities the model never saw during training
(test_unseen.csv), and for each:
  1. Enrolls one image as a reference embedding.
  2. Compares it against the person's other images (genuine pairs).
  3. Compares it against other unseen people's images (impostor pairs).

Plots the genuine vs. impostor cosine-similarity distributions, to show
the model separates "same unseen person" from "different unseen person"
well -- direct visual evidence of generalization, not just a summary
accuracy number.

Usage:
    python src/unseen_identity_demo.py --config configs/baseline.yaml \
        --checkpoint ../checkpoints/baseline/epoch_29.pt \
        --output-dir ../docs
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

from data.dataset import build_transform
from eval import build_model, extract_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--n-identities", type=int, default=30, help="unseen identities to enroll")
    parser.add_argument("--min-images", type=int, default=3, help="min images an identity needs to be included")
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

    by_identity: dict[str, list[str]] = defaultdict(list)
    with open(splits_dir / "test_unseen.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_identity[row["identity_name"]].append(row["image_path"])

    eligible = [name for name, imgs in by_identity.items() if len(imgs) >= args.min_images]
    rng = random.Random(args.seed)
    enrolled_names = rng.sample(eligible, min(args.n_identities, len(eligible)))
    print(f"enrolling {len(enrolled_names)} never-trained-on identities")

    all_paths = sorted({p for name in enrolled_names for p in by_identity[name]})
    embeddings_by_path = extract_embeddings(model, all_paths, transform, device)

    reference = {}  # identity -> reference embedding
    probes = defaultdict(list)  # identity -> list of other embeddings
    for name in enrolled_names:
        imgs = by_identity[name]
        reference[name] = embeddings_by_path[imgs[0]]
        probes[name] = [embeddings_by_path[p] for p in imgs[1:]]

    genuine_sims = []
    for name in enrolled_names:
        for emb in probes[name]:
            genuine_sims.append(float(np.dot(reference[name], emb)))

    impostor_sims = []
    for name in enrolled_names:
        for emb in probes[name]:
            for other_name in enrolled_names:
                if other_name == name:
                    continue
                impostor_sims.append(float(np.dot(reference[other_name], emb)))

    genuine_sims = np.array(genuine_sims)
    impostor_sims = np.array(impostor_sims)

    print(f"genuine pairs: {len(genuine_sims)}, mean={genuine_sims.mean():.4f}, std={genuine_sims.std():.4f}")
    print(f"impostor pairs: {len(impostor_sims)}, mean={impostor_sims.mean():.4f}, std={impostor_sims.std():.4f}")

    # Zoom to the actual data range (with padding) rather than the full [-1, 1]
    # cosine range -- ArcFace-trained embeddings typically occupy a narrow high-
    # similarity band (a "narrow cone" on the hypersphere), so a full-range plot
    # would compress the real, meaningful separation into a sliver and make it
    # look like near-total overlap when it isn't.
    lo = min(genuine_sims.min(), impostor_sims.min())
    hi = max(genuine_sims.max(), impostor_sims.max())
    pad = (hi - lo) * 0.1
    bins = np.linspace(lo - pad, hi + pad, 80)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(impostor_sims, bins=bins, alpha=0.6, label=f"impostor (n={len(impostor_sims)})", color="tab:red", density=True)
    ax.hist(genuine_sims, bins=bins, alpha=0.6, label=f"genuine (n={len(genuine_sims)})", color="tab:blue", density=True)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"Genuine vs. impostor similarity, {len(enrolled_names)} never-trained-on identities")
    ax.legend()
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "unseen_identity_similarity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
