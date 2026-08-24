"""Exports embedding visualization data (2D projections only, no images) as
JSON, for the interactive HTML viewer. Reuses the exact same sampling as
visualize_embeddings.py (same seed) so results match the static PNGs.

Deliberately does NOT embed any face images/thumbnails, even downscaled --
CASIA-WebFace is restricted to non-commercial research/educational use (see
docs/DATASET.md), and the viewer this feeds may end up more widely shared
than a purely local analysis script, so no image data derived from the
dataset is included here at all.

Usage:
    python src/export_embedding_viz_data.py --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline/epoch_29.pt --output docs/embedding_viz_data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data.dataset import build_transform
from eval import build_model, extract_embeddings
from visualize_embeddings import sample_identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--n-seen-identities", type=int, default=10)
    parser.add_argument("--n-unseen-identities", type=int, default=10)
    parser.add_argument("--images-per-identity", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
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

    all_paths, identity_ids, identity_names, is_unseen = [], [], [], []
    id_counter = 0
    for name, paths in seen.items():
        for p in paths:
            all_paths.append(p)
            identity_ids.append(id_counter)
            identity_names.append(f"seen #{id_counter}")
            is_unseen.append(False)
        id_counter += 1
    for name, paths in unseen.items():
        for p in paths:
            all_paths.append(p)
            identity_ids.append(id_counter)
            identity_names.append(f"unseen #{id_counter}")
            is_unseen.append(True)
        id_counter += 1

    embeddings_by_path = extract_embeddings(model, all_paths, transform, device)
    embeddings = np.stack([embeddings_by_path[p] for p in all_paths])
    print(f"embedded {len(all_paths)} images, shape {embeddings.shape}")

    pca_coords = PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)
    tsne_coords = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(embeddings) // 4)).fit_transform(
        embeddings
    )

    def normalize(coords):
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        return ((coords - lo) / (hi - lo + 1e-9)).tolist()

    points = []
    for i, path in enumerate(all_paths):
        points.append(
            {
                "id": identity_ids[i],
                "name": identity_names[i],
                "unseen": is_unseen[i],
                "pca": None,  # filled below
                "tsne": None,
            }
        )

    pca_norm = normalize(pca_coords)
    tsne_norm = normalize(tsne_coords)
    for i, pt in enumerate(points):
        pt["pca"] = [round(v, 4) for v in pca_norm[i]]
        pt["tsne"] = [round(v, 4) for v in tsne_norm[i]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"points": points}, f)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
