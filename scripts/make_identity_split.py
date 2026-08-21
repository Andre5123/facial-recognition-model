"""Build an identity-disjoint train/val/test split for CASIA-WebFace.

Expects the dataset root to contain one subdirectory per identity, each
holding that identity's face images (the standard CASIA-WebFace layout).
This has not yet been verified against the actual downloaded dataset -- run
`scripts/inspect_dataset.py` first and adjust this script if the real
layout differs.

Produces three manifests (CSV: image_path,identity_label) so that identity
labels stay consistent (0..num_train_identities-1) for the ArcFace head:

  - train.csv               images used for training
  - val_seen.csv             held-out images of TRAINING identities
                              ("seen identity, unseen image" eval)
  - test_unseen.csv          all images of identities NEVER used in training
                              ("unseen identity" eval)

The split is fully determined by `--seed`, so it is reproducible.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_identities(root: Path) -> dict[str, list[Path]]:
    identities: dict[str, list[Path]] = {}
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in identity_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            identities[identity_dir.name] = images
    return identities


def build_split(
    identities: dict[str, list[Path]],
    unseen_identity_fraction: float,
    held_out_image_fraction: float,
    seed: int,
):
    rng = random.Random(seed)

    identity_names = list(identities.keys())
    rng.shuffle(identity_names)

    n_unseen = max(1, round(len(identity_names) * unseen_identity_fraction))
    unseen_identities = set(identity_names[:n_unseen])
    train_eligible_identities = identity_names[n_unseen:]

    train_rows = []
    val_seen_rows = []
    test_unseen_rows = []

    # Training identity labels are assigned only over train-eligible
    # identities, in a fixed (shuffled) order, so label ids are stable
    # given the same seed.
    label_of = {name: idx for idx, name in enumerate(train_eligible_identities)}

    for name in train_eligible_identities:
        images = list(identities[name])
        rng.shuffle(images)
        n_held_out = max(1, round(len(images) * held_out_image_fraction)) if len(images) > 1 else 0
        held_out, kept = images[:n_held_out], images[n_held_out:]
        if not kept:
            # Never leave an identity with zero training images.
            kept, held_out = held_out, []
        label = label_of[name]
        for img in kept:
            train_rows.append((str(img), label))
        for img in held_out:
            val_seen_rows.append((str(img), label))

    for name in unseen_identities:
        for img in identities[name]:
            # Unseen identities have no training label; use the identity
            # name itself so verification pairs can be formed per-identity.
            test_unseen_rows.append((str(img), name))

    return train_rows, val_seen_rows, test_unseen_rows, label_of


def write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="CASIA-WebFace root (identity-per-folder)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--unseen-identity-fraction", type=float, default=0.05)
    parser.add_argument("--held-out-image-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    identities = discover_identities(args.data_root)
    if not identities:
        raise SystemExit(f"No identity subdirectories with images found under {args.data_root}")

    train_rows, val_seen_rows, test_unseen_rows, label_of = build_split(
        identities, args.unseen_identity_fraction, args.held_out_image_fraction, args.seed
    )

    write_csv(args.output_dir / "train.csv", train_rows, ["image_path", "label"])
    write_csv(args.output_dir / "val_seen.csv", val_seen_rows, ["image_path", "label"])
    write_csv(args.output_dir / "test_unseen.csv", test_unseen_rows, ["image_path", "identity_name"])
    write_csv(
        args.output_dir / "label_map.csv",
        sorted(label_of.items(), key=lambda kv: kv[1]),
        ["identity_name", "label"],
    )

    print(f"identities total:        {len(identities)}")
    print(f"train identities:        {len(label_of)}")
    print(f"unseen identities:       {len(identities) - len(label_of)}")
    print(f"train images:            {len(train_rows)}")
    print(f"val_seen images:         {len(val_seen_rows)}")
    print(f"test_unseen images:      {len(test_unseen_rows)}")
    print(f"seed:                    {args.seed}")
    print(f"wrote manifests to:      {args.output_dir}")


if __name__ == "__main__":
    main()
