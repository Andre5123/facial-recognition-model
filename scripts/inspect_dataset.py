"""Inspect a downloaded dataset root before it's incorporated into the pipeline.

Reports directory layout, identity/image counts, image modes/sizes, and
flags anything that suggests the assumed "one folder per identity" layout
(as used by scripts/make_identity_split.py) doesn't hold, so the loader can
be adjusted before any training code depends on it.

Usage:
    python scripts/inspect_dataset.py --root <path-to-extracted-dataset>
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-images", type=int, default=20, help="number of images to open and inspect")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    top_level = sorted(root.iterdir())
    dirs = [p for p in top_level if p.is_dir()]
    files = [p for p in top_level if p.is_file()]

    print(f"root: {root}")
    print(f"top-level dirs: {len(dirs)}, top-level files: {len(files)}")
    if files:
        print(f"  example top-level files: {[p.name for p in files[:5]]}")
    if dirs:
        print(f"  example top-level dirs: {[p.name for p in dirs[:5]]}")

    if not dirs:
        print("No subdirectories found at top level -- layout is not 'one folder per identity'. Inspect manually.")
        return

    image_counts = Counter()
    ext_counts = Counter()
    total_images = 0
    for identity_dir in dirs:
        images = [p for p in identity_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        image_counts[identity_dir.name] = len(images)
        for img in images:
            ext_counts[img.suffix.lower()] += 1
        total_images += len(images)

    n_identities = len(dirs)
    counts = list(image_counts.values())
    print(f"\nidentities: {n_identities}")
    print(f"total images: {total_images}")
    print(f"images per identity: min={min(counts)}, max={max(counts)}, mean={total_images / n_identities:.1f}")
    print(f"file extensions: {dict(ext_counts)}")

    empty_identities = [name for name, c in image_counts.items() if c == 0]
    if empty_identities:
        print(f"\nWARNING: {len(empty_identities)} identity folders contain no recognized images "
              f"(e.g. {empty_identities[:5]})")

    # Sample a few images to check mode/size consistency.
    sample_paths = []
    for identity_dir in dirs:
        for img in identity_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                sample_paths.append(img)
                break
        if len(sample_paths) >= args.sample_images:
            break

    print(f"\nsampling {len(sample_paths)} images for mode/size:")
    sizes = Counter()
    modes = Counter()
    for p in sample_paths:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
    print(f"  sizes: {dict(sizes)}")
    print(f"  modes: {dict(modes)}")

    for name in ("LICENSE", "license", "LICENSE.txt", "README", "README.md", "readme.txt"):
        candidate = root / name
        if candidate.exists():
            print(f"\nFound license/readme file: {candidate} -- read this before using the dataset.")


if __name__ == "__main__":
    main()
