"""Downloads a small, filtered subset of LFW (people with many photos each)
and saves it as real JPG files, one folder per person -- ready to point the
demo's "Bulk enroll from a folder" feature at directly (see docs/DEMO.md).

Uses scikit-learn's built-in LFW fetcher, which downloads the full LFW
archive (~200MB, one-time, cached at ~/scikit_learn_data) and filters
locally -- there's no smaller server-side download available for this.

Usage:
    python scripts/fetch_demo_test_photos.py --min-faces-per-person 50 --output-dir demo_test_photos
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from sklearn.datasets import fetch_lfw_people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-faces-per-person",
        type=int,
        default=50,
        help="only keep people with at least this many photos (higher = fewer, more-photographed people)",
    )
    parser.add_argument("--max-per-person", type=int, default=10, help="cap saved photos per person")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_test_photos"))
    args = parser.parse_args()

    print(f"fetching LFW subset (min_faces_per_person={args.min_faces_per_person}) -- may download on first run...")
    lfw = fetch_lfw_people(min_faces_per_person=args.min_faces_per_person, color=True, resize=1.0)

    print(f"{len(lfw.target_names)} people, {len(lfw.images)} total images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for image_arr, target_idx in zip(lfw.images, lfw.target):
        name = lfw.target_names[target_idx].replace(" ", "_")
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > args.max_per_person:
            continue
        person_dir = args.output_dir / name
        person_dir.mkdir(exist_ok=True)
        # lfw.images is float32 in [0, 1] (not [0, 255]) -- scale before
        # converting to uint8, or every pixel truncates to near-black.
        img = Image.fromarray((image_arr * 255).round().astype("uint8"))
        img.save(person_dir / f"{counts[name]}.jpg", quality=90)

    print(f"wrote photos to {args.output_dir}/<person_name>/*.jpg")
    for name in sorted(counts):
        print(f"  {name}: {min(counts[name], args.max_per_person)} photos saved")


if __name__ == "__main__":
    main()
