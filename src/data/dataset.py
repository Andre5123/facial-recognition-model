"""PyTorch Dataset for the CASIA-WebFace identity-disjoint split.

Reads the manifests produced by scripts/make_identity_split.py
(image_path,label CSVs). Images are already aligned to a fixed size (112x112
for the Kaggle webface-112x112 mirror -- see docs/DATASET.md), so no face
detection/alignment happens here, only normalization and light augmentation.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size: int, train: bool, horizontal_flip: bool) -> transforms.Compose:
    ops = []
    if image_size != 112:
        ops.append(transforms.Resize((image_size, image_size)))
    if train and horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.append(transforms.ToTensor())  # -> [0, 1]
    ops.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # -> [-1, 1]
    return transforms.Compose(ops)


class FaceDataset(Dataset):
    """Loads (image, integer label) pairs from a train.csv/val_seen.csv manifest."""

    def __init__(self, manifest_csv: str | Path, transform: transforms.Compose):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image_path"], int(row["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self) -> int:
        return max(label for _, label in self.samples) + 1
