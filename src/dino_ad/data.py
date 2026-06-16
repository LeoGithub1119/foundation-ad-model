from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VisaSample:
    category: str
    split: str
    label_name: str
    image_path: Path
    mask_path: Path | None

    @property
    def target(self) -> int:
        return 0 if self.label_name.lower() == "normal" else 1


def load_visa_split(
    dataset_root: Path,
    split_csv: Path,
    split: str,
    categories: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> list[VisaSample]:
    dataset_root = dataset_root.expanduser().resolve()
    split_csv = split_csv if split_csv.is_absolute() else dataset_root / split_csv
    wanted_categories = set(categories or [])
    samples: list[VisaSample] = []

    with split_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            if wanted_categories and row["object"] not in wanted_categories:
                continue

            image_path = dataset_root / row["image"]
            mask_path = dataset_root / row["mask"] if row.get("mask") else None
            samples.append(
                VisaSample(
                    category=row["object"],
                    split=row["split"],
                    label_name=row["label"],
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

    missing = [str(sample.image_path) for sample in samples if not sample.image_path.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} images. First missing paths:\n{preview}")

    return samples


def summarize_samples(samples: list[VisaSample]) -> dict[str, object]:
    by_label = {"normal": 0, "anomaly": 0}
    by_category: dict[str, dict[str, int]] = {}
    for sample in samples:
        label_key = "normal" if sample.target == 0 else "anomaly"
        by_label[label_key] += 1
        by_category.setdefault(sample.category, {"normal": 0, "anomaly": 0})
        by_category[sample.category][label_key] += 1
    return {
        "total": len(samples),
        "by_label": by_label,
        "by_category": by_category,
    }


class VisaImageDataset(Dataset):
    def __init__(self, samples: list[VisaSample], image_processor):
        self.samples = samples
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            encoded = self.image_processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "label": float(sample.target),
            "label_name": sample.label_name,
            "category": sample.category,
            "path": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else "",
        }
