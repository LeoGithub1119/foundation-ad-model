from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from dino_ad.data import VisaSample, load_visa_split, summarize_samples
from dino_ad.metrics import compute_binary_metrics_from_scores, histogram_binary_metrics


@dataclass(frozen=True)
class EvalSample:
    category: str
    split: str
    label_name: str
    image_path: Path
    mask_path: Path | None

    @property
    def target(self) -> int:
        return 0 if self.label_name.lower() in {"normal", "good"} else 1


class ImageDataset(Dataset):
    def __init__(self, samples: list[EvalSample], image_processor):
        self.samples = samples
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            encoded = self.image_processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "label": float(sample.target),
            "label_name": sample.label_name,
            "category": sample.category,
            "path": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else "",
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "label_names": [item["label_name"] for item in batch],
        "categories": [item["category"] for item in batch],
        "paths": [item["path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 normal memory-bank anomaly decoder.")
    parser.add_argument("--dataset", choices=["visa", "mvtec"], default=os.environ.get("MEMORY_DATASET", "visa"))
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "data/VisA")))
    parser.add_argument("--split-csv", type=Path, default=Path(os.environ.get("SPLIT_CSV", "split_csv/2cls_highshot.csv")))
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", "models/dinov3-vitb16-pretrain-lvd1689m")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUT_DIR", "outputs/dino_memory_bank")))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-bank-patches", type=int, default=int(os.environ.get("MAX_BANK_PATCHES", "20000")))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "6")))
    parser.add_argument("--eval-size", type=int, default=int(os.environ.get("HEATMAP_EVAL_SIZE", "224")))
    parser.add_argument("--hist-bins", type=int, default=int(os.environ.get("HIST_BINS", "4096")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--distance-chunk-size", type=int, default=int(os.environ.get("DISTANCE_CHUNK_SIZE", "1024")))
    return parser.parse_args()


def visa_to_eval(sample: VisaSample) -> EvalSample:
    return EvalSample(
        category=sample.category,
        split=sample.split,
        label_name=sample.label_name,
        image_path=sample.image_path,
        mask_path=sample.mask_path,
    )


def load_visa_samples(args: argparse.Namespace) -> tuple[list[EvalSample], list[EvalSample]]:
    train = [
        visa_to_eval(sample)
        for sample in load_visa_split(args.dataset_root, args.split_csv, split="train", categories=args.categories)
        if sample.target == 0
    ]
    test = [
        visa_to_eval(sample)
        for sample in load_visa_split(args.dataset_root, args.split_csv, split="test", categories=args.categories)
    ]
    return train[: args.max_train_samples], test[: args.max_test_samples]


def load_mvtec_samples(args: argparse.Namespace) -> tuple[list[EvalSample], list[EvalSample]]:
    root = args.dataset_root.expanduser().resolve()
    categories = sorted([path.name for path in root.iterdir() if path.is_dir()])
    if args.categories:
        wanted = set(args.categories)
        categories = [category for category in categories if category in wanted]

    train: list[EvalSample] = []
    test: list[EvalSample] = []
    for category in categories:
        train_dir = root / category / "train" / "good"
        if train_dir.exists():
            for image_path in sorted(train_dir.glob("*.png")):
                train.append(EvalSample(category, "train", "good", image_path, None))

        test_root = root / category / "test"
        if not test_root.exists():
            continue
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            label = "good" if defect_dir.name == "good" else defect_dir.name
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if label != "good":
                    candidate = root / category / "ground_truth" / label / f"{image_path.stem}_mask.png"
                    if candidate.exists():
                        mask_path = candidate
                test.append(EvalSample(category, "test", label, image_path, mask_path))

    return train[: args.max_train_samples], test[: args.max_test_samples]


def summarize_eval_samples(samples: Iterable[EvalSample]) -> dict[str, object]:
    by_label = {"normal": 0, "anomaly": 0}
    by_category: dict[str, dict[str, int]] = {}
    for sample in samples:
        label = "normal" if sample.target == 0 else "anomaly"
        by_label[label] += 1
        by_category.setdefault(sample.category, {"normal": 0, "anomaly": 0})
        by_category[sample.category][label] += 1
    return {"total": sum(by_label.values()), "by_label": by_label, "by_category": by_category}


def extract_patch_features(encoder, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = encoder(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state
    num_register_tokens = getattr(encoder.config, "num_register_tokens", 0)
    patch_start = 1 + int(num_register_tokens)
    return tokens[:, patch_start:]


def build_memory_bank(
    encoder,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_bank_patches: int,
    seed: int,
) -> torch.Tensor:
    features = []
    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="build memory bank", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                patch_features = extract_patch_features(encoder, pixel_values)
            patch_features = F.normalize(patch_features.detach().cpu().float(), dim=-1)
            features.append(patch_features.reshape(-1, patch_features.shape[-1]))

    bank = torch.cat(features, dim=0)
    if bank.shape[0] > max_bank_patches:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(bank.shape[0], generator=generator)[:max_bank_patches]
        bank = bank[indices]
    return bank.contiguous()


def nearest_distances(features: torch.Tensor, bank: torch.Tensor, chunk_size: int, device: torch.device) -> torch.Tensor:
    bank = bank.to(device)
    distances = []
    with torch.no_grad():
        for start in range(0, features.shape[0], chunk_size):
            chunk = features[start : start + chunk_size].to(device)
            sims = chunk @ bank.T
            dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sims.max(dim=1).values, min=0.0))
            distances.append(dist.cpu())
    return torch.cat(distances, dim=0)


def load_mask(mask_path: str, eval_size: int) -> torch.Tensor:
    if not mask_path:
        return torch.zeros((eval_size, eval_size), dtype=torch.bool)
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        resampling = getattr(Image, "Resampling", Image).NEAREST
        mask = mask.resize((eval_size, eval_size), resampling)
        values = torch.tensor(list(mask.getdata()), dtype=torch.uint8).reshape(eval_size, eval_size)
        return values > 0


def add_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> None:
    scores = scores.detach().cpu().float().clamp(0.0, 1.0)
    labels = labels.detach().cpu().bool()
    if labels.any():
        pos_hist += torch.histc(scores[labels], bins=pos_hist.numel(), min=0.0, max=1.0).long()
    if (~labels).any():
        neg_hist += torch.histc(scores[~labels], bins=neg_hist.numel(), min=0.0, max=1.0).long()


def format_metric(value: object) -> str:
    if not isinstance(value, float) or math.isnan(value):
        return "nan"
    return f"{value * 100:.2f}"


def write_markdown(
    path: Path,
    image_metrics: dict[str, object],
    dense_metrics: dict[str, object],
    title: str,
    dataset: str,
) -> None:
    lines = [
        f"# {title} Memory-Bank Decoder 指標",
        "",
        f"Dataset: `{dataset}`. DINOv3 frozen features, normal-only memory bank, nearest-normal patch distance.",
        "",
        "| Image AUROC | Image AUPRC | Image F1max | Dense AUROC | Dense AUPRC | Dense F1max |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {format_metric(image_metrics['auroc'])} | {format_metric(image_metrics['auprc'])} | {format_metric(image_metrics['best_f1'])} | "
        f"{format_metric(dense_metrics['auroc'])} | {format_metric(dense_metrics['auprc'])} | {format_metric(dense_metrics['best_f1'])} |",
    ]
    path.write_text("\n".join(lines) + "\n")


def evaluate(
    encoder,
    loader: DataLoader,
    bank: torch.Tensor,
    device: torch.device,
    amp: bool,
    top_k: int,
    eval_size: int,
    hist_bins: int,
    distance_chunk_size: int,
) -> tuple[dict[str, object], dict[str, object]]:
    image_labels: list[int] = []
    image_scores: list[float] = []
    pos_hist = torch.zeros(hist_bins, dtype=torch.long)
    neg_hist = torch.zeros(hist_bins, dtype=torch.long)
    bank = bank.to(device)

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="memory bank eval", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                patch_features = extract_patch_features(encoder, pixel_values)
            patch_features = F.normalize(patch_features.detach().cpu().float(), dim=-1)
            batch_size, num_patches, hidden_size = patch_features.shape
            grid_size = int(num_patches**0.5)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"Expected square patch grid, got {num_patches} patches")

            distances = nearest_distances(
                patch_features.reshape(-1, hidden_size),
                bank=bank,
                chunk_size=distance_chunk_size,
                device=device,
            ).reshape(batch_size, num_patches)
            dense_scores = F.interpolate(
                distances.reshape(batch_size, 1, grid_size, grid_size),
                size=(eval_size, eval_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            # L2-normalized features have cosine distance in [0, 2], giving a stable histogram scale.
            dense_scores_norm = (dense_scores / 2.0).clamp(0.0, 1.0)

            k = min(max(int(top_k), 1), num_patches)
            batch_image_scores = distances.topk(k, dim=1).values.mean(dim=1).tolist()
            for label, mask_path, image_score, dense_score in zip(
                [int(x) for x in batch["labels"].tolist()],
                batch["mask_paths"],
                batch_image_scores,
                dense_scores_norm,
            ):
                image_labels.append(label)
                image_scores.append(float(image_score))
                mask = load_mask(mask_path, eval_size)
                add_hist(pos_hist, neg_hist, dense_score, mask)

    return (
        compute_binary_metrics_from_scores(image_labels, image_scores),
        histogram_binary_metrics(pos_hist.tolist(), neg_hist.tolist()),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == "visa":
        train_samples, test_samples = load_visa_samples(args)
    else:
        train_samples, test_samples = load_mvtec_samples(args)
    print("[DATA] train", json.dumps(summarize_eval_samples(train_samples), ensure_ascii=False))
    print("[DATA] test", json.dumps(summarize_eval_samples(test_samples), ensure_ascii=False))

    image_processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    train_loader = DataLoader(
        ImageDataset(train_samples, image_processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        ImageDataset(test_samples, image_processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    device = torch.device(args.device)
    encoder = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    bank = build_memory_bank(
        encoder=encoder,
        loader=train_loader,
        device=device,
        amp=args.amp,
        max_bank_patches=args.max_bank_patches,
        seed=args.seed,
    )
    bank_path = args.output_dir / "memory_bank_summary.json"
    bank_path.write_text(json.dumps({
        "dataset": args.dataset,
        "num_bank_patches": int(bank.shape[0]),
        "hidden_size": int(bank.shape[1]),
        "max_bank_patches": args.max_bank_patches,
        "top_k": args.top_k,
        "eval_size": args.eval_size,
        "hist_bins": args.hist_bins,
    }, indent=2, ensure_ascii=False))

    image_metrics, dense_metrics = evaluate(
        encoder=encoder,
        loader=test_loader,
        bank=bank,
        device=device,
        amp=args.amp,
        top_k=args.top_k,
        eval_size=args.eval_size,
        hist_bins=args.hist_bins,
        distance_chunk_size=args.distance_chunk_size,
    )
    dense_metrics["eval_size"] = args.eval_size
    dense_metrics["hist_bins"] = args.hist_bins
    dense_metrics["num_bank_patches"] = int(bank.shape[0])

    image_path = args.output_dir / f"memory_bank_image_overall_{args.dataset}.json"
    dense_path = args.output_dir / f"memory_bank_dense_overall_{args.dataset}.json"
    markdown_path = args.output_dir / f"memory_bank_summary_{args.dataset}.md"
    image_path.write_text(json.dumps(image_metrics, indent=2, ensure_ascii=False))
    dense_path.write_text(json.dumps(dense_metrics, indent=2, ensure_ascii=False))
    write_markdown(markdown_path, image_metrics, dense_metrics, args.output_dir.name, args.dataset)

    print("[IMAGE_OVERALL]", json.dumps(image_metrics, ensure_ascii=False))
    print("[DENSE_OVERALL]", json.dumps(dense_metrics, ensure_ascii=False))
    print("[DONE]", json.dumps({
        "bank": str(bank_path),
        "image": str(image_path),
        "dense": str(dense_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
