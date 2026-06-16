from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from dino_ad.data import VisaImageDataset, load_visa_split, summarize_samples
from dino_ad.metrics import compute_binary_metrics


class DINOClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, head: str, feature: str):
        super().__init__()
        self.encoder = encoder
        self.feature = feature
        if head == "linear":
            self.head = nn.Linear(hidden_size, 1)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
            )
        else:
            raise ValueError(f"Unsupported head: {head}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        if self.feature == "cls":
            features = tokens[:, 0]
        elif self.feature == "mean_patch":
            # DINOv3 has CLS then register tokens before patch tokens.
            num_register_tokens = getattr(self.encoder.config, "num_register_tokens", 0)
            patch_start = 1 + int(num_register_tokens)
            features = tokens[:, patch_start:].mean(dim=1)
        else:
            raise ValueError(f"Unsupported feature: {self.feature}")
        return self.head(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINOv3 supervised VisA image-level AD baseline.")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "data/VisA")))
    parser.add_argument("--split-csv", type=Path, default=Path("split_csv/2cls_highshot.csv"))
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", "models/dinov3-vitb16-pretrain-lvd1689m")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUT_DIR", "outputs/dino_visa_a0_linear")))
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--feature", choices=["cls", "mean_patch"], default="cls")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-train-samples-per-label", type=int, default=None)
    parser.add_argument("--max-test-samples-per-label", type=int, default=None)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "label_names": [item["label_name"] for item in batch],
        "categories": [item["category"] for item in batch],
        "paths": [item["path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
    }


def limit_per_label(samples, max_per_label: int | None):
    if max_per_label is None:
        return samples
    counts = {0: 0, 1: 0}
    limited = []
    for sample in samples:
        target = sample.target
        if counts[target] >= max_per_label:
            continue
        limited.append(sample)
        counts[target] += 1
        if counts[0] >= max_per_label and counts[1] >= max_per_label:
            break
    return limited


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    logits: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            batch_logits = model(pixel_values)
            logits.extend(batch_logits.detach().cpu().float().tolist())
            labels.extend(int(x) for x in batch["labels"].tolist())
    return compute_binary_metrics(labels, logits)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    hidden_size = int(encoder.config.hidden_size)

    train_samples = load_visa_split(
        args.dataset_root,
        args.split_csv,
        split="train",
        categories=args.categories,
        max_samples=args.max_train_samples,
    )
    train_samples = limit_per_label(train_samples, args.max_train_samples_per_label)
    test_samples = load_visa_split(
        args.dataset_root,
        args.split_csv,
        split="test",
        categories=args.categories,
        max_samples=args.max_test_samples,
    )
    test_samples = limit_per_label(test_samples, args.max_test_samples_per_label)

    print("[DATA] train", json.dumps(summarize_samples(train_samples), ensure_ascii=False))
    print("[DATA] test", json.dumps(summarize_samples(test_samples), ensure_ascii=False))

    train_loader = DataLoader(
        VisaImageDataset(train_samples, image_processor),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        VisaImageDataset(test_samples, image_processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    model = DINOClassifier(encoder=encoder, hidden_size=hidden_size, head=args.head, feature=args.feature)
    if not args.unfreeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()

    device = torch.device(args.device)
    model.to(device)

    num_pos = sum(sample.target for sample in train_samples)
    num_neg = len(train_samples) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = vars(args).copy()
    config.update(
        {
            "dataset_root": str(args.dataset_root),
            "split_csv": str(args.split_csv),
            "model_path": str(args.model_path),
            "output_dir": str(args.output_dir),
            "hidden_size": hidden_size,
            "train_summary": summarize_samples(train_samples),
            "test_summary": summarize_samples(test_samples),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    history: list[dict[str, object]] = []
    best_auroc = -1.0
    best_path = args.output_dir / "best_head.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.unfreeze_encoder:
            model.encoder.eval()
        running_loss = 0.0
        num_seen = 0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(pixel_values)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels.shape[0])
            running_loss += float(loss.detach().cpu()) * batch_size
            num_seen += batch_size

        metrics = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(num_seen, 1),
            "test_metrics": metrics,
        }
        history.append(record)
        print("[EPOCH]", json.dumps(record, ensure_ascii=False))

        auroc = metrics.get("auroc")
        if isinstance(auroc, float) and auroc > best_auroc:
            best_auroc = auroc
            torch.save(
                {
                    "head": model.head.state_dict(),
                    "encoder_model_path": str(args.model_path),
                    "head_type": args.head,
                    "feature": args.feature,
                    "hidden_size": hidden_size,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                best_path,
            )

        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))

    print(f"[DONE] best_auroc={best_auroc:.6f} best_head={best_path}")


if __name__ == "__main__":
    main()
