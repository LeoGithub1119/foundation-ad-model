from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from dino_ad.data import VisaImageDataset, load_visa_split, summarize_samples
from dino_ad.metrics import compute_binary_metrics_from_scores, sigmoid
from dino_ad.train_visa import DINOClassifier, collate_batch, limit_per_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DINOv3 VisA image-level AD checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "data/VisA")))
    parser.add_argument("--split-csv", type=Path, default=Path("split_csv/2cls_highshot.csv"))
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", "models/dinov3-vitb16-pretrain-lvd1689m")))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("CHECKPOINT", "outputs/dino_visa_a0_linear/best_head.pt")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUT_DIR", "outputs/dino_visa_a0_linear")))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-label", type=int, default=None)
    parser.add_argument("--categories", nargs="*", default=None)
    return parser.parse_args()


def format_metric(value: object) -> str:
    if not isinstance(value, float) or math.isnan(value):
        return "nan"
    return f"{value * 100:.2f}"


def write_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    fieldnames = ["category", "path", "label", "label_name", "logit", "score", "mask_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: row[key] for key in fieldnames})


def grouped_metrics(predictions: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        groups[str(row["category"])].append(row)

    metrics = {}
    for category in sorted(groups):
        rows = groups[category]
        labels = [int(row["label"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        metrics[category] = compute_binary_metrics_from_scores(labels, scores)
    return metrics


def write_markdown_table(path: Path, overall: dict[str, object], by_category: dict[str, dict[str, object]]) -> None:
    lines = [
        "# EXP-001 Image-Level Per-Category 指標",
        "",
        "表內數值為百分比。`F1max` 是在此 split 上掃過 score threshold 後得到的最佳 F1。",
        "",
        "## Overall 結果",
        "",
        "| AUROC | AUPRC | F1 @ 0.5 | F1max | F1max Threshold |",
        "| --- | --- | --- | --- | --- |",
        f"| {format_metric(overall['auroc'])} | {format_metric(overall['auprc'])} | {format_metric(overall['f1_at_0_5'])} | {format_metric(overall['best_f1'])} | {overall['best_f1_threshold']:.4f} |",
        "",
        "## 各類別結果",
        "",
        "| 類別 | AUROC | AUPRC | F1max | F1max Threshold | Normal | Anomaly |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for category, metrics in by_category.items():
        lines.append(
            f"| {category} | {format_metric(metrics['auroc'])} | {format_metric(metrics['auprc'])} | "
            f"{format_metric(metrics['best_f1'])} | {metrics['best_f1_threshold']:.4f} | "
            f"{metrics['num_normal']} | {metrics['num_anomaly']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> list[dict[str, object]]:
    model.eval()
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(pixel_values)
            batch_logits = logits.detach().cpu().float().tolist()
            labels = [int(x) for x in batch["labels"].tolist()]
            label_names = batch.get("label_names") or ["anomaly" if label else "normal" for label in labels]
            mask_paths = batch.get("mask_paths") or [""] * len(labels)
            for category, path, label, label_name, mask_path, logit in zip(
                batch["categories"],
                batch["paths"],
                labels,
                label_names,
                mask_paths,
                batch_logits,
            ):
                predictions.append(
                    {
                        "category": category,
                        "path": path,
                        "label": label,
                        "label_name": label_name,
                        "logit": logit,
                        "score": sigmoid(logit),
                        "mask_path": mask_path,
                    }
                )
    return predictions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    head_type = checkpoint.get("head_type", "linear")
    feature = checkpoint.get("feature", "cls")
    hidden_size = int(checkpoint.get("hidden_size", 768))

    image_processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    model = DINOClassifier(encoder=encoder, hidden_size=hidden_size, head=head_type, feature=feature)
    model.head.load_state_dict(checkpoint["head"])
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()

    device = torch.device(args.device)
    model.to(device)

    samples = load_visa_split(
        args.dataset_root,
        args.split_csv,
        split=args.split,
        categories=args.categories,
        max_samples=args.max_samples,
    )
    samples = limit_per_label(samples, args.max_samples_per_label)
    print("[DATA]", args.split, json.dumps(summarize_samples(samples), ensure_ascii=False))

    loader = DataLoader(
        VisaImageDataset(samples, image_processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    predictions = evaluate(model, loader, device, args.amp)
    labels = [int(row["label"]) for row in predictions]
    scores = [float(row["score"]) for row in predictions]
    overall = compute_binary_metrics_from_scores(labels, scores)
    by_category = grouped_metrics(predictions)

    predictions_path = args.output_dir / f"predictions_{args.split}.csv"
    overall_path = args.output_dir / f"image_level_overall_{args.split}.json"
    category_path = args.output_dir / f"image_level_by_category_{args.split}.json"
    markdown_path = args.output_dir / f"image_level_by_category_{args.split}.md"

    write_predictions(predictions_path, predictions)
    overall_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    category_path.write_text(json.dumps(by_category, indent=2, ensure_ascii=False))
    write_markdown_table(markdown_path, overall, by_category)

    print("[OVERALL]", json.dumps(overall, ensure_ascii=False))
    print("[DONE]", json.dumps({
        "predictions": str(predictions_path),
        "overall": str(overall_path),
        "by_category": str(category_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
