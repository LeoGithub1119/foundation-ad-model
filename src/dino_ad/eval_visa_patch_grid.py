from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from dino_ad.data import VisaImageDataset, load_visa_split, summarize_samples
from dino_ad.metrics import compute_binary_metrics_from_scores
from dino_ad.train_visa import DINOClassifier, collate_batch, limit_per_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisA patch-grid localization from patch-token logits.")
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
    parser.add_argument("--write-rows", action="store_true", help="Write one CSV row per patch. Slower; disabled by default.")
    return parser.parse_args()


def format_metric(value: object) -> str:
    if not isinstance(value, float) or math.isnan(value):
        return "nan"
    return f"{value * 100:.2f}"


def load_checkpoint_model(args: argparse.Namespace) -> DINOClassifier:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    head_type = checkpoint.get("head_type", "linear")
    feature = checkpoint.get("feature", "cls")
    if not str(feature).startswith("patch_"):
        raise ValueError(f"Patch-grid evaluation requires a patch feature checkpoint, got feature={feature!r}")

    top_k = int(checkpoint.get("top_k", 6))
    patch_projector_depth = int(checkpoint.get("patch_projector_depth", 6))
    patch_projector_heads = int(checkpoint.get("patch_projector_heads", 8))
    patch_projector_dropout = float(checkpoint.get("patch_projector_dropout", 0.1))
    hidden_size = int(checkpoint.get("hidden_size", 768))

    encoder = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    model = DINOClassifier(
        encoder=encoder,
        hidden_size=hidden_size,
        head=head_type,
        feature=feature,
        top_k=top_k,
        patch_projector_depth=patch_projector_depth,
        patch_projector_heads=patch_projector_heads,
        patch_projector_dropout=patch_projector_dropout,
    )
    if model.patch_projector is not None and checkpoint.get("patch_projector") is not None:
        model.patch_projector.load_state_dict(checkpoint["patch_projector"])
    model.head.load_state_dict(checkpoint["head"])
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    return model


def mask_to_grid(mask_path: str, grid_size: int) -> list[int]:
    if not mask_path:
        return [0] * (grid_size * grid_size)
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        mask = mask.resize((grid_size, grid_size), resampling)
        return [1 if value > 0 else 0 for value in mask.getdata()]


def grouped_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)

    metrics = {}
    for category in sorted(groups):
        category_rows = groups[category]
        labels = [int(row["label"]) for row in category_rows]
        scores = [float(row["score"]) for row in category_rows]
        metrics[category] = compute_binary_metrics_from_scores(labels, scores)
    return metrics


def write_markdown(path: Path, overall: dict[str, object], by_category: dict[str, dict[str, object]], title: str) -> None:
    lines = [
        f"# {title} Patch-Grid Localization 指標",
        "",
        "這是 DINO patch-token grid 上的 localization proxy：將 VisA mask resize 到 patch grid 後計算指標。",
        "它用來快速檢查 patch score 是否對到瑕疵區域，尚不等同 publication-grade full-resolution pixel metric。",
        "",
        "## Overall 結果",
        "",
        "| Patch AUROC | Patch AUPRC | F1 @ 0.5 | F1max | F1max Threshold |",
        "| --- | --- | --- | --- | --- |",
        f"| {format_metric(overall['auroc'])} | {format_metric(overall['auprc'])} | {format_metric(overall['f1_at_0_5'])} | {format_metric(overall['best_f1'])} | {overall['best_f1_threshold']:.4f} |",
        "",
        "## 各類別結果",
        "",
        "| 類別 | Patch AUROC | Patch AUPRC | F1max | Normal Pixels | Anomaly Pixels |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category, metrics in by_category.items():
        lines.append(
            f"| {category} | {format_metric(metrics['auroc'])} | {format_metric(metrics['auprc'])} | "
            f"{format_metric(metrics['best_f1'])} | {metrics['num_normal']} | {metrics['num_anomaly']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["category", "image_path", "mask_path", "patch_index", "label", "score"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[list[dict[str, object]], int]:
    model.eval()
    rows: list[dict[str, object]] = []
    grid_size = -1
    with torch.no_grad():
        for batch in tqdm(loader, desc="patch-grid eval", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                patch_logits = model.forward_patch_logits(pixel_values)
            patch_scores = torch.sigmoid(patch_logits.detach().cpu().float())
            num_patches = int(patch_scores.shape[1])
            grid_size = int(num_patches**0.5)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"Expected square patch grid, got {num_patches} patches")

            for category, image_path, mask_path, scores in zip(
                batch["categories"],
                batch["paths"],
                batch["mask_paths"],
                patch_scores.tolist(),
            ):
                labels = mask_to_grid(mask_path, grid_size)
                for patch_index, (label, score) in enumerate(zip(labels, scores)):
                    rows.append(
                        {
                            "category": category,
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "patch_index": patch_index,
                            "label": label,
                            "score": score,
                        }
                    )
    return rows, grid_size


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    model = load_checkpoint_model(args)
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
    rows, grid_size = evaluate(model, loader, device, args.amp)
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    overall = compute_binary_metrics_from_scores(labels, scores)
    overall["grid_size"] = grid_size
    by_category = grouped_metrics(rows)

    overall_path = args.output_dir / f"patch_grid_overall_{args.split}.json"
    category_path = args.output_dir / f"patch_grid_by_category_{args.split}.json"
    markdown_path = args.output_dir / f"patch_grid_by_category_{args.split}.md"

    rows_path = None
    if args.write_rows:
        rows_path = args.output_dir / f"patch_grid_predictions_{args.split}.csv"
        write_rows(rows_path, rows)
    overall_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    category_path.write_text(json.dumps(by_category, indent=2, ensure_ascii=False))
    write_markdown(markdown_path, overall, by_category, args.output_dir.name)

    print("[PATCH_GRID_OVERALL]", json.dumps(overall, ensure_ascii=False))
    print("[DONE]", json.dumps({
        "rows": str(rows_path) if rows_path is not None else "",
        "overall": str(overall_path),
        "by_category": str(category_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
