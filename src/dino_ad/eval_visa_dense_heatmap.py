from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from dino_ad.data import VisaImageDataset, load_visa_split, summarize_samples
from dino_ad.metrics import compute_binary_metrics_from_scores, histogram_binary_metrics
from dino_ad.train_visa import DINOClassifier, collate_batch, limit_per_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense VisA heatmaps from patch-token logits.")
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
    parser.add_argument("--eval-size", type=int, default=int(os.environ.get("HEATMAP_EVAL_SIZE", "224")))
    parser.add_argument("--hist-bins", type=int, default=int(os.environ.get("HIST_BINS", "4096")))
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
        raise ValueError(f"Dense heatmap evaluation requires a patch feature checkpoint, got feature={feature!r}")

    encoder = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    model = DINOClassifier(
        encoder=encoder,
        hidden_size=int(checkpoint.get("hidden_size", 768)),
        head=head_type,
        feature=feature,
        top_k=int(checkpoint.get("top_k", 6)),
        patch_projector_depth=int(checkpoint.get("patch_projector_depth", 6)),
        patch_projector_heads=int(checkpoint.get("patch_projector_heads", 8)),
        patch_projector_dropout=float(checkpoint.get("patch_projector_dropout", 0.1)),
    )
    if model.patch_projector is not None and checkpoint.get("patch_projector") is not None:
        model.patch_projector.load_state_dict(checkpoint["patch_projector"])
    model.head.load_state_dict(checkpoint["head"])
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()
    return model


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


def grouped_hist_metrics(group_hists: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, dict[str, object]]:
    metrics = {}
    for category in sorted(group_hists):
        pos_hist, neg_hist = group_hists[category]
        metrics[category] = histogram_binary_metrics(pos_hist.tolist(), neg_hist.tolist())
    return metrics


def write_markdown(
    path: Path,
    image_metrics: dict[str, object],
    dense_metrics: dict[str, object],
    by_category: dict[str, dict[str, object]],
    title: str,
    eval_size: int,
) -> None:
    lines = [
        f"# {title} Dense Heatmap 指標",
        "",
        f"Heatmap 由 DINO patch logits bilinear upsample 到 `{eval_size}x{eval_size}` 後計算。",
        "此指標比 14x14 patch-grid proxy 更接近 pixel-level，但仍不是原始解析度 full-sort metric。",
        "",
        "## Overall 結果",
        "",
        "| Image AUROC | Image AUPRC | Image F1max | Dense AUROC | Dense AUPRC | Dense F1max |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {format_metric(image_metrics['auroc'])} | {format_metric(image_metrics['auprc'])} | {format_metric(image_metrics['best_f1'])} | "
        f"{format_metric(dense_metrics['auroc'])} | {format_metric(dense_metrics['auprc'])} | {format_metric(dense_metrics['best_f1'])} |",
        "",
        "## 各類別 Dense 結果",
        "",
        "| 類別 | Dense AUROC | Dense AUPRC | Dense F1max | Normal Pixels | Anomaly Pixels |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category, metrics in by_category.items():
        lines.append(
            f"| {category} | {format_metric(metrics['auroc'])} | {format_metric(metrics['auprc'])} | "
            f"{format_metric(metrics['best_f1'])} | {metrics['num_normal']} | {metrics['num_anomaly']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    eval_size: int,
    hist_bins: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    model.eval()
    image_labels: list[int] = []
    image_scores: list[float] = []
    pos_hist = torch.zeros(hist_bins, dtype=torch.long)
    neg_hist = torch.zeros(hist_bins, dtype=torch.long)
    group_hists: dict[str, tuple[torch.Tensor, torch.Tensor]] = defaultdict(
        lambda: (torch.zeros(hist_bins, dtype=torch.long), torch.zeros(hist_bins, dtype=torch.long))
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="dense heatmap eval", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                patch_logits = model.forward_patch_logits(pixel_values)
            patch_scores = torch.sigmoid(patch_logits.detach().cpu().float())
            num_patches = int(patch_scores.shape[1])
            grid_size = int(num_patches**0.5)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"Expected square patch grid, got {num_patches} patches")

            dense_scores = F.interpolate(
                patch_scores.reshape(-1, 1, grid_size, grid_size),
                size=(eval_size, eval_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            top_k = min(max(int(getattr(model, "top_k", 6)), 1), num_patches)
            batch_image_scores = torch.sigmoid(
                patch_logits.detach().cpu().float().topk(top_k, dim=1).values.mean(dim=1)
            ).tolist()

            for category, label, mask_path, image_score, dense_score in zip(
                batch["categories"],
                [int(x) for x in batch["labels"].tolist()],
                batch["mask_paths"],
                batch_image_scores,
                dense_scores,
            ):
                mask = load_mask(mask_path, eval_size)
                image_labels.append(label)
                image_scores.append(float(image_score))
                add_hist(pos_hist, neg_hist, dense_score, mask)
                group_pos, group_neg = group_hists[str(category)]
                add_hist(group_pos, group_neg, dense_score, mask)

    image_metrics = compute_binary_metrics_from_scores(image_labels, image_scores)
    dense_metrics = histogram_binary_metrics(pos_hist.tolist(), neg_hist.tolist())
    by_category = grouped_hist_metrics(group_hists)
    return image_metrics, dense_metrics, by_category


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
    image_metrics, dense_metrics, by_category = evaluate(
        model=model,
        loader=loader,
        device=device,
        amp=args.amp,
        eval_size=args.eval_size,
        hist_bins=args.hist_bins,
    )
    dense_metrics["eval_size"] = args.eval_size
    dense_metrics["hist_bins"] = args.hist_bins

    image_path = args.output_dir / f"dense_heatmap_image_overall_{args.split}.json"
    dense_path = args.output_dir / f"dense_heatmap_overall_{args.split}.json"
    category_path = args.output_dir / f"dense_heatmap_by_category_{args.split}.json"
    markdown_path = args.output_dir / f"dense_heatmap_by_category_{args.split}.md"

    image_path.write_text(json.dumps(image_metrics, indent=2, ensure_ascii=False))
    dense_path.write_text(json.dumps(dense_metrics, indent=2, ensure_ascii=False))
    category_path.write_text(json.dumps(by_category, indent=2, ensure_ascii=False))
    write_markdown(markdown_path, image_metrics, dense_metrics, by_category, args.output_dir.name, args.eval_size)

    print("[IMAGE_OVERALL]", json.dumps(image_metrics, ensure_ascii=False))
    print("[DENSE_HEATMAP_OVERALL]", json.dumps(dense_metrics, ensure_ascii=False))
    print("[DONE]", json.dumps({
        "image": str(image_path),
        "dense": str(dense_path),
        "by_category": str(category_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
