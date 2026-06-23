# EXP-002/003 Patch Component Comparison

本頁整理固定 DINOv3 encoder 後的 downstream anomaly component 實驗。目的不是再證明 DINOv3 是不是最佳 encoder，也不是完整復現特定 paper，而是在相同 VisA image-level supervised protocol 下，建立可比較的 component baseline。

## Controlled Setup

| 項目 | 設定 |
| --- | --- |
| Dataset | VisA `split_csv/2cls_highshot.csv` |
| Train set | 6,493 張影像，5,773 normal / 720 anomaly |
| Test set | 4,328 張影像，3,848 normal / 480 anomaly |
| Encoder | DINOv3 `facebook/dinov3-vitb16-pretrain-lvd1689m`，固定不訓練 |
| Metrics | Image-level AUROC / AUPRC / F1 @ 0.5 / F1max |
| Top-K | `K=6` for patch-token aggregation |

## Component Comparison

表內數值為百分比。`EXP-002` 是 patch-token Top-K control；`EXP-003` 在相同 encoder / split / Top-K aggregation 下加入 ViT projector，測試 token interaction module 的收益。

| ID | Feature / Module | LR | Batch | AUROC | AUPRC | F1 @ 0.5 | F1max | Δ AUPRC vs EXP-001 | 結論 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | CLS token + linear head | `1e-3` | 16 | 90.99 | 69.17 | 53.92 | 61.78 | baseline | 最小可用 baseline |
| EXP-002 | Patch head + Top-K=6 | `1e-3` | 16 | 90.82 | 70.22 | 55.36 | 62.84 | +1.05 | patch evidence 有用，但類別穩定性不足 |
| EXP-003 | 6-layer ViT projector + Top-K=6 | `1e-4` | 8 | 96.44 | 87.41 | 79.91 | 80.78 | +18.24 | 目前最佳 image-level component |

## Per-Category AUPRC

| 類別 | EXP-001 CLS-linear | EXP-002 Patch Top-K | EXP-003 ViT Top-K |
| --- | --- | --- | --- |
| candle | 78.81 | 85.56 | 95.22 |
| capsules | 76.48 | 66.24 | 73.04 |
| cashew | 89.71 | 93.53 | 94.01 |
| chewinggum | 96.72 | 94.56 | 95.86 |
| fryum | 80.25 | 84.54 | 95.24 |
| macaroni1 | 47.78 | 56.21 | 93.24 |
| macaroni2 | 55.37 | 52.28 | 81.50 |
| pcb1 | 60.06 | 69.54 | 86.84 |
| pcb2 | 55.63 | 19.43 | 76.19 |
| pcb3 | 63.66 | 42.14 | 73.37 |
| pcb4 | 80.92 | 83.72 | 91.54 |
| pipe_fryum | 95.64 | 96.03 | 97.79 |

## Interpretation

`EXP-002` is the minimal patch-token control. It removes CLS pooling and scores patch tokens directly with Top-K aggregation. The result only slightly improves over the CLS baseline, and some categories such as `pcb2` and `pcb3` get worse. This means patch-token evidence alone is not enough.

`EXP-003` adds a ViT projector before patch scoring. This is comparable to `EXP-002` as a downstream component comparison because both use the same DINOv3 encoder, VisA split, patch tokens, and Top-K aggregation. It is not a strict one-variable experiment because the projector introduces more trainable parameters and uses a different stable LR / batch size.

The main conclusion is that token interaction matters. `EXP-003` improves over `EXP-001` by +5.45 AUROC, +18.24 AUPRC, and +19.00 F1max points, and improves over `EXP-002` by +17.19 AUPRC and +17.94 F1max points. This supports the component direction: keep DINOv3 fixed, use patch tokens, and add a projector/token-interaction module.

## Research Value

These three experiments give a concrete baseline chain:

1. `EXP-001` establishes that frozen DINOv3 already gives a usable supervised image-level AD baseline on VisA, so DINOv3 can remain the fixed encoder for the next stage.
2. `EXP-002` tests the simplest spatial alternative: score patch tokens directly and aggregate the most suspicious patches with Top-K. The small and unstable gain shows that patch tokens alone are not sufficient.
3. `EXP-003` adds a ViT projector before Top-K scoring. The large improvement suggests that anomaly detection benefits from a trainable token-interaction module on top of frozen DINOv3 features.

The defensible research claim at this stage is: under the same DINOv3 / VisA image-level supervised setup, the main value is not just switching from CLS to patch tokens, but adding an anomaly-specific projector that can re-organize patch-token evidence before image-level scoring.

This is not yet a final paper-level claim. The current evidence is image-level, single-split, and supervised. The next required step is pixel-level heatmap evaluation, because `EXP-003` naturally produces patch scores that can be upsampled into localization maps.

## Output Artifacts

| Run | Output |
| --- | --- |
| EXP-002 | `WORK_DIR/outputs/dino_visa_a1_patch_topk` |
| EXP-003 | `WORK_DIR/outputs/dino_visa_a2_patch_vit_topk_lr1e4` |

## Next Step

Use `EXP-003` as the current image-level component baseline. The next experiment should convert patch logits into heatmaps and evaluate VisA pixel-level metrics against masks. This is the bridge from image-level supervised AD toward localization and FoundAD-style anomaly scoring.
