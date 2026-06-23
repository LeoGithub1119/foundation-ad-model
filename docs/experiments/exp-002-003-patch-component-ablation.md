# EXP-002/003 Patch Component Ablation

本頁整理固定 DINOv3 encoder 後的 downstream anomaly component 實驗。目的不是再證明 DINOv3 是不是最佳 encoder，而是比較不同方式如何把 DINOv3 features 轉成 anomaly score。

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

表內數值為百分比。`EXP-003a` 是同架構但 LR 過高的負結果；`EXP-003b` 是降低 LR 後的穩定結果。

| ID | Feature / Module | LR | Batch | AUROC | AUPRC | F1 @ 0.5 | F1max | Δ AUPRC vs EXP-001 | 結論 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | CLS token + linear head | `1e-3` | 16 | 90.99 | 69.17 | 53.92 | 61.78 | baseline | 最小可用 baseline |
| EXP-002 | Patch head + Top-K=6 | `1e-3` | 16 | 90.82 | 70.22 | 55.36 | 62.84 | +1.05 | patch evidence 有用，但類別穩定性不足 |
| EXP-003a | 6-layer ViT projector + Top-K=6 | `1e-3` | 8 | 66.41 | 23.35 | 1.65 | 27.34 | -45.82 | learning rate 過高導致訓練不穩 |
| EXP-003b | 6-layer ViT projector + Top-K=6 | `1e-4` | 8 | 96.44 | 87.41 | 79.91 | 80.78 | +18.24 | 目前最佳 image-level component |

## Per-Category AUPRC

| 類別 | EXP-001 CLS-linear | EXP-002 Patch Top-K | EXP-003b ViT Top-K |
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

`EXP-002` confirms that moving from global CLS features to patch-level evidence is useful, but a naive patch linear head can be unstable. It improves overall AUPRC/F1max, yet hurts `pcb2` and `pcb3`.

`EXP-003a` shows that a ViT projector with LR `1e-3` is not stable under this supervised BCE setup. The loss increases sharply and the output scores collapse for many samples.

`EXP-003b` shows the same ViT projector becomes effective after lowering LR to `1e-4`. It improves over `EXP-001` by +5.45 AUROC, +18.24 AUPRC, and +19.00 F1max points. This supports the component direction: keep DINOv3 fixed, use patch tokens, and add a projector/token-interaction module.

## Output Artifacts

| Run | Output |
| --- | --- |
| EXP-002 | `WORK_DIR/outputs/dino_visa_a1_patch_topk` |
| EXP-003a | `WORK_DIR/outputs/dino_visa_a2_patch_vit_topk` |
| EXP-003b | `WORK_DIR/outputs/dino_visa_a2_patch_vit_topk_lr1e4` |

## Next Step

Use `EXP-003b` as the current image-level component baseline. The next experiment should convert patch logits into heatmaps and evaluate VisA pixel-level metrics against masks. This is the bridge from image-level supervised AD toward localization and FoundAD-style anomaly scoring.
