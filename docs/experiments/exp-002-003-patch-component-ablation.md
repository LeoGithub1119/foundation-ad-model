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

The working hypothesis is that token interaction matters. `EXP-003` improves over `EXP-001` by +5.45 AUROC, +18.24 AUPRC, and +19.00 F1max points, and improves over `EXP-002` by +17.19 AUPRC and +17.94 F1max points. This supports the component direction: keep DINOv3 fixed, use patch tokens, and test projector/token-interaction modules more carefully.

## Research Value

These three experiments give a concrete baseline chain, but they should be treated as a direction-finding result, not a final proof:

1. `EXP-001` establishes that frozen DINOv3 already gives a usable supervised image-level AD baseline on VisA, so DINOv3 can remain the fixed encoder for the next stage.
2. `EXP-002` tests the simplest spatial alternative: score patch tokens directly and aggregate the most suspicious patches with Top-K. The small and unstable gain shows that patch tokens alone are not sufficient.
3. `EXP-003` adds a ViT projector before Top-K scoring. The large improvement makes projector-based patch scoring the strongest current candidate component, but does not by itself prove that ViT structure is the reason.

The defensible research claim at this stage is: under the same DINOv3 / VisA image-level supervised setup, simply switching from CLS to raw patch-token Top-K is not enough; a trainable patch-token projector is a promising direction and should become the next controlled study.

This is not yet a final paper-level claim. The current evidence is image-level, single-split, and supervised. The next required step is pixel-level heatmap evaluation, because `EXP-003` naturally produces patch scores that can be upsampled into localization maps.

## What This Does Not Prove Yet

This comparison does not prove that "ViT is uniquely better" or that "adding any projector is sufficient." `EXP-003` changes multiple factors at once: model capacity, token interaction, stable learning rate, and batch size. Therefore, the result is useful for choosing the next direction, but not enough for a final ablation table.

The next controlled comparisons should separate these factors:

| Control | Purpose |
| --- | --- |
| Patch MLP projector + Top-K | Test whether extra capacity without token mixing explains the gain |
| ViT projector depth / head sweep | Test whether token interaction strength matters |
| Top-K sweep on EXP-002 and EXP-003 | Test whether the gain comes from aggregation policy |
| Pixel-level heatmap evaluation | Test whether the image-level gain corresponds to actual localization |

## Output Artifacts

| Run | Output |
| --- | --- |
| EXP-002 | `WORK_DIR/outputs/dino_visa_a1_patch_topk` |
| EXP-003 | `WORK_DIR/outputs/dino_visa_a2_patch_vit_topk_lr1e4` |

## Next Step

Use `EXP-003` as the current image-level component baseline, but do not present it as a final proof. The follow-up module controls are summarized in [EXP-004+ Projector Module Study](exp-004-projector-module-study.md), and the dense heatmap / memory-bank follow-up is summarized in [EXP-007/008/009 Visual Module Follow-up](exp-007-009-visual-module-followup.md).
