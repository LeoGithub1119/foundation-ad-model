# EXP-004+ Projector Module Study

本頁整理固定 DINOv3 encoder 後的 projector / aggregation study。這一組實驗的目的不是接 VLM，而是在進入 VLM decoder 前，先確認 visual anomaly module 是否有值得保留的設計。

## Design Motivation

近期方向支持先把 visual module 做乾淨：

- FoundAD 使用 foundation visual encoders，並把 nonlinear projection operator 作為 anomaly detection 的核心工具。
- SuperADD / CVPR 2026 VAND Industrial Track 走 training-free class-agnostic anomaly segmentation，包含 DINOv3 backbone、memory-bank subsampling、overlapping patch processing 與 morphology。

因此本階段分成兩條路線：

| Route | Purpose | Status |
| --- | --- | --- |
| Supervised projector on VisA | 比較 DINOv3 patch tokens 後接哪種 trainable projector | 本頁已完成第一輪 |
| Normal memory-bank decoder on VisA/MVTec | 對齊 MVTec official normal-only protocol 與近期 training-free AD 方法 | 下一步 |

## Controlled Setup

| 項目 | 設定 |
| --- | --- |
| Dataset | VisA `split_csv/2cls_highshot.csv` |
| Train set | 6,493 images, 5,773 normal / 720 anomaly |
| Test set | 4,328 images, 3,848 normal / 480 anomaly |
| Encoder | DINOv3 `facebook/dinov3-vitb16-pretrain-lvd1689m`, frozen |
| Trainable modules | projector / patch scoring head only |
| Image metrics | AUROC / AUPRC / F1 @ 0.5 / F1max |
| Localization proxy | Patch-grid metrics: resize VisA mask to DINO 14x14 patch grid |

Patch-grid metrics are a fast localization proxy, not publication-grade full-resolution pixel metrics.

## Projector Comparison

| ID | Module | K | Image AUROC | Image AUPRC | Image F1max | Patch-grid AUROC | Patch-grid AUPRC | Patch-grid F1max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-002 | Raw patch linear | 6 | 90.82 | 70.22 | 62.84 | 93.69 | 11.55 | 18.99 |
| EXP-003 | Transformer projector | 6 | 96.44 | 87.41 | 80.78 | 87.91 | 13.57 | 26.24 |
| EXP-004 | Token-wise MLP projector | 6 | 95.20 | 84.78 | 77.53 | 94.18 | 13.26 | 24.91 |
| EXP-005 | Local Conv projector | 6 | 94.50 | 73.69 | 67.29 | 53.12 | 0.17 | 0.36 |

## Top-K Sweep

| ID | Module | K | Image AUROC | Image AUPRC | Image F1max |
| --- | --- | --- | --- | --- | --- |
| EXP-006a | Transformer projector | 1 | 95.27 | 81.34 | 75.26 |
| EXP-003 | Transformer projector | 6 | 96.44 | 87.41 | 80.78 |
| EXP-006b | Transformer projector | 12 | 96.03 | 84.70 | 76.07 |

`K=6` is not proven globally optimal, but it is stronger than `K=1` and `K=12` in this first sweep.

## Interpretation

The strongest result is not "Transformer projector is solved." The stronger and more defensible reading is:

- Raw patch Top-K alone is insufficient.
- A token-wise nonlinear MLP projector explains most of the gain over raw patch scoring.
- Transformer self-attention still gives a small but consistent advantage over MLP on both image F1max and patch-grid F1max.
- Local Conv projector is weak in this setup and should not be the main path without redesign.

This means the next controlled study should compare parameter-matched MLP vs Transformer projectors more carefully before claiming token interaction is essential.

## Current Module Candidates

| Candidate | Keep? | Reason |
| --- | --- | --- |
| Token-wise MLP projector | Yes | strong, simple, cheaper than Transformer |
| Transformer projector | Yes | best current score, but needs cost/control study |
| Local Conv projector | No for now | localization proxy nearly fails |
| Raw patch Top-K | Baseline only | useful lower bound |

## Next Experiments

1. Parameter-match MLP and Transformer projector depth/width.
2. Add DINOv3 normal memory-bank decoder for VisA and MVTec-AD.
3. Replace patch-grid proxy with full-resolution pixel metric and heatmap overlays.
4. Keep VLM decoder postponed until the visual module has a robust image/pixel story.

## Output Artifacts

| Run | Output |
| --- | --- |
| EXP-004 | `WORK_DIR/outputs/dino_visa_a3_patch_mlp_topk` |
| EXP-005 | `WORK_DIR/outputs/dino_visa_a4_patch_conv_topk` |
| EXP-006a | `WORK_DIR/outputs/dino_visa_a5_patch_vit_topk_k1` |
| EXP-006b | `WORK_DIR/outputs/dino_visa_a6_patch_vit_topk_k12` |
