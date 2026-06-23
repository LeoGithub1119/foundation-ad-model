# 實驗總表

本頁記錄每個實驗的目的、設定、結果位置與結論。後續每週進度、國科會成果報告與 paper baseline table 都可以從這裡整理。

## 命名規則

每個實驗使用 `EXP-###` 編號。raw logs、checkpoints 與敏感路徑不放在 `docs/`；公開頁面只呈現可公開摘要與整理後表格。

## 實驗列表

| ID | 狀態 | 目的 | 資料集 | 模型 / Head | 指標 | 結果位置 | 結論 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | 已完成 | 確認 DINOv3 feature extraction 與 token shape | VisA sample image | `facebook/dinov3-vitb16-pretrain-lvd1689m` frozen encoder | shape sanity | local smoke | CLS `(1, 768)`，patch tokens `(1, 196, 768)` |
| EXP-001 | 已完成 | Image-level supervised AD baseline A0 | VisA `2cls_highshot.csv` | DINOv3 frozen + linear head | AUROC, AUPRC, F1 | `WORK_DIR/outputs/dino_visa_a0_linear` | 第一個完整 GPU baseline 已完成：AUROC `0.9099`，AUPRC `0.6915`，best F1 `0.6176` |
| EXP-002 | 已完成 | Patch-token Top-K supervised AD | VisA `2cls_highshot.csv` | DINOv3 frozen + patch linear head + Top-K | AUROC, AUPRC, F1 | `WORK_DIR/outputs/dino_visa_a1_patch_topk` | AUPRC/F1max 小幅優於 CLS baseline，但部分 PCB 類別不穩 |
| EXP-003 | 已完成 | ViT projector patch-token Top-K supervised AD | VisA `2cls_highshot.csv` | DINOv3 frozen + 6-layer ViT projector + patch head + Top-K | AUROC, AUPRC, F1 | `WORK_DIR/outputs/dino_visa_a2_patch_vit_topk_lr1e4` | LR `1e-4` 穩定後明顯優於 EXP-001/002：AUROC `0.9644`，AUPRC `0.8741`，F1max `0.8078` |
| EXP-004 | 已完成 | Token-wise MLP projector control | VisA `2cls_highshot.csv` | DINOv3 frozen + patch MLP projector + Top-K=6 | image-level + patch-grid metrics | `WORK_DIR/outputs/dino_visa_a3_patch_mlp_topk` | 大部分提升可由 token-wise nonlinear projector 解釋：AUPRC `0.8478`，F1max `0.7753` |
| EXP-005 | 已完成 | Local Conv projector control | VisA `2cls_highshot.csv` | DINOv3 frozen + patch Conv projector + Top-K=6 | image-level + patch-grid metrics | `WORK_DIR/outputs/dino_visa_a4_patch_conv_topk` | image-level 較弱且 patch-grid localization 幾乎失效；暫不作主線 |
| EXP-006 | 已完成 | Transformer projector Top-K sweep | VisA `2cls_highshot.csv` | DINOv3 frozen + Transformer projector + Top-K `{1,6,12}` | image-level metrics | `WORK_DIR/outputs/dino_visa_a5_patch_vit_topk_k1`, `WORK_DIR/outputs/dino_visa_a6_patch_vit_topk_k12` | `K=6` 在 1/6/12 中 image AUPRC/F1max 最好，但仍不是完整 K sweep |
| EXP-007 | 規劃中 | DINOv3 normal memory-bank decoder | VisA + MVTec-AD | DINOv3 dense features + normal memory bank / PatchCore-style scoring | image-level + full-res pixel metrics | TBD | MVTec official protocol 應走 normal-only / training-free 路線，不混入 supervised test anomalies |

## 標記完成前必須記錄

- 資料集版本與 split。
- 模型 checkpoint 或 Hugging Face model ID。
- 可訓練模組。
- Loss function。
- 評估指標與 threshold policy。
- Seed 與 config path。
- Output path。
- 一段可放入報告的結論。

## Smoke Test 驗證紀錄

DINOv3 feature extraction smoke：

```text
pixel_values=(1, 3, 224, 224)
last_hidden_state=(1, 201, 768)
cls=(1, 768)
patch_tokens=(1, 196, 768)
num_register_tokens=4
```

Tiny CPU training-loop smoke 使用 2 張 normal 與 2 張 anomaly 做 train/test。此結果只驗證程式路徑，不代表有效模型表現。

## Slurm 紀錄

- `237545`：第一次提交後立即失敗，原因是 sbatch script 從 Slurm spool copy 推 `REPO_ROOT`，導致路徑錯誤。
- `237546`：修正為使用 `SLURM_SUBMIT_DIR` 後，在 `hgpn18` 成功完成，耗時 `00:05:57`。
- `237589`：image-level per-category evaluation 已在 `hgpn04` 完成，耗時 `00:00:37`，產出 P14-style image-level 表與 prediction dump。
- `243976`：EXP-002 patch-token Top-K 訓練完成，耗時 `00:06:00`。
- `243985`：EXP-002 per-category evaluation 完成，耗時 `00:00:37`。
- `243991`：EXP-003 ViT projector，LR `1e-4`，穩定完成，耗時 `00:06:06`。
- `243997`：EXP-003 stable run per-category evaluation 完成，耗時 `00:00:38`。
- `244054`：EXP-004 token-wise MLP projector 訓練完成，耗時 `00:06:16`。
- `244055`：EXP-005 local Conv projector 訓練完成，耗時 `00:06:15`。
- `244056`：EXP-002 patch-grid localization proxy 完成，耗時 `00:29:48`。
- `244057`：EXP-003 patch-grid localization proxy 完成，耗時 `00:17:37`。
- `244059`：EXP-006 Transformer projector `K=1` 訓練完成，耗時 `00:05:58`。
- `244060`：EXP-006 Transformer projector `K=12` 訓練完成，耗時 `00:06:05`。
- `244065`：EXP-004 image-level per-category evaluation 完成，耗時 `00:00:41`。
- `244066`：EXP-004 patch-grid localization proxy 完成，耗時 `00:19:57`。
- `244067`：EXP-005 image-level per-category evaluation 完成，耗時 `00:02:28`。
- `244068`：EXP-005 patch-grid localization proxy 完成，耗時 `00:27:35`。
- `244103`：EXP-006 `K=1` image-level per-category evaluation 完成，耗時 `00:00:38`。
- `244104`：EXP-006 `K=12` image-level per-category evaluation 完成，耗時 `00:00:38`。

## EXP-001 結果摘要

設定：

- Split：VisA `split_csv/2cls_highshot.csv`
- Train set：6,493 張影像，其中 5,773 張 normal、720 張 anomaly。
- Test set：4,328 張影像，其中 3,848 張 normal、480 張 anomaly。
- Encoder：local DINOv3 `facebook/dinov3-vitb16-pretrain-lvd1689m`，固定不訓練。
- Feature：CLS token，hidden size 768。
- Head：linear binary anomaly classifier。
- Epochs：5，batch size 16，learning rate `1e-3`，weight decay `1e-4`，seed 42，啟用 AMP。

以 AUROC 最高的 epoch 作為目前摘要：

| Epoch | Train Loss | AUROC | AUPRC | F1 @ 0.5 | Best F1 | Best F1 Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.7409 | 0.9099 | 0.6915 | 0.5395 | 0.6176 | 0.6364 |

初步解讀：frozen DINOv3 CLS feature 在 VisA 上已能形成可用的 supervised image-level baseline，但 thresholded performance 仍有改善空間。下一步應先比較 head capacity、feature pooling 與各類別 breakdown，再決定是否改變研究方向。

### Image-Level 表格產出狀態

已新增 evaluation 工具，可從 `best_head.pt` 重跑 test set 並輸出：

- `predictions_test.csv`：每張圖的 category、label、path、logit、score。
- `image_level_overall_test.json`：overall AUROC / AUPRC / F1。
- `image_level_by_category_test.json`：per-category AUROC / AUPRC / F1max。
- `image_level_by_category_test.md`：可貼入網站或報告的 Markdown 表。

完整 test set evaluation 已由 Slurm job `237589` 完成。公開表格整理於 [EXP-001 Image-Level 表格](exp-001-image-level-table.md)。

## EXP-002 / EXP-003 結果摘要

EXP-002/003 固定 DINOv3 encoder 與 VisA split，比較 patch-token Top-K control 與加入 ViT projector 的 downstream component。完整比較整理於 [EXP-002/003 Patch Component Comparison](exp-002-003-patch-component-ablation.md)。

| ID | Feature / Module | LR | AUROC | AUPRC | F1 @ 0.5 | F1max | 結論 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | CLS + linear head | `1e-3` | 90.99 | 69.17 | 53.92 | 61.78 | 最小 image-level baseline |
| EXP-002 | Patch head + Top-K=6 | `1e-3` | 90.82 | 70.22 | 55.36 | 62.84 | AUPRC/F1 小幅提升，但 category stability 不足 |
| EXP-003 | 6-layer ViT projector + Top-K=6 | `1e-4` | 96.44 | 87.41 | 79.91 | 80.78 | 明顯優於 EXP-001/002，成為下一階段主線 |

## EXP-004 / EXP-006 Projector Module Study

完整整理於 [EXP-004+ Projector Module Study](exp-004-projector-module-study.md)。

| ID | Module | K | Image AUROC | Image AUPRC | Image F1max | Patch-grid AUROC | Patch-grid AUPRC | Patch-grid F1max | 結論 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-002 | Raw patch linear | 6 | 90.82 | 70.22 | 62.84 | 93.69 | 11.55 | 18.99 | raw patch Top-K 不足 |
| EXP-003 | Transformer projector | 6 | 96.44 | 87.41 | 80.78 | 87.91 | 13.57 | 26.24 | image/localization 綜合最佳 |
| EXP-004 | Token-wise MLP projector | 6 | 95.20 | 84.78 | 77.53 | 94.18 | 13.26 | 24.91 | 大部分提升來自 nonlinear projector |
| EXP-005 | Local Conv projector | 6 | 94.50 | 73.69 | 67.29 | 53.12 | 0.17 | 0.36 | 暫不作主線 |
| EXP-006a | Transformer projector | 1 | 95.27 | 81.34 | 75.26 | TBD | TBD | TBD | K 太小，召回較不穩 |
| EXP-006b | Transformer projector | 12 | 96.03 | 84.70 | 76.07 | TBD | TBD | TBD | K=12 接近但低於 K=6 |
