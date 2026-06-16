# 實驗總表

本頁記錄每個實驗的目的、設定、結果位置與結論。後續每週進度、國科會成果報告與 paper baseline table 都可以從這裡整理。

## 命名規則

每個實驗使用 `EXP-###` 編號。raw logs、checkpoints 與敏感路徑不放在 `docs/`；公開頁面只呈現可公開摘要與整理後表格。

## 實驗列表

| ID | 狀態 | 目的 | 資料集 | 模型 / Head | 指標 | 結果位置 | 結論 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | 已完成 | 確認 DINOv3 feature extraction 與 token shape | VisA sample image | `facebook/dinov3-vitb16-pretrain-lvd1689m` frozen encoder | shape sanity | local smoke | CLS `(1, 768)`，patch tokens `(1, 196, 768)` |
| EXP-001 | 已完成 | Image-level supervised AD baseline A0 | VisA `2cls_highshot.csv` | DINOv3 frozen + linear head | AUROC, AUPRC, F1 | `WORK_DIR/outputs/dino_visa_a0_linear` | 第一個完整 GPU baseline 已完成：AUROC `0.9099`，AUPRC `0.6915`，best F1 `0.6176` |
| EXP-002 | 規劃中 | 比較 MLP head 與 linear head | VisA `2cls_highshot.csv` | DINOv3 frozen + MLP head | AUROC, AUPRC, F1 | TBD | TBD |
| EXP-003 | 規劃中 | 比較 CLS token 與 pooled patch tokens | VisA `2cls_highshot.csv` | DINOv3 frozen + selected pooling | AUROC, AUPRC, F1 | TBD | TBD |
| EXP-004 | 規劃中 | 探索 patch-token aggregation 作為 image score | VisA `2cls_highshot.csv` | DINOv3 frozen + patch MLP / pooling | image-level AUROC, AUPRC, F1 | TBD | TBD |
| EXP-005 | 規劃中 | Pixel-level anomaly map baseline | VisA mask annotations | DINOv3 patch tokens + aggregation | Pix-AUROC, Pix-AUPRC, Pix-F1max | TBD | 另開實驗；不混入目前 image-level classifier |

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
