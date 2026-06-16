# EXP-001 Image-Level 表格

本頁整理 `EXP-001` 的 image-level supervised anomaly detection 結果。表格型態對齊簡報 P14 的 image-level per-category table；目前尚未包含 pixel-level 指標與 Deco-Diff 對照 Δ。

## 實驗設定

| 項目 | 設定 |
| --- | --- |
| Dataset | VisA `split_csv/2cls_highshot.csv` |
| Train set | 6,493 張影像，5,773 normal / 720 anomaly |
| Test set | 4,328 張影像，3,848 normal / 480 anomaly |
| Encoder | DINOv3 `facebook/dinov3-vitb16-pretrain-lvd1689m`，固定不訓練 |
| Feature | CLS token |
| Head | Linear binary anomaly classifier |
| Checkpoint | `WORK_DIR/outputs/dino_visa_a0_linear/best_head.pt` |
| Eval job | Slurm `237589`，`hgpn04`，耗時 `00:00:37` |

## Overall Image-Level 結果

表內數值為百分比。`F1max` 是在 test split 上掃過 score threshold 後得到的最佳 F1。

| AUROC | AUPRC | F1 @ 0.5 | F1max | F1max Threshold |
| --- | --- | --- | --- | --- |
| 90.99 | 69.17 | 53.92 | 61.78 | 0.6108 |

## Per-Category Image-Level 結果

| 類別 | AUROC | AUPRC | F1max | F1max Threshold | Normal | Anomaly |
| --- | --- | --- | --- | --- | --- | --- |
| candle | 96.34 | 78.81 | 68.42 | 0.6169 | 400 | 40 |
| capsules | 90.67 | 76.48 | 72.00 | 0.5789 | 241 | 40 |
| cashew | 95.34 | 89.71 | 83.78 | 0.8531 | 200 | 40 |
| chewinggum | 98.63 | 96.72 | 92.68 | 0.5905 | 201 | 40 |
| fryum | 89.70 | 80.25 | 76.92 | 0.6565 | 200 | 40 |
| macaroni1 | 85.91 | 47.78 | 48.78 | 0.6584 | 400 | 40 |
| macaroni2 | 86.59 | 55.37 | 54.55 | 0.7286 | 400 | 40 |
| pcb1 | 93.31 | 60.06 | 57.14 | 0.5502 | 402 | 40 |
| pcb2 | 86.47 | 55.63 | 54.55 | 0.5318 | 400 | 40 |
| pcb3 | 86.14 | 63.66 | 63.64 | 0.5567 | 402 | 40 |
| pcb4 | 97.69 | 80.92 | 78.05 | 0.6285 | 402 | 40 |
| pipe_fryum | 98.02 | 95.64 | 91.14 | 0.7017 | 200 | 40 |

## 初步觀察

- `chewinggum`、`pipe_fryum`、`cashew` 的 image-level 表現較好，AUPRC 分別為 `96.72`、`95.64`、`89.71`。
- `macaroni1`、`macaroni2`、`pcb2` 的 AUPRC 與 F1max 較低，是下一步 failure-case inspection 的優先類別。
- 目前結果只代表 image-level classifier；不能等同 pixel-level localization 能力。

## 輸出檔案

| 檔案 | 說明 |
| --- | --- |
| `WORK_DIR/outputs/dino_visa_a0_linear/predictions_test.csv` | 每張 test image 的 category、label、logit、score |
| `WORK_DIR/outputs/dino_visa_a0_linear/image_level_overall_test.json` | Overall image-level 指標 |
| `WORK_DIR/outputs/dino_visa_a0_linear/image_level_by_category_test.json` | Per-category image-level 指標 |
| `WORK_DIR/outputs/dino_visa_a0_linear/image_level_by_category_test.md` | 自動產出的 Markdown 表格 |
