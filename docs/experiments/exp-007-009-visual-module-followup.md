# EXP-007/008/009 Visual Module Follow-up

本頁整理 2026-07-08 這輪實驗。目的不是接 VLM，而是在固定 DINOv3 encoder 之後，繼續回答 visual anomaly module 是否已經有值得保留的設計。

## 實驗問題

| ID | 問題 | 設計 |
| --- | --- | --- |
| EXP-007 | image-level 分數高的 projector 是否也能產生 localization 訊號？ | 將 patch logits sigmoid 後 upsample 到 `224x224`，與 resized mask 計算 dense heatmap proxy |
| EXP-008 | Transformer 的收益是否只是參數量 / depth？ | 比較 depth9 MLP 與 depth2 Transformer，固定 DINOv3、VisA split、Top-K=6 |
| EXP-009 | normal-only memory-bank decoder 能否作為進入 MVTec 的 baseline？ | DINOv3 patch features + 20k normal patch memory + nearest-normal distance |

Dense heatmap 指標是 `224x224` proxy，不是原始解析度 full-sort pixel metric。它比 14x14 patch-grid 更接近 pixel-level，但仍不能直接宣稱等同簡報中的 pixel-level protocol。

## Supervised Projector 結果

| ID | Module | Image AUROC | Image AUPRC | Image F1max | Dense AUROC | Dense AUPRC | Dense F1max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-003 | 6-layer Transformer projector | 96.44 | 87.41 | 80.78 | 92.03 | 15.01 | 24.12 |
| EXP-004 | depth2 token-wise MLP | 95.20 | 84.78 | 77.53 | 97.07 | 14.32 | 23.70 |
| EXP-008a | depth9 token-wise MLP | 94.63 | 83.21 | 76.77 | 96.14 | 14.35 | 22.80 |
| EXP-008b | depth2 Transformer projector | 96.97 | 86.99 | 79.70 | 97.14 | 9.70 | 18.80 |

## Normal Memory-Bank 結果

| ID | Dataset | Image AUROC | Image AUPRC | Image F1max | Dense AUROC | Dense AUPRC | Dense F1max | Bank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-009a | VisA | 79.66 | 44.47 | 41.28 | 95.49 | 22.36 | 35.19 | 20k normal patches |
| EXP-009b | MVTec-AD | 85.86 | 94.74 | 86.98 | 92.52 | 40.44 | 39.49 | 20k normal patches |

MVTec image AUPRC 較高需注意 test split anomaly 比例較高，因此 AUROC 與 dense 指標更適合當主要解讀。

## Findings

1. 加深 MLP 沒有帶來收益。`EXP-008a` depth9 MLP 低於原本 depth2 MLP，表示目前增益不是單純由 MLP depth 或參數量解釋。
2. depth2 Transformer 在 image AUROC 達到最高，但 image AUPRC/F1max 仍略低於 6-layer Transformer，且 dense F1max 明顯較弱。
3. 6-layer Transformer 仍是目前最平衡的 supervised projector baseline：image AUPRC/F1max 最好，dense F1max 也最好。
4. Memory-bank branch 在 VisA image-level 不適合直接取代 supervised projector，但 dense localization proxy 明顯更強。
5. MVTec 應保留為 normal-only memory-bank / PatchCore-style 路線，不應硬套 VisA supervised anomaly head。

## 與博士班進度的對齊

目前只在任務層對齊：同樣關心 VisA / MVTec 類工業 anomaly detection 與 image/pixel 指標。尚未在方法與 protocol 層對齊。

BLIP-3 + Anomaly-OV 簡報 P13-P14 的 supervised VisA result 已接近 image AUROC `99.96`、pixel AUROC `99.54`。本輪 DINOv3 visual-module 結果仍低於該數字，也沒有重現其 Anomaly-OV / LTFM / VLM pipeline。因此不能宣稱已達成或對齊博士班的 `99` 分成果。

較精確的說法是：我們建立了一條固定 DINOv3 encoder 的 visual-module baseline，已證明 downstream projector 與 memory-bank decoder 都有可觀察訊號，但距離博士班的 oracle supervised / full pipeline 還有明確 gap。

## 下一步

| 方向 | 目的 |
| --- | --- |
| Class-specific memory bank | 避免跨類別 normal distribution 混在同一個 bank，改善 VisA image score |
| Coreset / neighborhood-restricted scoring | 對齊 PatchCore / DINOSaur 類 memory-bank 改良，降低雜訊 |
| Spatial smoothing / morphology | 對齊 SuperADD 類 segmentation 後處理，提高 dense F1 |
| Per-category dense report + heatmap overlays | 找出 localization 成功與失敗類別，補明天報告圖 |
| Score aggregation study | 比較 max / Top-K / percentile / category calibration，改善 memory-bank image-level 分數 |

暫不接 VLM decoder。VLM 應等 visual module 能穩定產生可用 anomaly map 後，再作文字化報告、reasoning 或 open-vocabulary extension。

## Artifacts

| Run | Output |
| --- | --- |
| EXP-003 dense | `WORK_DIR/outputs/dino_visa_a2_patch_vit_topk_lr1e4` |
| EXP-004 dense | `WORK_DIR/outputs/dino_visa_a3_patch_mlp_topk` |
| EXP-008a | `WORK_DIR/outputs/dino_visa_a7_patch_mlp_topk_depth9` |
| EXP-008b | `WORK_DIR/outputs/dino_visa_a8_patch_vit_topk_depth2` |
| EXP-009a | `WORK_DIR/outputs/dino_visa_a9_memory_bank` |
| EXP-009b | `WORK_DIR/outputs/dino_mvtec_a1_memory_bank` |
