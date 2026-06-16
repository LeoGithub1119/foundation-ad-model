# 技術決策紀錄

本頁記錄可公開的技術決策與理由，方便回溯研究方向為什麼這樣安排。

## DEC-001：先建立 DINOv3 固定 Encoder Baseline

日期：2026-06-15

狀態：已採納

決策：第一個 visual encoder 使用 `facebook/dinov3-vitb16-pretrain-lvd1689m`，初期 supervised anomaly detection baseline 先固定 encoder，只訓練分類 head。

理由：

- DINOv3 提供強的 pretrained visual representation，計算成本仍可控。
- 第一個 baseline 需要足夠簡單，才能快速 debug 與建立可回溯結果。
- 避免太早把 visual encoder 選擇、LLM reasoning 與 full VLM alignment 混在一起，導致問題難以定位。

預期驗證：

- Feature extraction smoke test。
- VisA image-level AUROC / AUPRC / F1，先比較 linear head 與 MLP head。

## DEC-002：GitHub Pages 只發布可公開研究進度

日期：2026-06-15

狀態：已採納

決策：`docs/` 作為 GitHub Pages 公開來源。內部原始討論紀錄放在 `research_ops/`，不直接發布。

理由：

- 研究過程包含實驗室內部討論與尚未公開結果。
- GitHub Pages 適合放公開或可分享文件。
- 公開摘要已足夠支援進度追蹤與後續報告整理。

## DEC-003：延後完整 VLM 與 Anomaly-OV 整合

日期：2026-06-15

狀態：已採納

決策：暫不從 LLM reasoning、完整 Anomaly-OV reimplementation 或 multi-dataset training 開始。先完成乾淨的 visual anomaly detection baseline。

理由：

- 目前優先目標是 image-level anomaly detection。
- Anomaly-OV 含有超出簡單 encoder-projector-LLM stack 的組件，例如 LTFM 與 token selection。
- 先建立乾淨 supervised baseline，後續 ablation 才能有可辯護的比較基準。
