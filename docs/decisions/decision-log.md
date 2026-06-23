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
- 先建立乾淨 supervised baseline，後續 component comparison 才能有可辯護的比較基準。

## DEC-004：固定 DINOv3，將主線轉向 Patch-Level Downstream Component

日期：2026-06-23

狀態：已採納

決策：短期不再把主要實驗資源放在 DINOv3 encoder 本身的可用性驗證。`EXP-001` 已建立 frozen DINOv3 CLS-linear image-level baseline；`EXP-002/003` 改為固定 DINOv3，測試 patch-token scoring、Top-K aggregation 與 ViT projector。

理由：

- 研究對外敘事已經以 DINOv3 作為主要 visual encoder，繼續只做 encoder selection 的邊際價值偏低。
- `EXP-002` 顯示 patch-level evidence 相對 CLS baseline 有小幅提升，但 naive patch head 在部分 PCB 類別不穩。
- `EXP-003` 顯示 patch tokens 加上 ViT projector 與 Top-K aggregation 後，image-level AUPRC 從 `69.17` 提升到 `87.41`，F1max 從 `61.78` 提升到 `80.78`。
- 這個結果支持下一階段把問題定義為 anomaly-specific component design，而不是重新打開 encoder selection。

後續動作：

- 以 `EXP-003` 作為目前 image-level component baseline。
- 從 patch logits 產生 heatmaps，補 VisA pixel-level AUROC / AUPRC / F1max。
- 若 localization 指標不足，再評估 FoundAD-style projection-distance objective 或其他 anomaly-specific losses。

## DEC-005：VLM Decoder 暫緩，先完成 Visual Projector / Memory-Bank Module

日期：2026-06-24

狀態：已採納

決策：短期不接 VLM decoder，不把 CLIP/SigLIP 作為主線 encoder 對照。研究主線固定 DINOv3，先完成 visual anomaly module：supervised projector comparison、patch-grid / pixel-level localization、normal memory-bank decoder。

理由：

- 目前任務是 VisA / MVTec 類工業瑕疵，主要是局部幾何、紋理與表面缺陷，不是 open-vocabulary semantic anomaly。
- `EXP-004` 顯示 token-wise MLP projector 已接近 Transformer projector，表示需要先釐清 projector capacity / token interaction，而不是直接增加 VLM 複雜度。
- `EXP-005` 顯示 naive local Conv projector 在 localization proxy 上失效，代表 decoder/projector 需要更嚴格設計。
- MVTec official train set 以 normal samples 為主，不適合直接混入目前 VisA supervised head；更合理的是 DINOv3 normal memory-bank / PatchCore-style decoder。
- 近期 industrial AD 方向仍重視 DINOv3 dense features、memory bank、patch-wise scoring 與 morphology；VLM 應留到 visual module 已可穩定產出 anomaly map 後再作文字輸出。

後續動作：

- 補 MLP vs Transformer projector 的 parameter-matched comparison。
- 實作 normal-only memory-bank decoder，先跑 VisA / MVTec-AD。
- 將 patch-grid proxy 升級為 full-resolution pixel metric 與 heatmap overlays。
- VLM 只作後段 report / reasoning / open-vocabulary extension，不作近期 EXP-004/005 主線。
