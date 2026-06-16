# 評估指標定義

本頁整理公開頁面使用的評估指標定義，讓每週進度、實驗總表、國科會報告與 paper draft 使用一致符號。

## 二元影像層級瑕疵偵測

每張影像有 ground-truth label \(y_i \in \{0, 1\}\)，其中 `1` 代表 anomaly。模型輸出 anomaly score \(s_i\)，目前由 logit 經 sigmoid 得到：

\[
s_i = \sigma(z_i)
\]

第一階段 supervised baseline 同時報告 ranking metrics 與 thresholded classification metrics：

| 指標 | 目的 | 備註 |
| --- | --- | --- |
| AUROC | 評估所有 threshold 下的排序能力 | 不依賴單一 operating point |
| AUPRC / AP | 評估 anomaly class 的 retrieval quality | 對 anomaly prevalence 較敏感 |
| F1 | 評估指定 threshold 下 precision/recall 平衡 | 目前同時報告 `0.5` 與 best threshold |

## Threshold 後的統計量

給定 threshold \(\tau\)，prediction 定義為：

\[
\hat{y}_i(\tau)=\mathbb{1}[s_i \ge \tau]
\]

Precision、recall 與 F1 定義如下：

\[
\mathrm{Precision} = \frac{TP}{TP + FP}
\]

\[
\mathrm{Recall} = \frac{TP}{TP + FN}
\]

\[
F_1 = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
\]

公開進度頁只放 aggregate metrics 與整理後 qualitative observations。實際 failure cases、內部討論脈絡與敏感路徑不放在 `docs/`。
