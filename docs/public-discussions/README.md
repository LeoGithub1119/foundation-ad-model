# 公開討論摘要

本區放由實驗室內部討論整理出的公開摘要。它不是逐字稿，也不包含未確認或敏感內容。

## 收錄原則

可以放：

- 可公開的研究方向。
- 已確認的技術決策。
- 高層次實驗規劃。
- 經整理後可公開的結果摘要。

不放：

- 內部討論逐字稿。
- 與未審閱意見綁在一起的具名內容。
- 不應公開的未發布資料細節。
- raw logs、private paths、tokens、account names 或 credentials。
- 尚未經實驗驗證的宣稱。

## 目前摘要

目前研究方向是先建立乾淨的 DINOv3-based visual anomaly detection baseline，再進入 full VLM reasoning 或 Anomaly-OV-style modules。第一個目標是在 VisA 上做 supervised image-level anomaly detection，使用 frozen DINOv3 encoder 搭配簡單可訓練 head。
