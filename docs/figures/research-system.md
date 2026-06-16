# 研究系統圖

本頁整理可公開的研究圖。後續可轉成 paper figure、簡報圖或國科會成果報告圖。

## 目前 Baseline

```mermaid
flowchart TB
    subgraph data[資料層]
        visa[VisA 2cls_highshot split]
        trainset[訓練集: normal + anomaly]
        testset[測試集: normal + anomaly]
        visa --> trainset
        visa --> testset
    end

    subgraph model[模型層]
        image[影像 224x224]
        processor[DINOv3 image processor]
        encoder[DINOv3 ViT-B/16 frozen encoder]
        features[CLS token / patch tokens]
        head[可訓練 linear anomaly head]
        logit[Image-level anomaly logit]
        image --> processor --> encoder --> features --> head --> logit
    end

    subgraph eval[評估層]
        score[Sigmoid 異常分數]
        auroc[AUROC]
        auprc[AUPRC]
        f1[F1]
        logit --> score
        score --> auroc
        score --> auprc
        score --> f1
    end

    trainset --> image
    testset --> image
```

## 文件與實驗更新流程

```mermaid
flowchart LR
    discussion[內部討論] --> raw[research_ops 原始紀錄]
    raw --> sanitized[可公開摘要]
    sanitized --> docs[docs/ MkDocs 網站]
    exp[實驗執行] --> registry[實驗總表]
    registry --> docs
    decision[技術決策] --> log[決策紀錄]
    log --> docs
    docs --> push[git push]
    push --> pages[GitHub Pages]
```

## 圖表待辦

| 圖表 | 用途 | 狀態 |
| --- | --- | --- |
| Baseline pipeline | 說明 DINOv3 frozen encoder supervised AD | 已用 Mermaid 起草 |
| Experiment lifecycle | 說明內部筆記到 GitHub Pages 的更新流程 | 已用 Mermaid 起草 |
| Future VLM architecture | 比較 DINO 路線與 Anomaly-OV style route | 規劃中 |
| Paper-ready pipeline | 轉成 paper 可用向量圖 | 規劃中 |
