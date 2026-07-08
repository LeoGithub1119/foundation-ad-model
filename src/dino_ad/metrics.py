from __future__ import annotations

import math
from typing import Sequence


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def binary_counts(labels: list[int], scores: list[float], threshold: float) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def f1_from_counts(counts: dict[str, int]) -> float:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def auroc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_pos = 0.0
    index = 0
    while index < len(ranked):
        next_index = index + 1
        while next_index < len(ranked) and ranked[next_index][0] == ranked[index][0]:
            next_index += 1
        avg_rank = (index + 1 + next_index) / 2.0
        positives_in_tie = sum(label for _, label in ranked[index:next_index])
        rank_sum_pos += positives_in_tie * avg_rank
        index = next_index

    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def best_f1(labels: list[int], scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"f1": float("nan"), "threshold": float("nan")}
    candidates = sorted(set(scores))
    best = {"f1": -1.0, "threshold": candidates[0]}
    for threshold in candidates:
        f1 = f1_from_counts(binary_counts(labels, scores, threshold))
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": threshold}
    return best


def compute_binary_metrics(labels: list[int], logits: list[float]) -> dict[str, object]:
    scores = [sigmoid(logit) for logit in logits]
    return compute_binary_metrics_from_scores(labels, scores)


def compute_binary_metrics_from_scores(labels: list[int], scores: list[float]) -> dict[str, object]:
    counts_05 = binary_counts(labels, scores, 0.5)
    best = best_f1(labels, scores)
    return {
        "auroc": auroc(labels, scores),
        "auprc": average_precision(labels, scores),
        "f1_at_0_5": f1_from_counts(counts_05),
        "counts_at_0_5": counts_05,
        "best_f1": best["f1"],
        "best_f1_threshold": best["threshold"],
        "num_samples": len(labels),
        "num_normal": len(labels) - sum(labels),
        "num_anomaly": sum(labels),
    }


def histogram_binary_metrics(pos_hist: Sequence[int], neg_hist: Sequence[int]) -> dict[str, object]:
    if len(pos_hist) != len(neg_hist):
        raise ValueError("Positive and negative histograms must have the same length")

    positives = int(sum(pos_hist))
    negatives = int(sum(neg_hist))
    total = positives + negatives
    if total == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1_at_0_5": float("nan"),
            "counts_at_0_5": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "best_f1": float("nan"),
            "best_f1_threshold": float("nan"),
            "num_samples": 0,
            "num_normal": 0,
            "num_anomaly": 0,
        }

    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auroc_area = 0.0
    precision_weighted_recall = 0.0
    best = {"f1": -1.0, "threshold": 1.0}
    counts_at_05 = None
    num_bins = len(pos_hist)

    for index in range(num_bins - 1, -1, -1):
        tp += int(pos_hist[index])
        fp += int(neg_hist[index])
        fn = positives - tp
        tn = negatives - fp
        tpr = 0.0 if positives == 0 else tp / positives
        fpr = 0.0 if negatives == 0 else fp / negatives
        auroc_area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr

        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall_delta = 0.0 if positives == 0 else int(pos_hist[index]) / positives
        precision_weighted_recall += precision * recall_delta

        f1 = 0.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
        threshold = index / max(num_bins - 1, 1)
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": threshold}
        if index == num_bins // 2:
            counts_at_05 = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    if counts_at_05 is None:
        counts_at_05 = {"tp": tp, "fp": fp, "tn": negatives - fp, "fn": positives - tp}

    return {
        "auroc": auroc_area if positives > 0 and negatives > 0 else float("nan"),
        "auprc": precision_weighted_recall if positives > 0 else float("nan"),
        "f1_at_0_5": f1_from_counts(counts_at_05),
        "counts_at_0_5": counts_at_05,
        "best_f1": best["f1"],
        "best_f1_threshold": best["threshold"],
        "num_samples": total,
        "num_normal": negatives,
        "num_anomaly": positives,
    }
