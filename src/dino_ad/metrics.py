from __future__ import annotations

import math


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
