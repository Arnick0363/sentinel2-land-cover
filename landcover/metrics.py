"""样本精度：混淆矩阵、总体精度、Kappa 与各类 F1。"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index = {label: i for i, label in enumerate(labels)}
    for truth, pred in zip(y_true, y_pred):
        if truth in index and pred in index:
            matrix[index[truth], index[pred]] += 1
    return matrix


def overall_accuracy(matrix: np.ndarray) -> float:
    total = matrix.sum()
    if total == 0:
        return float("nan")
    return float(np.trace(matrix) / total)


def cohen_kappa(matrix: np.ndarray) -> float:
    """Cohen's Kappa。matrix 的行是参考类，列是预测类。"""
    total = float(matrix.sum())
    if total == 0:
        return float("nan")
    observed = float(np.trace(matrix) / total)
    row = matrix.sum(axis=1).astype(np.float64)
    col = matrix.sum(axis=0).astype(np.float64)
    expected = float(np.sum(row * col) / (total * total))
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else 0.0
    return (observed - expected) / (1.0 - expected)


def per_class_scores(matrix: np.ndarray) -> list[dict]:
    """返回各类的 precision、recall、F1 与支持数（按行，即参考类）。"""
    scores = []
    for i in range(matrix.shape[0]):
        tp = float(matrix[i, i])
        support = float(matrix[i, :].sum())
        predicted = float(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        scores.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )
    return scores


def binary_built_labels(labels: np.ndarray, built_id: int) -> np.ndarray:
    """把多类标签收成非建成 0 / 建成 1，对齐 IndiaSAT 的 BU/NBU 评价。"""
    return (np.asarray(labels).astype(int) == built_id).astype(int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels)
    scores = per_class_scores(matrix)
    return {
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "oa": overall_accuracy(matrix),
        "kappa": cohen_kappa(matrix),
        "per_class": scores,
        "n": int(matrix.sum()),
    }
