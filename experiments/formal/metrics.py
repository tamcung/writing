#!/usr/bin/env python3
"""Metric helpers for formal experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def metrics_from_probs(probs: list[float] | np.ndarray, labels: list[int] | np.ndarray, threshold: float = 0.5) -> dict:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (probs >= threshold).astype(int)
    sqli_probs = probs[labels == 1]
    benign_probs = probs[labels == 0]
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "mean_sqli_prob": float(sqli_probs.mean()) if len(sqli_probs) else 0.0,
        "mean_benign_prob": float(benign_probs.mean()) if len(benign_probs) else 0.0,
        "p10_sqli_prob": float(np.quantile(sqli_probs, 0.10)) if len(sqli_probs) else 0.0,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
