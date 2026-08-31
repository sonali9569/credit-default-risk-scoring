"""
Imbalance-appropriate evaluation metrics. Accuracy is deliberately not implemented
here — at ~22% default rate it rewards a majority-class-only model, which is not a
mistake this project makes. See README for the reasoning.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Max separation between the score distributions of the two classes."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic


def gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_proba) - 1


def summarize(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """One call, every metric this project reports — keeps reporting consistent
    across the baseline, tuned model, and calibrated model."""
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "ks": float(ks_statistic(y_true, y_proba)),
        "gini": float(gini(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two score/feature distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e = np.histogram(expected, breakpoints)[0] / len(expected)
    a = np.histogram(actual, breakpoints)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
