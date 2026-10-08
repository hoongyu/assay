"""ROC AUC：随机挑一个异常帧和一个正常帧，异常帧分数更高的概率；分数相同算一半。"""

from __future__ import annotations

import math

import numpy as np


def roc_auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2  # 从 1 开始的平均秩
    ranks = np.empty(len(s))
    ranks[order] = avg_rank[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
