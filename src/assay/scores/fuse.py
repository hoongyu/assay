"""把几个逐帧分数放到同一尺度上等权相加。

尺度用一批示范全部帧的中位数和四分位距（不看标签）。不能每条示范各自标准化：
那样会抹掉「这条整体比那条更像异常」，2026-10-03 在 RoboVAD train 上试过，micro AUC 从 0.700 掉到 0.643。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from assay.checks import NotApplicable
from assay.types import Episode

FrameScorer = Callable[[Episode], np.ndarray]


def robust_stats(scorer: FrameScorer, episodes: Sequence[Episode]) -> tuple[float, float]:
    xs = []
    for ep in episodes:
        try:
            xs.append(np.asarray(scorer(ep), float))
        except NotApplicable:
            continue
    x = np.concatenate(xs) if xs else np.zeros(1)
    med = float(np.median(x))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    return med, iqr if iqr > 0 else 1.0


def fuse(parts: Sequence[FrameScorer], stats: Sequence[tuple[float, float]]) -> FrameScorer:
    w = 1.0 / len(parts)

    def scorer(ep: Episode) -> np.ndarray:
        return sum(w * (np.asarray(p(ep), float) - med) / iqr for p, (med, iqr) in zip(parts, stats))

    return scorer
