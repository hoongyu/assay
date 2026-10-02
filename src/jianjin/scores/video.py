"""把视频评判的进度曲线插值到每一帧，变成逐帧异常分（越高越像异常）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from jianjin.checks import NotApplicable
from jianjin.judges.base import Curve
from jianjin.judges.cache import load_curve
from jianjin.types import Episode

SMOOTH_S = 1.0


def per_frame(times: list[float], values: list[float], ep: Episode) -> np.ndarray:
    return np.interp(ep.timestamps, np.asarray(times, float), np.asarray(values, float))


def _smooth(x: np.ndarray, fps: float) -> np.ndarray:
    k = max(1, int(round(SMOOTH_S * fps)))
    return x if k == 1 or len(x) < k else np.convolve(x, np.ones(k) / k, mode="same")


def drop(c: Curve, ep: Episode) -> np.ndarray:
    """进度往回掉：东西掉了、放错了。"""
    p = per_frame(c.times, c.progress, ep)
    slope = np.gradient(p, ep.timestamps) if len(p) > 1 else np.zeros_like(p)
    return _smooth(-slope, ep.fps)


def low(c: Curve, ep: Episode) -> np.ndarray:
    """进度低。"""
    return 1.0 - per_frame(c.times, c.progress, ep)


def fail(c: Curve, ep: Episode) -> np.ndarray:
    """成功概率低。"""
    if c.success is None:
        raise NotApplicable(f"{c.model} 不给成功概率")
    return 1.0 - per_frame(c.times, c.success, ep)


CURVE_SCORES: dict[str, Callable[[Curve, Episode], np.ndarray]] = {"drop": drop, "low": low, "fail": fail}


def video_scorer(judge_name: str, kind: str, cache_dir: Path, camera: str | None = None):
    """从缓存读进度曲线的逐帧打分器；没缓存的示范记为打不了分。"""
    fn = CURVE_SCORES[kind]

    def scorer(ep: Episode) -> np.ndarray:
        c = load_curve(cache_dir, judge_name, ep, camera)
        if c is None:
            raise NotApplicable(f"缓存里没有 {judge_name} 对这条示范的结果")
        return fn(c, ep)

    return scorer
