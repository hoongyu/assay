"""关节信号的基本运算。所有速度都按关节活动幅度归一化，和单位无关。"""

from __future__ import annotations

import numpy as np


def joint_scale(state: np.ndarray) -> np.ndarray:
    """每个关节的活动幅度（5% 到 95% 分位之差）；没动过的关节记 inf，不参与比较。"""
    lo, hi = np.nanpercentile(state, [5, 95], axis=0)
    scale = np.asarray(hi - lo, dtype=np.float64)
    scale[~(scale > 1e-9)] = np.inf
    return scale


def speed(state: np.ndarray, fps: float, scale: np.ndarray | None = None) -> np.ndarray:
    """相邻两帧之间、动得最快的那个关节的归一化速度（幅度/秒），长度 T-1。"""
    if scale is None:
        scale = joint_scale(state)
    if len(state) < 2:
        return np.zeros(0)
    step = np.abs(np.diff(state, axis=0)) / scale
    return np.max(step, axis=1) * fps


def path_length(state: np.ndarray, scale: np.ndarray) -> float:
    """所有关节归一化后走过的路程之和。"""
    if len(state) < 2:
        return 0.0
    step = np.abs(np.diff(state, axis=0)) / scale
    return float(np.nansum(step))


def excursion(state: np.ndarray, scale: np.ndarray) -> float:
    """各关节离开头那一帧最远走了几个活动幅度，取走得最远的那个关节。"""
    if len(state) < 2:
        return 0.0
    d = np.abs(state - state[0]) / scale
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """mask 里连续为 True 的区段，返回 [start, end) 下标。"""
    if mask.size == 0:
        return []
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))
