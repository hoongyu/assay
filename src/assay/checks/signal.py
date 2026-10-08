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


def gripper_channels(names: list[str], gripper_index: int | None) -> list[int]:
    """夹爪是哪几个通道：名字里有 gripper 的都算（双臂两只手各一个）；名字认不出，用读入时定的 gripper_index。"""
    found = [i for i, n in enumerate(names) if "gripper" in n.lower()]
    if found:
        return found
    return [] if gripper_index is None else [gripper_index]


def _visits(x: np.ndarray, low: float, high: float, min_frames: int = 1) -> list[list[int]]:
    """不高于 low 算低档、不低于 high 算高档，夹在中间的（和 NaN）不算；返回依次去过的 [档, 进这一档的那一帧, 帧数]。
    中间那种在一档里待不够 min_frames 帧、又回到原来那一档的一趟不算（一闪就回去），前后同一档的接起来；
    开头、结尾那一趟再短也算（录制刚开始、刚停，不是去了又回来）。"""
    visits: list[list[int]] = []
    for i, v in enumerate(np.asarray(x, dtype=np.float64)):
        if v <= low:
            now = -1
        elif v >= high:
            now = 1
        else:  # 夹在中间，或者是 NaN
            continue
        if visits and visits[-1][0] == now:
            visits[-1][2] += 1
        else:
            visits.append([now, i, 1])
    kept: list[list[int]] = []
    for k, (level, start, n) in enumerate(visits):
        brief = 0 < k < len(visits) - 1 and n < min_frames
        if not brief and (not kept or kept[-1][0] != level):
            kept.append([level, start, n])
    return kept


def level_changes(x: np.ndarray, lo: float, hi: float, band: float, min_frames: int = 1) -> list[int]:
    """读数按 [lo, hi] 归一化，低于 0.5 - band 算一档、高于 0.5 + band 算另一档，夹在中间的不算；
    返回每次换到另一档的那一帧的下标。读数是 NaN 的帧跳过；hi 不比 lo 大时返回空。
    一闪就回去的一趟不算，见 _visits。"""
    span = hi - lo
    if not span > 1e-9:
        return []
    z = (np.asarray(x, dtype=np.float64) - lo) / span
    return [start for _, start, _ in _visits(z, 0.5 - band, 0.5 + band, min_frames)[1:]]


def arm_channels(names: list[str]) -> list[int]:
    """手臂关节：名字里没有 gripper、hand 的通道（夹爪、灵巧手的开合不算手臂在动）。"""
    return [i for i, n in enumerate(names) if "gripper" not in n.lower() and "hand" not in n.lower()]


def stops(v: np.ndarray, still: float, min_frames: int) -> list[tuple[int, int]]:
    """速度 v 低于 still、连续至少 min_frames 帧的区段 [start, end)；开头、结尾那一段不算（摆姿势、收手）。"""
    return [(s, e) for s, e in runs(np.asarray(v) < still) if s > 0 and e < len(v) and e - s >= min_frames]


def bursts(v: np.ndarray, low: float, high: float, min_frames: int = 1) -> list[int]:
    """动作分成几段：速度从不高于 low 升到不低于 high 算一段，返回每段开始的那一帧；一开头就在高档的也算一段。
    一闪就回去的一趟不算（中途一闪慢下来又快起来，还是一段），见 _visits。"""
    return [start for level, start, _ in _visits(v, low, high, min_frames) if level == 1]


def jerk(state: np.ndarray, fps: float, scale: np.ndarray) -> np.ndarray:
    """逐帧的 jerk（位置的三阶差分乘 fps³）：各关节按幅度归一化后取最急的那个，长度 T-3。和单位无关。"""
    if len(state) < 4:
        return np.zeros(0)
    d3 = np.abs(np.diff(state, n=3, axis=0)) / scale
    return np.max(d3, axis=1) * fps**3
