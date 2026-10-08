"""逐帧异常分数，越高越像异常。不用任何标签训练；都按关节活动幅度归一化，和单位无关。"""

from __future__ import annotations

import numpy as np

from assay.checks import NotApplicable
from assay.checks.signal import joint_scale, speed
from assay.types import Episode

SMOOTH_S = 0.3  # 平滑窗口，秒


def _smooth(x: np.ndarray, fps: float) -> np.ndarray:
    k = max(1, int(round(SMOOTH_S * fps)))
    if k == 1 or len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _pad_to(x: np.ndarray, n: int) -> np.ndarray:
    """差分会少帧，前后补边上的值，对齐到 n 帧。"""
    if len(x) == 0:
        return np.zeros(n)
    left = (n - len(x)) // 2
    return np.pad(x, (left, n - len(x) - left), mode="edge")


def _need_action(ep: Episode) -> np.ndarray:
    if ep.action is None or ep.action.shape != ep.state.shape:
        raise NotApplicable("没有和关节状态同维度的动作（主臂指令）")
    return ep.action


def tracking(ep: Episode) -> np.ndarray:
    """从臂跟不上主臂：碰撞、被挡住时变大。"""
    action = _need_action(ep)
    err = np.nan_to_num(np.abs(action - ep.state) / joint_scale(ep.state))
    return _smooth(err.max(axis=1), ep.fps)


def gripper_tracking(ep: Episode) -> np.ndarray:
    """夹爪合不到主臂要的位置：夹着东西时大，抓空或掉了时小。"""
    action = _need_action(ep)
    g = ep.gripper_index
    if g is None:
        raise NotApplicable("认不出夹爪通道")
    scale = joint_scale(ep.state)[g]
    err = np.nan_to_num(np.abs(action[:, g] - ep.state[:, g]) / scale)
    return _smooth(err, ep.fps)


def accel(ep: Episode) -> np.ndarray:
    """加速度：急停、撞击。"""
    scale = joint_scale(ep.state)
    v = np.diff(ep.state, axis=0) / scale * ep.fps
    a = np.nan_to_num(np.abs(np.diff(v, axis=0)) * ep.fps).max(axis=1) if len(v) > 1 else np.zeros(0)
    return _smooth(_pad_to(a, len(ep.state)), ep.fps)


def calm(ep: Episode) -> np.ndarray:
    """负的加速度，越平稳分越高。2026-10-02 在 RoboVAD validation 上看到加速度的 AUC 只有 0.33（异常时操作员往往放慢、停下）后加的候选。"""
    return -accel(ep)


def stillness(ep: Episode) -> np.ndarray:
    """越不动分越高：卡住、犹豫。"""
    return _smooth(-_pad_to(speed(ep.state, ep.fps), len(ep.state)), ep.fps)
