"""视频评判的统一接口：看一段视频和任务描述，给出进度曲线（0 到 1），有的还给成功概率。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Curve:
    model: str
    version: str
    times: list[float]  # 秒，相对示范开头
    progress: list[float]
    success: list[float] | None = None


class Judge(Protocol):
    name: str
    version: str

    def curve(self, frames: np.ndarray, times: np.ndarray, task: str) -> Curve: ...


class FakeJudge:
    """测试用：进度等于到这一帧为止的平均亮度占最终平均亮度的比例。"""

    name = "fake"

    def __init__(self, version: str = "1"):
        self.version = version
        self.calls = 0

    def curve(self, frames: np.ndarray, times: np.ndarray, task: str) -> Curve:
        self.calls += 1
        b = frames.reshape(len(frames), -1).mean(axis=1)
        p = np.cumsum(b) / max(float(np.sum(b)), 1e-9)
        return Curve(self.name, self.version, [float(t) for t in times], [float(x) for x in p])
