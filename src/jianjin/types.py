"""拣金的核心数据结构。设计见设计文档 4.2。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


class Cause(StrEnum):
    """失败原因类别，见设计文档第 5 节。"""

    INCOMPLETE = "没做完"
    EMPTY = "空录"
    MISPLACED = "定位不准"
    GRASP_MISS = "夹爪没合上"
    REGRASP = "二次抓取"
    DROP = "物体掉落"
    EXTRA_MOTION = "多余动作"
    LONG_PAUSE = "停顿过长"
    WRONG_TASK = "指令不符"
    DATA_ERROR = "数据异常"


@dataclass(frozen=True)
class VideoRef:
    """一路视频在文件里的位置；LeRobot v3 多条示范共用一个 mp4。"""

    path: Path
    from_s: float
    to_s: float | None


@dataclass
class Episode:
    id: str
    source: str
    task: str
    fps: float
    state: np.ndarray  # (T, D)
    action: np.ndarray | None  # (T, D') 或没有
    timestamps: np.ndarray  # (T,) 秒
    state_names: list[str]
    gripper_index: int | None = None
    cameras: dict[str, VideoRef] = field(default_factory=dict)
    operator_id: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.state) / self.fps

    @property
    def gripper(self) -> np.ndarray | None:
        if self.gripper_index is None:
            return None
        return self.state[:, self.gripper_index]


@dataclass(frozen=True)
class Finding:
    check: str
    cause: Cause
    t0: float
    t1: float
    score: float
    evidence: str
    hard: bool = False
    source: str = "joints"  # 哪个独立来源：关节规则记 joints，视频评判记模型名


SOURCE_NAMES = {"joints": "关节", "meta": "元数据", "robometer": "Robometer", "topreward": "TOPReward"}


def source_name(source: str) -> str:
    return SOURCE_NAMES.get(source, source)


_CJK_LATIN = re.compile(r"([\u4e00-\u9fff])([A-Za-z])|([A-Za-z])([\u4e00-\u9fff])")


def spaced(text: str) -> str:
    """汉字和英文字母挨着时中间加一个空格，比如「Robometer报」写成「Robometer 报」。"""
    return _CJK_LATIN.sub(lambda m: f"{m[1]} {m[2]}" if m[1] else f"{m[3]} {m[4]}", text)


@dataclass
class LoadResult:
    episodes: list[Episode]
    failures: list[tuple[str, str]] = field(default_factory=list)  # (哪个文件, 为什么)
