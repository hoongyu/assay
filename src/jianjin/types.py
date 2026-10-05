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
    JERKY = "动作不流畅"
    WRONG_TASK = "指令不符"
    DATA_ERROR = "数据异常"


CAUSE_EN = {
    Cause.INCOMPLETE: "incomplete", Cause.EMPTY: "empty recording", Cause.MISPLACED: "mispositioned",
    Cause.GRASP_MISS: "missed grasp", Cause.REGRASP: "regrasp", Cause.DROP: "object dropped",
    Cause.EXTRA_MOTION: "extra motion", Cause.LONG_PAUSE: "long pause", Cause.JERKY: "unsmooth motion",
    Cause.WRONG_TASK: "wrong task", Cause.DATA_ERROR: "data error",
}


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
    moments: tuple[float, ...] = ()  # 发现落在哪几个时刻（秒），比如二次抓取的换档；配关键帧时按它取


SOURCE_NAMES = {"joints": "关节", "meta": "元数据", "robometer": "Robometer", "topreward": "TOPReward"}
SOURCE_EN = {"joints": "joints", "meta": "metadata", "robometer": "Robometer", "topreward": "TOPReward"}


def source_name(source: str, lang: str = "zh") -> str:
    return (SOURCE_EN if lang == "en" else SOURCE_NAMES).get(source, source)


class Bi(str):
    """\u5e26\u82f1\u6587\u7684\u4e2d\u6587\uff1a\u5f53\u666e\u901a\u5b57\u7b26\u4e32\u7528\u65f6\u5c31\u662f\u4e2d\u6587\uff0c.en \u662f\u540c\u4e00\u53e5\u8bdd\u7684\u82f1\u6587\uff08\u82f1\u6587\u62a5\u544a\u7528\uff09\u3002

    \u8bc1\u636e\u3001\u5224\u5b9a\u7406\u7531\u3001\u6ca1\u67e5\u7684\u539f\u56e0\u90fd\u7528\u5b83\uff0c\u4e2d\u82f1\u4e24\u4efd\u5728\u540c\u4e00\u5904\u3001\u7528\u540c\u4e00\u7ec4\u6570\u5199\u51fa\u6765\uff0c\u4e0d\u7ecf\u8fc7\u6a21\u578b\uff1b
    \u53ea\u8ba4 str \u7684\u8001\u4ee3\u7801\u7167\u65e7\u62ff\u5230\u4e2d\u6587\u3002\u8bbe\u8ba1\u89c1 docs/superpowers/specs/2026-10-05-english-report-design.md\u3002
    """

    en: str

    def __new__(cls, zh: str, en: str = ""):
        obj = super().__new__(cls, zh)
        obj.en = en
        return obj


def en(text: str) -> str:
    """\u53d6\u82f1\u6587\uff1b\u6ca1\u6709\u82f1\u6587\u7684\u9000\u56de\u539f\u6587\u3002"""
    return getattr(text, "en", "") or text


_CJK_LATIN = re.compile(r"([\u4e00-\u9fff])([A-Za-z])|([A-Za-z])([\u4e00-\u9fff])")


def spaced(text: str) -> str:
    """汉字和英文字母挨着时中间加一个空格，比如「Robometer报」写成「Robometer 报」。"""
    return _CJK_LATIN.sub(lambda m: f"{m[1]} {m[2]}" if m[1] else f"{m[3]} {m[4]}", text)


@dataclass
class LoadResult:
    episodes: list[Episode]
    failures: list[tuple[str, str]] = field(default_factory=list)  # (哪个文件, 为什么)
