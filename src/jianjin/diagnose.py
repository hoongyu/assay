"""诊断：把关节规则和视频评判的发现合成一个判定（设计文档 4.3）。

- 作废：客观硬规则触发；或者两个不同来源指向同一处问题。
  「同一处」是时间区间重叠，而且原因相同，或者两条都只占示范的一段。
  只看重叠不够：「指令不符」这类发现覆盖整条示范，和什么都重叠（2026-10-03 改）。
  整条的「空录」和整条的「指令不符」算同一处问题（2026-10-03 加）。
- 存疑：有发现，但够不上作废，要人看。
- 合格：什么都没报。没查的项目记在 not_checked，不当成通过。
错杀一条好数据比漏掉一条坏数据贵，所以判得偏保守。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations

from jianjin.checks import CheckConfig, run_checks
from jianjin.checks.video import Curves, run_video_checks
from jianjin.judges.base import Curve
from jianjin.types import Cause, Episode, Finding, source_name, spaced


class Label(StrEnum):
    PASS = "合格"
    DOUBT = "存疑"
    REJECT = "作废"


@dataclass
class Verdict:
    episode_id: str
    label: Label
    reasons: list[str]
    causes: list[Finding]
    not_checked: dict[str, str] = field(default_factory=dict)
    progress: Curve | None = None


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _whole(f: Finding, duration: float) -> bool:
    return f.t0 <= 0.0 and f.t1 >= duration - 1e-9


# 原因不同、但说的是同一件事的：两个来源各报一条，而且都覆盖整条示范，就算指向同一处问题。
# 关节的「空录」（手臂几乎没动）和视频的「指令不符」（没看到在做这个任务），都是说这条没做指令里的任务。
SAME_PROBLEM = {frozenset({Cause.EMPTY, Cause.WRONG_TASK}): "整条没做指令里的任务"}


def _agree(a: Finding, b: Finding, duration: float) -> bool:
    if a.source == b.source or not overlaps((a.t0, a.t1), (b.t0, b.t1)):
        return False
    whole_a, whole_b = _whole(a, duration), _whole(b, duration)
    if whole_a and whole_b:
        return a.cause == b.cause or frozenset({a.cause, b.cause}) in SAME_PROBLEM
    return a.cause == b.cause or not (whole_a or whole_b)


def label_of(findings: list[Finding], duration: float) -> tuple[Label, list[str]]:
    hard = [f for f in findings if f.hard]
    if hard:
        return Label.REJECT, [f"客观硬规则：{f.evidence}" for f in hard]
    agree = []
    for a, b in combinations(findings, 2):
        if not _agree(a, b, duration):
            continue
        when = f"第 {max(a.t0, b.t0):.1f} 到 {min(a.t1, b.t1):.1f} 秒"
        if a.cause == b.cause:
            agree.append(f"{source_name(a.source)}和{source_name(b.source)}都报了「{a.cause.value}」，{when}重叠")
        elif frozenset({a.cause, b.cause}) in SAME_PROBLEM:
            agree.append(f"{source_name(a.source)}报「{a.cause.value}」、{source_name(b.source)}报「{b.cause.value}」，"
                         f"说的都是{SAME_PROBLEM[frozenset({a.cause, b.cause})]}")
        else:
            agree.append(f"{source_name(a.source)}报「{a.cause.value}」、{source_name(b.source)}报「{b.cause.value}」，"
                         f"{when}重叠，指向同一处问题")
    if agree:
        return Label.REJECT, [spaced(r) for r in agree]
    if findings:
        names = "、".join(sorted({source_name(f.source) for f in findings}))
        if len({f.source for f in findings}) == 1:
            return Label.DOUBT, [spaced(f"只有{names}报了问题，没有第二个来源印证，要人看")]
        return Label.DOUBT, [spaced(f"{names}报的问题对不上（时间不重叠，或者原因不同），要人看")]
    return Label.PASS, []


def diagnose(episode_id: str, duration: float, findings: list[Finding],
             not_checked: dict[str, str] | None = None, progress: Curve | None = None) -> Verdict:
    label, reasons = label_of(findings, duration)
    causes = sorted(findings, key=lambda f: (f.t0, f.t1))
    return Verdict(episode_id, label, reasons, causes, dict(not_checked or {}), progress)


PROGRESS_FROM = ("robometer", "topreward")  # 报告里画谁的进度曲线，按这个顺序找


def diagnose_batch(episodes: list[Episode], curves: dict[str, Curves] | None = None,
                   cfg: CheckConfig | None = None, reference: list[Episode] | None = None) -> list[Verdict]:
    """关节规则加视频规则，逐条出判定。curves 是 {模型名: {示范 id: 曲线}}，没有就只跑关节体检。"""
    curves = curves or {}
    checks = {r.episode_id: r for r in run_checks(episodes, cfg, reference=reference)}
    video = run_video_checks(episodes, curves, cfg, reference)
    order = [j for j in PROGRESS_FROM if j in curves] + [j for j in curves if j not in PROGRESS_FROM]
    out = []
    for ep in episodes:
        r = checks[ep.id]
        vf, vn = video[ep.id]
        not_checked = {**r.not_checked, **vn}
        if not curves:
            not_checked["视频评判"] = "没跑视频评判（没有 GPU，或者缓存里没有结果）"
        progress = next((curves[j][ep.id] for j in order if ep.id in curves[j]), None)
        out.append(diagnose(ep.id, ep.duration, r.findings + vf, not_checked, progress))
    return out
