"""视频评判的规则：从缓存的进度曲线里找问题。每个视频模型算一个独立来源（Finding.source 记模型名，
证据文字里不再重复模型名）。

阈值都在开发数据上定，出处写在阈值表旁边；某个模型还没定阈值的规则，记进 not_checked。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.judges.base import Curve
from jianjin.types import Cause, Episode, Finding, source_name, spaced

Found = dict[str, list[Finding]]
Skipped = dict[str, str]
Curves = dict[str, Curve]  # 示范 id -> 进度曲线

SPREAD_FLOOR = 0.01  # 进度的 1 个百分点：一批太齐时，别把一点小差别放大成异常


def _tail(x: list[float], k: int = 3) -> float:
    return float(np.mean(x[-k:]))


def _success_end(c: Curve) -> float:
    if c.success is None:
        raise NotApplicable(f"{c.model} 不给成功概率")
    return 1.0 - _tail(c.success)


# 分数越高越像「做的不是这个任务」；都是 1 减去某个 0 到 1 的量
BASE_SCORES: dict[str, Callable[[Curve], float]] = {
    "end": lambda c: 1.0 - _tail(c.progress),       # 结尾进度低
    "peak": lambda c: 1.0 - float(max(c.progress)),  # 全程最高进度低
    "mean": lambda c: 1.0 - float(np.mean(c.progress)),
    "success": _success_end,                         # 结尾成功概率低
}
SCORE_LABELS = {"end": "结尾进度", "peak": "全程最高进度", "mean": "平均进度", "success": "结尾的完成概率"}

# 指令不符：模型 -> (分数, 阈值)。和同任务那批比，差出阈值个四分位距以上就报。
WRONG_TASK: dict[str, tuple[str, float]] = {
    # 2026-10-03 考卷 A「换掉任务描述」dev 上定，原样报出不超过 5%（results/exam_a_swap_dev.json）
    "robometer": ("peak", 1.5477136091635564),
    # 2026-10-04 同上，TOPReward 对话格式（results/exam_a_swap_dev_topreward.json）；
    # 只在规则能用的 rel_ 分数里选，lerobot 默认格式只有 45/103，没选（设计文档 TOPReward 一节）
    "topreward": ("success", 3.320597328099884),
}

# 进度回落：模型 -> 阈值。进度从此前最高点往下掉超过这么多就报。
PROGRESS_DROP: dict[str, float] = {
    # 2026-10-03 改：去掉回原位那段以后，RoboVAD train 正常示范（307 条）进度回落的 95 分位
    # （results/exam_b_episode_train.json）。原来不去掉时是 0.345。
    "robometer": 0.23904454708099365,
    # 2026-10-04 TOPReward 对话格式，RoboVAD train 正常示范（307 条）进度回落的 95 分位
    # （results/exam_b_episode_train_topreward.json）。进度是每条示范里最小最大归一化的，所以阈值高
    "topreward": 0.7070742137558385,
}


def relative(values, ref) -> np.ndarray:
    """按参照那批的中位数和四分位距换算：比常态差多少个四分位距。"""
    ref = np.asarray(ref, dtype=np.float64)
    med = float(np.median(ref))
    spread = max(float(np.subtract(*np.percentile(ref, [75, 25]))), SPREAD_FLOOR)
    return (np.asarray(values, dtype=np.float64) - med) / spread


def _all_skipped(episodes: list[Episode], why: str) -> tuple[Found, Skipped]:
    return {}, {ep.id: why for ep in episodes}


def wrong_task(episodes: list[Episode], curves: Curves, judge: str, cfg: CheckConfig,
               reference: list[Episode] | None = None) -> tuple[Found, Skipped]:
    """进度比同一任务描述下那批示范差得多：做的可能不是这个任务。参照默认是这一批自己。"""
    name = source_name(judge)
    if judge not in WRONG_TASK:
        return _all_skipped(episodes, f"还没在开发数据上给 {name} 定「指令不符」的阈值")
    score_name, thr = WRONG_TASK[judge]
    fn = BASE_SCORES[score_name]
    groups: dict[str, list[float]] = defaultdict(list)
    for ep in reference if reference is not None else episodes:
        if ep.id in curves:
            groups[ep.task].append(fn(clip(curves[ep.id], ep.duration)))
    found: Found = {}
    skipped: Skipped = {}
    for ep in episodes:
        c = curves.get(ep.id)
        if c is None:
            skipped[ep.id] = f"缓存里没有 {name} 对这条示范的结果"
            continue
        ref = groups.get(ep.task, [])
        if len(ref) < cfg.min_task_episodes:
            skipped[ep.id] = f"同任务有 {name} 结果的只有 {len(ref)} 条，少于 {cfg.min_task_episodes} 条，没法比"
            continue
        z = float(relative([fn(clip(c, ep.duration))], ref)[0])
        if z <= thr:
            continue
        value, med = 1.0 - fn(clip(c, ep.duration)), 1.0 - float(np.median(ref))
        found[ep.id] = [Finding(
            check="wrong_task", cause=Cause.WRONG_TASK, t0=0.0, t1=ep.duration, score=z,
            evidence=spaced(f"{SCORE_LABELS[score_name]} {value:.0%}，同任务 {len(ref)} 条示范的中位数是 "
                            f"{med:.0%}；做的可能不是「{ep.task}」"),
            source=judge,
        )]
    return found, skipped


def drawdown(c: Curve) -> tuple[float, float, float, float, float]:
    """进度从此前最高点往下掉得最多的一次：(掉了多少, 最高点时刻, 最低点时刻, 最高, 最低)。"""
    p = np.asarray(c.progress, dtype=np.float64)
    t = np.asarray(c.times, dtype=np.float64)
    run = np.maximum.accumulate(p)
    k = int(np.argmax(run - p))
    i = int(np.argmax(p[: k + 1]))
    return float(run[k] - p[k]), float(t[i]), float(t[k]), float(p[i]), float(p[k])


# 回原位那段：关节（不含夹爪）离开头姿态的距离，最后一次不小于全程最远的 RETURN_FROM 倍之后。
# 手臂收回时进度模型常认不出已经做完（2026-10-03 核视频：叠方块报的 12 条、叠毛巾抽看的 3 条都是），
# 「进度回落」只看这之前。0.7 在 RoboVAD train 和考卷 A 开发集上选：去掉的那段里不能有标注异常的开头，
# 满足的取值里开发集原样报出最少（设计文档「核视频」一节）。没回到开头附近的示范，这个时刻落在结尾。
RETURN_FROM = 0.7


def return_start(ep: Episode, frac: float = RETURN_FROM) -> float:
    """手臂开始回原位的时刻（秒）；没回原位、没动过的，返回整条的时长。"""
    idx = [i for i, n in enumerate(ep.state_names) if i != ep.gripper_index and "grip" not in n.lower()]
    if not idx or len(ep.state) == 0:
        return ep.duration
    q = ep.state[:, idx]
    dist = np.nan_to_num(np.linalg.norm(q - q[0], axis=1))
    far = float(dist.max())
    if far <= 0:
        return ep.duration
    i = int(np.flatnonzero(dist >= frac * far)[-1])
    return ep.duration if i == len(dist) - 1 else float(ep.timestamps[i])


def clip(c: Curve, t_end: float) -> Curve:
    """只留 t_end 以前（含）的点；一个都不剩时留第一个。"""
    keep = [i for i, t in enumerate(c.times) if t <= t_end + 1e-6] or [0]

    def pick(xs):
        return None if xs is None else [xs[i] for i in keep]

    return Curve(c.model, c.version, pick(c.times), pick(c.progress), pick(c.success))


def rule_drawdown(ep: Episode, c: Curve) -> tuple[float, float, float, float, float]:
    """「进度回落」看的那段：关节数据有的时间里，手臂开始回原位以前。"""
    return drawdown(clip(c, min(ep.duration, return_start(ep))))


def progress_drop(episodes: list[Episode], curves: Curves, judge: str, cfg: CheckConfig,
                  reference: list[Episode] | None = None, threshold: float | None = None) -> tuple[Found, Skipped]:
    """进度突然往回掉：东西掉了，或者退回去重做。只看关节数据有的时间里、手臂开始回原位以前（见 RETURN_FROM）。
    不和同任务比，cfg、reference 只为和别的规则同一个签名。"""
    name = source_name(judge)
    thr = threshold if threshold is not None else PROGRESS_DROP.get(judge)
    if thr is None:
        return _all_skipped(episodes, f"还没在开发数据上给 {name} 定「进度回落」的阈值")
    found: Found = {}
    skipped: Skipped = {}
    for ep in episodes:
        c = curves.get(ep.id)
        if c is None:
            skipped[ep.id] = f"缓存里没有 {name} 对这条示范的结果"
            continue
        d, t0, t1, hi, lo = rule_drawdown(ep, c)
        if d > thr:
            found[ep.id] = [Finding(
                check="progress_drop", cause=Cause.DROP, t0=t0, t1=t1, score=d,
                evidence=f"进度从 {hi:.0%} 掉到 {lo:.0%}，可能东西掉了或者退回去重做",
                source=judge,
            )]
    return found, skipped


VIDEO_RULES = {"wrong_task": wrong_task, "progress_drop": progress_drop}


def run_video_checks(episodes: list[Episode], curves: dict[str, Curves], cfg: CheckConfig | None = None,
                     reference: list[Episode] | None = None) -> dict[str, tuple[list[Finding], dict[str, str]]]:
    """curves：{模型名: {示范 id: 曲线}}。返回 {示范 id: (发现, 没查的)}，没查的键是「模型.规则」。"""
    cfg = cfg or CheckConfig()
    out: dict[str, tuple[list[Finding], dict[str, str]]] = {ep.id: ([], {}) for ep in episodes}
    for judge, cs in curves.items():
        for name, rule in VIDEO_RULES.items():
            found, skipped = rule(episodes, cs, judge, cfg, reference=reference)
            for eid, fs in found.items():
                out[eid][0].extend(fs)
            for eid, why in skipped.items():
                out[eid][1][f"{judge}.{name}"] = why
    return out
