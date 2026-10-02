"""要和同任务其他示范比才看得出来的问题。

比较对象默认是这一批自己；考卷里整批都注入了缺陷，就传 reference（原样示范）进来比。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from jianjin.checks.config import CheckConfig
from jianjin.checks.signal import joint_scale, path_length
from jianjin.types import Cause, Episode, Finding

Found = dict[str, list[Finding]]
Skipped = dict[str, str]


def _key(ep: Episode) -> tuple[str, int]:
    return (ep.task, ep.state.shape[1])


def _groups(episodes: list[Episode]) -> dict[tuple[str, int], list[Episode]]:
    groups: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for ep in episodes:
        groups[_key(ep)].append(ep)
    return groups


def _compare(episodes, cfg, reference, summarize, judge) -> tuple[Found, Skipped]:
    """按任务分组；参照组太小的记 not_checked，否则用 summarize(参照组) 的统计量逐条 judge。"""
    ref = _groups(reference if reference is not None else episodes)
    stats: dict = {}
    found: Found = {}
    skipped: Skipped = {}
    for ep in episodes:
        group = ref.get(_key(ep), [])
        if len(group) < cfg.min_task_episodes:
            skipped[ep.id] = f"同任务只有 {len(group)} 条，少于 {cfg.min_task_episodes} 条，没法比"
            continue
        if _key(ep) not in stats:
            stats[_key(ep)] = summarize(group)
        finding = judge(ep, stats[_key(ep)], len(group))
        if finding is not None:
            found[ep.id] = [finding]
    return found, skipped


def extra_motion(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[Found, Skipped]:
    """路程远超同任务中位数，算多余动作。"""

    def summarize(group):
        scale = joint_scale(np.concatenate([e.state for e in group]))
        return scale, float(np.median([path_length(e.state, scale) for e in group]))

    def judge(ep, stat, n):
        scale, med = stat
        if med <= 0:
            return None
        ratio = path_length(ep.state, scale) / med
        if ratio <= cfg.extra_factor:
            return None
        return Finding(
            check="extra_motion",
            cause=Cause.EXTRA_MOTION,
            t0=0.0,
            t1=ep.duration,
            score=float(ratio),
            evidence=f"关节走过的路程是同任务 {n} 条示范中位数的 {ratio:.1f} 倍",
        )

    return _compare(episodes, cfg, reference, summarize, judge)


def end_pose(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[Found, Skipped]:
    """最后一帧的姿态离同任务通常的收尾姿态太远，像是没做完就停了。"""

    def summarize(group):
        scale = joint_scale(np.concatenate([e.state for e in group]))
        return scale, np.median(np.array([e.state[-1] for e in group]), axis=0)

    def judge(ep, stat, n):
        scale, med = stat
        off = np.nan_to_num(np.abs(ep.state[-1] - med) / scale)
        j = int(np.argmax(off))
        if off[j] <= cfg.end_pose_max:
            return None
        t1 = ep.duration
        return Finding(
            check="end_pose",
            cause=Cause.INCOMPLETE,
            t0=max(0.0, t1 - 1.0),
            t1=t1,
            score=float(off[j]),
            evidence=(
                f"最后停的姿态离同任务 {n} 条示范的常见收尾姿态差得远："
                f"{ep.state_names[j]} 差了 {off[j]:.0%} 个活动幅度，像是没做完就停了"
            ),
        )

    return _compare(episodes, cfg, reference, summarize, judge)
