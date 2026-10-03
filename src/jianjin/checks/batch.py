"""要和同任务其他示范比才看得出来的问题。

比较对象默认是这一批自己；考卷里整批都注入了缺陷，就传 reference（原样示范）进来比。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.checks.signal import excursion, joint_scale, path_length
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
            try:
                stats[_key(ep)] = summarize(group)
            except NotApplicable as exc:  # 这一批本身不满足规则的前提
                stats[_key(ep)] = exc
        stat = stats[_key(ep)]
        if isinstance(stat, NotApplicable):
            skipped[ep.id] = str(stat)
            continue
        finding = judge(ep, stat, len(group))
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
    """最后一帧的姿态离同任务通常的收尾姿态太远，像是没做完就停了。

    每个关节按同任务那批收尾位置的四分位距换算（下限是活动幅度的 end_pose_floor 倍）：收尾本来就不固定的关节
    （比如停稳以后的手腕转角）差一点不算。整批收尾都不固定的（各关节收尾位置的四分位距按活动幅度算、取中位，
    超过 end_spread_max），这条规则查不了，记进没查。2026-10-03 改，见设计文档考卷 A「第三批结果」。
    """

    def summarize(group):
        scale = joint_scale(np.concatenate([e.state for e in group]))
        finals = np.array([e.state[-1] for e in group], dtype=np.float64)
        q75, q25 = np.nanpercentile(finals, [75, 25], axis=0)
        iqr = q75 - q25
        moving = np.isfinite(scale)
        spread = float(np.median(iqr[moving] / scale[moving])) if moving.any() else 0.0
        if spread > cfg.end_spread_max:
            raise NotApplicable(
                f"同任务 {len(group)} 条示范停下来的姿态本来就不固定（各关节收尾位置一般差开活动幅度的 {spread:.0%}），"
                "没法拿收尾姿态判没做完"
            )
        return np.nanmedian(finals, axis=0), np.maximum(iqr, cfg.end_pose_floor * scale)

    def judge(ep, stat, n):
        med, unit = stat
        off = np.nan_to_num(np.abs(ep.state[-1] - med) / unit)
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
                f"最后停的姿态和同任务 {n} 条示范的常见收尾姿态差得远：{ep.state_names[j]} 差出的距离"
                f"是示范之间常见出入的 {off[j]:.1f} 倍，像是没做完就停了"
            ),
        )

    return _compare(episodes, cfg, reference, summarize, judge)


def empty_recording(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[Found, Skipped]:
    """手臂整条几乎没动：离开头姿态最远只到同任务中位数的一小部分，录下的不是示范（空录，或者录的是人在复位）。

    「停顿过长」抓不到这种：它按本条示范自己的活动幅度换算速度，而且不算开头、结尾的静止。
    2026-10-03 开发集上，空录和复位是 0.001 以下，正常示范在 0.75 以上，阈值取 0.2。
    """

    def summarize(group):
        scale = joint_scale(np.concatenate([e.state for e in group]))
        return scale, float(np.median([excursion(e.state, scale) for e in group]))

    def judge(ep, stat, n):
        scale, med = stat
        if med <= 0 or ep.duration < cfg.min_duration_s:  # 过短的已经由「示范过短」判了
            return None
        ratio = excursion(ep.state, scale) / med
        if ratio >= cfg.empty_max:
            return None
        return Finding(
            check="empty_recording",
            cause=Cause.EMPTY,
            t0=0.0,
            t1=ep.duration,
            score=float(ratio),
            evidence=f"手臂几乎没动：离开头姿态最远只到同任务 {n} 条示范中位数的 {ratio:.0%}",
        )

    return _compare(episodes, cfg, reference, summarize, judge)
