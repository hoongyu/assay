"""要和同任务其他示范比才看得出来的问题。

比较对象默认是这一批自己；考卷里整批都注入了缺陷，就传 reference（原样示范）进来比。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.checks.signal import (
    arm_channels,
    bursts,
    excursion,
    gripper_channels,
    jerk,
    joint_scale,
    level_changes,
    path_length,
    speed,
    stops,
)
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


@dataclass(frozen=True)
class RegraspScore:
    """一条示范和参照那批比，多开合了几次（取多出最多的那只手）。"""

    n: int  # 参照那批有几条
    excess: float  # 开合次数减常见上限
    channel: str  # 哪只手的夹爪通道
    count: int  # 开合次数（张开、合上各算一次）
    upper: float  # 常见上限：参照那批开合次数的 regrasp_quantile 分位
    times: tuple[float, ...]  # 每次换档的时刻，秒，从示范开头算


def _grip(ep: Episode, source: str) -> np.ndarray:
    """source 是 action、主臂数据在而且宽度和从臂一样，就读主臂；否则读从臂。"""
    if source == "action" and ep.action is not None and ep.action.shape == ep.state.shape:
        return ep.action
    return ep.state


def _hand(name: str) -> str:
    low = name.lower()
    return "左手夹爪" if "left" in low else ("右手夹爪" if "right" in low else "夹爪")


def _changes(ep: Episode, c: int, lo: float, hi: float, cfg: CheckConfig) -> list[int]:
    """这条示范第 c 个通道换档的帧；在一档里待不够 grip_dwell_s 秒的那一趟不算。"""
    min_frames = max(1, round(cfg.grip_dwell_s * ep.fps))
    return level_changes(_grip(ep, cfg.regrasp_source)[:, c], lo, hi, cfg.grip_band, min_frames)


def regrasp_scores(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[dict[str, RegraspScore], Skipped]:
    """每条能比的示范多开合了几次；没法比的写原因。「二次抓取」规则和考卷都用它。"""

    def summarize(group):
        chans = gripper_channels(group[0].state_names, group[0].gripper_index)
        if not chans:
            raise NotApplicable("认不出夹爪通道")
        stats = {}
        for c in chans:
            lo, hi = np.nanpercentile(np.concatenate([_grip(e, cfg.regrasp_source)[:, c] for e in group]), [5, 95])
            counts = [len(_changes(e, c, lo, hi, cfg)) for e in group]
            if np.median(counts) == 0:  # 这只手基本不用
                continue
            stats[c] = (float(lo), float(hi), float(np.quantile(counts, cfg.regrasp_quantile, method="higher")))
        if not stats:
            raise NotApplicable(f"同任务 {len(group)} 条示范的夹爪基本不开合，不查二次抓取")
        return stats

    def judge(ep, stats, n):
        best = None
        for c, (lo, hi, upper) in stats.items():
            idx = _changes(ep, c, lo, hi, cfg)
            if best is None or len(idx) - upper > best.excess:
                best = RegraspScore(n, len(idx) - upper, ep.state_names[c], len(idx), upper,
                                    tuple(i / ep.fps for i in idx))
        return best

    found, skipped = _compare(episodes, cfg, reference, summarize, judge)
    return {k: v[0] for k, v in found.items()}, skipped


def regrasp(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[Found, Skipped]:
    """夹爪开合次数超过同任务那批的「常见上限 + regrasp_margin」：像是抓了不止一回（二次抓取）。

    次数：读数按参照那批的 5 到 95 分位归一化，两档滞回数换档（signal.level_changes），在一档里待不够
    grip_dwell_s 秒的那一趟不算；常见上限是参照那批
    开合次数的 regrasp_quantile 分位。时间段写整条示范（和「多余动作」一样），换档时刻写进证据。
    2026-10-04 加，见 docs/superpowers/specs/2026-10-04-regrasp-design.md。
    """
    scores, skipped = regrasp_scores(episodes, cfg, reference)
    found: Found = {}
    for ep in episodes:
        s = scores.get(ep.id)
        if s is None or s.excess <= cfg.regrasp_margin:
            continue
        shown = "、".join(f"{t:.1f}" for t in s.times[:12]) + ("......" if len(s.times) > 12 else "")
        found[ep.id] = [Finding(
            check="regrasp",
            cause=Cause.REGRASP,
            t0=0.0,
            t1=ep.duration,
            score=float(s.excess),
            evidence=(f"{_hand(s.channel)}开合了 {s.count} 次（张开、合上各算一次），同任务 {s.n} 条示范常见不超过 "
                      f"{s.upper:.0f} 次，像是抓了不止一回；换档在 {shown} 秒"),
        )]
    return found, skipped


# ---- 动作不流畅、速度过快 ----
# 2026-10-05 加，见 docs/superpowers/specs/2026-10-05-motion-quality-design.md。motion_scores 一次算出全部候选，
# 考卷 C 拿它比；规则只用选中的那个。

# 候选名：(哪一类问题, 阈值怎么比)。ratio 是和参照那批中位数的比值，count 是次数减参照那批的常见上限。
MOTION_CANDIDATES = {
    "duration": ("jerky", "ratio"),  # 基线：示范比同任务长多少
    "stops": ("jerky", "count"),
    "bursts": ("jerky", "count"),
    "jerk": ("jerky", "ratio"),
    "duration_short": ("fast", "ratio"),  # 基线：示范比同任务短多少
    "peak_speed": ("fast", "ratio"),
    "mean_speed": ("fast", "ratio"),
}


@dataclass(frozen=True)
class MotionScore:
    """一条示范和参照那批比，在各候选上的分数（越大越像有问题），以及写证据用的细节。"""

    n: int  # 参照那批有几条
    scores: dict[str, float]
    detail: dict


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def motion_scores(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[dict[str, MotionScore], Skipped]:
    """只看手臂关节（夹爪、灵巧手不算），按参照那批每个关节的幅度归一化；逐帧速度是动得最快的那个关节的。"""

    def basics(e: Episode, arm: list[int], scale: np.ndarray):
        s = e.state[:, arm]
        v = speed(s, e.fps, scale)
        j = jerk(s, e.fps, scale)
        st = stops(v, cfg.still_rate, max(1, round(cfg.stop_s * e.fps)))
        p95 = float(np.nanpercentile(v, 95)) if len(v) else 0.0
        jm = float(np.nanmean(j)) if len(j) else 0.0
        return v, st, p95, jm, path_length(s, scale) / e.duration

    def segments(v: np.ndarray, peak: float, fps: float) -> list[int]:
        return bursts(v, cfg.burst_low * peak, cfg.burst_high * peak, max(1, round(cfg.burst_dwell_s * fps)))

    def summarize(group):
        arm = arm_channels(group[0].state_names)
        if not arm:
            raise NotApplicable("认不出手臂关节")
        scale = joint_scale(np.concatenate([e.state[:, arm] for e in group]))
        if not np.isfinite(scale).any():
            raise NotApplicable(f"同任务 {len(group)} 条示范的手臂基本没动，不查动作顺不顺、快不快")
        b = [basics(e, arm, scale) for e in group]
        peak = float(np.median([x[2] for x in b]))

        def upper(xs):
            return float(np.quantile(xs, cfg.motion_quantile, method="higher"))

        return {
            "arm": arm,
            "scale": scale,
            "peak": peak,
            "duration": float(np.median([e.duration for e in group])),
            "stop_upper": upper([len(x[1]) for x in b]),
            "burst_upper": upper([len(segments(x[0], peak, e.fps)) for x, e in zip(b, group)]),
            "jerk": float(np.median([x[3] for x in b])),
            "mean_speed": float(np.median([x[4] for x in b])),
        }

    def judge(ep, ref, n):
        v, st, p95, jm, mean_speed = basics(ep, ref["arm"], ref["scale"])
        seg = segments(v, ref["peak"], ep.fps)
        scores = {
            "duration": _ratio(ep.duration, ref["duration"]),
            "stops": len(st) - ref["stop_upper"],
            "bursts": len(seg) - ref["burst_upper"],
            "jerk": _ratio(jm, ref["jerk"]),
            "duration_short": _ratio(ref["duration"], ep.duration),
            "peak_speed": _ratio(p95, ref["peak"]),
            "mean_speed": _ratio(mean_speed, ref["mean_speed"]),
        }
        detail = {
            "duration": ep.duration,
            "ref_duration": ref["duration"],
            "n_stops": len(st),
            "stop_upper": ref["stop_upper"],
            "stop_times": [(s / ep.fps, e / ep.fps) for s, e in st],
            "n_bursts": len(seg),
            "burst_upper": ref["burst_upper"],
        }
        return MotionScore(n, scores, detail)

    found, skipped = _compare(episodes, cfg, reference, summarize, judge)
    return {k: v[0] for k, v in found.items()}, skipped
