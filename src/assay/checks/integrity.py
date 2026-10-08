"""客观硬规则：数据本身坏了。一条触发就能判作废（设计文档 4.3）。"""

from __future__ import annotations

import numpy as np

from assay.checks.config import CheckConfig, NotApplicable
from assay.checks.signal import runs
from assay.types import Bi, Cause, Episode, Finding


def missing_values(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    out = []
    arrays = {("关节状态", "Joint state"): ep.state}
    if ep.action is not None:
        arrays[("动作", "Action")] = ep.action
    for (name, name_en), arr in arrays.items():
        bad = ~np.isfinite(arr).all(axis=1)
        if not bad.any():
            continue
        idx = np.flatnonzero(bad)
        out.append(
            Finding(
                check="missing_values",
                cause=Cause.DATA_ERROR,
                t0=idx[0] / ep.fps,
                t1=(idx[-1] + 1) / ep.fps,
                score=float(len(idx)),
                evidence=Bi(f"{name}有 {len(idx)} 帧读数缺失（NaN 或无穷）",
                            f"{name_en} has {len(idx)} frames with missing readings (NaN or infinite)"),
                hard=True,
            )
        )
    return out


def too_short(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    if ep.duration >= cfg.min_duration_s:
        return []
    return [
        Finding(
            check="too_short",
            cause=Cause.DATA_ERROR,
            t0=0.0,
            t1=ep.duration,
            score=ep.duration,
            evidence=Bi(f"整条只有 {ep.duration:.2f} 秒，短于 {cfg.min_duration_s} 秒",
                        f"The whole episode lasts only {ep.duration:.2f} s, shorter than {cfg.min_duration_s} s"),
            hard=True,
        )
    ]


def timestamp_gaps(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    ts = ep.timestamps
    if len(ts) < 2:
        return []
    dt = np.diff(ts)
    bad = (dt > cfg.max_gap_factor / ep.fps) | (dt <= 0)
    out = []
    for s, e in runs(bad):
        seg = dt[s:e]
        lost = int(np.sum(np.clip(np.round(seg * ep.fps) - 1, 0, None)))
        backwards = bool((seg <= 0).any())
        evidence = (Bi("时间戳倒退或重复", "Timestamps go backwards or repeat") if backwards
                    else Bi(f"丢了约 {lost} 帧", f"About {lost} frames were dropped"))
        out.append(
            Finding(
                check="timestamp_gaps",
                cause=Cause.DATA_ERROR,
                t0=float(ts[s]),
                t1=float(ts[e]),
                score=float(seg.max()),
                evidence=evidence,
                hard=True,
            )
        )
    return out


def _cams(bad: dict[str, float], known: dict[str, float]) -> tuple[str, str]:
    if len(bad) == len(known) and len(known) > 1:
        return f"{len(known)} 路相机都这样", f"all {len(known)} cameras"
    names = [c.split(".")[-1] for c in sorted(bad)]
    return "、".join(names), ", ".join(names)


def video_length(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    """元数据里每路视频的时长要和关节数据一样长。只读元数据，不解码视频。

    视频短：最后一段关节数据没有画面，训练时取不到帧，判作废。
    视频长：多出来的画面对不上关节数据，要人看（叠毛巾 ep83 多出的 8 秒是复位画面）。
    """
    if not ep.cameras:
        return []
    known = {cam: ref.to_s - ref.from_s for cam, ref in ep.cameras.items() if ref.to_s is not None}
    if not known:
        raise NotApplicable(Bi("元数据里没写视频时长，没法和关节数据对",
                               "The metadata gives no video length to compare with the joint data"))
    slack = cfg.video_slack_frames / ep.fps
    short = {c: ep.duration - n for c, n in known.items() if n < ep.duration - slack}
    long = {c: n - ep.duration for c, n in known.items() if n > ep.duration + slack}
    out = []
    if short:
        gap = max(short.values())
        zh, en = _cams(short, known)
        out.append(Finding(
            check="video_length", cause=Cause.DATA_ERROR, t0=max(0.0, ep.duration - gap), t1=ep.duration,
            score=gap, evidence=Bi(f"视频比关节数据短 {gap:.1f} 秒（{zh}），最后这段关节数据没有画面",
                                   f"The video is {gap:.1f} s shorter than the joint data ({en}); "
                                   "the last part of the joint data has no picture"),
            hard=True, source="meta",
        ))
    if long:
        extra = max(long.values())
        zh, en = _cams(long, known)
        out.append(Finding(
            check="video_length", cause=Cause.DATA_ERROR, t0=0.0, t1=ep.duration, score=extra,
            evidence=Bi(f"视频比关节数据长 {extra:.1f} 秒（{zh}），两路时长对不上，要看多出来的画面是什么",
                        f"The video is {extra:.1f} s longer than the joint data ({en}); the two do not match, "
                        "so check what the extra footage shows"),
            source="meta",
        ))
    return out
