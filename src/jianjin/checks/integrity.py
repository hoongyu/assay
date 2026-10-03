"""客观硬规则：数据本身坏了。一条触发就能判作废（设计文档 4.3）。"""

from __future__ import annotations

import numpy as np

from jianjin.checks.config import CheckConfig
from jianjin.checks.signal import runs
from jianjin.types import Cause, Episode, Finding


def missing_values(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    out = []
    arrays = {"关节状态": ep.state}
    if ep.action is not None:
        arrays["动作"] = ep.action
    for name, arr in arrays.items():
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
                evidence=f"{name}有 {len(idx)} 帧读数缺失（NaN 或无穷）",
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
            evidence=f"整条只有 {ep.duration:.2f} 秒，短于 {cfg.min_duration_s} 秒",
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
        evidence = "时间戳倒退或重复" if backwards else f"丢了约 {lost} 帧"
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
