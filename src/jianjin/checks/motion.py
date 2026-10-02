"""看关节怎么动：中途停太久、录到最后还在动。都是软信号，单独一条只判存疑。"""

from __future__ import annotations

import numpy as np

from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.checks.signal import runs, speed
from jianjin.types import Cause, Episode, Finding


def long_pause(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    v = speed(ep.state, ep.fps)
    out = []
    for s, e in runs(v < cfg.still_rate):
        if s == 0 or e == len(v):  # 开头摆姿势、结尾收手时的静止是正常的
            continue
        dur = (e - s) / ep.fps
        if dur < cfg.pause_s:
            continue
        t0, t1 = s / ep.fps, e / ep.fps
        out.append(
            Finding(
                check="long_pause",
                cause=Cause.LONG_PAUSE,
                t0=t0,
                t1=t1,
                score=dur,
                evidence=f"第 {t0:.1f} 到 {t1:.1f} 秒关节几乎不动，停了 {dur:.1f} 秒",
            )
        )
    return out


def end_in_motion(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    v = speed(ep.state, ep.fps)
    k = max(1, int(round(cfg.end_window_s * ep.fps)))
    if len(v) < 2 * k:
        raise NotApplicable("示范太短，看不了结尾")
    tail = float(np.mean(v[-k:]))
    if tail <= cfg.end_moving_rate:
        return []
    t1 = ep.duration
    return [
        Finding(
            check="end_in_motion",
            cause=Cause.INCOMPLETE,
            t0=t1 - k / ep.fps,
            t1=t1,
            score=tail,
            evidence=f"录到最后 {k / ep.fps:.1f} 秒手臂还在动（每秒 {tail:.2f} 个幅度），像是没做完就停了",
        )
    ]
