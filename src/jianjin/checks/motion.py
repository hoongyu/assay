"""看关节怎么动：中途停太久。软信号，单独一条只判存疑。"""

from __future__ import annotations

from jianjin.checks.config import CheckConfig
from jianjin.checks.signal import runs, speed
from jianjin.types import Bi, Cause, Episode, Finding


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
                evidence=Bi(f"关节几乎不动，停了 {dur:.1f} 秒", f"The joints barely moved for {dur:.1f} s"),
            )
        )
    return out

