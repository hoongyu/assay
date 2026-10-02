"""要和同任务其他示范比才看得出来的问题。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from jianjin.checks.config import CheckConfig
from jianjin.checks.signal import joint_scale, path_length
from jianjin.types import Cause, Episode, Finding


def extra_motion(
    episodes: list[Episode], cfg: CheckConfig
) -> tuple[dict[str, list[Finding]], dict[str, str]]:
    """路程远超同任务中位数，算多余动作。返回 (发现, 没查的及原因)。"""
    groups: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for ep in episodes:
        groups[(ep.task, ep.state.shape[1])].append(ep)

    found: dict[str, list[Finding]] = {}
    skipped: dict[str, str] = {}
    for (task, _), group in groups.items():
        if len(group) < cfg.min_task_episodes:
            for ep in group:
                skipped[ep.id] = f"同任务只有 {len(group)} 条，少于 {cfg.min_task_episodes} 条，没法比"
            continue
        scale = joint_scale(np.concatenate([ep.state for ep in group]))
        lengths = np.array([path_length(ep.state, scale) for ep in group])
        med = float(np.median(lengths))
        if med <= 0:
            continue
        for ep, length in zip(group, lengths):
            ratio = length / med
            if ratio > cfg.extra_factor:
                found[ep.id] = [
                    Finding(
                        check="extra_motion",
                        cause=Cause.EXTRA_MOTION,
                        t0=0.0,
                        t1=ep.duration,
                        score=float(ratio),
                        evidence=f"关节走过的路程是同任务 {len(group)} 条示范中位数的 {ratio:.1f} 倍",
                    )
                ]
    return found, skipped
