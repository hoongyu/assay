"""考卷 A：在真机录的成功示范上动手脚，每条的答案已知。这是造出来的假失败，不能说成真实失败。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from jianjin.types import Cause, Episode, VideoRef


@dataclass(frozen=True)
class Defect:
    kind: str
    cause: Cause
    t0: float
    t1: float


def _rebuild(ep: Episode, state: np.ndarray, action: np.ndarray | None, kind: str, **changes) -> Episode:
    return replace(
        ep,
        id=f"{ep.id}+{kind}",
        state=state,
        action=action,
        timestamps=np.arange(len(state)) / ep.fps,
        meta={**ep.meta, "injected": kind},
        **changes,
    )


def truncate_end(ep: Episode, rng: np.random.Generator) -> tuple[Episode, Defect]:
    """只留前 50% 到 80%，对应「没做完」。"""
    keep = int(len(ep.state) * rng.uniform(0.5, 0.8))
    action = ep.action[:keep] if ep.action is not None else None
    dur = keep / ep.fps
    cams = {k: VideoRef(v.path, v.from_s, v.from_s + dur) for k, v in ep.cameras.items()}
    new = _rebuild(ep, ep.state[:keep], action, "truncate_end", cameras=cams)
    return new, Defect("truncate_end", Cause.INCOMPLETE, max(0.0, dur - 1.0), dur)


def insert_pause(ep: Episode, rng: np.random.Generator) -> tuple[Episode, Defect]:
    """在 20% 到 80% 之间某一帧，重复这一帧 4 到 8 秒，对应「停顿过长」。"""
    n = len(ep.state)
    k = int(rng.integers(int(0.2 * n), int(0.8 * n)))
    hold = int(round(rng.uniform(4.0, 8.0) * ep.fps))

    def stretch(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a[: k + 1], np.repeat(a[k : k + 1], hold, axis=0), a[k + 1 :]])

    action = stretch(ep.action) if ep.action is not None else None
    new = _rebuild(ep, stretch(ep.state), action, "insert_pause", cameras={})
    return new, Defect("insert_pause", Cause.LONG_PAUSE, k / ep.fps, (k + hold) / ep.fps)


def swap_task(ep: Episode, rng: np.random.Generator, pool: list[str]) -> tuple[Episode, Defect]:
    """换成别的任务描述，关节和视频不动，对应「做的不是这个任务」。只有视频评判查得出来。"""
    others = sorted(t for t in set(pool) if t != ep.task)
    if not others:
        raise ValueError(f"任务池里除了「{ep.task}」没有别的任务")
    task = others[int(rng.integers(len(others)))]
    new = replace(ep, id=f"{ep.id}+swap_task", task=task,
                  meta={**ep.meta, "injected": "swap_task", "original_task": ep.task})
    return new, Defect("swap_task", Cause.WRONG_TASK, 0.0, ep.duration)


# 只动关节数据的注入；swap_task 要任务池和视频评判，单独在 task_swap.py 里跑
INJECTORS = {
    "truncate_end": truncate_end,
    "insert_pause": insert_pause,
}
