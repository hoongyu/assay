from __future__ import annotations

import numpy as np

from jianjin.types import Episode


def find_gripper(names: list[str]) -> int | None:
    for i, n in enumerate(names):
        if "gripper" in n.lower():
            return i
    return None


def make_episode(
    state: np.ndarray,
    fps: float = 30.0,
    task: str = "pick the cube",
    names: list[str] | None = None,
    ep_id: str = "ep0",
    action: np.ndarray | None = None,
) -> Episode:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state[:, None]
    names = names or [f"j{i}" for i in range(state.shape[1])]
    return Episode(
        id=ep_id,
        source="test",
        task=task,
        fps=fps,
        state=state,
        action=action,
        timestamps=np.arange(len(state)) / fps,
        state_names=names,
        gripper_index=find_gripper(names),
    )
