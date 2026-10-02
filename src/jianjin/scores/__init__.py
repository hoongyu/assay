from collections.abc import Callable

import numpy as np

from jianjin.scores.frame import accel, calm, gripper_tracking, stillness, tracking
from jianjin.types import Episode

FrameScorer = Callable[[Episode], np.ndarray]

SCORERS: dict[str, FrameScorer] = {
    "tracking": tracking,
    "gripper_tracking": gripper_tracking,
    "accel": accel,
    "calm": calm,
    "stillness": stillness,
}

__all__ = ["SCORERS", "FrameScorer"]
