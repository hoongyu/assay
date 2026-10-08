from collections.abc import Callable

import numpy as np

from assay.scores.frame import accel, calm, gripper_tracking, stillness, tracking
from assay.types import Episode

FrameScorer = Callable[[Episode], np.ndarray]

SCORERS: dict[str, FrameScorer] = {
    "tracking": tracking,
    "gripper_tracking": gripper_tracking,
    "accel": accel,
    "calm": calm,
    "stillness": stillness,
}

__all__ = ["SCORERS", "FrameScorer"]
