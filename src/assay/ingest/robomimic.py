"""读 robomimic 的 HDF5（data/demo_*）。MH 数据集的 mask/ 里有操作员水平（better/okay/worse）和具体操作员。"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from assay.types import Episode, LoadResult

QUALITIES = ("better", "okay", "worse")
JOINTS = [f"joint_{i}" for i in range(7)]


def is_robomimic(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as h:
            return "data" in h and any(k.startswith("demo_") for k in h["data"].keys())
    except OSError:
        return False


def _operators(h: h5py.File) -> tuple[dict[str, str], dict[str, str]]:
    """每条示范的水平和操作员：只认 better、better_operator_1 这类键，train、20_percent 之类的划分不管。"""
    quality, operator = {}, {}
    for key in h.get("mask", {}):
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h["mask"][key][()]]
        if key in QUALITIES:
            quality.update(dict.fromkeys(names, key))
        elif key.split("_operator_")[0] in QUALITIES and key.rsplit("_", 1)[1].isdigit():
            operator.update(dict.fromkeys(names, key))
    return quality, operator


def read_robomimic(path: str | Path) -> LoadResult:
    path = Path(path)
    prefix = f"{path.parent.parent.name}/{path.parent.name}"
    episodes, failures = [], []
    with h5py.File(path, "r") as h:
        env = json.loads(h["data"].attrs.get("env_args", "{}"))
        fps = float(env.get("env_kwargs", {}).get("control_freq", 20))
        task = env.get("env_name", "")
        quality, operator = _operators(h)
        demos = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for name in demos:
            g = h["data"][name]
            try:
                joints = np.asarray(g["obs/robot0_joint_pos"][()], dtype=np.float64)
                fingers = np.asarray(g["obs/robot0_gripper_qpos"][()], dtype=np.float64)
                action = np.asarray(g["actions"][()], dtype=np.float64)
            except KeyError as exc:
                failures.append((f"{prefix}/{name}", f"读不了：{exc}"))
                continue
            width = fingers[:, :1] - fingers[:, 1:2]  # 两指对称，开口 = 左指 - 右指
            episodes.append(
                Episode(
                    id=f"{prefix}/{name}",
                    source="robomimic",
                    task=task,
                    fps=fps,
                    state=np.hstack([joints, width]),
                    action=action,
                    timestamps=np.arange(len(joints)) / fps,
                    state_names=[*JOINTS, "gripper"],
                    gripper_index=7,
                    operator_id=operator.get(name),
                    meta={"robomimic_quality": quality.get(name), "file": path.name},
                )
            )
    return LoadResult(episodes, failures)
