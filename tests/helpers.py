from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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


# ---- LeRobot v3 ----

SO101_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _matrix_column(m: np.ndarray) -> pa.Array:
    m = np.asarray(m, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel()), m.shape[1])


def make_lerobot_v3(root: Path, episodes: list[tuple[str, np.ndarray]], fps: int = 30) -> Path:
    """在 root 下造一个最小的 LeRobot v3 数据集。episodes 是 (任务描述, 关节状态 T×6)。"""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "names": SO101_NAMES},
            "action": {"dtype": "float32", "shape": [6], "names": SO101_NAMES},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    tasks = sorted({t for t, _ in episodes})
    pq.write_table(
        pa.table({"task_index": list(range(len(tasks))), "task": tasks}),
        root / "meta" / "tasks.parquet",
    )

    cols: dict[str, list] = {k: [] for k in ["episode_index", "frame_index", "timestamp", "task_index", "index"]}
    states, video_from, video_to = [], [], []
    t_video = 0.0
    for ei, (task, state) in enumerate(episodes):
        n = len(state)
        cols["episode_index"] += [ei] * n
        cols["frame_index"] += list(range(n))
        cols["timestamp"] += (np.arange(n) / fps).tolist()
        cols["task_index"] += [tasks.index(task)] * n
        cols["index"] += list(range(len(cols["index"]), len(cols["index"]) + n))
        states.append(state)
        video_from.append(t_video)
        t_video += n / fps
        video_to.append(t_video)
    m = np.concatenate(states)
    table = pa.table(
        {
            **{k: pa.array(v) for k, v in cols.items()},
            "observation.state": _matrix_column(m),
            "action": _matrix_column(m),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    k = "observation.images.front"
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(len(episodes))),
                "tasks": [[t] for t, _ in episodes],
                "length": [len(s) for _, s in episodes],
                f"videos/{k}/chunk_index": [0] * len(episodes),
                f"videos/{k}/file_index": [0] * len(episodes),
                f"videos/{k}/from_timestamp": video_from,
                f"videos/{k}/to_timestamp": video_to,
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    return root


# ---- RoboMIND ----


def make_robomind_h5(path: Path, state: np.ndarray | None = None, left_right: np.ndarray | None = None) -> Path:
    """造一个最小的 RoboMIND trajectory.hdf5。state 是单臂 T×8；left_right 是双臂 (左 T×7, 右 T×7) 叠成 2×T×7。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        if state is not None:
            h["puppet/joint_position"] = state
            h["master/joint_position"] = state
        if left_right is not None:
            h["puppet/joint_position_left"] = left_right[0]
            h["puppet/joint_position_right"] = left_right[1]
        h.create_group("observations/rgb_images/camera_top")
    return path


# ---- RoboVAD ----


def make_robovad(
    root: Path,
    episodes: dict[str, tuple[np.ndarray, np.ndarray]],
    annotations: dict | None = None,
    evaluation: dict[str, list[tuple[str, int, int, int]]] | None = None,
    validation: list[str] | None = None,
    t0: float = 1.1,
) -> Path:
    """episodes：{示范名: (state T×6, action T×6)}；evaluation：{设置名: [(示范名, 起, 止, 标签)]}。"""
    (root / "proprioception").mkdir(parents=True)
    for name, (state, action) in episodes.items():
        n = len(state)
        pq.write_table(
            pa.table(
                {
                    "frame_index": pa.array(np.arange(n), pa.int64()),
                    "timestamp": pa.array(t0 + np.arange(n) / 30.0),
                    "observation.state": pa.array([list(r) for r in np.asarray(state, float)], pa.large_list(pa.float64())),
                    "action": pa.array([list(r) for r in np.asarray(action, float)], pa.large_list(pa.float64())),
                }
            ),
            root / "proprioception" / f"{name}.parquet",
        )
    (root / "annotations.json").write_text(json.dumps(annotations or {}), encoding="utf-8")
    (root / "evaluation").mkdir()
    for setting, rows in (evaluation or {}).items():
        lines = ["episode_name,start_frame,end_frame_exclusive,label"] + [f"{a},{b},{c},{d}" for a, b, c, d in rows]
        (root / "evaluation" / f"{setting}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "splits").mkdir()
    (root / "splits" / "validation.csv").write_text(
        "\n".join(["episode_name", *(validation or [])]) + "\n", encoding="utf-8"
    )
    return root
