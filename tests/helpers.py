from __future__ import annotations

import json
import re
from pathlib import Path

import av
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


def make_robomind_failure_h5(path: Path, arm: np.ndarray, hand: np.ndarray) -> Path:
    """RoboMIND 公开失败示范的真机布局：puppet/arm_joint_position 加 puppet/hand_joint_position，master 同构。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        for g in ("puppet", "master"):
            h[f"{g}/arm_joint_position"] = arm
            h[f"{g}/hand_joint_position"] = hand
        h["observations/rgb_images/camera_top"] = np.zeros(0, dtype=np.uint8)
        h["language_instruction"] = b"do it"
    return path


def make_tiangong_h5(path: Path, arms: tuple[np.ndarray, np.ndarray], hands: tuple[np.ndarray, np.ndarray] | None,
                     sim_time: np.ndarray | None = None) -> Path:
    """RoboMIND 公开失败示范里天工仿真的布局：tiangong/{left,right}_{arm,hand}_joint_pos_seq，没有主臂。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["sim"] = True
        for side, a in zip(("left", "right"), arms):
            h[f"tiangong/{side}_arm_joint_pos_seq"] = a
        for side, a in zip(("left", "right"), hands or ()):
            h[f"tiangong/{side}_hand_joint_pos_seq"] = a
        if sim_time is not None:
            h["sim_time_seq"] = sim_time
        h["observations/rgb_images/camera_head"] = np.zeros(0, dtype=np.uint8)
    return path


# ---- RoboVAD ----


def make_robovad(
    root: Path,
    episodes: dict[str, tuple[np.ndarray, np.ndarray]],
    annotations: dict | None = None,
    evaluation: dict[str, list[tuple[str, int, int, int]]] | None = None,
    validation: list[str] | None = None,
    t0: float = 1.1,
    splits: dict[str, list[str]] | None = None,
) -> Path:
    """episodes：{示范名: (state T×6, action T×6)}；evaluation：{设置名: [(示范名, 起, 止, 标签)]}；
    splits：{划分名: [示范名]}，写成 splits/<划分名>.csv（会盖掉默认的空 train.csv）。"""
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
    (root / "splits" / "train.csv").write_text("episode_name\n", encoding="utf-8")
    for split, names in (splits or {}).items():
        (root / "splits" / f"{split}.csv").write_text("\n".join(["episode_name", *names]) + "\n", encoding="utf-8")
    return root


# ---- 视频 ----


def make_mp4(path: Path, n: int = 300, fps: int = 30) -> Path:
    """第 i 帧亮度是 min(i, 255)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(n):
            img = np.full((48, 64, 3), min(i, 255), dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


# ---- 缓存的进度曲线 ----


def write_cached_curve(cache: Path, ep: Episode, camera: str, progress: list[float],
                       judge: str = "robometer", times: list[float] | None = None,
                       success: list[float] | None = None) -> Path:
    """直接写一份缓存 JSON，形状和 judge_episode 写的一样（本机没视频时按示范名和任务描述认）。"""
    from jianjin.judges.cache import _path

    path = _path(cache, judge, ep, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = times if times is not None else list(np.linspace(0, ep.duration, len(progress)))
    path.write_text(json.dumps({"episode": ep.id, "camera": camera, "task": ep.task, "model": judge,
                                "version": "test", "times": [float(t) for t in times],
                                "progress": progress, "success": success}), encoding="utf-8")
    return path


HAN = re.compile(r"[一-鿿]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def assert_bilingual(text: str) -> None:
    """中文旁边带了英文（jianjin.types.Bi）：英文不含汉字，两份里的数字按顺序一样。"""
    from jianjin.types import en

    english = en(text)
    assert english != text, f"没有英文：{text}"
    assert not HAN.search(english), f"英文里有汉字：{english}"
    assert NUMBER.findall(english) == NUMBER.findall(text), f"数字对不上：{text} / {english}"
