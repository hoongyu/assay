"""读 RoboVAD 的关节数据（proprioception/*.parquet）和异常标注（annotations.json）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from assay.ingest.lerobot import _matrix
from assay.types import Episode, LoadResult, VideoRef

# 论文只给了任务名，下面的英文描述是我们按任务名写的，给视频模型当任务描述用
TASKS = {
    "h_pickandplace_multi": "Pick up each cube and place it in the container.",
    "h_pickandplace": "Pick up the cube and place it in the container.",
    "h_stackcubes": "Stack one cube on top of the other cube.",
    "h_pour_rice": "Pour the rice into the bowl.",
    "h_rings_on_peg": "Put the ring onto the peg.",
}
SO101_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def task_key(name: str) -> str | None:
    for prefix in sorted(TASKS, key=len, reverse=True):
        if name.startswith(prefix + "_"):
            return prefix
    return None


def read_robovad(root: str | Path, names: list[str] | None = None) -> LoadResult:
    root = Path(root)
    ann_path = root / "annotations.json"
    annotations = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    wanted = set(names) if names is not None else None
    episodes, failures = [], []
    for f in sorted((root / "proprioception").glob("*.parquet")):
        name = f.stem
        if wanted is not None and name not in wanted:
            continue
        try:
            t = pq.read_table(f)
            order = np.argsort(t["frame_index"].to_numpy(), kind="stable")
            frame_index = t["frame_index"].to_numpy()[order]
            ts = t["timestamp"].to_numpy().astype(np.float64)[order]
            state = _matrix(t["observation.state"])[order]
            action = _matrix(t["action"])[order]
        except Exception as exc:  # 一条坏了不中断整批
            failures.append((name, f"读不了：{exc}"))
            continue
        key = task_key(name)
        fps = float(round(1.0 / np.median(np.diff(ts)))) if len(ts) > 1 else 30.0
        cams = {}
        for side in ("left", "right"):
            video = root / "videos" / f"{name}_{side}.mp4"
            if video.exists():
                cams[side] = VideoRef(video, 0.0, None)
        episodes.append(
            Episode(
                id=name,
                source="robovad",
                task=TASKS.get(key, ""),
                fps=fps,
                state=state,
                action=action,
                timestamps=ts - ts[0],
                state_names=list(SO101_NAMES),
                gripper_index=5,
                cameras=cams,
                meta={"task_key": key, "anomalies": annotations.get(name, []), "frame_index": frame_index},
            )
        )
    return LoadResult(episodes, failures)
