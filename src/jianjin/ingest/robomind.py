"""直接读 RoboMIND 的 trajectory.hdf5。key 结构见官方 static/all_robot_h5_info.md。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jianjin.types import Episode, LoadResult

ASSUMED_FPS = 30.0  # 文件说明里没写帧率，拿到真数据后核

FAILURE_CAUSES = {
    "动作不流畅", "二次抓取", "触碰多余", "夹取前碰撞", "目标移位", "失败放置",
    "轨迹异常", "动作不规范", "速度过快", "机械臂异常", "夹爪出境",
}


def _joints(h: h5py.File, group: str) -> tuple[np.ndarray, list[str], int | None] | None:
    single = f"{group}/joint_position"
    left, right = f"{group}/joint_position_left", f"{group}/joint_position_right"
    if single in h:
        arr = np.asarray(h[single][()], dtype=np.float64)
        d = arr.shape[1]
        if d in (7, 8):  # Franka、UR：最后一维是夹爪
            return arr, [f"j{i}" for i in range(d - 1)] + ["gripper"], d - 1
        return arr, [f"j{i}" for i in range(d)], None  # 天工等：手的位置不固定，不认
    if left in h and right in h:
        a, b = (np.asarray(h[k][()], dtype=np.float64) for k in (left, right))
        names = [f"left_j{i}" for i in range(a.shape[1] - 1)] + ["left_gripper"]
        names += [f"right_j{i}" for i in range(b.shape[1] - 1)] + ["right_gripper"]
        return np.concatenate([a, b], axis=1), names, a.shape[1] - 1
    return None


def _context(f: Path, root: Path) -> tuple[str, dict]:
    ep_dir = f.parent.parent if f.parent.name == "data" else f.parent
    base = root if root.is_dir() else root.parent
    try:
        rel = ep_dir.relative_to(base)
    except ValueError:
        rel = Path(ep_dir.name)
    parts = ep_dir.parts
    meta: dict = {}
    robot = next((p for p in parts if p.startswith("h5_")), None)
    if robot:
        meta["robot"] = robot
    if "success_episodes" in parts:
        task = parts[parts.index("success_episodes") - 1]
    elif ep_dir.parent.name in FAILURE_CAUSES:
        meta["robomind_failure_cause"] = ep_dir.parent.name
        task = ep_dir.parent.parent.name
    else:
        task = ep_dir.parent.name
    return rel.as_posix(), {"task_dir": task, **meta}


def read_robomind(root: Path, fps: float = ASSUMED_FPS) -> LoadResult:
    root = Path(root)
    files = [root] if root.is_file() else sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])
    episodes, failures = [], []
    for f in files:
        ep_id, meta = _context(f, root)
        try:
            with h5py.File(f, "r") as h:
                puppet = _joints(h, "puppet")
                if puppet is None:
                    raise KeyError("没有 puppet/joint_position（也没有 _left/_right）")
                master = _joints(h, "master")
                cams = sorted(h["observations/rgb_images"].keys()) if "observations/rgb_images" in h else []
        except Exception as exc:  # 一条坏了不中断整批
            failures.append((ep_id, f"读不了：{exc}"))
            continue
        state, names, gripper = puppet
        action = master[0] if master is not None and master[0].shape == state.shape else None
        episodes.append(
            Episode(
                id=ep_id,
                source="robomind",
                task=meta["task_dir"].replace("_", " "),
                fps=fps,
                state=state,
                action=action,
                timestamps=np.arange(len(state)) / fps,
                state_names=names,
                gripper_index=gripper,
                meta={**meta, "fps_assumed": fps == ASSUMED_FPS, "image_keys": cams},
            )
        )
    return LoadResult(episodes, failures)
