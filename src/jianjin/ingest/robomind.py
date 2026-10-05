"""直接读 RoboMIND 的 trajectory.hdf5。key 结构见官方 static/all_robot_h5_info.md。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jianjin.types import Episode, LoadResult

# 真机文件里没有时间戳，官方也没写帧率；2026-10-05 核：any4lerobot（常用的 RoboMIND 转 LeRobot 工具）按 30 Hz
# 转，注明是控制频率。仿真文件带 sim_time_seq，按它估（约 120 Hz）。
ASSUMED_FPS = 30.0

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


def _arm_names(d: int, two: bool) -> list[str]:
    if not two:
        return [f"j{i}" for i in range(d)]
    half = d // 2
    return [f"left_j{i}" for i in range(half)] + [f"right_j{i}" for i in range(d - half)]


def _same_column(arm: np.ndarray, col: np.ndarray, lo: int, hi: int) -> int | None:
    """arm 的 [lo, hi) 列里和 col 是同一路读数的那一列：差的中位数最小的那列，且不到 col 幅度的 2%。
    从臂那边两路的采样时刻差一点，换档那几帧差得多（实测最大差是幅度的 3% 到 10%），所以不看最大差；
    挑最小的而不是第一个够近的：dual_04 有一条第 6 列碰巧也在 2% 以内。"""
    if hi <= lo:
        return None
    lo_q, hi_q = np.nanpercentile(col, [5, 95]) if np.isfinite(col).any() else (0.0, 0.0)
    tol = max(1e-4, 0.02 * float(hi_q - lo_q))
    diffs = [float(np.nanmedian(np.abs(arm[:, j] - col))) for j in range(lo, hi)]
    j = int(np.argmin(diffs))
    return lo + j if diffs[j] <= tol else None


def _arm_hand(h: h5py.File, group: str) -> tuple[np.ndarray, list[str], int | None] | None:
    """公开失败示范的真机布局：arm_joint_position 加 hand_joint_position。手那几列已经在臂里（双臂 16 列里的
    第 7、15 列就是两只夹爪）就不重复拼，不在就拼到后面。一只、两只手的算夹爪；更多列的是灵巧手，不当夹爪。"""
    arm_k, hand_k = f"{group}/arm_joint_position", f"{group}/hand_joint_position"
    if arm_k not in h:
        return None
    arm = np.asarray(h[arm_k][()], dtype=np.float64)
    hand = np.asarray(h[hand_k][()], dtype=np.float64) if hand_k in h else np.zeros((len(arm), 0))
    if hand.ndim == 1:
        hand = hand[:, None]
    n = min(len(arm), len(hand))
    arm, hand = arm[:n], hand[:n]
    k, d = hand.shape[1], arm.shape[1]
    two = k == 2
    names = _arm_names(d, two)
    spans = [(0, d // 2), (d // 2, d)] if two else [(0, d)]
    inside = [_same_column(arm, hand[:, i], *spans[i]) for i in range(k)] if k in (1, 2) else []
    if inside and all(j is not None for j in inside):
        for side, j in zip(("left_", "right_") if two else ("",), inside):
            names[j] = f"{side}gripper"
        return arm, names, inside[0]
    if k in (1, 2):
        return np.concatenate([arm, hand], axis=1), names + (["left_gripper", "right_gripper"] if two else ["gripper"]), d
    return np.concatenate([arm, hand], axis=1), names + [f"hand_j{i}" for i in range(k)], None


TIANGONG_ARMS = ("tiangong/left_arm_joint_pos_seq", "tiangong/right_arm_joint_pos_seq")
TIANGONG_HANDS = ("tiangong/left_hand_joint_pos_seq", "tiangong/right_hand_joint_pos_seq")


def _tiangong(h: h5py.File) -> tuple[np.ndarray, list[str], None] | None:
    """公开失败示范里天工仿真的布局：左右臂各 7 维，灵巧手各 12 维（不当夹爪）；各路帧数会差一两帧，截到最短。"""
    if not all(k in h for k in TIANGONG_ARMS):
        return None
    keys = TIANGONG_ARMS + (TIANGONG_HANDS if all(k in h for k in TIANGONG_HANDS) else ())
    parts = [np.asarray(h[k][()], dtype=np.float64) for k in keys]
    n = min(len(p) for p in parts)
    names = [f"{s}_j{i}" for s, p in zip(("left", "right"), parts[:2]) for i in range(p.shape[1])]
    names += [f"{s}_hand_j{i}" for s, p in zip(("left", "right"), parts[2:]) for i in range(p.shape[1])]
    return np.concatenate([p[:n] for p in parts], axis=1), names, None


def _sim_fps(h: h5py.File) -> float | None:
    """仿真文件带 sim_time_seq（秒）：按相邻间隔的中位数估帧率。"""
    if "sim_time_seq" not in h:
        return None
    d = np.diff(np.asarray(h["sim_time_seq"][()], dtype=np.float64))
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else None


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
                layout = "joint_position"
                puppet = _joints(h, "puppet")
                if puppet is None and (puppet := _arm_hand(h, "puppet")) is not None:
                    layout = "arm_hand"
                if puppet is None and (puppet := _tiangong(h)) is not None:
                    layout = "tiangong"
                if puppet is None and (puppet := _joints(h, "franka")) is not None:  # 仿真 Franka
                    layout = "franka_sim"
                if puppet is None:
                    raise KeyError("没有 puppet/joint_position（也没有 _left/_right）、puppet/arm_joint_position，"
                                   "也不是仿真的 tiangong/*_arm_joint_pos_seq、franka/joint_position")
                master = _joints(h, "master") or _arm_hand(h, "master")
                sim_fps = _sim_fps(h)
                sim = bool(h.attrs.get("sim", False))
                cams = sorted(h["observations/rgb_images"].keys()) if "observations/rgb_images" in h else []
        except Exception as exc:  # 一条坏了不中断整批
            failures.append((ep_id, f"读不了：{exc}"))
            continue
        state, names, gripper = puppet
        action = master[0][: len(state)] if master is not None and master[0].shape[1:] == state.shape[1:] else None
        if action is not None and len(action) < len(state):
            state = state[: len(action)]
        ep_fps = sim_fps or fps
        episodes.append(
            Episode(
                id=ep_id,
                source="robomind",
                task=meta["task_dir"].replace("_", " "),
                fps=ep_fps,
                state=state,
                action=action,
                timestamps=np.arange(len(state)) / ep_fps,
                state_names=names,
                gripper_index=gripper,
                meta={**meta, "fps_assumed": sim_fps is None and fps == ASSUMED_FPS, "image_keys": cams,
                      "layout": layout, "sim": sim},
            )
        )
    return LoadResult(episodes, failures)
