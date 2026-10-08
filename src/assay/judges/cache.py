"""模型输出按「视频指纹 + 任务描述」做键缓存成旁路 JSON，文件里记着模型版本和采样参数；不改原数据。

重跑时键、版本、采样参数都对上才复用；读结果（打分）时只看键，采样参数照文件里记的用。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from assay.checks import NotApplicable
from assay.judges.base import Curve, Judge
from assay.types import Episode, VideoRef
from assay.video import read_clip

CHUNK = 1 << 20


def fingerprint(ref: VideoRef) -> str:
    """视频文件大小、首尾各 1 MB 的哈希，加上这条示范的起止时间。"""
    path = Path(ref.path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(CHUNK))
        f.seek(max(0, size - CHUNK))
        h.update(f.read(CHUNK))
    h.update(f"{ref.from_s:.3f}-{ref.to_s}".encode())
    return h.hexdigest()


def _key(ref: VideoRef, task: str) -> str:
    raw = json.dumps([fingerprint(ref), task])
    return hashlib.sha256(raw.encode()).hexdigest()


def _path(cache_dir: Path, judge_name: str, ep: Episode, camera: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", f"{ep.id}__{camera}")
    return Path(cache_dir) / judge_name / f"{safe}.json"


def pick_camera(ep: Episode, camera: str | None) -> str:
    if not ep.cameras:
        raise NotApplicable("这条示范没有视频")
    if camera is None:
        return next(iter(ep.cameras))
    if camera not in ep.cameras:
        raise NotApplicable(f"没有相机 {camera}，只有 {', '.join(ep.cameras)}")
    return camera


def _curve(data: dict) -> Curve:
    return Curve(data["model"], data["version"], data["times"], data["progress"], data.get("success"))


def load_curve(cache_dir: Path, judge_name: str, ep: Episode, camera: str | None = None) -> Curve | None:
    """有本地视频就核对视频指纹；没有（视频只在 GPU 机器上）就退一步，只核对示范名、相机和任务描述。"""
    if camera is None:
        camera = pick_camera(ep, camera)
    path = _path(cache_dir, judge_name, ep, camera)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ref = ep.cameras.get(camera)
    if ref is not None and Path(ref.path).exists():
        if data.get("key") != _key(ref, ep.task):
            return None
    elif data.get("episode") != ep.id or data.get("task") != ep.task:
        return None
    return _curve(data)


def load_curves(episodes: list[Episode], cache_dir: Path, judge_name: str,
                camera: str | None = None) -> dict[str, Curve]:
    """一批示范从缓存里读出来的曲线；没缓存、没视频的跳过。"""
    out = {}
    for ep in episodes:
        try:
            c = load_curve(Path(cache_dir), judge_name, ep, camera)
        except NotApplicable:
            continue
        if c is not None:
            out[ep.id] = c
    return out


def judge_episode(judge: Judge, ep: Episode, camera: str | None, cache_dir: Path,
                  sample_fps: float = 2.0, max_side: int = 448) -> Curve:
    camera = pick_camera(ep, camera)
    path = _path(cache_dir, judge.name, ep, camera)
    key = _key(ep.cameras[camera], ep.task)
    params = {"sample_fps": sample_fps, "max_side": max_side}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        same = data.get("key") == key and data.get("version") == judge.version
        if same and all(data.get(k) == v for k, v in params.items()):
            return _curve(data)
    frames, times = read_clip(ep.cameras[camera], sample_fps=sample_fps, max_side=max_side)
    if len(frames) == 0:
        raise NotApplicable("视频里读不到帧")
    curve = judge.curve(frames, times, ep.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "episode": ep.id, "camera": camera, "task": ep.task, **params, **asdict(curve)}), encoding="utf-8")
    return curve
