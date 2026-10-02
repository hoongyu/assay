"""模型输出按「视频指纹 + 任务描述 + 模型版本 + 采样参数」缓存成旁路 JSON，不改原数据；换了模型不会读到旧结果。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from jianjin.checks import NotApplicable
from jianjin.judges.base import Curve, Judge
from jianjin.types import Episode, VideoRef
from jianjin.video import read_clip

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


def _key(ref: VideoRef, task: str, sample_fps: float, max_side: int) -> str:
    raw = json.dumps([fingerprint(ref), task, sample_fps, max_side])
    return hashlib.sha256(raw.encode()).hexdigest()


def _path(cache_dir: Path, judge_name: str, ep: Episode, camera: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", f"{ep.id}__{camera}")
    return Path(cache_dir) / judge_name / f"{safe}.json"


def _pick_camera(ep: Episode, camera: str | None) -> str:
    if not ep.cameras:
        raise NotApplicable("这条示范没有视频")
    if camera is None:
        return next(iter(ep.cameras))
    if camera not in ep.cameras:
        raise NotApplicable(f"没有相机 {camera}，只有 {', '.join(ep.cameras)}")
    return camera


def _curve(data: dict) -> Curve:
    return Curve(data["model"], data["version"], data["times"], data["progress"], data.get("success"))


def load_curve(cache_dir: Path, judge_name: str, ep: Episode, camera: str | None = None,
               sample_fps: float = 2.0, max_side: int = 448) -> Curve | None:
    camera = _pick_camera(ep, camera)
    path = _path(cache_dir, judge_name, ep, camera)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("key") != _key(ep.cameras[camera], ep.task, sample_fps, max_side):
        return None
    return _curve(data)


def judge_episode(judge: Judge, ep: Episode, camera: str | None, cache_dir: Path,
                  sample_fps: float = 2.0, max_side: int = 448) -> Curve:
    camera = _pick_camera(ep, camera)
    path = _path(cache_dir, judge.name, ep, camera)
    key = _key(ep.cameras[camera], ep.task, sample_fps, max_side)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("key") == key and data.get("version") == judge.version:
            return _curve(data)
    frames, times = read_clip(ep.cameras[camera], sample_fps=sample_fps, max_side=max_side)
    if len(frames) == 0:
        raise NotApplicable("视频里读不到帧")
    curve = judge.curve(frames, times, ep.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "episode": ep.id, "camera": camera, **asdict(curve)}), encoding="utf-8")
    return curve
