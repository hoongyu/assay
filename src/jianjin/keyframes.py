"""给存疑、作废的体检单配几张画面，人看一眼就知道报得对不对。

画面在本机视频里按时刻取，编成 JPEG 嵌进报告（data URI），报告还是一个文件、断网能开。
合格的不配，报告不至于太大。
"""

from __future__ import annotations

import base64
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from jianjin.checks.config import NotApplicable
from jianjin.diagnose import Label, Verdict
from jianjin.judges.cache import pick_camera
from jianjin.types import Episode
from jianjin.video import frames_at

Shots = dict[str, list[tuple[float, str]]]  # 示范 id -> [(秒, data URI)]


def keyframe_times(v: Verdict, duration: float, k: int = 4, gap: float = 0.5) -> list[float]:
    """每条发现取起止两个时刻；整条的发现取开头、中间、结尾。相隔不到 gap 秒的并成一个，最多 k 个。"""
    end = max(duration - 0.05, 0.0)  # 最后一帧在示范结束前一点
    ts = []
    for f in v.causes:
        if f.t0 <= 0.0 and f.t1 >= duration - 1e-9:
            ts += [0.0, duration / 2, end]
        else:
            ts += [f.t0, f.t1]
    out: list[float] = []
    for t in sorted(min(max(t, 0.0), end) for t in ts):
        if not out or t - out[-1] >= gap:
            out.append(t)
    if len(out) > k:
        out = [out[round(i * (len(out) - 1) / (k - 1))] for i in range(k)]
    return out


def jpeg_data_uri(img: np.ndarray) -> str:
    cc = av.CodecContext.create("mjpeg", "w")
    cc.width, cc.height, cc.pix_fmt = img.shape[1], img.shape[0], "yuvj420p"
    cc.time_base = Fraction(1, 25)
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").reformat(format="yuvj420p")
    data = b"".join(bytes(p) for p in [*cc.encode(frame), *cc.encode(None)])
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def shots_for(verdicts: list[Verdict], episodes: list[Episode], camera: str | None = None,
              max_side: int = 240) -> tuple[Shots, int]:
    """存疑、作废的示范各配几张关键帧。返回（关键帧，没有本地视频或读不出画面的条数）。"""
    by_id = {ep.id: ep for ep in episodes}
    shots: Shots = {}
    missing = 0
    for v in verdicts:
        if v.label == Label.PASS:
            continue
        ep = by_id[v.episode_id]
        try:
            ref = ep.cameras[pick_camera(ep, camera)]
        except NotApplicable:
            missing += 1
            continue
        times = keyframe_times(v, ep.duration)
        extra = max((f.score for f in v.causes if f.check == "video_length" and not f.hard), default=0.0)
        if extra > 0:  # 视频比关节数据长：多出来那段的正中再取一张，看录进了什么
            times.append(ep.duration + extra / 2)
        if not times or not Path(ref.path).exists():
            missing += 1
            continue
        try:
            imgs = frames_at(ref, times, max_side=max_side)
        except (av.error.FFmpegError, ValueError):
            missing += 1
            continue
        shots[v.episode_id] = [(t, jpeg_data_uri(img)) for t, img in zip(times, imgs)]
    return shots, missing
