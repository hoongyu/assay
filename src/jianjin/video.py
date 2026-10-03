"""按采样率读一条示范的视频帧。LeRobot v3 多条示范拼在一个 mp4 里，按 VideoRef 的起止时间截。"""

from __future__ import annotations

import av
import numpy as np

from jianjin.types import VideoRef


def read_clip(ref: VideoRef, sample_fps: float = 2.0, max_side: int = 448) -> tuple[np.ndarray, np.ndarray]:
    """返回 (帧 N×H×W×3 uint8, 每帧相对示范开头的秒数)。长边超过 max_side 时等比缩小。"""
    a = ref.from_s
    b = ref.to_s
    frames, times = [], []
    next_t = a
    with av.open(str(ref.path)) as c:
        s = c.streams.video[0]
        if a > 0:
            c.seek(int(a / s.time_base), stream=s, backward=True)
        w, h = s.codec_context.width, s.codec_context.height
        scale = min(1.0, max_side / max(w, h))
        tw, th = max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)
        for fr in c.decode(s):
            t = fr.time
            if t is None or t < a - 1e-3:
                continue
            if b is not None and t >= b - 1e-3:
                break
            if t + 1e-3 < next_t:
                continue
            img = fr.reformat(width=tw, height=th) if scale < 1.0 else fr
            frames.append(img.to_ndarray(format="rgb24"))
            times.append(t - a)
            next_t += 1.0 / sample_fps
    if not frames:
        return np.zeros((0, th, tw, 3), dtype=np.uint8), np.zeros(0)
    return np.stack(frames), np.asarray(times)
