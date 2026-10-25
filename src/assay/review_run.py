"""在命令行里跑视频复核：从本地视频切出规则指定的片段，调 Nebius Token Factory 上的 Nemotron 3.5 Super VL。

assay/review.py 是纯函数，网页也用；这里补上命令行这一侧：PyAV 切片段和取关键帧，带缓存、记用量的请求。
回答按请求内容缓存，同一段视频同一个问题第二次不再花钱。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Callable

import av
import numpy as np

from assay.diagnose import Verdict
from assay.keyframes import jpeg_data_uri
from assay.review import MODEL, PROMPT_VERSION, ReviewResult, review, review_requests
from assay.summary import BASE_URL, token_factory_body, token_factory_text
from assay.types import Episode, VideoRef
from assay.video import frames_at, read_clip

VIDEO_FPS = 10  # 片段按这个帧率重新编码；模型大约每秒取 2 帧
MAX_SIDE = 448
N_FRAMES = 8
Media = Callable[[VideoRef, float, float], tuple[bytes, list[bytes], list[float]]]


def _mp4(frames: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        s = out.add_stream("libx264", rate=VIDEO_FPS)
        s.height, s.width = frames.shape[1:3]
        s.pix_fmt = "yuv420p"
        for img in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
                out.mux(p)
        for p in s.encode(None):
            out.mux(p)
    return buf.getvalue()


def clip_media(ref: VideoRef, t0: float, t1: float) -> tuple[bytes, list[bytes], list[float]]:
    """示范里 [t0, t1] 秒这段：重新编码的 mp4，和均匀分布的 N_FRAMES 张关键帧（JPEG）及其在片段里的秒数。"""
    frames, _ = read_clip(VideoRef(ref.path, ref.from_s + t0, ref.from_s + t1), sample_fps=VIDEO_FPS, max_side=MAX_SIDE)
    times = [t0 + i * (t1 - t0) / (N_FRAMES - 1) for i in range(N_FRAMES)]
    keys = frames_at(ref, [min(t, max(t0, t1 - 0.05)) for t in times], max_side=MAX_SIDE)
    jpegs = [base64.b64decode(jpeg_data_uri(k).split(",", 1)[1]) for k in keys]
    return _mp4(frames), jpegs, [round(t - t0, 1) for t in times]


class CachedChat:
    """Token Factory 的请求，回答按请求内容缓存在 cache_dir/super_vl/，每次调用的用量记在 usage 里。"""

    def __init__(self, key: str | None, cache_dir: str | Path, base_url: str = BASE_URL, timeout: float = 180.0,
                 read_cache: bool = True):
        self.key, self.base_url, self.timeout, self.read_cache = key, base_url, timeout, read_cache
        self.dir = Path(cache_dir) / "super_vl"
        self.usage: list[dict] = []

    def __call__(self, messages: list[dict], model: str) -> str:
        body = token_factory_body(messages, model, temperature=0.0, max_tokens=60)
        digest = hashlib.sha256(json.dumps([PROMPT_VERSION, body], sort_keys=True).encode("utf-8")).hexdigest()[:24]
        path = self.dir / f"{digest}.json"
        if self.read_cache and path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))
            self.usage.append({**hit["usage"], "cached": True})
            return hit["text"]
        if self.key is None:
            raise RuntimeError("NEBIUS_API_KEY is not set and this request is not cached")
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = token_factory_text(data)
        usage = {**(data.get("usage") or {}), "seconds": round(time.time() - t, 1)}
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "prompt": PROMPT_VERSION, "text": text, "usage": usage}),
                        encoding="utf-8")
        self.usage.append(usage)
        return text


def run_reviews(verdicts: list[Verdict], episodes: list[Episode], chat, *, media: Media = clip_media,
                camera: str | None = None, mode: str = "video", model: str = MODEL,
                on_result: Callable[[ReviewResult], None] | None = None) -> tuple[list[Verdict], list[str]]:
    """每条能复核、本地有视频的存疑或作废示范问一次。返回挂好复核结论的判定，和因为没有本地视频而跳过的示范。"""
    by_id = {e.id: e for e in episodes}
    results: dict[str, ReviewResult] = {}
    skipped = []
    for req in review_requests(verdicts, episodes):
        cams = by_id[req.episode_id].cameras
        refs = [r for c, r in cams.items() if (camera is None or c == camera) and Path(r.path).exists()]
        if not refs:
            skipped.append(req.episode_id)
            continue
        video, frames, times = media(refs[0], req.t0, req.t1)
        kw = {"video": video} if mode == "video" else {"frames": frames, "frame_times": times}
        res = review(req, chat, model, **kw)
        results[req.episode_id] = res
        if on_result:
            on_result(res)
    out = [replace(v, reviews=v.reviews + [results[v.episode_id]]) if v.episode_id in results else v for v in verdicts]
    return out, skipped
