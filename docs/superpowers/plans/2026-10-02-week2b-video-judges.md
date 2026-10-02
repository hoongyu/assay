# 拣金第二周 2b：视频评判 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把视频评判模型（Robometer、TOPReward）接进拣金：读视频、算进度曲线、按「示范内容加模型版本」缓存，再把进度曲线变成逐帧异常分，跑考卷 B 的视频部分。

**Architecture:** 模型只在有 GPU 的地方跑（Colab），结果写成缓存 JSON；本机读缓存打分、算 AUC，不需要 GPU。`video.py` 用 PyAV 按采样率读帧并缩小；`judges/` 定义 `Judge` 接口和缓存；`scores/video.py` 把进度曲线插值到每一帧；`jianjin judge` 命令在 Colab 上批量跑。

**Tech Stack:** 新增核心依赖 `av`（PyAV）；可选依赖组 `judges`（torch、lerobot），只在 Colab 上装。

**分三段做：**

1. Task 1 到 5：本机，用假模型做测试驱动开发。
2. Task 6：Colab 摸底笔记本，用户来跑，测出 Robometer 和 TOPReward 的真实接口、速度和显存。
3. 按摸底结果写真适配器、跑考卷 B 的视频部分。这一段的代码依赖摸底结果，摸底回来后补进本计划的「第三段」。

**事实依据（2026-10-02 核）：**

- PyAV 19.0.0 在本机能解 RoboVAD 的 H.264（1920×1080、30 fps）和 LeRobot 的 AV1（libdav1d，640×480）；60 帧全分辨率约 1 到 1.5 秒。
- LeRobot v3 一个 mp4 里拼着多条示范，要按 `VideoRef.from_s` 跳过去读。
- LeRobot 文档：`RobometerRewardModel.compute_reward(batch)` 只返回最后一帧的进度（或按阈值判成功）；模型内部有逐帧进度和成功两个头。TOPReward 的 `compute_reward` 每条返回一个对数概率。要拿整条曲线，保底办法是对前 k 帧逐段调用（TOPReward 官方的离线打标也是 15 个锚点）；Robometer 能不能一次拿到逐帧输出，Task 6 摸底时看。
- 2a 的教训：打分器在 RoboVAD 的 **train** 划分（367 条）上选，不用 validation（82 条），test 只跑一次。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/jianjin/video.py` | 按采样率读一段视频、缩小 |
| `src/jianjin/judges/__init__.py` | 评判注册表 |
| `src/jianjin/judges/base.py` | `Curve`、`Judge` 接口、`FakeJudge` |
| `src/jianjin/judges/cache.py` | 视频指纹、缓存读写、`judge_episode` |
| `src/jianjin/scores/video.py` | 进度曲线 → 逐帧分数；从缓存造打分器 |
| `src/jianjin/eval/exam_b.py` | 支持 `video:<评判>:<分数>` 形式的打分器名 |
| `src/jianjin/cli.py` | `jianjin judge` |
| `notebooks/colab_probe.ipynb` | 摸底笔记本 |
| `tests/test_video.py`、`tests/test_judges.py`、`tests/test_video_scores.py` | 测试 |

---

### Task 1：读视频

**Files:**
- Modify: `pyproject.toml`（核心依赖加 `av>=13`）
- Create: `src/jianjin/video.py`、`tests/test_video.py`

- [ ] **Step 1：加依赖**

`pyproject.toml` 的 `dependencies` 改成：

```toml
dependencies = ["numpy>=2.0", "h5py>=3.11", "pyarrow>=17", "av>=13"]
```

Run：`uv sync --all-extras`

- [ ] **Step 2：写失败的测试**

`tests/test_video.py`：

```python
import av
import numpy as np
import pytest

from jianjin.types import VideoRef
from jianjin.video import read_clip


@pytest.fixture
def mp4(tmp_path):
    """造一个 10 秒、30 fps、64×48 的视频，第 i 帧的亮度是 i。"""
    path = tmp_path / "v.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(300):
            img = np.full((48, 64, 3), min(i, 255), dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def test_samples_at_requested_rate(mp4):
    frames, times = read_clip(VideoRef(mp4, 0.0, 10.0), sample_fps=2.0)
    assert frames.dtype == np.uint8 and frames.shape[1:] == (48, 64, 3)
    assert len(frames) == 20
    np.testing.assert_allclose(times[:3], [0.0, 0.5, 1.0], atol=0.04)


def test_reads_only_this_episode(mp4):
    frames, times = read_clip(VideoRef(mp4, 4.0, 6.0), sample_fps=2.0)
    assert len(frames) == 4
    assert times[0] < 0.05  # 相对这条示范的开头
    assert abs(float(frames[0].mean()) - 120) < 12  # 第 4 秒是第 120 帧


def test_downscales(mp4):
    frames, _ = read_clip(VideoRef(mp4, 0.0, 1.0), sample_fps=2.0, max_side=32)
    assert max(frames.shape[1:3]) == 32


def test_open_ended_ref(mp4):
    frames, _ = read_clip(VideoRef(mp4, 0.0, None), sample_fps=1.0)
    assert len(frames) == 10
```

- [ ] **Step 3：跑测试，确认失败**

Run：`uv run pytest tests/test_video.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.video'`。

- [ ] **Step 4：写实现**

`src/jianjin/video.py`：

```python
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
```

- [ ] **Step 5：跑测试，确认通过**

Run：`uv run pytest tests/test_video.py -q`
Expected：`4 passed`。

- [ ] **Step 6：Commit**

```bash
git add pyproject.toml uv.lock src/jianjin/video.py tests/test_video.py
git commit -m "feat: 按采样率读示范视频（PyAV）"
```

---

### Task 2：评判接口、假模型、缓存

**Files:**
- Create: `src/jianjin/judges/__init__.py`、`src/jianjin/judges/base.py`、`src/jianjin/judges/cache.py`、`tests/test_judges.py`

- [ ] **Step 1：写失败的测试**

`tests/test_judges.py`（复用 `tests/test_video.py` 里造视频的办法，挪进 `helpers.py` 的 `make_mp4`）。先在 `tests/helpers.py` 末尾加：

```python
# ---- 视频 ----

import av  # noqa: E402


def make_mp4(path: Path, n: int = 300, fps: int = 30) -> Path:
    """第 i 帧亮度是 min(i, 255)。"""
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
```

并把 `tests/test_video.py` 的 fixture 改成 `return make_mp4(tmp_path / "v.mp4")`（`from helpers import make_mp4`），删掉它自己的造视频代码。

`tests/test_judges.py`：

```python
import json

import numpy as np
import pytest

from helpers import make_episode, make_mp4
from jianjin.checks import NotApplicable
from jianjin.judges import JUDGES
from jianjin.judges.base import Curve, FakeJudge
from jianjin.judges.cache import fingerprint, judge_episode, load_curve
from jianjin.types import VideoRef


@pytest.fixture
def ep(tmp_path):
    e = make_episode(np.zeros((300, 2)))
    e.cameras = {"front": VideoRef(make_mp4(tmp_path / "v.mp4"), 0.0, 10.0)}
    return e


def test_fake_judge_curve_rises_with_brightness(ep, tmp_path):
    c = judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert isinstance(c, Curve)
    assert len(c.times) == len(c.progress) == 10
    assert c.progress[-1] > c.progress[0]
    assert c.model == "fake" and c.version == "1"


def test_second_call_hits_cache(ep, tmp_path):
    j = FakeJudge()
    judge_episode(j, ep, "front", tmp_path / "cache", sample_fps=1.0)
    judge_episode(j, ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert j.calls == 1


def test_new_version_does_not_reuse_cache(ep, tmp_path):
    judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    j2 = FakeJudge(version="2")
    judge_episode(j2, ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert j2.calls == 1


def test_load_curve_checks_fingerprint(ep, tmp_path):
    judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert load_curve(tmp_path / "cache", "fake", ep, "front") is not None
    ep.task = "something else"  # 任务描述变了，旧结果不能用
    assert load_curve(tmp_path / "cache", "fake", ep, "front") is None


def test_cache_file_is_plain_json(ep, tmp_path):
    judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    (f,) = list((tmp_path / "cache" / "fake").glob("*.json"))
    data = json.loads(f.read_text(encoding="utf-8"))
    assert {"key", "model", "version", "times", "progress"} <= set(data)


def test_no_camera(tmp_path):
    with pytest.raises(NotApplicable):
        judge_episode(FakeJudge(), make_episode(np.zeros((30, 2))), None, tmp_path)


def test_fingerprint_changes_with_slice(ep):
    ref = ep.cameras["front"]
    assert fingerprint(ref) != fingerprint(VideoRef(ref.path, 1.0, 10.0))


def test_registry_has_fake():
    assert "fake" in JUDGES
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_judges.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.judges'`。

- [ ] **Step 3：写实现**

`src/jianjin/judges/base.py`：

```python
"""视频评判的统一接口：看一段视频和任务描述，给出进度曲线（0 到 1），有的还给成功概率。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Curve:
    model: str
    version: str
    times: list[float]  # 秒，相对示范开头
    progress: list[float]
    success: list[float] | None = None


class Judge(Protocol):
    name: str
    version: str

    def curve(self, frames: np.ndarray, times: np.ndarray, task: str) -> Curve: ...


class FakeJudge:
    """测试用：进度等于到这一帧为止的平均亮度占最终平均亮度的比例。"""

    name = "fake"

    def __init__(self, version: str = "1"):
        self.version = version
        self.calls = 0

    def curve(self, frames: np.ndarray, times: np.ndarray, task: str) -> Curve:
        self.calls += 1
        b = frames.reshape(len(frames), -1).mean(axis=1)
        p = np.cumsum(b) / max(float(np.sum(b)), 1e-9)
        return Curve(self.name, self.version, [float(t) for t in times], [float(x) for x in p])
```

`src/jianjin/judges/cache.py`：

```python
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


def load_curve(cache_dir: Path, judge_name: str, ep: Episode, camera: str | None = None,
               sample_fps: float = 2.0, max_side: int = 448) -> Curve | None:
    camera = _pick_camera(ep, camera)
    path = _path(cache_dir, judge_name, ep, camera)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("key") != _key(ep.cameras[camera], ep.task, sample_fps, max_side):
        return None
    return Curve(data["model"], data["version"], data["times"], data["progress"], data.get("success"))


def judge_episode(judge: Judge, ep: Episode, camera: str | None, cache_dir: Path,
                  sample_fps: float = 2.0, max_side: int = 448) -> Curve:
    camera = _pick_camera(ep, camera)
    path = _path(cache_dir, judge.name, ep, camera)
    key = _key(ep.cameras[camera], ep.task, sample_fps, max_side)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("key") == key and data.get("version") == judge.version:
            return Curve(data["model"], data["version"], data["times"], data["progress"], data.get("success"))
    frames, times = read_clip(ep.cameras[camera], sample_fps=sample_fps, max_side=max_side)
    if len(frames) == 0:
        raise NotApplicable("视频里读不到帧")
    curve = judge.curve(frames, times, ep.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "episode": ep.id, "camera": camera, **asdict(curve)}), encoding="utf-8")
    return curve
```

`src/jianjin/judges/__init__.py`：

```python
"""评判注册表。真模型的适配器只在装了 judges 依赖组的机器上能实例化，所以注册的是构造函数。"""

from collections.abc import Callable

from jianjin.judges.base import Curve, FakeJudge, Judge

JUDGES: dict[str, Callable[[], Judge]] = {
    "fake": FakeJudge,
}

__all__ = ["JUDGES", "Curve", "FakeJudge", "Judge"]
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_judges.py tests/test_video.py -q`
Expected：`12 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/judges tests/helpers.py tests/test_video.py tests/test_judges.py
git commit -m "feat: 视频评判接口、假模型、按内容和模型版本缓存"
```

---

### Task 3：进度曲线 → 逐帧分数

**Files:**
- Create: `src/jianjin/scores/video.py`、`tests/test_video_scores.py`

- [ ] **Step 1：写失败的测试**

`tests/test_video_scores.py`：

```python
import numpy as np
import pytest

from helpers import make_episode
from jianjin.checks import NotApplicable
from jianjin.judges.base import Curve
from jianjin.scores.video import CURVE_SCORES, per_frame


def ep300():
    return make_episode(np.zeros((300, 2)))  # 10 秒


def test_per_frame_interpolates():
    c = Curve("m", "1", [0.0, 5.0, 10.0], [0.0, 0.5, 1.0])
    p = per_frame(c.times, c.progress, ep300())
    assert p.shape == (300,)
    assert abs(p[150] - 0.5) < 0.01


def test_drop_peaks_where_progress_falls():
    c = Curve("m", "1", [0, 2, 4, 6, 8, 10], [0.0, 0.4, 0.8, 0.3, 0.6, 1.0])
    s = CURVE_SCORES["drop"](c, ep300())
    assert 120 <= int(np.argmax(s)) <= 180


def test_low_is_one_minus_progress():
    c = Curve("m", "1", [0.0, 10.0], [0.2, 0.2])
    np.testing.assert_allclose(CURVE_SCORES["low"](c, ep300()), 0.8)


def test_fail_needs_success_head():
    c = Curve("m", "1", [0.0, 10.0], [0.0, 1.0])
    with pytest.raises(NotApplicable):
        CURVE_SCORES["fail"](c, ep300())
    c2 = Curve("m", "1", [0.0, 10.0], [0.0, 1.0], success=[0.9, 0.1])
    s = CURVE_SCORES["fail"](c2, ep300())
    assert s[-1] > s[0]
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_video_scores.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/scores/video.py`：

```python
"""把视频评判的进度曲线插值到每一帧，变成逐帧异常分（越高越像异常）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from jianjin.checks import NotApplicable
from jianjin.judges.base import Curve
from jianjin.judges.cache import load_curve
from jianjin.types import Episode

SMOOTH_S = 1.0


def per_frame(times: list[float], values: list[float], ep: Episode) -> np.ndarray:
    return np.interp(ep.timestamps, np.asarray(times, float), np.asarray(values, float))


def _smooth(x: np.ndarray, fps: float) -> np.ndarray:
    k = max(1, int(round(SMOOTH_S * fps)))
    return x if k == 1 or len(x) < k else np.convolve(x, np.ones(k) / k, mode="same")


def drop(c: Curve, ep: Episode) -> np.ndarray:
    """进度往回掉：东西掉了、放错了。"""
    p = per_frame(c.times, c.progress, ep)
    slope = np.gradient(p, ep.timestamps) if len(p) > 1 else np.zeros_like(p)
    return _smooth(-slope, ep.fps)


def low(c: Curve, ep: Episode) -> np.ndarray:
    """进度低。"""
    return 1.0 - per_frame(c.times, c.progress, ep)


def fail(c: Curve, ep: Episode) -> np.ndarray:
    """成功概率低。"""
    if c.success is None:
        raise NotApplicable(f"{c.model} 不给成功概率")
    return 1.0 - per_frame(c.times, c.success, ep)


CURVE_SCORES: dict[str, Callable[[Curve, Episode], np.ndarray]] = {"drop": drop, "low": low, "fail": fail}


def video_scorer(judge_name: str, kind: str, cache_dir: Path, camera: str | None = None):
    """从缓存读进度曲线的逐帧打分器；没缓存的示范记为打不了分。"""
    fn = CURVE_SCORES[kind]

    def scorer(ep: Episode) -> np.ndarray:
        c = load_curve(cache_dir, judge_name, ep, camera)
        if c is None:
            raise NotApplicable(f"缓存里没有 {judge_name} 对这条示范的结果")
        return fn(c, ep)

    return scorer
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_video_scores.py -q`
Expected：`4 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/scores/video.py tests/test_video_scores.py
git commit -m "feat: 进度曲线转逐帧异常分（掉进度、进度低、成功率低）"
```

---

### Task 4：考卷 B 支持视频打分器、`jianjin judge`

打分器名写成 `video:<评判名>:<分数名>`，例如 `video:robometer:drop`。`jianjin judge` 在有 GPU 的机器上跑，把曲线写进缓存目录。

**Files:**
- Modify: `src/jianjin/eval/exam_b.py`、`src/jianjin/cli.py`、`tests/test_exam_b.py`

- [ ] **Step 1：写失败的测试**

在 `tests/test_exam_b.py` 末尾加：

```python
def test_video_scorer_from_cache(tmp_path):
    from helpers import make_mp4
    from jianjin.cli import main

    root = setup(tmp_path)
    (root / "videos").mkdir()
    for name in ["h_stackcubes_ep1", "h_stackcubes_ep2", "h_pour_rice_ep1"]:
        make_mp4(root / "videos" / f"{name}_left.mp4", n=150)
    cache = tmp_path / "cache"
    assert main(["judge", "robovad", "--robovad-root", str(root), "--judge", "fake", "--split", "validation",
                 "--camera", "left", "--cache", str(cache)]) == 0
    assert len(list((cache / "fake").glob("*.json"))) == 2  # validation 里两条

    out = run_exam_b(root, "validation", "video:fake:low", cache_dir=cache, camera="left")
    (r,) = out["results"]
    assert r["n_episodes"] == 2 and r["not_applicable"] == []
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_exam_b.py::test_video_scorer_from_cache -q`
Expected：FAIL（`invalid choice: 'judge'`）。

- [ ] **Step 3：改 `src/jianjin/eval/exam_b.py`**

在文件顶部的 import 里加：

```python
from jianjin.scores.video import video_scorer
```

把 `run_exam_b` 换成：

```python
def make_scorer(name: str, cache_dir: str | Path = "cache", camera: str | None = None):
    """关节打分器写名字（如 calm）；视频打分器写 video:<评判>:<分数>（如 video:robometer:drop）。"""
    if name.startswith("video:"):
        _, judge_name, kind = name.split(":")
        return video_scorer(judge_name, kind, Path(cache_dir), camera)
    return SCORERS[name]


def split_names(root: Path, split: str) -> list[str]:
    with open(root / "splits" / f"{split}.csv", encoding="utf-8", newline="") as f:
        return [r["episode_name"] for r in csv.DictReader(f)]


def run_exam_b(root: str | Path, split: str, scorer_name: str,
               cache_dir: str | Path = "cache", camera: str | None = None) -> dict:
    root = Path(root)
    res = read_robovad(root)
    episodes = {e.id: e for e in res.episodes}
    if split == "test":
        populations = {s: read_segments(root / "evaluation" / f"{s}.csv") for s in SETTINGS}
    else:  # train 或 validation：用标注造逐帧标签，全部帧都评
        populations = {split: _annotated_segments(split_names(root, split), episodes)}
    scorer = make_scorer(scorer_name, cache_dir, camera)
    results = [{"setting": k, **score_population(v, episodes, scorer)} for k, v in populations.items()]
    return {
        "exam": "B（RoboVAD 帧级异常检测；拣金零样本，不在 RoboVAD 上训练）",
        "split": split,
        "scorer": scorer_name,
        "camera": camera,
        "read_failures": res.failures,
        "results": results,
    }
```

把 `_validation_segments` 换成（按名单造，train 和 validation 共用）：

```python
def _annotated_segments(names: list[str], episodes: dict) -> Segments:
    out: Segments = {}
    for name in names:
        ep = episodes.get(name)
        if ep is None:
            out[name] = []
            continue
        n = int(ep.meta["frame_index"].max()) + 1
        y = labels_from_annotations(n, ep.meta["anomalies"])
        out[name] = [(i, i + 1, int(v)) for i, v in enumerate(y)]
    return out
```

并把 `_cmd_eval` 里调用 `run_exam_b` 的那一行改成：

```python
        out = run_exam_b(args.robovad_root, split, args.scorer, cache_dir=args.cache, camera=args.camera)
```

`--split` 的 choices 加上 `"train"`，`eval` 子命令加两个参数：

```python
    e.add_argument("--cache", default="cache")
    e.add_argument("--camera", default=None)
```

`tests/helpers.py` 的 `make_robovad` 里，`splits/validation.csv` 之外再写一个空的 `splits/train.csv`：

```python
    (root / "splits" / "train.csv").write_text("episode_name\n", encoding="utf-8")
```

- [ ] **Step 4：加 `jianjin judge`**

在 `src/jianjin/cli.py` 加：

```python
def _cmd_judge(args: argparse.Namespace) -> int:
    from jianjin.checks import NotApplicable
    from jianjin.judges import JUDGES
    from jianjin.judges.cache import judge_episode

    judge = JUDGES[args.judge]()
    if args.what == "robovad":
        from jianjin.eval.exam_b import split_names
        from jianjin.ingest.robovad import read_robovad

        names = [n for s in args.split.split(",") for n in split_names(Path(args.robovad_root), s)]
        episodes = read_robovad(args.robovad_root, names=names).episodes
    else:
        episodes = load(args.what).episodes
    if args.limit:
        episodes = episodes[: args.limit]
    done = skipped = 0
    for i, ep in enumerate(episodes, 1):
        try:
            judge_episode(judge, ep, args.camera, Path(args.cache), sample_fps=args.sample_fps)
            done += 1
        except NotApplicable as exc:
            skipped += 1
            print(f"  跳过 {ep.id}：{exc}")
        if i % 10 == 0 or i == len(episodes):
            print(f"{i}/{len(episodes)}，完成 {done}，跳过 {skipped}", flush=True)
    return 0
```

`build_parser()` 里 `return p` 前加：

```python
    j = sub.add_parser("judge", help="在有 GPU 的机器上跑视频评判，结果写进缓存")
    j.add_argument("what", help="robovad，或者一个数据集路径")
    j.add_argument("--judge", default="robometer")
    j.add_argument("--robovad-root", default="data/robovad")
    j.add_argument("--split", default="train", help="robovad 的划分，逗号分隔，如 train,test")
    j.add_argument("--camera", default=None)
    j.add_argument("--cache", default="cache")
    j.add_argument("--sample-fps", type=float, default=2.0)
    j.add_argument("--limit", type=int, default=0)
    j.set_defaults(func=_cmd_judge)
```

`.gitignore` 加一行 `cache/`。

- [ ] **Step 5：跑全部测试**

Run：`uv run pytest -q`
Expected：全部通过。`tests/test_exam_b.py` 里原来调用 `run_exam_b(..., "validation", ...)` 的测试不变。

- [ ] **Step 6：Commit**

```bash
git add src tests .gitignore
git commit -m "feat: jianjin judge；考卷 B 支持视频打分器和 train 划分"
```

---

### Task 5：打 wheel，给 Colab 用

仓库没有远程，Colab 拿代码用 wheel：本机打包，用户把文件拖进 Colab。

- [ ] **Step 1：打包**

```bash
uv build --wheel
```

Expected：`dist/jianjin-0.1.0-py3-none-any.whl`（`dist/` 已在 `.gitignore`）。

---

### Task 6：Colab 摸底笔记本（用户来跑）

**Files:**
- Create: `notebooks/colab_probe.ipynb`

笔记本做五件事，每格的输出就是摸底结论，用户跑完把整个笔记本（含输出）下载回来，放到 `notebooks/colab_probe.out.ipynb`：

1. 看 GPU 型号和显存（`nvidia-smi`）。
2. 装 `lerobot` 的 robometer 和 topreward 两个可选依赖、装 jianjin 的 wheel；打印 lerobot 版本。
3. 从 Zenodo 用 Range 取 RoboVAD 的关节数据和 3 条示范的视频（`h_pickandplace_ep9` 有掉落、一条正常的、一条套环的），用 `jianjin.video.read_clip` 按每秒 2 帧读。
4. Robometer：按文档加载（T4 不支持 BF16，用 FP16），对一条视频调 `compute_reward`，打印返回值的类型和形状；再用 `forward` 看模型内部输出里有没有逐帧进度；对前 k 帧逐段调用 8 次，记下每次耗时和显存峰值。
5. TOPReward：先试默认的 Qwen3-VL-8B，装不下就换 `Qwen/Qwen3-VL-4B-Instruct`；同样记返回值、耗时、显存峰值。

笔记本的具体格子在执行这一项时按上面五步写，每格只做一件事、打印要的数字；出错不中断后面的格子（每格包 try/except 打印异常）。

- [ ] **Step 1：写笔记本**
- [ ] **Step 2：Commit**

```bash
git add notebooks/colab_probe.ipynb
git commit -m "chore: Colab 摸底笔记本"
```

- [ ] **Step 3：交给用户跑**，等结果。

---

## 第三段（摸底回来后补）

按摸底结果补：`judges/robometer.py`、`judges/topreward.py` 的真实现和注册；Colab 上批量跑 RoboVAD 的 train 和 test（如果太慢，先只跑 test 的一个设置）；在 train 上选视频打分器，test 跑一次；考卷 A 加「换掉任务描述」的注入。
