# 拣金第二周 2a：考卷 B（RoboVAD）的关节部分 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 不训练、只用关节数据，在 RoboVAD 上给每一帧打异常分，按论文协议报帧级 micro 和 macro AUC，和论文表 II 并列。

**Architecture:** RoboVAD 的压缩包 6.86 GB，其中关节数据只有 23 MB。用 HTTP Range 只取标注、评测清单和关节数据。`ingest/robovad.py` 读成 `Episode`；`scores/frame.py` 里每个打分器是一个纯函数 `Episode -> (T,) 分数`；`eval/exam_b.py` 按评测 csv 列出的帧收集分数和标签，算 AUC。打分器在 validation 划分（82 条）上选，test 的四种设置只跑一次。

**Tech Stack:** 同第一周；只用标准库 urllib 做 Range 下载，AUC 自己写（不引 sklearn）。

**范围：** 视频评判（Robometer、TOPReward）、视频解码、Colab 笔记本、关节和视频的融合，都在 2b。

**事实依据（2026-10-02 核）：**

- 压缩包 `https://zenodo.org/records/22754659/files/RoboVAD.zip?download=1`，支持 Range。里面是 `RoboVAD/` 下的 `README.md`、`license.txt`、`annotations.json`、`splits/{train,validation,test}.csv`、`evaluation/{in_domain,cross_anomaly,cross_task,cross_task+anomaly}.csv`、`proprioception/<示范>.parquet`（1,079 个文件，共 22.9 MB）、`videos/<示范>_{left,right}.mp4`（2,156 个）；另有 `__MACOSX/` 和 `.DS_Store`，要跳过。
- 关节 parquet 的列：`frame_index`（int64）、`timestamp`（double，不从 0 开始，比如 1.1）、`observation.state` 和 `action`（large_list<double>，6 维，SO-101 的关节顺序，最后一维是夹爪）。
- `annotations.json`：`{示范名: [{start_frame, end_frame, type, categories}]}`，区间左闭右开，正常示范不出现。
- 评测 csv：`episode_name,start_frame,end_frame_exclusive,label`，只评列出的帧。
- `splits/*.csv` 只有一列 `episode_name`。
- 示范名前缀就是任务：`h_pickandplace`、`h_pickandplace_multi`、`h_stackcubes`、`h_pour_rice`、`h_rings_on_peg`。论文只给了任务名，没给自然语言描述。
- 论文表 II（帧级 AUC，micro/macro）：

| 方法 | in_domain | cross_anomaly | cross_task | cross_task+anomaly |
|---|---|---|---|---|
| FlowJigsaw-Supervised | 94.6/94.4 | 69.0/71.9 | 72.0/73.8 | 66.7/66.8 |
| UR-DMU | 88.5/94.4 | 61.3/71.0 | 71.8/77.5 | 63.9/71.9 |
| Jigsaw-VAD | 64.1/65.2 | 51.7/51.1 | 66.9/70.9 | 57.6/57.9 |
| FIDeL | 56.3/54.0 | 52.1/53.4 | 58.1/53.7 | 63.2/51.2 |

  这四行是网页摘要工具读出来的，Task 7 第 1 步回原文核一遍全表；调研 agent 先前报过「最难设置最好 67.4%」，和这里的 66.7 对不上，以原文为准。

**打分的思路（为什么只用关节也有戏）：** RoboVAD 的五类异常里，碰撞会让从臂跟不上主臂的指令，跟随误差（`action - state`）突然变大；抓空时夹爪能合到底，夹住东西时会被物体挡住；掉落时夹爪的跟随误差突然消失。所以候选打分器是：全关节跟随误差、夹爪跟随误差、加速度、静止程度。都按关节活动幅度归一化，不用任何 RoboVAD 的标签训练。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/jianjin/eval/auc.py` | 带并列处理的 ROC AUC |
| `src/jianjin/eval/robovad_fetch.py` | HTTP Range 读 zip，只解出需要的成员 |
| `src/jianjin/ingest/robovad.py` | 读 RoboVAD 关节数据和标注 |
| `src/jianjin/scores/__init__.py`、`src/jianjin/scores/frame.py` | 逐帧异常打分器 |
| `src/jianjin/eval/exam_b.py` | 评测清单、收集帧、算 AUC、出表 |
| `src/jianjin/cli.py` | `jianjin fetch robovad`、`jianjin eval b` |
| `scripts/choose_exam_b_scorer.py` | 在 validation 上比打分器 |
| `tests/helpers.py` | 加 `make_robovad` |

---

### Task 1：AUC

**Files:**
- Create: `src/jianjin/eval/auc.py`、`tests/test_auc.py`

- [ ] **Step 1：写失败的测试**

`tests/test_auc.py`：

```python
import math

import numpy as np

from jianjin.eval.auc import roc_auc


def test_known_value():
    # 和 sklearn.metrics.roc_auc_score 的文档示例一致
    assert math.isclose(roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)


def test_perfect_and_inverse():
    assert roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert roc_auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0


def test_ties_count_half():
    assert roc_auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_one_class_is_nan():
    assert math.isnan(roc_auc([1, 2], [0, 0]))


def test_large_random_is_about_half():
    rng = np.random.default_rng(0)
    assert abs(roc_auc(rng.random(20000), rng.integers(0, 2, 20000)) - 0.5) < 0.02
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_auc.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/eval/auc.py`：

```python
"""ROC AUC：随机挑一个异常帧和一个正常帧，异常帧分数更高的概率；分数相同算一半。"""

from __future__ import annotations

import math

import numpy as np


def roc_auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2  # 从 1 开始的平均秩
    ranks = np.empty(len(s))
    ranks[order] = avg_rank[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_auc.py -q`
Expected：`5 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/eval/auc.py tests/test_auc.py
git commit -m "feat: ROC AUC（并列算一半）"
```

---

### Task 2：只取 RoboVAD 压缩包里需要的文件

**Files:**
- Create: `src/jianjin/eval/robovad_fetch.py`、`tests/test_robovad_fetch.py`

- [ ] **Step 1：写失败的测试**

网络部分不在单元测试里跑，测的是「从一个 zip 里按前缀解出成员」。

`tests/test_robovad_fetch.py`：

```python
import zipfile

from jianjin.eval.robovad_fetch import SMALL_PREFIXES, extract


def make_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("RoboVAD/", "")
        z.writestr("RoboVAD/annotations.json", "{}")
        z.writestr("RoboVAD/evaluation/in_domain.csv", "episode_name,start_frame,end_frame_exclusive,label\n")
        z.writestr("RoboVAD/proprioception/h_pickandplace_ep1.parquet", b"x")
        z.writestr("RoboVAD/videos/h_pickandplace_ep1_left.mp4", b"v")
        z.writestr("RoboVAD/.DS_Store", b"")
        z.writestr("__MACOSX/RoboVAD/._annotations.json", b"")
    return path


def test_extract_small_parts_only(tmp_path):
    z = zipfile.ZipFile(make_zip(tmp_path / "r.zip"))
    n = extract(z, tmp_path / "out", SMALL_PREFIXES)
    assert n == 3
    assert (tmp_path / "out/annotations.json").exists()
    assert (tmp_path / "out/evaluation/in_domain.csv").exists()
    assert (tmp_path / "out/proprioception/h_pickandplace_ep1.parquet").exists()
    assert not (tmp_path / "out/videos").exists()
    assert not (tmp_path / "out/.DS_Store").exists()


def test_extract_skips_existing_same_size(tmp_path):
    z = zipfile.ZipFile(make_zip(tmp_path / "r.zip"))
    extract(z, tmp_path / "out", SMALL_PREFIXES)
    assert extract(z, tmp_path / "out", SMALL_PREFIXES) == 0
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_robovad_fetch.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/eval/robovad_fetch.py`：

```python
"""从 Zenodo 上的 RoboVAD.zip 里按需取文件。压缩包 6.86 GB，关节数据只有 23 MB，用 HTTP Range 只读要的部分。

RoboVAD 授权是 CC BY-NC-ND 4.0：只拿来评测，不改编、不再分发数据。
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

ROBOVAD_URL = "https://zenodo.org/records/22754659/files/RoboVAD.zip?download=1"
ROOT = "RoboVAD/"
SMALL_PREFIXES = (
    "RoboVAD/README.md",
    "RoboVAD/license.txt",
    "RoboVAD/annotations.json",
    "RoboVAD/splits/",
    "RoboVAD/evaluation/",
    "RoboVAD/proprioception/",
)
VIDEO_PREFIX = "RoboVAD/videos/"


class HttpRangeFile(io.RawIOBase):
    """把远程文件当成可 seek 的本地文件读，每次 read 发一个 Range 请求。"""

    def __init__(self, url: str):
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.url = head.url  # 跟随重定向后的真实地址
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def extract(z: zipfile.ZipFile, dest: Path, prefixes: tuple[str, ...]) -> int:
    """解出以 prefixes 开头的成员（去掉 RoboVAD/ 前缀）；本地已有且大小相同的跳过。返回新解出的个数。"""
    n = 0
    for info in z.infolist():
        name = info.filename
        if name.endswith("/") or name.startswith("__MACOSX") or name.endswith(".DS_Store"):
            continue
        if not name.startswith(prefixes):
            continue
        target = Path(dest) / name[len(ROOT):]
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(name))
        n += 1
    return n


def fetch_robovad(data_root: str | Path, videos: bool = False) -> Path:
    dest = Path(data_root) / "robovad"
    prefixes = SMALL_PREFIXES + ((VIDEO_PREFIX,) if videos else ())
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ROBOVAD_URL), buffer_size=1 << 20))
    extract(z, dest, prefixes)
    return dest
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_robovad_fetch.py -q`
Expected：`2 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/eval/robovad_fetch.py tests/test_robovad_fetch.py
git commit -m "feat: 按需从 RoboVAD 压缩包取文件（HTTP Range）"
```

---

### Task 3：读 RoboVAD

**Files:**
- Create: `src/jianjin/ingest/robovad.py`、`tests/test_ingest_robovad.py`
- Modify: `tests/helpers.py`（加 `make_robovad`）

- [ ] **Step 1：在 `tests/helpers.py` 末尾加造数据的函数**

```python
# ---- RoboVAD ----


def make_robovad(
    root: Path,
    episodes: dict[str, tuple[np.ndarray, np.ndarray]],
    annotations: dict | None = None,
    evaluation: dict[str, list[tuple[str, int, int, int]]] | None = None,
    validation: list[str] | None = None,
    t0: float = 1.1,
) -> Path:
    """episodes：{示范名: (state T×6, action T×6)}；evaluation：{设置名: [(示范名, 起, 止, 标签)]}。"""
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
    return root
```

- [ ] **Step 2：写失败的测试**

`tests/test_ingest_robovad.py`：

```python
import numpy as np

from helpers import make_robovad
from jianjin.ingest.robovad import read_robovad, task_key


def arm(n=90):
    s = np.tile(np.linspace(0, 1, n)[:, None], (1, 6))
    return s, s + 0.01


def test_task_key_prefers_longest_prefix():
    assert task_key("h_pickandplace_multi_ep3") == "h_pickandplace_multi"
    assert task_key("h_pickandplace_ep3") == "h_pickandplace"


def test_reads_episodes_and_annotations(tmp_path):
    ann = {"h_stackcubes_ep2": [{"start_frame": 10, "end_frame": 20, "type": "collision", "categories": {}}]}
    root = make_robovad(tmp_path / "rv", {"h_stackcubes_ep2": arm(), "h_pour_rice_ep5": arm(60)}, annotations=ann)
    res = read_robovad(root)
    assert res.failures == []
    eps = {e.id: e for e in res.episodes}
    e = eps["h_stackcubes_ep2"]
    assert e.source == "robovad" and e.fps == 30.0
    assert e.state.shape == (90, 6) and e.action.shape == (90, 6)
    assert e.gripper_index == 5
    assert np.isclose(e.timestamps[0], 0.0)
    assert e.meta["task_key"] == "h_stackcubes"
    assert e.meta["anomalies"][0]["type"] == "collision"
    np.testing.assert_array_equal(e.meta["frame_index"], np.arange(90))
    assert eps["h_pour_rice_ep5"].meta["anomalies"] == []
    assert "rice" in eps["h_pour_rice_ep5"].task


def test_only_requested_episodes(tmp_path):
    root = make_robovad(tmp_path / "rv", {"h_stackcubes_ep1": arm(), "h_stackcubes_ep2": arm()})
    assert [e.id for e in read_robovad(root, names=["h_stackcubes_ep2"]).episodes] == ["h_stackcubes_ep2"]
```

- [ ] **Step 3：跑测试，确认失败**

Run：`uv run pytest tests/test_ingest_robovad.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 4：写实现**

`src/jianjin/ingest/robovad.py`：

```python
"""读 RoboVAD 的关节数据（proprioception/*.parquet）和异常标注（annotations.json）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from jianjin.ingest.lerobot import _matrix
from jianjin.types import Episode, LoadResult, VideoRef

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
```

- [ ] **Step 5：跑测试，确认通过**

Run：`uv run pytest tests/test_ingest_robovad.py -q`
Expected：`3 passed`。

- [ ] **Step 6：Commit**

```bash
git add src/jianjin/ingest/robovad.py tests/helpers.py tests/test_ingest_robovad.py
git commit -m "feat: 读 RoboVAD 关节数据和标注"
```

---

### Task 4：逐帧异常打分器

**Files:**
- Create: `src/jianjin/scores/__init__.py`、`src/jianjin/scores/frame.py`、`tests/test_frame_scores.py`

- [ ] **Step 1：写失败的测试**

`tests/test_frame_scores.py`：

```python
import numpy as np
import pytest

from helpers import make_episode
from jianjin.checks import NotApplicable
from jianjin.scores import SCORERS
from jianjin.scores.frame import accel, gripper_tracking, stillness, tracking

NAMES = ["a", "b", "gripper"]


def follower(n=120):
    """从臂完全跟住主臂，只有第 50 到 60 帧被挡住（像撞到东西）。"""
    action = np.stack([np.linspace(0, 1, n), np.linspace(0, 2, n), np.linspace(0, 1, n)], axis=1)
    state = action.copy()
    state[50:60, 0] -= 0.3
    return state, action


def test_tracking_peaks_where_blocked():
    state, action = follower()
    s = tracking(make_episode(state, names=NAMES, action=action))
    assert s.shape == (120,)
    assert 45 <= int(np.argmax(s)) <= 64
    assert s[:40].max() < s[50:60].min()


def test_gripper_tracking_ignores_other_joints():
    state, action = follower()
    s = gripper_tracking(make_episode(state, names=NAMES, action=action))
    assert np.allclose(s, 0)
    state[80:90, 2] += 0.2  # 夹爪被物体挡住
    s = gripper_tracking(make_episode(state, names=NAMES, action=action))
    assert 75 <= int(np.argmax(s)) <= 94


def test_accel_peaks_at_jerk():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(60)])  # 第 60 帧急停
    s = accel(make_episode(np.stack([x, x], axis=1)))
    assert s.shape == (120,)
    assert 55 <= int(np.argmax(s)) <= 64


def test_stillness_high_when_still():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(60)])
    s = stillness(make_episode(np.stack([x, x], axis=1)))
    assert s[90] > s[30]


def test_needs_action():
    state, _ = follower()
    with pytest.raises(NotApplicable):
        tracking(make_episode(state, names=NAMES))


def test_registry():
    assert set(SCORERS) == {"tracking", "gripper_tracking", "accel", "stillness"}
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_frame_scores.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.scores'`。

- [ ] **Step 3：写实现**

`src/jianjin/scores/frame.py`：

```python
"""逐帧异常分数，越高越像异常。不用任何标签训练；都按关节活动幅度归一化，和单位无关。"""

from __future__ import annotations

import numpy as np

from jianjin.checks import NotApplicable
from jianjin.checks.signal import joint_scale, speed
from jianjin.types import Episode

SMOOTH_S = 0.3  # 平滑窗口，秒


def _smooth(x: np.ndarray, fps: float) -> np.ndarray:
    k = max(1, int(round(SMOOTH_S * fps)))
    if k == 1 or len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _pad_to(x: np.ndarray, n: int) -> np.ndarray:
    """差分会少帧，前后补边上的值，对齐到 n 帧。"""
    if len(x) == 0:
        return np.zeros(n)
    left = (n - len(x)) // 2
    return np.pad(x, (left, n - len(x) - left), mode="edge")


def _need_action(ep: Episode) -> np.ndarray:
    if ep.action is None or ep.action.shape != ep.state.shape:
        raise NotApplicable("没有和关节状态同维度的动作（主臂指令）")
    return ep.action


def tracking(ep: Episode) -> np.ndarray:
    """从臂跟不上主臂：碰撞、被挡住时变大。"""
    action = _need_action(ep)
    err = np.nan_to_num(np.abs(action - ep.state) / joint_scale(ep.state))
    return _smooth(err.max(axis=1), ep.fps)


def gripper_tracking(ep: Episode) -> np.ndarray:
    """夹爪合不到主臂要的位置：夹着东西时大，抓空或掉了时小。"""
    action = _need_action(ep)
    g = ep.gripper_index
    if g is None:
        raise NotApplicable("认不出夹爪通道")
    scale = joint_scale(ep.state)[g]
    err = np.nan_to_num(np.abs(action[:, g] - ep.state[:, g]) / scale)
    return _smooth(err, ep.fps)


def accel(ep: Episode) -> np.ndarray:
    """加速度：急停、撞击。"""
    scale = joint_scale(ep.state)
    v = np.diff(ep.state, axis=0) / scale * ep.fps
    a = np.nan_to_num(np.abs(np.diff(v, axis=0)) * ep.fps).max(axis=1) if len(v) > 1 else np.zeros(0)
    return _smooth(_pad_to(a, len(ep.state)), ep.fps)


def stillness(ep: Episode) -> np.ndarray:
    """越不动分越高：卡住、犹豫。"""
    return _smooth(-_pad_to(speed(ep.state, ep.fps), len(ep.state)), ep.fps)
```

`src/jianjin/scores/__init__.py`：

```python
from collections.abc import Callable

import numpy as np

from jianjin.scores.frame import accel, gripper_tracking, stillness, tracking
from jianjin.types import Episode

FrameScorer = Callable[[Episode], np.ndarray]

SCORERS: dict[str, FrameScorer] = {
    "tracking": tracking,
    "gripper_tracking": gripper_tracking,
    "accel": accel,
    "stillness": stillness,
}

__all__ = ["SCORERS", "FrameScorer"]
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_frame_scores.py -q`
Expected：`6 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/scores tests/test_frame_scores.py
git commit -m "feat: 逐帧异常打分器（跟随误差、夹爪、加速度、静止）"
```

---

### Task 5：考卷 B runner

**Files:**
- Create: `src/jianjin/eval/exam_b.py`、`tests/test_exam_b.py`

- [ ] **Step 1：写失败的测试**

`tests/test_exam_b.py`：

```python
import math

import numpy as np

from helpers import make_robovad
from jianjin.eval.exam_b import SETTINGS, format_table, labels_from_annotations, run_exam_b


def blocked(n, lo, hi):
    action = np.tile(np.linspace(0, 1, n)[:, None], (1, 6))
    state = action.copy()
    state[lo:hi, 0] -= 0.3
    return state, action


def setup(tmp_path):
    eps = {
        "h_stackcubes_ep1": blocked(150, 60, 80),
        "h_stackcubes_ep2": blocked(150, 0, 0),
        "h_pour_rice_ep1": blocked(150, 100, 120),
    }
    ann = {
        "h_stackcubes_ep1": [{"start_frame": 60, "end_frame": 80, "type": "collision", "categories": {}}],
        "h_pour_rice_ep1": [{"start_frame": 100, "end_frame": 120, "type": "collision", "categories": {}}],
    }
    rows = [
        ("h_stackcubes_ep1", 0, 60, 0), ("h_stackcubes_ep1", 60, 80, 1), ("h_stackcubes_ep1", 90, 150, 0),
        ("h_stackcubes_ep2", 0, 150, 0),
        ("missing_ep", 0, 10, 0),
    ]
    evaluation = {s: rows for s in SETTINGS}
    return make_robovad(tmp_path / "rv", eps, ann, evaluation, validation=["h_pour_rice_ep1", "h_stackcubes_ep2"])


def test_labels_from_annotations():
    y = labels_from_annotations(10, [{"start_frame": 2, "end_frame": 4}, {"start_frame": 8, "end_frame": 20}])
    assert y.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]


def test_test_split_uses_listed_frames_only(tmp_path):
    out = run_exam_b(setup(tmp_path), "test", "tracking")
    assert [r["setting"] for r in out["results"]] == SETTINGS
    r = out["results"][0]
    assert r["n_episodes"] == 2
    assert r["n_frames"] == 60 + 20 + 60 + 150  # 80 到 90 帧没列出，不评
    assert r["n_anomalous"] == 20
    assert r["missing_episodes"] == ["missing_ep"]
    assert r["micro_auc"] > 0.95
    assert r["n_macro_episodes"] == 1  # 只有 ep1 同时有正常和异常帧


def test_validation_split_from_annotations(tmp_path):
    out = run_exam_b(setup(tmp_path), "validation", "tracking")
    (r,) = out["results"]
    assert r["setting"] == "validation"
    assert r["n_episodes"] == 2 and r["n_frames"] == 300 and r["n_anomalous"] == 20
    assert r["micro_auc"] > 0.95


def test_table_has_denominators_and_paper(tmp_path):
    out = run_exam_b(setup(tmp_path), "test", "tracking")
    t = format_table(out)
    assert "零样本" in t and "FIDeL" in t and "帧" in t
    assert not math.isnan(out["results"][0]["micro_auc"])
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_exam_b.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/eval/exam_b.py`：

```python
"""考卷 B：RoboVAD 帧级异常检测。按论文协议只评 csv 里列出的帧；micro AUC 把所有被评的帧合起来算。

拣金零样本：不在 RoboVAD 上训练；打分器在 validation 上选，test 只跑一次。
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from jianjin.checks import NotApplicable
from jianjin.eval.auc import roc_auc
from jianjin.ingest.robovad import read_robovad
from jianjin.scores import SCORERS

SETTINGS = ["in_domain", "cross_anomaly", "cross_task", "cross_task+anomaly"]

# 论文 arXiv 2609.17843 表 II，帧级 AUC（micro, macro），百分数。这些方法都在 RoboVAD 上训练过。
PAPER_TABLE_II = {
    "FlowJigsaw-Supervised": {"in_domain": (94.6, 94.4), "cross_anomaly": (69.0, 71.9), "cross_task": (72.0, 73.8), "cross_task+anomaly": (66.7, 66.8)},
    "UR-DMU": {"in_domain": (88.5, 94.4), "cross_anomaly": (61.3, 71.0), "cross_task": (71.8, 77.5), "cross_task+anomaly": (63.9, 71.9)},
    "Jigsaw-VAD": {"in_domain": (64.1, 65.2), "cross_anomaly": (51.7, 51.1), "cross_task": (66.9, 70.9), "cross_task+anomaly": (57.6, 57.9)},
    "FIDeL": {"in_domain": (56.3, 54.0), "cross_anomaly": (52.1, 53.4), "cross_task": (58.1, 53.7), "cross_task+anomaly": (63.2, 51.2)},
}

Segments = dict[str, list[tuple[int, int, int]]]  # 示范名 -> [(起, 止, 标签)]


def read_segments(path: Path) -> Segments:
    out: Segments = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["episode_name"]].append(
                (int(row["start_frame"]), int(row["end_frame_exclusive"]), int(row["label"]))
            )
    return dict(out)


def labels_from_annotations(n: int, anomalies: list[dict]) -> np.ndarray:
    y = np.zeros(n, dtype=np.int8)
    for a in anomalies:
        y[max(0, a["start_frame"]) : min(n, a["end_frame"])] = 1
    return y


def _validation_segments(root: Path, episodes: dict) -> Segments:
    with open(root / "splits" / "validation.csv", encoding="utf-8", newline="") as f:
        names = [r["episode_name"] for r in csv.DictReader(f)]
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


def score_population(segments: Segments, episodes: dict, scorer) -> dict:
    all_s, all_y, per_ep = [], [], []
    missing, not_applicable, missing_frames = [], [], 0
    for name, segs in segments.items():
        ep = episodes.get(name)
        if ep is None:
            missing.append(name)
            continue
        try:
            s = scorer(ep)
        except NotApplicable:
            not_applicable.append(name)
            continue
        fi = ep.meta["frame_index"]
        by_frame = np.full(int(max(fi.max() + 1, max((e for _, e, _ in segs), default=0))), np.nan)
        by_frame[fi] = s
        ss, ys = [], []
        for start, end, label in segs:
            chunk = by_frame[start:end]
            ss.append(chunk)
            ys.append(np.full(len(chunk), label, dtype=np.int8))
        ss, ys = np.concatenate(ss) if ss else np.zeros(0), np.concatenate(ys) if ys else np.zeros(0)
        ok = np.isfinite(ss)
        missing_frames += int((~ok).sum())
        all_s.append(ss[ok])
        all_y.append(ys[ok])
        per_ep.append(roc_auc(ss[ok], ys[ok]))
    s = np.concatenate(all_s) if all_s else np.zeros(0)
    y = np.concatenate(all_y) if all_y else np.zeros(0)
    macro = [a for a in per_ep if not math.isnan(a)]
    return {
        "n_episodes": len(all_s),
        "n_frames": int(len(y)),
        "n_anomalous": int(y.sum()),
        "micro_auc": roc_auc(s, y),
        "macro_auc": float(np.mean(macro)) if macro else math.nan,
        "n_macro_episodes": len(macro),
        "missing_episodes": missing,
        "not_applicable": not_applicable,
        "missing_frames": missing_frames,
    }


def run_exam_b(root: str | Path, split: str, scorer_name: str) -> dict:
    root = Path(root)
    res = read_robovad(root)
    episodes = {e.id: e for e in res.episodes}
    if split == "validation":
        populations = {"validation": _validation_segments(root, episodes)}
    else:
        populations = {s: read_segments(root / "evaluation" / f"{s}.csv") for s in SETTINGS}
    scorer = SCORERS[scorer_name]
    results = [{"setting": k, **score_population(v, episodes, scorer)} for k, v in populations.items()]
    return {
        "exam": "B（RoboVAD 帧级异常检测；拣金零样本，只用关节数据）",
        "split": split,
        "scorer": scorer_name,
        "read_failures": res.failures,
        "results": results,
    }


def _pct(x: float) -> str:
    return "—" if math.isnan(x) else f"{x * 100:.1f}"


def format_table(out: dict) -> str:
    lines = [f"考卷 {out['exam']}", f"切分：{out['split']}；打分器：{out['scorer']}"]
    for r in out["results"]:
        lines += [
            f"[{r['setting']}] micro AUC {_pct(r['micro_auc'])}，macro AUC {_pct(r['macro_auc'])}"
            f"（macro 只算同时有正常和异常帧的 {r['n_macro_episodes']} 条）",
            f"  分母：{r['n_episodes']} 条示范、{r['n_frames']} 帧，其中异常 {r['n_anomalous']} 帧；"
            f"缺关节数据 {len(r['missing_episodes'])} 条，打不了分 {len(r['not_applicable'])} 条，"
            f"对不上帧号 {r['missing_frames']} 帧",
        ]
        if r["setting"] in SETTINGS:
            paper = "；".join(f"{m} {v[r['setting']][0]}/{v[r['setting']][1]}" for m, v in PAPER_TABLE_II.items())
            lines.append(f"  论文表 II（micro/macro，都训练过）：{paper}")
    return "\n".join(lines)
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_exam_b.py -q`
Expected：`4 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/eval/exam_b.py tests/test_exam_b.py
git commit -m "feat: 考卷 B runner（RoboVAD 协议，帧级 AUC）"
```

---

### Task 6：命令行

**Files:**
- Modify: `src/jianjin/cli.py`、`tests/test_exam_b.py`

- [ ] **Step 1：写失败的测试**

在 `tests/test_exam_b.py` 末尾加：

```python
def test_eval_b_command(tmp_path, capsys):
    import json

    from jianjin.cli import main

    root = setup(tmp_path)
    out = tmp_path / "b.json"
    assert main(["eval", "b", "--robovad-root", str(root), "--split", "test", "--scorer", "tracking", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["scorer"] == "tracking"
    assert "micro AUC" in capsys.readouterr().out
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_exam_b.py::test_eval_b_command -q`
Expected：FAIL，`argument exam: invalid choice: 'b'`。

- [ ] **Step 3：改 `src/jianjin/cli.py`**

把 `_cmd_eval` 换成：

```python
def _cmd_eval(args: argparse.Namespace) -> int:
    if args.exam == "a":
        from jianjin.eval.exam_a import format_table, load_config, run_exam_a

        split = args.split or "dev"
        out = run_exam_a(load_config(args.config), args.data_root, split, seed=args.seed)
    else:
        from jianjin.eval.exam_b import format_table, run_exam_b

        split = args.split or "validation"
        out = run_exam_b(args.robovad_root, split, args.scorer)
    print(format_table(out))
    path = Path(args.out or f"results/exam_{args.exam}_{split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写到 {path}")
    return 0
```

把 `_cmd_fetch` 换成：

```python
def _cmd_fetch(args: argparse.Namespace) -> int:
    if args.what == "robovad":
        from jianjin.eval.robovad_fetch import fetch_robovad

        print(f"已取到 {fetch_robovad(args.data_root, videos=args.videos)}")
        return 0
    from jianjin.eval.exam_a import load_config
    from jianjin.eval.fetch import fetch

    for p in fetch(load_config(args.config), args.data_root, with_videos=args.videos):
        print(f"已下载到 {p}")
    return 0
```

`build_parser()` 里 `eval` 和 `fetch` 两段换成：

```python
    e = sub.add_parser("eval", help="跑考卷")
    e.add_argument("exam", choices=["a", "b"])
    e.add_argument("--config", default="configs/exam_a.toml")
    e.add_argument("--data-root", default="data")
    e.add_argument("--robovad-root", default="data/robovad")
    e.add_argument("--split", choices=["dev", "test", "validation"], default=None)
    e.add_argument("--scorer", default="tracking")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_eval)

    f = sub.add_parser("fetch", help="下载考卷数据")
    f.add_argument("what", nargs="?", choices=["exam_a", "robovad"], default="exam_a")
    f.add_argument("--config", default="configs/exam_a.toml")
    f.add_argument("--data-root", default="data")
    f.add_argument("--videos", action="store_true", help="连视频一起下")
    f.set_defaults(func=_cmd_fetch)
```

- [ ] **Step 4：跑全部测试**

Run：`uv run pytest -q`
Expected：全部通过。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/cli.py tests/test_exam_b.py
git commit -m "feat: jianjin eval b、jianjin fetch robovad"
```

---

### Task 7：真数据：取数、选打分器、跑 test 一次

取数只走 Range，约 23 MB，不下视频。

**Files:**
- Create: `scripts/choose_exam_b_scorer.py`、`results/exam_b_validation.json`、`results/exam_b_test.json`
- Modify: `src/jianjin/eval/exam_b.py`（论文数字有出入时）、设计文档第 6 节

- [ ] **Step 1：核论文表 II**

打开 https://arxiv.org/html/2609.17843 ，找表 II，逐格对 `PAPER_TABLE_II`；表里还有别的方法（比如只用视频的），把最好的那几行补进去，在常量上方注明哪些用了关节数据。

- [ ] **Step 2：取数、核读入**

```bash
uv run jianjin fetch robovad
uv run python -c "from jianjin.ingest.robovad import read_robovad; r = read_robovad('data/robovad'); print(len(r.episodes), len(r.failures)); e = r.episodes[0]; print(e.id, e.fps, e.state.shape, e.meta['frame_index'][:3])"
```

Expected：读到 1,078 条左右（README 说 1,078；压缩包里有 1,079 个关节文件，多出来的那个查一下是什么）、读取失败 0；fps 30。

- [ ] **Step 3：写选打分器的脚本**

`scripts/choose_exam_b_scorer.py`：

```python
"""在 RoboVAD 的 validation 划分上比打分器，按 micro AUC 选一个；test 只用选中的那个跑一次。"""

from __future__ import annotations

import sys

from jianjin.eval.exam_b import run_exam_b
from jianjin.scores import SCORERS


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    best = None
    for name in SCORERS:
        (r,) = run_exam_b("data/robovad", "validation", name)["results"]
        print(f"{name}: micro {r['micro_auc']:.3f}  macro {r['macro_auc']:.3f}  ({r['n_episodes']} 条，{r['n_frames']} 帧，异常 {r['n_anomalous']} 帧)")
        if best is None or r["micro_auc"] > best[1]:
            best = (name, r["micro_auc"])
    print(f"选：{best}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4：在 validation 上选**

Run：`uv run python scripts/choose_exam_b_scorer.py`

然后：

```bash
uv run jianjin eval b --split validation --scorer <选中的>
```

- [ ] **Step 5：test 跑一次**

```bash
uv run jianjin eval b --split test --scorer <选中的>
```

只跑这一次。之后要换打分器，回 validation 重选，并在结论里写明 test 已经看过一次。

- [ ] **Step 6：写回文档、Commit**

在设计文档第 6 节「考卷 B」下加「第一批结果（日期）」：四种设置的 micro 和 macro、分母、和论文表 II 并列；写明拣金零样本、只用关节数据、打分器在 validation 上选的；validation 上各打分器的成绩也列出来。

```bash
git add scripts results src docs
git commit -m "feat: 考卷 B 关节部分第一批数字"
```

---

## 本计划之后

2b：视频解码、评判适配器接口和缓存（按示范内容哈希加模型版本）、Robometer 和 TOPReward 适配器、Colab 笔记本（用户来跑）、关节和视频分数融合、考卷 A 的「换掉任务描述」注入。
