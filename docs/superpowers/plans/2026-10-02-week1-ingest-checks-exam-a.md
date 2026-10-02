# 拣金第一周：读入、关节体检、考卷 A 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 读入 LeRobot v3 和 RoboMIND HDF5 两种格式，跑只用 CPU 的关节体检，在真机示范上注入缺陷，出考卷 A 第一批带分母的数字。

**Architecture:** `ingest` 把两种格式读成同一个 `Episode`；`checks` 里每条规则是一个纯函数 `Episode -> list[Finding]`，缺数据时抛 `NotApplicable`，由 runner 记进 `not_checked`；`eval` 对原样示范注入缺陷、各跑一遍体检、按缺陷类别算检出率和原样示范上的报出率。所有速度阈值都用「这条示范里每个关节的活动幅度」归一化，所以 SO-101 的度数和 RoboMIND 的弧度能用同一套阈值。

**Tech Stack:** Python 3.12、uv、numpy、pyarrow、h5py、pytest；下载公开数据用可选依赖 huggingface_hub。

**范围：** 对应设计文档第 8 节的 10-03 至 10-09。视频评判（Robometer、TOPReward）、`diagnose` 判定、网页报告、考卷 B 和 C、「换掉任务描述」这一种注入，都在后面的计划里。夹爪类规则（抓空、掉落）要和视频进度合着判，放到诊断那份计划。

**和设计文档的出入（Task 1 做完时写回设计文档 4.2）：**

- `Episode` 多了 `timestamps`、`state_names`、`gripper_index` 三个字段，`gripper` 改成从 `state` 里按 `gripper_index` 取的属性，这样注入缺陷时夹爪通道不会和关节状态错位。
- `Finding` 多了 `hard` 字段，标记「客观硬规则」，诊断时一条硬规则就能判作废（设计文档 4.3）。
- 失败原因多了一类「数据异常」，给读数缺失、丢帧、示范过短这类硬规则用。
- `Verdict` 本周不建，留给诊断那份计划。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 依赖、命令行入口、pytest 配置 |
| `src/jianjin/types.py` | `Cause`、`VideoRef`、`Episode`、`Finding`、`LoadResult` |
| `src/jianjin/ingest/__init__.py` | `load(path)`：认格式、分派 |
| `src/jianjin/ingest/lerobot.py` | 读 LeRobot v3 |
| `src/jianjin/ingest/robomind.py` | 读 RoboMIND HDF5 |
| `src/jianjin/checks/config.py` | `CheckConfig`（全部阈值）、`NotApplicable` |
| `src/jianjin/checks/signal.py` | 关节幅度归一化、速度、连续区段、路程 |
| `src/jianjin/checks/integrity.py` | 硬规则：读数缺失、示范过短、丢帧 |
| `src/jianjin/checks/motion.py` | 停顿过长、结尾还在动 |
| `src/jianjin/checks/batch.py` | 同任务比较：多余动作 |
| `src/jianjin/checks/runner.py` | `CheckResult`、`run_checks` |
| `src/jianjin/checks/__init__.py` | 对外导出 |
| `src/jianjin/eval/inject.py` | 截掉结尾、插入静止帧 |
| `src/jianjin/eval/metrics.py` | Wilson 区间、区间重叠、按缺陷打分 |
| `src/jianjin/eval/exam_a.py` | 读配置、注入、跑体检、出表 |
| `src/jianjin/eval/fetch.py` | 按配置从 Hugging Face 下载 meta 和 data |
| `src/jianjin/cli.py` | `jianjin check`、`jianjin eval a`、`jianjin fetch` |
| `configs/exam_a.toml` | 考卷 A 用哪些数据集、各属开发集还是测试集、授权 |
| `scripts/tune_exam_a.py` | 在开发集上扫阈值 |
| `tests/helpers.py` | `make_episode`、`make_lerobot_v3`、`make_robomind_h5` |
| `tests/test_*.py` | 每个模块一份 |

---

### Task 0：项目骨架

**Files:**
- Create: `pyproject.toml`、`.python-version`、`src/jianjin/__init__.py`、`tests/test_smoke.py`
- Modify: `.gitignore`

- [ ] **Step 1：写 `pyproject.toml`**

```toml
[project]
name = "jianjin"
version = "0.1.0"
description = "数采场示范数据的自动质检"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0", "h5py>=3.11", "pyarrow>=17"]

[project.optional-dependencies]
hub = ["huggingface_hub>=0.30"]

[project.scripts]
jianjin = "jianjin.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jianjin"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
```

- [ ] **Step 2：写包和冒烟测试**

`src/jianjin/__init__.py`：

```python
"""拣金：数采场示范数据的自动质检。"""

__version__ = "0.1.0"
```

`tests/test_smoke.py`：

```python
import jianjin


def test_version():
    assert jianjin.__version__ == "0.1.0"
```

- [ ] **Step 3：`.gitignore` 末尾加两行**

```
data/
.pytest_cache/
```

`results/` 不忽略，考卷成绩要进仓库。

- [ ] **Step 4：装环境、跑测试**

```bash
uv python pin 3.12
uv sync --all-extras
uv run pytest -q
```

Expected：`1 passed`。

- [ ] **Step 5：Commit**

```bash
git add pyproject.toml .python-version uv.lock .gitignore src tests
git commit -m "chore: 项目骨架（uv、pytest）"
```

---

### Task 1：数据结构

**Files:**
- Create: `src/jianjin/types.py`、`tests/helpers.py`、`tests/test_types.py`
- Modify: `docs/superpowers/specs/2026-10-02-jianjin-design.md`（4.2 和第 5 节）

- [ ] **Step 1：写失败的测试**

`tests/test_types.py`：

```python
import numpy as np

from helpers import make_episode
from jianjin.types import Cause, Finding


def test_duration_and_gripper():
    state = np.zeros((60, 3))
    state[:, 2] = np.linspace(0, 1, 60)
    ep = make_episode(state, fps=30, names=["a", "b", "gripper.pos"])
    assert ep.duration == 2.0
    assert ep.gripper_index == 2
    np.testing.assert_allclose(ep.gripper, state[:, 2])


def test_no_gripper():
    ep = make_episode(np.zeros((30, 2)), names=["a", "b"])
    assert ep.gripper is None


def test_finding_defaults_soft():
    f = Finding("x", Cause.LONG_PAUSE, 1.0, 2.0, 1.0, "停了")
    assert f.hard is False
    assert Cause.DATA_ERROR.value == "数据异常"
```

`tests/helpers.py`（后面的 Task 会往里加函数）：

```python
from __future__ import annotations

import numpy as np

from jianjin.types import Episode


def find_gripper(names: list[str]) -> int | None:
    for i, n in enumerate(names):
        if "gripper" in n.lower():
            return i
    return None


def make_episode(
    state: np.ndarray,
    fps: float = 30.0,
    task: str = "pick the cube",
    names: list[str] | None = None,
    ep_id: str = "ep0",
    action: np.ndarray | None = None,
) -> Episode:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state[:, None]
    names = names or [f"j{i}" for i in range(state.shape[1])]
    return Episode(
        id=ep_id,
        source="test",
        task=task,
        fps=fps,
        state=state,
        action=action,
        timestamps=np.arange(len(state)) / fps,
        state_names=names,
        gripper_index=find_gripper(names),
    )
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_types.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.types'`。

- [ ] **Step 3：写 `src/jianjin/types.py`**

```python
"""拣金的核心数据结构。设计见设计文档 4.2。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


class Cause(StrEnum):
    """失败原因类别，见设计文档第 5 节。"""

    INCOMPLETE = "没做完"
    MISPLACED = "定位不准"
    GRASP_MISS = "夹爪没合上"
    DROP = "物体掉落"
    EXTRA_MOTION = "多余动作"
    LONG_PAUSE = "停顿过长"
    WRONG_TASK = "指令不符"
    DATA_ERROR = "数据异常"


@dataclass(frozen=True)
class VideoRef:
    """一路视频在文件里的位置；LeRobot v3 多条示范共用一个 mp4。"""

    path: Path
    from_s: float
    to_s: float | None


@dataclass
class Episode:
    id: str
    source: str
    task: str
    fps: float
    state: np.ndarray  # (T, D)
    action: np.ndarray | None  # (T, D') 或没有
    timestamps: np.ndarray  # (T,) 秒
    state_names: list[str]
    gripper_index: int | None = None
    cameras: dict[str, VideoRef] = field(default_factory=dict)
    operator_id: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.state) / self.fps

    @property
    def gripper(self) -> np.ndarray | None:
        if self.gripper_index is None:
            return None
        return self.state[:, self.gripper_index]


@dataclass(frozen=True)
class Finding:
    check: str
    cause: Cause
    t0: float
    t1: float
    score: float
    evidence: str
    hard: bool = False


@dataclass
class LoadResult:
    episodes: list[Episode]
    failures: list[tuple[str, str]] = field(default_factory=list)  # (哪个文件, 为什么)
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_types.py -q`
Expected：`3 passed`。

- [ ] **Step 5：写回设计文档**

在设计文档 4.2 节末尾加：

```markdown
实现时的出入（2026-10-02，第一周计划）：`Episode` 加了 `timestamps`、`state_names`、`gripper_index`，`gripper` 改成按 `gripper_index` 从 `state` 取的属性，注入缺陷时夹爪不会和关节错位；`Finding` 加了 `hard`，标记客观硬规则。
```

在第 5 节表格末尾加一行：

```markdown
| 数据异常 | 读数缺失、丢帧、示范过短；客观硬规则，一条就能判作废 |
```

- [ ] **Step 6：Commit**

```bash
git add src/jianjin/types.py tests/helpers.py tests/test_types.py docs/superpowers/specs/2026-10-02-jianjin-design.md
git commit -m "feat: Episode、Finding、Cause 数据结构"
```

---

### Task 2：信号工具

**Files:**
- Create: `src/jianjin/checks/__init__.py`（先空着）、`src/jianjin/checks/config.py`、`src/jianjin/checks/signal.py`、`tests/test_signal.py`

- [ ] **Step 1：写失败的测试**

`tests/test_signal.py`：

```python
import numpy as np

from jianjin.checks.signal import joint_scale, path_length, runs, speed


def test_runs():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    assert runs(m) == [(1, 3), (4, 5), (7, 8)]
    assert runs(np.zeros(0, dtype=bool)) == []


def test_joint_scale_ignores_still_joint():
    s = np.stack([np.linspace(0, 10, 101), np.full(101, 3.0)], axis=1)
    sc = joint_scale(s)
    assert np.isclose(sc[0], 9.0)  # 95 分位减 5 分位
    assert np.isinf(sc[1])


def test_speed_is_unit_free():
    deg = np.linspace(0, 90, 31)[:, None]
    rad = np.deg2rad(deg)
    np.testing.assert_allclose(speed(deg, 30), speed(rad, 30))
    assert speed(deg, 30).shape == (30,)


def test_path_length_doubles_for_back_and_forth():
    there = np.linspace(0, 1, 50)[:, None]
    back = np.concatenate([there, there[::-1], there])
    scale = np.array([1.0])
    assert np.isclose(path_length(back, scale), 3 * path_length(there, scale))
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_signal.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/checks/__init__.py` 先写一个空文件（Task 6 再填）。

`src/jianjin/checks/config.py`：

```python
"""关节体检的全部阈值。默认值在考卷 A 的开发集上定，测试集只用来报成绩。"""

from __future__ import annotations

from dataclasses import dataclass


class NotApplicable(Exception):
    """缺了某种数据，这条规则查不了；记进 not_checked，不当成通过。"""


@dataclass(frozen=True)
class CheckConfig:
    # 速度单位是「每秒走过本条示范里该关节活动幅度的几分之几」
    still_rate: float = 0.02  # 低于它算没动
    pause_s: float = 3.0  # 中途连续不动超过这么久，算停顿过长
    end_window_s: float = 0.5  # 看结尾多长一段
    end_moving_rate: float = 0.1  # 结尾平均速度高于它，算没做完就停了
    min_duration_s: float = 1.0  # 短于它，硬规则判过短
    max_gap_factor: float = 1.5  # 相邻时间戳间隔超过 factor/fps，算丢帧
    extra_factor: float = 2.0  # 路程超过同任务中位数这么多倍，算多余动作
    min_task_episodes: int = 5  # 同任务少于这么多条，不做同任务比较
```

`src/jianjin/checks/signal.py`：

```python
"""关节信号的基本运算。所有速度都按关节活动幅度归一化，和单位无关。"""

from __future__ import annotations

import numpy as np


def joint_scale(state: np.ndarray) -> np.ndarray:
    """每个关节的活动幅度（5% 到 95% 分位之差）；没动过的关节记 inf，不参与比较。"""
    lo, hi = np.nanpercentile(state, [5, 95], axis=0)
    scale = np.asarray(hi - lo, dtype=np.float64)
    scale[~(scale > 1e-9)] = np.inf
    return scale


def speed(state: np.ndarray, fps: float, scale: np.ndarray | None = None) -> np.ndarray:
    """相邻两帧之间、动得最快的那个关节的归一化速度（幅度/秒），长度 T-1。"""
    if scale is None:
        scale = joint_scale(state)
    if len(state) < 2:
        return np.zeros(0)
    step = np.abs(np.diff(state, axis=0)) / scale
    return np.max(step, axis=1) * fps


def path_length(state: np.ndarray, scale: np.ndarray) -> float:
    """所有关节归一化后走过的路程之和。"""
    if len(state) < 2:
        return 0.0
    step = np.abs(np.diff(state, axis=0)) / scale
    return float(np.nansum(step))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """mask 里连续为 True 的区段，返回 [start, end) 下标。"""
    if mask.size == 0:
        return []
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_signal.py -q`
Expected：`4 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/checks tests/test_signal.py
git commit -m "feat: 关节信号归一化、速度、区段工具"
```

---

### Task 3：硬规则（读数缺失、示范过短、丢帧）

**Files:**
- Create: `src/jianjin/checks/integrity.py`、`tests/test_integrity.py`

- [ ] **Step 1：写失败的测试**

`tests/test_integrity.py`：

```python
import numpy as np

from helpers import make_episode
from jianjin.checks.config import CheckConfig
from jianjin.checks.integrity import missing_values, timestamp_gaps, too_short
from jianjin.types import Cause

CFG = CheckConfig()


def ramp(n=90, d=2):
    return np.tile(np.linspace(0, 1, n)[:, None], (1, d))


def test_missing_values_found_and_hard():
    s = ramp()
    s[30:33, 1] = np.nan
    fs = missing_values(make_episode(s), CFG)
    assert len(fs) == 1
    f = fs[0]
    assert f.cause is Cause.DATA_ERROR and f.hard
    assert np.isclose(f.t0, 1.0) and np.isclose(f.t1, 1.1)


def test_missing_values_checks_action_too():
    a = ramp()
    a[0, 0] = np.inf
    assert len(missing_values(make_episode(ramp(), action=a), CFG)) == 1


def test_clean_episode_has_no_integrity_findings():
    ep = make_episode(ramp())
    assert missing_values(ep, CFG) == []
    assert too_short(ep, CFG) == []
    assert timestamp_gaps(ep, CFG) == []


def test_too_short():
    fs = too_short(make_episode(ramp(n=20)), CFG)
    assert len(fs) == 1 and fs[0].hard


def test_timestamp_gap():
    ep = make_episode(ramp())
    ts = ep.timestamps.copy()
    ts[45:] += 0.5  # 第 1.5 秒附近丢了约 15 帧
    ep.timestamps = ts
    fs = timestamp_gaps(ep, CFG)
    assert len(fs) == 1
    assert fs[0].hard and fs[0].cause is Cause.DATA_ERROR
    assert np.isclose(fs[0].t0, 44 / 30)
    assert "15" in fs[0].evidence


def test_timestamp_going_backwards_is_a_gap():
    ep = make_episode(ramp())
    ts = ep.timestamps.copy()
    ts[50] = ts[48]
    ep.timestamps = ts
    assert len(timestamp_gaps(ep, CFG)) >= 1
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_integrity.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.checks.integrity'`。

- [ ] **Step 3：写实现**

`src/jianjin/checks/integrity.py`：

```python
"""客观硬规则：数据本身坏了。一条触发就能判作废（设计文档 4.3）。"""

from __future__ import annotations

import numpy as np

from jianjin.checks.config import CheckConfig
from jianjin.checks.signal import runs
from jianjin.types import Cause, Episode, Finding


def missing_values(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    out = []
    arrays = {"关节状态": ep.state}
    if ep.action is not None:
        arrays["动作"] = ep.action
    for name, arr in arrays.items():
        bad = ~np.isfinite(arr).all(axis=1)
        if not bad.any():
            continue
        idx = np.flatnonzero(bad)
        out.append(
            Finding(
                check="missing_values",
                cause=Cause.DATA_ERROR,
                t0=idx[0] / ep.fps,
                t1=(idx[-1] + 1) / ep.fps,
                score=float(len(idx)),
                evidence=f"{name}有 {len(idx)} 帧读数缺失（NaN 或无穷）",
                hard=True,
            )
        )
    return out


def too_short(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    if ep.duration >= cfg.min_duration_s:
        return []
    return [
        Finding(
            check="too_short",
            cause=Cause.DATA_ERROR,
            t0=0.0,
            t1=ep.duration,
            score=ep.duration,
            evidence=f"整条只有 {ep.duration:.2f} 秒，短于 {cfg.min_duration_s} 秒",
            hard=True,
        )
    ]


def timestamp_gaps(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    ts = ep.timestamps
    if len(ts) < 2:
        return []
    dt = np.diff(ts)
    bad = (dt > cfg.max_gap_factor / ep.fps) | (dt <= 0)
    out = []
    for s, e in runs(bad):
        seg = dt[s:e]
        lost = int(np.sum(np.clip(np.round(seg * ep.fps) - 1, 0, None)))
        backwards = bool((seg <= 0).any())
        evidence = (
            f"第 {ts[s]:.1f} 秒附近时间戳倒退或重复"
            if backwards
            else f"第 {ts[s]:.1f} 秒附近丢了约 {lost} 帧"
        )
        out.append(
            Finding(
                check="timestamp_gaps",
                cause=Cause.DATA_ERROR,
                t0=float(ts[s]),
                t1=float(ts[e]),
                score=float(seg.max()),
                evidence=evidence,
                hard=True,
            )
        )
    return out
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_integrity.py -q`
Expected：`6 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/checks/integrity.py tests/test_integrity.py
git commit -m "feat: 硬规则（读数缺失、过短、丢帧）"
```

---

### Task 4：停顿过长、结尾还在动

**Files:**
- Create: `src/jianjin/checks/motion.py`、`tests/test_motion.py`

- [ ] **Step 1：写失败的测试**

`tests/test_motion.py`：

```python
import numpy as np
import pytest

from helpers import make_episode
from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.checks.motion import end_in_motion, long_pause
from jianjin.types import Cause

CFG = CheckConfig()


def two_joints(x):
    return np.stack([x, np.full_like(x, 5.0)], axis=1)  # 第二个关节不动，不该影响结果


def test_pause_in_middle_found():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(150), np.linspace(1, 2, 60)])
    fs = long_pause(make_episode(two_joints(x)), CFG)
    assert len(fs) == 1
    f = fs[0]
    assert f.cause is Cause.LONG_PAUSE and not f.hard
    assert abs(f.t0 - 2.0) < 0.1 and abs(f.t1 - 7.0) < 0.1


def test_still_at_start_and_end_is_normal():
    x = np.concatenate([np.zeros(150), np.linspace(0, 1, 60), np.ones(150)])
    assert long_pause(make_episode(two_joints(x)), CFG) == []


def test_short_pause_ignored():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(30), np.linspace(1, 2, 60)])
    assert long_pause(make_episode(two_joints(x)), CFG) == []


def test_end_in_motion_found():
    x = np.linspace(0, 1, 90)
    fs = end_in_motion(make_episode(two_joints(x)), CFG)
    assert len(fs) == 1
    f = fs[0]
    assert f.cause is Cause.INCOMPLETE
    assert np.isclose(f.t1, 3.0) and np.isclose(f.t0, 2.5)


def test_end_at_rest_is_fine():
    x = np.concatenate([np.linspace(0, 1, 90), np.ones(30)])
    assert end_in_motion(make_episode(two_joints(x)), CFG) == []


def test_end_in_motion_needs_enough_frames():
    with pytest.raises(NotApplicable):
        end_in_motion(make_episode(two_joints(np.linspace(0, 1, 10))), CFG)
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_motion.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.checks.motion'`。

- [ ] **Step 3：写实现**

`src/jianjin/checks/motion.py`：

```python
"""看关节怎么动：中途停太久、录到最后还在动。都是软信号，单独一条只判存疑。"""

from __future__ import annotations

import numpy as np

from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.checks.signal import runs, speed
from jianjin.types import Cause, Episode, Finding


def long_pause(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    v = speed(ep.state, ep.fps)
    out = []
    for s, e in runs(v < cfg.still_rate):
        if s == 0 or e == len(v):  # 开头摆姿势、结尾收手时的静止是正常的
            continue
        dur = (e - s) / ep.fps
        if dur < cfg.pause_s:
            continue
        t0, t1 = s / ep.fps, e / ep.fps
        out.append(
            Finding(
                check="long_pause",
                cause=Cause.LONG_PAUSE,
                t0=t0,
                t1=t1,
                score=dur,
                evidence=f"第 {t0:.1f} 到 {t1:.1f} 秒关节几乎不动，停了 {dur:.1f} 秒",
            )
        )
    return out


def end_in_motion(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    v = speed(ep.state, ep.fps)
    k = max(1, int(round(cfg.end_window_s * ep.fps)))
    if len(v) < 2 * k:
        raise NotApplicable("示范太短，看不了结尾")
    tail = float(np.mean(v[-k:]))
    if tail <= cfg.end_moving_rate:
        return []
    t1 = ep.duration
    return [
        Finding(
            check="end_in_motion",
            cause=Cause.INCOMPLETE,
            t0=t1 - k / ep.fps,
            t1=t1,
            score=tail,
            evidence=f"录到最后 {k / ep.fps:.1f} 秒手臂还在动（每秒 {tail:.2f} 个幅度），像是没做完就停了",
        )
    ]
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_motion.py -q`
Expected：`6 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/checks/motion.py tests/test_motion.py
git commit -m "feat: 停顿过长、结尾还在动"
```

---

### Task 5：多余动作（同任务比较）

**Files:**
- Create: `src/jianjin/checks/batch.py`、`tests/test_batch.py`

- [ ] **Step 1：写失败的测试**

`tests/test_batch.py`：

```python
import numpy as np

from helpers import make_episode
from jianjin.checks.batch import extra_motion
from jianjin.checks.config import CheckConfig
from jianjin.types import Cause

CFG = CheckConfig()


def there():
    return np.linspace(0, 1, 60)


def test_back_and_forth_flagged():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    wobble = np.concatenate([there(), there()[::-1], there(), there()[::-1], there()])
    eps.append(make_episode(wobble, ep_id="bad"))
    found, skipped = extra_motion(eps, CFG)
    assert skipped == {}
    assert set(found) == {"bad"}
    f = found["bad"][0]
    assert f.cause is Cause.EXTRA_MOTION and not f.hard
    assert f.score > 4


def test_small_task_not_checked():
    eps = [make_episode(there(), ep_id=f"e{i}") for i in range(3)]
    found, skipped = extra_motion(eps, CFG)
    assert found == {}
    assert set(skipped) == {"e0", "e1", "e2"}
    assert "3" in skipped["e0"]


def test_tasks_compared_separately():
    a = [make_episode(there(), ep_id=f"a{i}", task="A") for i in range(5)]
    b = [make_episode(np.linspace(0, 1, 60) * 5, ep_id=f"b{i}", task="B") for i in range(5)]
    found, _ = extra_motion(a + b, CFG)
    assert found == {}
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_batch.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/checks/batch.py`：

```python
"""要和同任务其他示范比才看得出来的问题。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from jianjin.checks.config import CheckConfig
from jianjin.checks.signal import joint_scale, path_length
from jianjin.types import Cause, Episode, Finding


def extra_motion(
    episodes: list[Episode], cfg: CheckConfig
) -> tuple[dict[str, list[Finding]], dict[str, str]]:
    """路程远超同任务中位数，算多余动作。返回 (发现, 没查的及原因)。"""
    groups: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for ep in episodes:
        groups[(ep.task, ep.state.shape[1])].append(ep)

    found: dict[str, list[Finding]] = {}
    skipped: dict[str, str] = {}
    for (task, _), group in groups.items():
        if len(group) < cfg.min_task_episodes:
            for ep in group:
                skipped[ep.id] = f"同任务只有 {len(group)} 条，少于 {cfg.min_task_episodes} 条，没法比"
            continue
        scale = joint_scale(np.concatenate([ep.state for ep in group]))
        lengths = np.array([path_length(ep.state, scale) for ep in group])
        med = float(np.median(lengths))
        if med <= 0:
            continue
        for ep, length in zip(group, lengths):
            ratio = length / med
            if ratio > cfg.extra_factor:
                found[ep.id] = [
                    Finding(
                        check="extra_motion",
                        cause=Cause.EXTRA_MOTION,
                        t0=0.0,
                        t1=ep.duration,
                        score=float(ratio),
                        evidence=f"关节走过的路程是同任务 {len(group)} 条示范中位数的 {ratio:.1f} 倍",
                    )
                ]
    return found, skipped
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_batch.py -q`
Expected：`3 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/checks/batch.py tests/test_batch.py
git commit -m "feat: 多余动作（和同任务比路程）"
```

---

### Task 6：体检 runner

**Files:**
- Create: `src/jianjin/checks/runner.py`、`tests/test_runner.py`
- Modify: `src/jianjin/checks/__init__.py`

- [ ] **Step 1：写失败的测试**

`tests/test_runner.py`：

```python
import numpy as np

from helpers import make_episode
from jianjin.checks import CheckConfig, run_checks
from jianjin.types import Cause


def test_runner_collects_findings_and_not_checked():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(150), np.linspace(1, 2, 60), np.full(30, 2.0)])
    paused = make_episode(x, ep_id="paused")
    tiny = make_episode(np.linspace(0, 1, 10), ep_id="tiny")
    results = {r.episode_id: r for r in run_checks([paused, tiny], CheckConfig())}

    assert [f.cause for f in results["paused"].findings] == [Cause.LONG_PAUSE]
    assert "extra_motion" in results["paused"].not_checked

    tiny_causes = {f.check for f in results["tiny"].findings}
    assert "too_short" in tiny_causes
    assert "end_in_motion" in results["tiny"].not_checked


def test_runner_keeps_input_order():
    eps = [make_episode(np.linspace(0, 1, 60), ep_id=f"e{i}") for i in range(3)]
    assert [r.episode_id for r in run_checks(eps)] == ["e0", "e1", "e2"]
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_runner.py -q`
Expected：FAIL，`ImportError: cannot import name 'CheckConfig' from 'jianjin.checks'`。

- [ ] **Step 3：写实现**

`src/jianjin/checks/runner.py`：

```python
"""按顺序跑全部关节规则，把查不了的记进 not_checked。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from jianjin.checks.batch import extra_motion
from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.checks.integrity import missing_values, timestamp_gaps, too_short
from jianjin.checks.motion import end_in_motion, long_pause
from jianjin.types import Episode, Finding

EpisodeCheck = Callable[[Episode, CheckConfig], list[Finding]]
BatchCheck = Callable[
    [list[Episode], CheckConfig], tuple[dict[str, list[Finding]], dict[str, str]]
]

EPISODE_CHECKS: dict[str, EpisodeCheck] = {
    "missing_values": missing_values,
    "too_short": too_short,
    "timestamp_gaps": timestamp_gaps,
    "long_pause": long_pause,
    "end_in_motion": end_in_motion,
}

BATCH_CHECKS: dict[str, BatchCheck] = {
    "extra_motion": extra_motion,
}


@dataclass
class CheckResult:
    episode_id: str
    findings: list[Finding] = field(default_factory=list)
    not_checked: dict[str, str] = field(default_factory=dict)


def run_checks(episodes: list[Episode], cfg: CheckConfig | None = None) -> list[CheckResult]:
    cfg = cfg or CheckConfig()
    results = {ep.id: CheckResult(ep.id) for ep in episodes}
    for ep in episodes:
        r = results[ep.id]
        for name, fn in EPISODE_CHECKS.items():
            try:
                r.findings.extend(fn(ep, cfg))
            except NotApplicable as exc:
                r.not_checked[name] = str(exc)
    for name, fn in BATCH_CHECKS.items():
        found, skipped = fn(episodes, cfg)
        for eid, fs in found.items():
            results[eid].findings.extend(fs)
        for eid, why in skipped.items():
            results[eid].not_checked[name] = why
    return [results[ep.id] for ep in episodes]
```

`src/jianjin/checks/__init__.py`：

```python
from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.checks.runner import BATCH_CHECKS, EPISODE_CHECKS, CheckResult, run_checks

__all__ = [
    "BATCH_CHECKS",
    "EPISODE_CHECKS",
    "CheckConfig",
    "CheckResult",
    "NotApplicable",
    "run_checks",
]
```

- [ ] **Step 4：跑全部测试**

Run：`uv run pytest -q`
Expected：全部通过（`25 passed`）。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/checks tests/test_runner.py
git commit -m "feat: 关节体检 runner"
```

---

### Task 7：读 LeRobot v3

格式依据 LeRobot 文档（`docs/source/lerobot-dataset-v3.mdx`）：`meta/info.json` 有 `codebase_version`、`fps`、`features`、`data_path`、`video_path`；`data/chunk-XXX/file-XXX.parquet` 里多条示范拼在一起，按 `episode_index` 分；`meta/tasks.parquet` 存任务描述和 `task_index`；`meta/episodes/chunk-XXX/file-XXX.parquet` 存每条示范在视频文件里的起止时间。`meta/episodes` 的列名（`videos/<相机>/from_timestamp` 等）和 `tasks.parquet` 的存法，文档没写全；读不到时不报错，只是不填 `cameras`，Task 12 用真数据核。

**Files:**
- Create: `src/jianjin/ingest/__init__.py`（先空着）、`src/jianjin/ingest/lerobot.py`、`tests/test_ingest_lerobot.py`
- Modify: `tests/helpers.py`（加 `make_lerobot_v3`）

- [ ] **Step 1：在 `tests/helpers.py` 末尾加造数据的函数**

```python
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SO101_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _matrix_column(m: np.ndarray) -> pa.Array:
    m = np.asarray(m, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel()), m.shape[1])


def make_lerobot_v3(root: Path, episodes: list[tuple[str, np.ndarray]], fps: int = 30) -> Path:
    """在 root 下造一个最小的 LeRobot v3 数据集。episodes 是 (任务描述, 关节状态 T×6)。"""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "names": SO101_NAMES},
            "action": {"dtype": "float32", "shape": [6], "names": SO101_NAMES},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    tasks = sorted({t for t, _ in episodes})
    pq.write_table(
        pa.table({"task_index": list(range(len(tasks))), "task": tasks}),
        root / "meta" / "tasks.parquet",
    )

    cols: dict[str, list] = {k: [] for k in ["episode_index", "frame_index", "timestamp", "task_index", "index"]}
    states, video_from, video_to = [], [], []
    t_video = 0.0
    for ei, (task, state) in enumerate(episodes):
        n = len(state)
        cols["episode_index"] += [ei] * n
        cols["frame_index"] += list(range(n))
        cols["timestamp"] += (np.arange(n) / fps).tolist()
        cols["task_index"] += [tasks.index(task)] * n
        cols["index"] += list(range(len(cols["index"]), len(cols["index"]) + n))
        states.append(state)
        video_from.append(t_video)
        t_video += n / fps
        video_to.append(t_video)
    m = np.concatenate(states)
    table = pa.table(
        {
            **{k: pa.array(v) for k, v in cols.items()},
            "observation.state": _matrix_column(m),
            "action": _matrix_column(m),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    k = "observation.images.front"
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(len(episodes))),
                "tasks": [[t] for t, _ in episodes],
                "length": [len(s) for _, s in episodes],
                f"videos/{k}/chunk_index": [0] * len(episodes),
                f"videos/{k}/file_index": [0] * len(episodes),
                f"videos/{k}/from_timestamp": video_from,
                f"videos/{k}/to_timestamp": video_to,
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    return root
```

- [ ] **Step 2：写失败的测试**

`tests/test_ingest_lerobot.py`：

```python
import json

import numpy as np
import pytest

from helpers import make_lerobot_v3
from jianjin.ingest.lerobot import read_lerobot


def ramp6(n):
    return np.tile(np.linspace(0, 1, n)[:, None], (1, 6))


@pytest.fixture
def ds(tmp_path):
    return make_lerobot_v3(
        tmp_path / "so101_demo",
        [("pick the cube", ramp6(60)), ("place the cube", ramp6(90) * 2)],
    )


def test_reads_episodes(ds):
    res = read_lerobot(ds)
    assert res.failures == []
    assert [e.id for e in res.episodes] == ["so101_demo/ep000000", "so101_demo/ep000001"]
    e0, e1 = res.episodes
    assert e0.task == "pick the cube" and e1.task == "place the cube"
    assert e0.state.shape == (60, 6) and e1.state.shape == (90, 6)
    assert e0.action.shape == (60, 6)
    assert e0.fps == 30.0 and e0.source == "lerobot"
    assert e0.gripper_index == 5
    np.testing.assert_allclose(e1.timestamps[:2], [0, 1 / 30], atol=1e-6)


def test_video_refs(ds):
    e0, e1 = read_lerobot(ds).episodes
    ref = e1.cameras["observation.images.front"]
    assert ref.path == ds / "videos/observation.images.front/chunk-000/file-000.mp4"
    assert np.isclose(ref.from_s, 2.0) and np.isclose(ref.to_s, 5.0)


def test_rejects_old_version(ds):
    info = json.loads((ds / "meta/info.json").read_text(encoding="utf-8"))
    info["codebase_version"] = "v2.1"
    (ds / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="v2.1"):
        read_lerobot(ds)


def test_broken_data_file_is_recorded_not_raised(ds):
    (ds / "data/chunk-000/file-001.parquet").write_bytes(b"not a parquet file")
    res = read_lerobot(ds)
    assert len(res.episodes) == 2
    assert len(res.failures) == 1
    assert "file-001" in res.failures[0][0]
```

- [ ] **Step 3：跑测试，确认失败**

Run：`uv run pytest tests/test_ingest_lerobot.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.ingest'`。

- [ ] **Step 4：写实现**

`src/jianjin/ingest/__init__.py` 先写一个空文件（Task 9 再填）。

`src/jianjin/ingest/lerobot.py`：

```python
"""直接读 LeRobot v3 的文件，不 import lerobot。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from jianjin.types import Episode, LoadResult, VideoRef

STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _names(feature: dict, width: int) -> list[str]:
    names = feature.get("names")
    if isinstance(names, dict):  # 有的数据集写成 {"motors": [...]}
        names = next(iter(names.values()), None)
    if isinstance(names, list) and len(names) == width:
        return [str(n) for n in names]
    return [f"j{i}" for i in range(width)]


def _matrix(col: pa.ChunkedArray) -> np.ndarray:
    arr = col.combine_chunks()
    flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
    return flat.reshape(len(arr), -1)


def _read_tasks(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    t = pq.read_table(path)
    text_cols = [
        f.name
        for f in t.schema
        if f.name != "task_index" and (pa.types.is_string(f.type) or pa.types.is_large_string(f.type))
    ]
    if "task_index" not in t.column_names or not text_cols:
        return {}
    return dict(zip(t["task_index"].to_pylist(), t[text_cols[0]].to_pylist()))


def _read_video_refs(root: Path, info: dict) -> dict[int, dict[str, VideoRef]]:
    keys = [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]
    template = info.get("video_path")
    if not keys or not template:
        return {}
    out: dict[int, dict[str, VideoRef]] = {}
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        names = set(pq.read_schema(f).names)
        wanted = {"episode_index"}
        for k in keys:
            wanted |= {f"videos/{k}/{c}" for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}
        t = pq.read_table(f, columns=sorted(wanted & names))
        if "episode_index" not in t.column_names:
            continue
        rows = t.to_pylist()
        for row in rows:
            cams = {}
            for k in keys:
                cols = [f"videos/{k}/{c}" for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
                if not all(c in row for c in cols):
                    continue
                chunk, file, a, b = (row[c] for c in cols)
                path = root / template.format(video_key=k, chunk_index=chunk, file_index=file)
                cams[k] = VideoRef(path, float(a), float(b) if b is not None else None)
            out[int(row["episode_index"])] = cams
    return out


def read_lerobot(root: Path) -> LoadResult:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(f"只支持 LeRobot v3，这份数据是 {version or '未知版本'}")
    fps = float(info["fps"])
    feats = info.get("features", {})
    tasks = _read_tasks(root / "meta" / "tasks.parquet")
    videos = _read_video_refs(root, info)

    episodes: list[Episode] = []
    failures: list[tuple[str, str]] = []
    for f in sorted((root / "data").rglob("*.parquet")):
        rel = f.relative_to(root).as_posix()
        try:
            t = pq.read_table(f)
            if STATE_KEY not in t.column_names:
                raise KeyError(f"没有 {STATE_KEY} 列")
            ep_idx = t["episode_index"].to_numpy()
            frame_idx = t["frame_index"].to_numpy()
            ts = t["timestamp"].to_numpy().astype(np.float64)
            task_idx = t["task_index"].to_numpy() if "task_index" in t.column_names else None
            state = _matrix(t[STATE_KEY])
            action = _matrix(t[ACTION_KEY]) if ACTION_KEY in t.column_names else None
        except Exception as exc:  # 一个文件坏了不中断整批
            failures.append((rel, f"读不了：{exc}"))
            continue
        names = _names(feats.get(STATE_KEY, {}), state.shape[1])
        gripper = next((i for i, n in enumerate(names) if "gripper" in n.lower()), None)
        for e in np.unique(ep_idx):
            rows = np.flatnonzero(ep_idx == e)
            rows = rows[np.argsort(frame_idx[rows], kind="stable")]
            task = tasks.get(int(task_idx[rows[0]]), "") if task_idx is not None else ""
            episodes.append(
                Episode(
                    id=f"{root.name}/ep{int(e):06d}",
                    source="lerobot",
                    task=task,
                    fps=fps,
                    state=state[rows],
                    action=action[rows] if action is not None else None,
                    timestamps=ts[rows] - ts[rows[0]],
                    state_names=names,
                    gripper_index=gripper,
                    cameras=videos.get(int(e), {}),
                    meta={"dataset": root.name, "episode_index": int(e), "robot_type": info.get("robot_type")},
                )
            )
    episodes.sort(key=lambda ep: ep.id)
    return LoadResult(episodes, failures)
```

- [ ] **Step 5：跑测试，确认通过**

Run：`uv run pytest tests/test_ingest_lerobot.py -q`
Expected：`4 passed`。

- [ ] **Step 6：Commit**

```bash
git add src/jianjin/ingest tests/helpers.py tests/test_ingest_lerobot.py
git commit -m "feat: 读 LeRobot v3"
```

---

### Task 8：读 RoboMIND HDF5

格式依据 RoboMIND 官方仓库的 `static/all_robot_h5_info.md`（2026-10-02 核）：

- 每条示范一个 `.../<示范>/data/trajectory.hdf5`。
- 顶层三组：`master`（主臂，遥操作端）、`puppet`（从臂）、`observations/{rgb_images,depth_images}/<相机>`。
- Franka、UR：`puppet/joint_position` 8 维，最后一维是夹爪。AgileX：`puppet/joint_position_left`、`_right` 各 7 维，夹爪在最后一维。天工：14 或 16 维，手在关节向量里，位置不固定，本周不认夹爪。
- 没有单独的 action，动作取 `master` 的同名 key。
- 成功示范路径：`h5_<机型>/<任务>/success_episodes/<train|val>/<示范>/data/trajectory.hdf5`。
- 失败示范路径：`failure_data/<任务>/<失败原因>/<示范>/data/trajectory.hdf5`，原因是中文目录名，11 类。
- 帧率没在文件说明里写。本周按 30 读，`meta["fps_assumed"] = True`，拿到真数据后核（看有没有时间戳 key，或和视频帧数对）。

图像是逐帧编码的数组，不是 mp4，本周不读，只把相机名记进 `meta["image_keys"]`。

**Files:**
- Create: `src/jianjin/ingest/robomind.py`、`tests/test_ingest_robomind.py`
- Modify: `tests/helpers.py`（加 `make_robomind_h5`）

- [ ] **Step 1：在 `tests/helpers.py` 末尾加造数据的函数**

```python
import h5py


def make_robomind_h5(path: Path, state: np.ndarray | None = None, left_right: np.ndarray | None = None) -> Path:
    """造一个最小的 RoboMIND trajectory.hdf5。state 是单臂 T×8；left_right 是双臂 (左 T×7, 右 T×7) 叠成 2×T×7。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        if state is not None:
            h["puppet/joint_position"] = state
            h["master/joint_position"] = state
        if left_right is not None:
            h["puppet/joint_position_left"] = left_right[0]
            h["puppet/joint_position_right"] = left_right[1]
        h.create_group("observations/rgb_images/camera_top")
    return path
```

- [ ] **Step 2：写失败的测试**

`tests/test_ingest_robomind.py`：

```python
import numpy as np

from helpers import make_robomind_h5
from jianjin.ingest.robomind import read_robomind


def ramp(n, d):
    return np.tile(np.linspace(0, 1, n)[:, None], (1, d))


def test_success_episode(tmp_path):
    root = tmp_path / "RoboMIND"
    make_robomind_h5(
        root / "h5_franka_1rgb/apple_storage/success_episodes/train/ep_0001/data/trajectory.hdf5",
        state=ramp(90, 8),
    )
    res = read_robomind(root)
    assert res.failures == []
    (ep,) = res.episodes
    assert ep.id == "h5_franka_1rgb/apple_storage/success_episodes/train/ep_0001"
    assert ep.task == "apple storage"
    assert ep.source == "robomind"
    assert ep.state.shape == (90, 8) and ep.action.shape == (90, 8)
    assert ep.gripper_index == 7
    assert ep.meta["robot"] == "h5_franka_1rgb"
    assert ep.meta["fps_assumed"] is True
    assert ep.meta["image_keys"] == ["camera_top"]
    assert "robomind_failure_cause" not in ep.meta


def test_failure_episode_keeps_cause(tmp_path):
    root = tmp_path / "RoboMIND"
    make_robomind_h5(root / "failure_data/apple_storage/二次抓取/ep_7/data/trajectory.hdf5", state=ramp(60, 8))
    (ep,) = read_robomind(root).episodes
    assert ep.task == "apple storage"
    assert ep.meta["robomind_failure_cause"] == "二次抓取"


def test_dual_arm_concatenated(tmp_path):
    root = tmp_path / "r"
    lr = np.stack([ramp(60, 7), ramp(60, 7) * 2])
    make_robomind_h5(root / "h5_agilex_3rgb/fold/success_episodes/val/e1/data/trajectory.hdf5", left_right=lr)
    (ep,) = read_robomind(root).episodes
    assert ep.state.shape == (60, 14)
    assert ep.action is None
    assert ep.state_names[6] == "left_gripper" and ep.state_names[13] == "right_gripper"
    assert ep.gripper_index == 6


def test_missing_joints_is_a_failure(tmp_path):
    root = tmp_path / "r"
    make_robomind_h5(root / "h5_ur_1rgb/t/success_episodes/train/e/data/trajectory.hdf5")
    res = read_robomind(root)
    assert res.episodes == []
    assert len(res.failures) == 1 and "joint_position" in res.failures[0][1]


def test_single_file(tmp_path):
    f = make_robomind_h5(tmp_path / "h5_ur_1rgb/t/success_episodes/train/e/data/trajectory.hdf5", state=ramp(30, 8))
    assert len(read_robomind(f).episodes) == 1
```

- [ ] **Step 3：跑测试，确认失败**

Run：`uv run pytest tests/test_ingest_robomind.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.ingest.robomind'`。

- [ ] **Step 4：写实现**

`src/jianjin/ingest/robomind.py`：

```python
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
```

- [ ] **Step 5：跑测试，确认通过**

Run：`uv run pytest tests/test_ingest_robomind.py -q`
Expected：`5 passed`。

- [ ] **Step 6：Commit**

```bash
git add src/jianjin/ingest/robomind.py tests/helpers.py tests/test_ingest_robomind.py
git commit -m "feat: 读 RoboMIND HDF5（成功和失败示范）"
```

---

### Task 9：认格式、`jianjin check`

**Files:**
- Modify: `src/jianjin/ingest/__init__.py`
- Create: `src/jianjin/cli.py`、`tests/test_cli.py`

- [ ] **Step 1：写失败的测试**

`tests/test_cli.py`：

```python
import json

import numpy as np

from helpers import make_lerobot_v3
from jianjin.cli import main
from jianjin.ingest import load


def paused6():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(150), np.linspace(1, 2, 60), np.full(30, 2.0)])
    return np.tile(x[:, None], (1, 6))


def test_load_detects_lerobot(tmp_path):
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    assert len(load(ds).episodes) == 1


def test_load_unknown_format(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    try:
        load(tmp_path)
    except ValueError as exc:
        assert "认不出" in str(exc)
    else:
        raise AssertionError("应该报错")


def test_check_command_writes_json(tmp_path, capsys):
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    out = tmp_path / "findings.json"
    assert main(["check", str(ds), "--json", str(out)]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["episode_id"] == "d/ep000000"
    assert rows[0]["findings"][0]["cause"] == "停顿过长"
    printed = capsys.readouterr().out
    assert "1 条示范" in printed and "停顿过长" in printed
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_cli.py -q`
Expected：FAIL，`ImportError`。

- [ ] **Step 3：写实现**

`src/jianjin/ingest/__init__.py`：

```python
"""认出数据格式，分派给对应的读取器。"""

from __future__ import annotations

from pathlib import Path

from jianjin.ingest.lerobot import read_lerobot
from jianjin.ingest.robomind import read_robomind
from jianjin.types import LoadResult


def load(path: str | Path) -> LoadResult:
    path = Path(path)
    if (path / "meta" / "info.json").exists():
        return read_lerobot(path)
    if path.is_file() and path.suffix in {".hdf5", ".h5"}:
        return read_robomind(path)
    if path.is_dir() and (any(path.rglob("*.hdf5")) or any(path.rglob("*.h5"))):
        return read_robomind(path)
    raise ValueError(f"认不出 {path} 的格式：既不是 LeRobot v3（没有 meta/info.json），也没有 HDF5 文件")


__all__ = ["load"]
```

`src/jianjin/cli.py`：

```python
"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from jianjin.checks import run_checks
from jianjin.ingest import load


def _cmd_check(args: argparse.Namespace) -> int:
    res = load(args.path)
    results = run_checks(res.episodes)
    tasks = {ep.id: ep.task for ep in res.episodes}
    counts: Counter[str] = Counter()
    for r in results:
        for f in r.findings:
            counts[f.cause.value] += 1
    print(f"{len(res.episodes)} 条示范，读取失败 {len(res.failures)} 个文件")
    for cause, n in counts.most_common():
        print(f"  {cause}：{n}")
    for name, why in res.failures:
        print(f"  读取失败 {name}：{why}")
    if args.json:
        rows = [
            {
                "episode_id": r.episode_id,
                "task": tasks[r.episode_id],
                "findings": [{**asdict(f), "cause": f.cause.value} for f in r.findings],
                "not_checked": r.not_checked,
            }
            for r in results
        ]
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细写到 {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jianjin", description="数采场示范数据的自动质检")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="对一个数据集做关节体检")
    c.add_argument("path")
    c.add_argument("--json", help="把每条示范的发现写成 JSON")
    c.set_defaults(func=_cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台管道默认不是 UTF-8
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4：跑全部测试**

Run：`uv run pytest -q`
Expected：全部通过。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/ingest/__init__.py src/jianjin/cli.py tests/test_cli.py
git commit -m "feat: 自动认格式、jianjin check"
```

---

### Task 10：注入缺陷

**Files:**
- Create: `src/jianjin/eval/__init__.py`（空）、`src/jianjin/eval/inject.py`、`tests/test_inject.py`

- [ ] **Step 1：写失败的测试**

`tests/test_inject.py`：

```python
import numpy as np

from helpers import make_episode
from jianjin.checks import CheckConfig
from jianjin.checks.motion import long_pause
from jianjin.eval.inject import insert_pause, truncate_end
from jianjin.types import Cause, VideoRef


def base():
    x = np.linspace(0, 1, 300)  # 10 秒一直在动
    ep = make_episode(np.stack([x, x], axis=1), action=np.stack([x, x], axis=1))
    ep.cameras = {"front": VideoRef("v.mp4", 4.0, 14.0)}
    return ep


def test_truncate_end():
    rng = np.random.default_rng(0)
    ep, d = truncate_end(base(), rng)
    assert d.cause is Cause.INCOMPLETE and d.kind == "truncate_end"
    assert 150 <= len(ep.state) <= 240
    assert len(ep.action) == len(ep.state) == len(ep.timestamps)
    assert np.isclose(d.t1, ep.duration) and np.isclose(d.t0, ep.duration - 1.0)
    assert np.isclose(ep.cameras["front"].to_s, 4.0 + ep.duration)
    assert ep.id.endswith("+truncate_end")


def test_insert_pause_is_detected_where_injected():
    rng = np.random.default_rng(1)
    ep, d = insert_pause(base(), rng)
    assert d.cause is Cause.LONG_PAUSE
    assert 4.0 <= d.t1 - d.t0 <= 8.0
    assert len(ep.action) == len(ep.state)
    np.testing.assert_allclose(np.diff(ep.timestamps), 1 / 30)
    assert ep.cameras == {}  # 视频没法跟着插帧，本周先不带视频
    fs = long_pause(ep, CheckConfig())
    assert len(fs) == 1
    assert abs(fs[0].t0 - d.t0) < 0.1 and abs(fs[0].t1 - d.t1) < 0.1


def test_injection_is_deterministic():
    a = insert_pause(base(), np.random.default_rng(7))[1]
    b = insert_pause(base(), np.random.default_rng(7))[1]
    assert a == b
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_inject.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/eval/inject.py`：

```python
"""考卷 A：在真机录的成功示范上动手脚，每条的答案已知。这是造出来的假失败，不能说成真实失败。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from jianjin.types import Cause, Episode, VideoRef


@dataclass(frozen=True)
class Defect:
    kind: str
    cause: Cause
    t0: float
    t1: float


def _rebuild(ep: Episode, state: np.ndarray, action: np.ndarray | None, kind: str, **changes) -> Episode:
    return replace(
        ep,
        id=f"{ep.id}+{kind}",
        state=state,
        action=action,
        timestamps=np.arange(len(state)) / ep.fps,
        meta={**ep.meta, "injected": kind},
        **changes,
    )


def truncate_end(ep: Episode, rng: np.random.Generator) -> tuple[Episode, Defect]:
    """只留前 50% 到 80%，对应「没做完」。"""
    keep = int(len(ep.state) * rng.uniform(0.5, 0.8))
    action = ep.action[:keep] if ep.action is not None else None
    dur = keep / ep.fps
    cams = {k: VideoRef(v.path, v.from_s, v.from_s + dur) for k, v in ep.cameras.items()}
    new = _rebuild(ep, ep.state[:keep], action, "truncate_end", cameras=cams)
    return new, Defect("truncate_end", Cause.INCOMPLETE, max(0.0, dur - 1.0), dur)


def insert_pause(ep: Episode, rng: np.random.Generator) -> tuple[Episode, Defect]:
    """在 20% 到 80% 之间某一帧，重复这一帧 4 到 8 秒，对应「停顿过长」。"""
    n = len(ep.state)
    k = int(rng.integers(int(0.2 * n), int(0.8 * n)))
    hold = int(round(rng.uniform(4.0, 8.0) * ep.fps))

    def stretch(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a[: k + 1], np.repeat(a[k : k + 1], hold, axis=0), a[k + 1 :]])

    action = stretch(ep.action) if ep.action is not None else None
    new = _rebuild(ep, stretch(ep.state), action, "insert_pause", cameras={})
    return new, Defect("insert_pause", Cause.LONG_PAUSE, k / ep.fps, (k + hold) / ep.fps)


INJECTORS = {
    "truncate_end": truncate_end,
    "insert_pause": insert_pause,
}
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_inject.py -q`
Expected：`3 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/eval tests/test_inject.py
git commit -m "feat: 考卷 A 的缺陷注入（截掉结尾、插入静止帧）"
```

---

### Task 11：指标

**Files:**
- Create: `src/jianjin/eval/metrics.py`、`tests/test_metrics.py`

- [ ] **Step 1：写失败的测试**

`tests/test_metrics.py`：

```python
import math

from jianjin.checks import CheckResult
from jianjin.eval.inject import Defect
from jianjin.eval.metrics import overlaps, score_kind, wilson
from jianjin.types import Cause, Finding


def test_wilson():
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_overlaps():
    assert overlaps((1, 3), (2, 4))
    assert not overlaps((1, 2), (2.5, 4))


def pause(t0, t1):
    return Finding("long_pause", Cause.LONG_PAUSE, t0, t1, t1 - t0, "")


def test_score_kind():
    labels = [Defect("insert_pause", Cause.LONG_PAUSE, 2, 6)] * 3
    injected = [
        CheckResult("a", [pause(2.1, 6.0)]),  # 检出且位置对
        CheckResult("b", [pause(8.0, 12.0)]),  # 检出但位置错
        CheckResult("c", []),  # 漏了
    ]
    clean = [CheckResult("a"), CheckResult("b", [pause(1, 5)]), CheckResult("c"), CheckResult("d")]
    s = score_kind("insert_pause", labels, injected, clean)
    assert (s["n_injected"], s["detected"], s["localized"]) == (3, 2, 1)
    assert (s["n_clean"], s["clean_flagged"]) == (4, 1)
    assert math.isclose(s["detect_rate"], 2 / 3)
    assert math.isclose(s["clean_flag_rate"], 0.25)
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_metrics.py -q`
Expected：FAIL，`ModuleNotFoundError`。

- [ ] **Step 3：写实现**

`src/jianjin/eval/metrics.py`：

```python
"""按示范计的检出率、定位率、原样示范上的报出率，都带分母和 95% Wilson 区间。"""

from __future__ import annotations

import math

from jianjin.checks import CheckResult
from jianjin.eval.inject import Defect


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def score_kind(
    kind: str, labels: list[Defect], injected: list[CheckResult], clean: list[CheckResult]
) -> dict:
    cause = labels[0].cause if labels else None
    detected = localized = 0
    for d, r in zip(labels, injected, strict=True):
        hits = [f for f in r.findings if f.cause == d.cause]
        detected += bool(hits)
        localized += any(overlaps((f.t0, f.t1), (d.t0, d.t1)) for f in hits)
    flagged = sum(any(f.cause == cause for f in r.findings) for r in clean)
    n, m = len(labels), len(clean)
    return {
        "kind": kind,
        "cause": cause.value if cause else None,
        "n_injected": n,
        "detected": detected,
        "localized": localized,
        "n_clean": m,
        "clean_flagged": flagged,
        "detect_rate": detected / n if n else math.nan,
        "detect_ci": wilson(detected, n),
        "localize_rate": localized / n if n else math.nan,
        "localize_ci": wilson(localized, n),
        "clean_flag_rate": flagged / m if m else math.nan,
        "clean_flag_ci": wilson(flagged, m),
    }
```

- [ ] **Step 4：跑测试，确认通过**

Run：`uv run pytest tests/test_metrics.py -q`
Expected：`3 passed`。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/eval/metrics.py tests/test_metrics.py
git commit -m "feat: 考卷指标（带分母和 Wilson 区间）"
```

---

### Task 12：考卷 A runner、下载、命令行

**Files:**
- Create: `src/jianjin/eval/exam_a.py`、`src/jianjin/eval/fetch.py`、`configs/exam_a.toml`、`tests/test_exam_a.py`
- Modify: `src/jianjin/cli.py`

- [ ] **Step 1：写失败的测试**

`tests/test_exam_a.py`：

```python
import json

import numpy as np

from helpers import make_lerobot_v3
from jianjin.checks import CheckConfig
from jianjin.cli import main
from jianjin.eval.exam_a import DatasetSpec, format_table, local_dir, run_exam_a


def moving6(n, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.5, 1.5, n))  # 一直在动、速度有起伏
    return np.tile(x[:, None], (1, 6)) * np.arange(1, 7)


def setup(tmp_path):
    root = tmp_path / "data"
    for repo, task in [("org/dev_ds", "pick"), ("org/test_ds", "place")]:
        make_lerobot_v3(local_dir(root, repo), [(task, moving6(300, i)) for i in range(6)])
    specs = [
        DatasetSpec("org/dev_ds", "dev", "apache-2.0"),
        DatasetSpec("org/test_ds", "test", "mit"),
    ]
    return root, specs


def test_run_exam_a_dev_only_uses_dev(tmp_path):
    root, specs = setup(tmp_path)
    # 阈值写死在测试里，Task 13 重定默认阈值时这条测试不受影响
    cfg = CheckConfig(still_rate=0.02, pause_s=3.0, end_moving_rate=0.05, end_window_s=0.5)
    out = run_exam_a(specs, root, "dev", cfg=cfg, seed=0)
    assert [s["repo"] for s in out["datasets"]] == ["org/dev_ds"]
    assert out["n_episodes"] == 6
    kinds = {k["kind"]: k for k in out["kinds"]}
    assert kinds["insert_pause"]["n_injected"] == 6
    assert kinds["insert_pause"]["localized"] == 6
    assert kinds["truncate_end"]["detected"] == 6
    table = format_table(out)
    assert "insert_pause" in table and "/6" in table and "org/dev_ds" in table


def test_run_exam_a_is_reproducible(tmp_path):
    root, specs = setup(tmp_path)
    a = run_exam_a(specs, root, "dev", seed=3)
    b = run_exam_a(specs, root, "dev", seed=3)
    assert a["kinds"] == b["kinds"]


def test_eval_command(tmp_path, capsys):
    root, _ = setup(tmp_path)
    cfg = tmp_path / "exam_a.toml"
    cfg.write_text(
        '[[dataset]]\nrepo = "org/dev_ds"\nsplit = "dev"\nlicense = "apache-2.0"\n',
        encoding="utf-8",
    )
    out = tmp_path / "res.json"
    assert main(["eval", "a", "--config", str(cfg), "--data-root", str(root), "--split", "dev", "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["split"] == "dev" and saved["seed"] == 0
    assert "注入" in capsys.readouterr().out
```

- [ ] **Step 2：跑测试，确认失败**

Run：`uv run pytest tests/test_exam_a.py -q`
Expected：FAIL，`ModuleNotFoundError: No module named 'jianjin.eval.exam_a'`。

- [ ] **Step 3：写实现**

`src/jianjin/eval/exam_a.py`：

```python
"""考卷 A：注入缺陷。按数据集切开发集和测试集（一个数据集一个任务），不按示范随机切。"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jianjin.checks import CheckConfig, run_checks
from jianjin.eval.inject import INJECTORS
from jianjin.eval.metrics import score_kind
from jianjin.ingest import load

MIN_INJECT_S = 3.0  # 短于 3 秒的示范不注入，数量照实打印


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    split: str
    license: str


def load_config(path: str | Path) -> list[DatasetSpec]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return [DatasetSpec(d["repo"], d["split"], d["license"]) for d in data["dataset"]]


def local_dir(data_root: str | Path, repo: str) -> Path:
    return Path(data_root) / repo.replace("/", "__")


def run_exam_a(
    specs: list[DatasetSpec],
    data_root: str | Path,
    split: str,
    cfg: CheckConfig | None = None,
    seed: int = 0,
) -> dict:
    cfg = cfg or CheckConfig()
    episodes, datasets = [], []
    for s in specs:
        if s.split != split:
            continue
        res = load(local_dir(data_root, s.repo))
        episodes += res.episodes
        datasets.append(
            {
                "repo": s.repo,
                "license": s.license,
                "n_episodes": len(res.episodes),
                "tasks": sorted({e.task for e in res.episodes}),
                "robot_types": sorted({str(e.meta.get("robot_type")) for e in res.episodes}),
                "read_failures": res.failures,
            }
        )
    clean = run_checks(episodes, cfg)
    eligible = [i for i, e in enumerate(episodes) if e.duration >= MIN_INJECT_S]

    kinds = []
    for kind, inject in INJECTORS.items():
        rng = np.random.default_rng(seed)
        pairs = [inject(episodes[i], rng) for i in eligible]
        injected = run_checks([p[0] for p in pairs], cfg)
        kinds.append(score_kind(kind, [p[1] for p in pairs], injected, clean))

    return {
        "exam": "A（注入缺陷：在真机成功示范上人工造的失败，不是真实失败）",
        "split": split,
        "seed": seed,
        "config": asdict(cfg),
        "datasets": datasets,
        "n_episodes": len(episodes),
        "n_eligible": len(eligible),
        "kinds": kinds,
    }


def _pct(k: int, n: int, ci: tuple[float, float]) -> str:
    if n == 0:
        return "—"
    return f"{k}/{n} = {k / n:.0%}（95% 区间 {ci[0]:.0%} 到 {ci[1]:.0%}）"


def format_table(out: dict) -> str:
    lines = [
        f"考卷 {out['exam']}",
        f"切分：{out['split']}；随机种子 {out['seed']}",
        f"示范 {out['n_episodes']} 条，其中 {out['n_eligible']} 条长于 {MIN_INJECT_S:.0f} 秒、用来注入",
    ]
    for d in out["datasets"]:
        lines.append(
            f"  {d['repo']}（{d['license']}）：{d['n_episodes']} 条；机型 {', '.join(d['robot_types'])}；"
            f"任务 {'; '.join(d['tasks'])}；读取失败 {len(d['read_failures'])} 个文件"
        )
    for k in out["kinds"]:
        lines += [
            f"[{k['kind']} → {k['cause']}]",
            f"  检出：{_pct(k['detected'], k['n_injected'], k['detect_ci'])}",
            f"  时间点落在注入区间：{_pct(k['localized'], k['n_injected'], k['localize_ci'])}",
            f"  原样示范上也报了这一类：{_pct(k['clean_flagged'], k['n_clean'], k['clean_flag_ci'])}"
            "（不全是误报：原示范本身可能就有这个问题）",
        ]
    return "\n".join(lines)
```

`src/jianjin/eval/fetch.py`：

```python
"""按考卷配置下载公开数据集，只要 meta 和 data（关节数据），不下视频。"""

from __future__ import annotations

from pathlib import Path

from jianjin.eval.exam_a import DatasetSpec, local_dir


def fetch(specs: list[DatasetSpec], data_root: str | Path, with_videos: bool = False) -> list[Path]:
    from huggingface_hub import snapshot_download  # 可选依赖：uv sync --extra hub

    patterns = ["meta/*", "data/*"] + (["videos/*"] if with_videos else [])
    out = []
    for s in specs:
        target = local_dir(data_root, s.repo)
        snapshot_download(repo_id=s.repo, repo_type="dataset", local_dir=target, allow_patterns=patterns)
        out.append(target)
    return out
```

在 `src/jianjin/cli.py` 的 `_cmd_check` 后面加两个函数：

```python
def _cmd_eval(args: argparse.Namespace) -> int:
    from jianjin.eval.exam_a import format_table, load_config, run_exam_a

    args.out = args.out or f"results/exam_a_{args.split}.json"
    out = run_exam_a(load_config(args.config), args.data_root, args.split, seed=args.seed)
    print(format_table(out))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写到 {args.out}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from jianjin.eval.exam_a import load_config
    from jianjin.eval.fetch import fetch

    for p in fetch(load_config(args.config), args.data_root, with_videos=args.videos):
        print(f"已下载到 {p}")
    return 0
```

在 `build_parser()` 的 `return p` 前面加：

```python
    e = sub.add_parser("eval", help="跑考卷")
    e.add_argument("exam", choices=["a"])
    e.add_argument("--config", default="configs/exam_a.toml")
    e.add_argument("--data-root", default="data")
    e.add_argument("--split", choices=["dev", "test"], default="dev")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_eval)

    f = sub.add_parser("fetch", help="按考卷配置下载公开数据集")
    f.add_argument("--config", default="configs/exam_a.toml")
    f.add_argument("--data-root", default="data")
    f.add_argument("--videos", action="store_true", help="连视频一起下")
    f.set_defaults(func=_cmd_fetch)
```

`configs/exam_a.toml`（2026-10-02 核过 `meta/info.json`：都是 v3.0、30 fps、有夹爪通道，数据卡 YAML 写 Apache-2.0）：

```toml
# 考卷 A 用的公开数据集。一个数据集只属于一个切分；授权照抄数据卡。
# 开发集定阈值；测试集换了任务（分拣、乐高块）、换了机型（SO-101、双臂），只跑一次。
[[dataset]]
repo = "lerobot/svla_so100_pickplace"   # 50 条，把方块放进盒子
split = "dev"
license = "apache-2.0"

[[dataset]]
repo = "lerobot/svla_so100_stacking"    # 56 条，红块叠到蓝块上
split = "dev"
license = "apache-2.0"

[[dataset]]
repo = "lerobot/svla_so100_sorting"     # 52 条，红蓝块分进左右两个盒子
split = "test"
license = "apache-2.0"

[[dataset]]
repo = "lerobot/svla_so101_pickplace"   # 50 条，乐高块放进透明盒；robot_type 写的是 so100_follower
split = "test"
license = "apache-2.0"

[[dataset]]
repo = "CoRL2026-CSI/SO101-DualArm-teleop_fold_towel_100epi"  # 100 条，双臂叠毛巾
split = "test"
license = "apache-2.0"
```

- [ ] **Step 4：跑全部测试**

Run：`uv run pytest -q`
Expected：全部通过。

- [ ] **Step 5：Commit**

```bash
git add src/jianjin/eval src/jianjin/cli.py configs tests/test_exam_a.py
git commit -m "feat: 考卷 A runner、jianjin eval a、jianjin fetch"
```

---

### Task 13：真数据：下载、核读入、定阈值、出第一批数字

这一项要联网下载公开数据（只下 meta 和 data，每个数据集几 MB 到几十 MB）。

**Files:**
- Modify: `configs/exam_a.toml`、`src/jianjin/checks/config.py`（默认阈值）、`src/jianjin/ingest/lerobot.py`（如果真数据和假设不符）
- Create: `scripts/tune_exam_a.py`、`results/exam_a_dev.json`、`results/exam_a_test.json`

- [ ] **Step 1：核数据卡正文的授权**

`configs/exam_a.toml` 里的授权取自数据卡 YAML，正文还没逐个看。打开五个数据卡，正文和 YAML 不一致的，按更严的那个改，或者从清单里拿掉。

- [ ] **Step 2：下载并核读入**

```bash
uv run jianjin fetch
uv run jianjin check data/<第一个数据集目录> --json data/check_first.json
```

逐条核对，和假设不符就改 `src/jianjin/ingest/lerobot.py` 并在 `tests/helpers.py` 的 `make_lerobot_v3` 里补上真实写法，再加一条测试：

- 示范条数等于数据卡上的 episode 数。
- `task` 不是空串（不是的话看 `meta/tasks.parquet` 的真实列名）。
- `gripper_index` 认出来了。
- `cameras` 非空（是空的话打印 `pq.read_schema("data/.../meta/episodes/chunk-000/file-000.parquet").names` 看真实列名）。

把核出的真实列名写进设计文档 4.1 节末尾一句话。

- [ ] **Step 3：写扫阈值的脚本**

`scripts/tune_exam_a.py`：

```python
"""在考卷 A 开发集上扫关节体检的阈值。原则：原样示范上的报出率不超过 5%，在这个前提下检出率最高。"""

from __future__ import annotations

import itertools
from dataclasses import replace

from jianjin.checks import CheckConfig
from jianjin.eval.exam_a import load_config, run_exam_a

GRID = {
    "insert_pause": {"still_rate": [0.005, 0.01, 0.02, 0.04], "pause_s": [2.0, 3.0, 4.0]},
    "truncate_end": {"end_moving_rate": [0.02, 0.05, 0.1, 0.2], "end_window_s": [0.3, 0.5, 1.0]},
}
MAX_CLEAN_FLAG = 0.05


def main() -> None:
    specs = load_config("configs/exam_a.toml")
    for kind, grid in GRID.items():
        print(f"== {kind}")
        best = None
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            out = run_exam_a(specs, "data", "dev", cfg=replace(CheckConfig(), **params))
            k = next(x for x in out["kinds"] if x["kind"] == kind)
            ok = k["clean_flag_rate"] <= MAX_CLEAN_FLAG
            print(
                f"{params}  检出 {k['detected']}/{k['n_injected']}  定位 {k['localized']}/{k['n_injected']}"
                f"  原样报出 {k['clean_flagged']}/{k['n_clean']}{'' if ok else '  超限'}"
            )
            if ok and (best is None or k["localized"] > best[1]):
                best = (params, k["localized"])
        print(f"选：{best}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4：在开发集上扫**

Run：`uv run python scripts/tune_exam_a.py`

把选中的值写进 `src/jianjin/checks/config.py` 的默认值；如果某一类没有任何组合满足 5% 的上限，选原样报出率最低的那组，并在 Step 6 的结论里写明。跑 `uv run pytest -q`，单元测试用的合成数据如果因为阈值变了而失败，调测试数据让它仍然测同一件事，不要为了过测试改阈值。

- [ ] **Step 5：开发集出数，测试集跑一次**

```bash
uv run jianjin eval a --split dev
uv run jianjin eval a --split test
```

测试集只跑这一次。之后不再按测试集的结果改阈值；要改，就重新扫开发集，并在结论里写明测试集已经看过一次。

- [ ] **Step 6：写回文档、Commit**

在设计文档第 6 节「考卷 A」下面加「第一批结果（日期）」小节：两种缺陷各自的检出、定位、原样报出，每个数字带分母和区间，写明数据集、机型、任务，以及「这是在真数据上造出来的假失败」。

```bash
git add configs scripts src results docs
git commit -m "feat: 考卷 A 第一批数字（开发集定阈值、测试集跑一次）"
```

---

## 本计划之后

- 第二份计划（10-10 至 10-16）：视频解码、Robometer 和 TOPReward 适配器、「换掉任务描述」注入、考卷 B。
- 第三份计划（10-17 至 10-23）：`diagnose`（`Verdict`、两个独立信号才作废）、夹爪类规则、离线网页报告、考卷 C。
- 第四份计划（10-24 至 10-30）：Cosmos 3 适配器上 Nebius。
