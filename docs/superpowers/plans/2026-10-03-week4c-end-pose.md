# 第四周 c：「没做完」按同任务收尾的常见出入换算 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让「没做完」（收尾姿态）规则在双臂和收尾不固定的数据上不再乱报：收尾本来就不固定的批次记「查不了」，其余的每个关节按同任务收尾位置的四分位距换算。

**Architecture:** `end_pose` 的统计量从「离中位收尾姿态几个活动幅度」改成「几倍的收尾四分位距（下限是活动幅度的 floor 倍）」；同任务收尾位置普遍散开超过活动幅度的 15% 的批次，这条规则记进 not_checked。考卷 A 开发集加两个双臂数据集（`armnet` 排除叠毛巾那个任务），在开发集上按「每个数据集原样报出都不超过 5%、截断定位对的最多」重选阈值；测试集看第二次。

**Tech Stack:** Python 3.12、numpy、pytest。

---

## 背景（2026-10-03）

下了两个双臂 SO-101 数据集的关节数据（Apache-2.0，不含视频）：`jun-seo/bimanual_tilt`（倒水，60 条）、`armnet/armnetbench_v01_lerobot_bimanual_so101`（4 个任务 1,219 条，其中叠毛巾 350 条不用，和测试集同任务）。

现在的规则（离中位收尾姿态超过 0.3 个活动幅度就报）在开发集好示范上：

| 数据集 | 原样报出 | 截断检出 |
|---|---|---|
| SO-100 放方块 | 2/50 | 48/50 |
| SO-100 叠方块 | 0/56 | 51/53 |
| 双臂倒水 | 25/60 | 60/60 |
| armnet（3 个任务） | 860/869 | 859/869 |

每批收尾姿态固不固定（各关节收尾位置的四分位距除以活动幅度，取中位）：放方块 0.010、叠方块 0.012、倒水 0.062，armnet 三个任务 0.251、0.359、0.599。armnet 的示范停下来的姿态本来就五花八门，收尾姿态判不了没做完；门槛取 0.15（落在 0.062 和 0.251 之间）。

试算「按收尾四分位距换算」（下限 floor 倍活动幅度，阈值 thr）：floor 0.1、thr 3 时放方块 1/50、叠方块 0/56、倒水 3/60，截断检出 48/50、51/53、60/60；floor 0.05、thr 6 时一样。正式在 Task 3 用扫描脚本选。

测试集上「没做完」原样报出 11/202（全在叠毛巾，全是手腕触发）；改完在测试集上看第二次，要申明。

## 文件

- 改 `src/jianjin/eval/exam_a.py`：`DatasetSpec` 加 `exclude_tasks`；读配置、跑考卷时排除。
- 改 `configs/exam_a.toml`：开发集加两个双臂数据集（不写相机，不进换描述考卷）。
- 改 `src/jianjin/checks/batch.py`：`_compare` 允许 `summarize` 说「这批查不了」；`end_pose` 换统计量。
- 改 `src/jianjin/checks/config.py`：`end_pose_max` 改单位，加 `end_pose_floor`、`end_spread_max`。
- 改 `scripts/tune_exam_a.py`：可以只扫一类；按数据集分别要求原样报出不超过 5%。
- 测试：`tests/test_exam_a.py`、`tests/test_batch.py`。
- 结果：`results/exam_a_dev.json`、`results/exam_a_test.json`。
- 文档：设计文档考卷 A 一节、「整条判定」一节；`.hackertrip/STATUS.md`。

---

### Task 1: 考卷 A 能排除某个任务；开发集加双臂数据

**Files:**
- Modify: `src/jianjin/eval/exam_a.py`、`configs/exam_a.toml`
- Test: `tests/test_exam_a.py`

- [ ] **Step 1: 写失败的测试**

加到 `tests/test_exam_a.py` 末尾：

```python
def test_excluded_tasks_left_out(tmp_path):
    root = tmp_path / "data"
    make_lerobot_v3(local_dir(root, "org/mix"), [("pick", moving6(300, i)) for i in range(6)]
                    + [("fold", moving6(300, 10 + i)) for i in range(6)])
    spec = DatasetSpec("org/mix", "dev", "apache-2.0", exclude_tasks=("fold",))
    out = run_exam_a([spec], root, "dev", seed=0)
    assert out["n_episodes"] == 6 and out["datasets"][0]["tasks"] == ["pick"]
    assert out["datasets"][0]["excluded_tasks"] == ["fold"]


def test_config_reads_excluded_tasks(tmp_path):
    from jianjin.eval.exam_a import load_config

    cfg = tmp_path / "a.toml"
    cfg.write_text('[[dataset]]\nrepo = "o/m"\nsplit = "dev"\nlicense = "mit"\nexclude_tasks = ["fold"]\n',
                   encoding="utf-8")
    (spec,) = load_config(cfg)
    assert spec.exclude_tasks == ("fold",) and spec.camera is None
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest tests/test_exam_a.py -q`
Expected: FAIL，`TypeError: ... unexpected keyword argument 'exclude_tasks'`

- [ ] **Step 3: 实现**

`DatasetSpec` 加字段：

```python
    exclude_tasks: tuple[str, ...] = ()  # 这些任务不用（比如和测试集同任务的）
```

`load_config`：

```python
    return [DatasetSpec(d["repo"], d["split"], d["license"], d.get("camera"), tuple(d.get("exclude_tasks", ())))
            for d in data["dataset"]]
```

`run_exam_a` 里读数据集那段：

```python
        res = load(local_dir(data_root, s.repo))
        kept = [e for e in res.episodes if e.task not in s.exclude_tasks]
        episodes += kept
        datasets.append(
            {
                "repo": s.repo,
                "license": s.license,
                "n_episodes": len(kept),
                "tasks": sorted({e.task for e in kept}),
                "excluded_tasks": sorted(s.exclude_tasks),
                "robot_types": sorted({str(e.meta.get("robot_type")) for e in kept}),
                "read_failures": res.failures,
            }
        )
```

`configs/exam_a.toml`，在两个开发集后面加（不写 camera：本机没有它们的视频，也不进换描述考卷）：

```toml
[[dataset]]
repo = "jun-seo/bimanual_tilt"           # 60 条，双臂 SO-101 倒水；2026-10-03 加，修「没做完」用的双臂开发集
split = "dev"
license = "apache-2.0"

[[dataset]]
repo = "armnet/armnetbench_v01_lerobot_bimanual_so101"  # 双臂 SO-101，4 个任务 1,219 条；叠毛巾和测试集同任务，不用
split = "dev"
license = "apache-2.0"
exclude_tasks = ["Fold the brown tea towel"]
```

文件头的说明加一句：「双臂开发集只有关节数据（2026-10-03 加）。」

- [ ] **Step 4: 跑测试**

Run: `uv run pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/jianjin/eval/exam_a.py configs/exam_a.toml tests/test_exam_a.py
git commit -m "feat: 考卷 A 可以排除某个任务；开发集加两个双臂数据集"
```

---

### Task 2: 「没做完」按收尾的常见出入换算，收尾不固定的批次不查

**Files:**
- Modify: `src/jianjin/checks/batch.py`、`src/jianjin/checks/config.py`
- Test: `tests/test_batch.py`、`tests/test_exam_a.py`

- [ ] **Step 1: 写失败的测试**

加到 `tests/test_batch.py` 末尾：

```python
# 收尾规则的测试把参数写死，只验机制；默认值在 Task 3 重选
END = CheckConfig(end_pose_max=3.0, end_pose_floor=0.1, end_spread_max=0.15)


def ends_at(final_wrist, n=60):
    """六个关节：前五个从 0 走到 1；最后一个（像手腕转角）收尾停在 final_wrist。"""
    arm = np.tile(np.linspace(0, 1, n)[:, None], (1, 5))
    return np.concatenate([arm, np.linspace(0, final_wrist, n)[:, None]], axis=1)


def test_joint_that_never_ends_the_same_is_not_flagged():
    # 最后一个关节的收尾位置在好示范之间就差很多，同样的差法不该被当成没做完
    eps = [make_episode(ends_at(w), ep_id=f"ok{i}") for i, w in enumerate([0.2, 0.5, 0.8, 0.35, 0.65])]
    eps.append(make_episode(ends_at(0.95), ep_id="odd_wrist"))
    eps.append(make_episode(ends_at(0.5)[:30], ep_id="half"))
    found, skipped = end_pose(eps, END)
    assert skipped == {} and set(found) == {"half"}
    assert "倍" in found["half"][0].evidence


def test_batch_without_a_common_end_pose_is_not_checked():
    rng = np.random.default_rng(0)
    eps = [make_episode(np.stack([np.linspace(0, e, 60) for e in rng.uniform(0, 1, 3)], axis=1), ep_id=f"r{i}")
           for i in range(8)]
    found, skipped = end_pose(eps, END)
    assert found == {} and len(skipped) == 8
    assert "停下来的姿态本来就不固定" in skipped["r0"]
```

已有的收尾测试也改用 `END`：`test_tasks_compared_separately`、`test_end_pose_flags_stopping_halfway`、`test_reference_used_instead_of_batch` 里调 `end_pose(..., CFG)` 的地方换成 `END`（`test_small_task_not_checked` 不用改）。

`tests/test_exam_a.py` 的 `test_run_exam_a_dev_only_uses_dev` 里写死的 `end_pose_max=0.15` 改成 `end_pose_max=3.0, end_pose_floor=0.02`（新单位是收尾四分位距的倍数；这组造的数据收尾很齐，下限取小一点，只留 80% 的截断也要检出）。

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest tests/test_batch.py -q`
Expected: FAIL：`test_joint_that_never_ends_the_same_is_not_flagged` 把 `odd_wrist` 和好示范也报了；`test_batch_without_a_common_end_pose_is_not_checked` 的 skipped 是空的

- [ ] **Step 3: 实现**

`src/jianjin/checks/config.py`，`end_pose_max` 那行换成三行（数值在 Task 3 重选，先用试算时的 floor 0.1、thr 3）：

```python
    end_pose_max: float = 3.0  # 收尾姿态离同任务中位收尾姿态，超过收尾常见出入（四分位距）的这么多倍，算没做完
    end_pose_floor: float = 0.1  # 收尾常见出入至少按关节活动幅度的这么多算，免得一批太齐时一点小差别被放大
    end_spread_max: float = 0.15  # 同任务收尾位置普遍散开超过活动幅度的这么多，收尾本来就不固定，不查
```

`src/jianjin/checks/batch.py`：导入加 `NotApplicable`（`from jianjin.checks.config import CheckConfig, NotApplicable`）。`_compare` 里取统计量那段改成允许整批查不了：

```python
        if _key(ep) not in stats:
            try:
                stats[_key(ep)] = summarize(group)
            except NotApplicable as exc:  # 这一批本身不满足规则的前提
                stats[_key(ep)] = exc
        stat = stats[_key(ep)]
        if isinstance(stat, NotApplicable):
            skipped[ep.id] = str(stat)
            continue
        finding = judge(ep, stat, len(group))
```

`end_pose` 整个换成：

```python
def end_pose(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[Found, Skipped]:
    """最后一帧的姿态离同任务通常的收尾姿态太远，像是没做完就停了。

    每个关节按同任务那批收尾位置的四分位距换算（下限是活动幅度的 end_pose_floor 倍）：收尾本来就不固定的关节
    （比如停稳以后的手腕转角）差一点不算。整批收尾都不固定的（各关节收尾位置的四分位距按活动幅度算、取中位，
    超过 end_spread_max），这条规则查不了，记进没查。2026-10-03 改，见设计文档考卷 A「第三批结果」。
    """

    def summarize(group):
        scale = joint_scale(np.concatenate([e.state for e in group]))
        finals = np.array([e.state[-1] for e in group], dtype=np.float64)
        q75, q25 = np.nanpercentile(finals, [75, 25], axis=0)
        iqr = q75 - q25
        moving = np.isfinite(scale)
        spread = float(np.median(iqr[moving] / scale[moving])) if moving.any() else 0.0
        if spread > cfg.end_spread_max:
            raise NotApplicable(
                f"同任务 {len(group)} 条示范停下来的姿态本来就不固定（各关节收尾位置一般差开活动幅度的 {spread:.0%}），"
                "没法拿收尾姿态判没做完"
            )
        return np.nanmedian(finals, axis=0), np.maximum(iqr, cfg.end_pose_floor * scale)

    def judge(ep, stat, n):
        med, unit = stat
        off = np.nan_to_num(np.abs(ep.state[-1] - med) / unit)
        j = int(np.argmax(off))
        if off[j] <= cfg.end_pose_max:
            return None
        t1 = ep.duration
        return Finding(
            check="end_pose",
            cause=Cause.INCOMPLETE,
            t0=max(0.0, t1 - 1.0),
            t1=t1,
            score=float(off[j]),
            evidence=(
                f"最后停的姿态和同任务 {n} 条示范的常见收尾姿态差得远：{ep.state_names[j]} 差出的距离"
                f"是示范之间常见出入的 {off[j]:.1f} 倍，像是没做完就停了"
            ),
        )

    return _compare(episodes, cfg, reference, summarize, judge)
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/jianjin/checks tests/test_batch.py tests/test_exam_a.py
git commit -m "feat: 「没做完」按同任务收尾的常见出入换算；收尾本来不固定的批次不查"
```

---

### Task 3: 开发集上重选阈值，测试集看第二次

**Files:**
- Modify: `scripts/tune_exam_a.py`、`src/jianjin/checks/config.py`
- Results: `results/exam_a_dev.json`、`results/exam_a_test.json`

- [ ] **Step 1: 扫描脚本按数据集分别卡 5%，可以只扫一类**

`scripts/tune_exam_a.py`：

- 文件说明改成「原则：每个开发数据集的原样报出率都不超过 5%；在这个前提下定位对的最多；再打平就选原样报出最少的。」（数据集大小差很多，armnet 一家 869 条，合起来算会被它一家主导。）
- `GRID["truncate_end"]` 换成 `{"end_pose_max": [2.0, 3.0, 4.0, 6.0, 8.0], "end_pose_floor": [0.02, 0.05, 0.1]}`。
- `main()` 接受命令行参数：`python scripts/tune_exam_a.py truncate_end` 只扫这一类；不给就全扫。
- 每组参数对每个开发数据集单独跑 `run_exam_a([spec], ...)`（同任务比较本来就不跨数据集），各自算原样报出率，全部不超过 5% 才算合格；定位、原样报出按数据集加总。打印每个数据集的数。

- [ ] **Step 2: 只扫「截掉结尾」**

Run: `uv run python scripts/tune_exam_a.py truncate_end`
Expected：打印每组参数下四个开发数据集的原样报出、截断检出；armnet 三个任务都记「查不了」，原样报出和检出都是 0。按原则选出一组，写进 `CheckConfig` 默认值（`end_pose_max`、`end_pose_floor`），注释写选法和出处。`uv run pytest -q` 全过。

- [ ] **Step 3: 开发集、测试集各跑一次**

Run: `uv run jianjin eval a --split dev` 然后 `uv run jianjin eval a --split test`
Expected：结果写到 `results/exam_a_{dev,test}.json`。测试集是「没做完」第二次看，照实记：原来截断检出 198/201、原样报出 11/202。「停顿过长」的规则没改，两边的数应该不变，变了就查。

- [ ] **Step 4: Commit**

```bash
git add scripts/tune_exam_a.py src/jianjin/checks/config.py results/exam_a_dev.json results/exam_a_test.json
git commit -m "feat: 「没做完」在加了双臂的开发集上重选阈值；测试集第二次看"
```

---

### Task 4: 重跑报告，写进文档

- [ ] **Step 1:** 重跑六份报告（命令同第四周 a 计划 Task 5 Step 1），记下叠毛巾「没做完」从 11 条变成几条；放方块的 2 条有没有变。
- [ ] **Step 2:** 设计文档考卷 A 一节加「第三批结果：「没做完」换算方法（2026-10-03，加了双臂开发集）」：背景表、门槛、扫描结果、选中的参数、开发集和测试集成绩（测试集申明第二次看）；第一批结果下面「要修得另找一个双臂开发集」那句后面注明已在第三批修。「整条判定」一节的表格和叠毛巾那几条按重跑结果改。
- [ ] **Step 3:** STATUS 头部 `next` 去掉手腕误报那项；日志加一行。

```bash
git add docs/superpowers/specs/2026-10-02-jianjin-design.md
git commit -m "docs: 「没做完」换算方法改了以后的成绩"
```
