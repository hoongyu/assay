# TOPReward 定阈值 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 TOPReward（Qwen3-VL-4B）接成第二个视频来源：在 dev、train 上定「指令不符」「进度回落」的阈值，test 各跑一次，用两个视频模型重出六份报告。之后「两个视频模型都报」这条作废的路才通。

**Architecture:** 只动阈值表、选分规则、适配器的提示格式开关和命令行的默认输出路径；判定逻辑（diagnose）不动。

**Tech Stack:** Python 3.12、uv、pytest；缓存来自 Colab（`notebooks/colab_run_topreward.ipynb`，2026-10-04 跑）。

---

## 已经看过的（只看了 dev 和 train）

Colab 跑了两种提示格式，各 1,604 条（考卷 A 608、RoboVAD 996），一条没漏：

- lerobot 默认格式：「The answer is: True」接在用户那段话后面，版本号 `TOPReward/Qwen3-VL-4B-Instruct/lerobot-0.6.1/fp16/f32a15`。
- 对话格式（lerobot 的 `add_chat_template=True`，模型以助手身份回答）：版本号末尾加 `/chat`。

| 看什么 | 默认格式 | 对话格式 |
|---|---|---|
| dev 结尾「True」概率中位数（原描述 / 换后描述） | 0.0004 / 0.0002 | 0.83 / 0.05 |
| dev 原描述比换后描述高的对 | 49% | 77% |
| 考卷 A 换描述 dev，rel_success，原样报出不超过 5% 时的检出 | 45/103（AUC 0.568） | 49/103（AUC 0.940） |
| RoboVAD train 按示范计，进度回落，正常报出 5% 时的检出 | 2/60（AUC 0.514） | 14/60（AUC 0.598） |

- 默认格式下，叠方块的视频配上「放进盒子」的描述，「True」反而比配自己的描述高（中位数 -7.25 对 -8.14）：比的是描述本身的先验，不是视频。
- 对话格式在 dev 上绝对的 success 检出 57/103，比 rel_success 的 49 多；但「指令不符」规则（`checks/video.py` 的 `wrong_task`）只和同任务那批比，绝对分数用不上。选分程序原来在所有分数里挑，这次会挑一个规则用不了的分数。Robometer 选中的本来就是 rel_peak，改了也不变。

## 冻结的选择（看 test 之前）

1. TOPReward 用对话格式，`TOPRewardJudge(chat=True)` 是默认。
2. 换描述考卷只在 `rel_` 开头的分数里选（规则能用的），不带 `rel_` 的只列出来参考。
3. 「指令不符」：对话格式 dev 上选中的分数和阈值（预计 rel_success）。
4. 「进度回落」：对话格式 RoboVAD train 正常示范的 95 分位（预计 0.707）。
5. test 各跑一次；默认格式不跑 test。

---

### Task 1: 换描述考卷只在规则能用的分数里选

**Files:**
- Modify: `src/jianjin/eval/task_swap.py`（`run_swap` 里的选分、`format_swap`、模块说明）
- Test: `tests/test_task_swap.py`

- [ ] **Step 1: 写失败的测试**

```python
def test_choose_only_scores_the_rule_can_use():
    """「指令不符」规则只和同任务那批比：绝对分数检出再多也不选。"""
    from jianjin.eval.task_swap import _choose

    rows = [{"name": "success", "detected_at_max": 57, "auc": 0.884},
            {"name": "rel_success", "detected_at_max": 49, "auc": 0.940},
            {"name": "rel_end", "detected_at_max": 43, "auc": 0.703}]
    assert _choose(rows) == "rel_success"
    assert _choose([{"name": "end", "detected_at_max": 9, "auc": 0.9}]) == "rel_end"  # 没有能用的：规则查不了
```

- [ ] **Step 2: 跑测试，确认失败**（`_choose` 不存在）

Run: `uv run pytest tests/test_task_swap.py -q`

- [ ] **Step 3: 实现**

```python
def _choose(rows: list[dict]) -> str:
    """只在 rel_ 开头的分数里选：「指令不符」规则（checks/video.py 的 wrong_task）只和同任务那批比，
    不带 rel_ 的分数规则用不上，只列出来参考（2026-10-04 改）。先比原样报出不超过 5% 时的检出，再比 AUC。"""
    usable = [r for r in rows if r["name"].startswith("rel_")]
    if not usable:
        return "rel_end"

    def rank(r: dict) -> tuple:
        return (r["detected_at_max"], -1.0 if math.isnan(r["auc"]) else r["auc"])

    return max(usable, key=rank)["name"]
```

`run_swap` 里 `score = max(rows, key=rank)["name"] if rows else "end"` 换成 `score = _choose(rows)`；`format_swap` 里「各分数」那行加「（只在 rel_ 开头的里选：「指令不符」规则只和同任务那批比）」。

- [ ] **Step 4: 全部测试通过**，Robometer 的换描述成绩原样复现（dev 57/103、test 184/201）

Run: `uv run pytest -q`；`uv run jianjin eval a-swap --split dev --out <临时文件>` 对比 `results/exam_a_swap_dev.json`

- [ ] **Step 5: Commit**

### Task 2: TOPReward 的提示格式开关，默认对话格式

**Files:**
- Modify: `src/jianjin/judges/topreward.py`
- Test: `tests/test_topreward.py`

- [ ] **Step 1: 写失败的测试**

```python
def test_chat_format_is_default_and_in_version():
    """版本号要和 Colab 上跑出来的缓存对得上（2026-10-04）。"""
    assert TOPRewardJudge().version == "TOPReward/Qwen3-VL-4B-Instruct/lerobot-0.6.1/fp16/f32a15/chat"
    assert TOPRewardJudge(chat=False).version == "TOPReward/Qwen3-VL-4B-Instruct/lerobot-0.6.1/fp16/f32a15"
```

- [ ] **Step 2: 确认失败**；**Step 3: 实现**：`__init__` 加 `chat: bool = True`，版本号末尾按它加 `/chat`；`_load` 里 `TOPRewardEncoderProcessorStep(..., add_chat_template=self.chat)`；模块说明写清两种格式和选择依据。

- [ ] **Step 4: 测试通过；Step 5: Commit**

### Task 3: 不是 Robometer 的考卷成绩，默认文件名带评判名

**Files:**
- Modify: `src/jianjin/cli.py`（`_cmd_eval`）
- Test: `tests/test_task_swap.py::test_judge_then_eval_commands`

- [ ] **Step 1: 测试末尾加**

```python
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "a-swap", "--judge", "fake", "--split", "dev", *common]) == 0
    assert (tmp_path / "results" / "exam_a_swap_dev_fake.json").exists()  # 带评判名，免得盖掉 Robometer 的
```

- [ ] **Step 2: 确认失败；Step 3: 实现**

```python
    suffix = f"_{args.judge}" if args.exam in ("a-swap", "b-episode") and args.judge != "robometer" else ""
```

默认路径 `results/exam_{exam}_{split}{suffix}.json`；两处「test 要用 dev（train）定的……加 --dev-result」的提示里的文件名也带上 `suffix`。

- [ ] **Step 4: 测试通过；Step 5: Commit**

### Task 4: 缓存归位，在 dev、train 上定阈值

- [ ] **Step 1:** `cache/topreward`（默认格式）挪到 `cache/lerobot_default/topreward`，`cache/chat/topreward`（对话格式）挪到 `cache/topreward`。
- [ ] **Step 2:** 对话格式：

```bash
uv run jianjin eval a-swap --judge topreward --split dev
uv run jianjin eval b-episode --judge topreward --split train
```

- [ ] **Step 3:** 默认格式留作对照（只 dev、train）：

```bash
uv run jianjin eval a-swap --judge topreward --split dev --cache cache/lerobot_default --out results/exam_a_swap_dev_topreward_lerobot_default.json
uv run jianjin eval b-episode --judge topreward --split train --cache cache/lerobot_default --out results/exam_b_episode_train_topreward_lerobot_default.json
```

- [ ] **Step 4: Commit** 四个结果文件。

### Task 5: 阈值填进表

**Files:**
- Modify: `src/jianjin/checks/video.py`（`WRONG_TASK`、`PROGRESS_DROP`）
- Modify: `tests/test_video_checks.py`、`tests/test_diagnose.py`（原来靠「TOPReward 还没定阈值」测没查的路径，改用一个没定阈值的评判名 `newjudge`；diagnose 那条改成核「缓存里没有 TOPReward」）

- [ ] **Step 1:** 先改测试，跑一遍确认现在还能过（只换了评判名）。
- [ ] **Step 2:** 填表，出处写在旁边：`"topreward": ("success", <dev 阈值>)`、`"topreward": <train 阈值>`。
- [ ] **Step 3:** 全部测试通过；**Commit**。

### Task 6: test 各跑一次

```bash
uv run jianjin eval a-swap --judge topreward --split test --dev-result results/exam_a_swap_dev_topreward.json
uv run jianjin eval b-episode --judge topreward --split test --dev-result results/exam_b_episode_train_topreward.json
```

- [ ] 结果原样记下，不管好坏；**Commit**。

### Task 7: 两个视频模型重出六份报告

```bash
uv run jianjin check data/lerobot__svla_so100_pickplace --judges robometer,topreward --camera observation.images.top --out reports/svla_so100_pickplace.html --json reports/svla_so100_pickplace.json
uv run jianjin check data/lerobot__svla_so100_stacking --judges robometer,topreward --camera observation.images.top --out reports/svla_so100_stacking.html --json reports/svla_so100_stacking.json
uv run jianjin check data/lerobot__svla_so100_sorting --judges robometer,topreward --camera observation.images.top --out reports/svla_so100_sorting.html --json reports/svla_so100_sorting.json
uv run jianjin check data/lerobot__svla_so101_pickplace --judges robometer,topreward --camera observation.images.up --out reports/svla_so101_pickplace.html --json reports/svla_so101_pickplace.json
uv run jianjin check data/CoRL2026-CSI__SO101-DualArm-teleop_fold_towel_100epi --judges robometer,topreward --camera observation.images.left_top --out reports/fold_towel.html --json reports/fold_towel.json
uv run jianjin check data/robovad --judges robometer,topreward --camera left --out reports/robovad.html --json reports/robovad.json
```

- [ ] 按数据集数合格、存疑、作废和原因；对照已经看过视频的那些示范（叠方块 10 条、叠毛巾 ep 10、61、62、76、83），看 TOPReward 多报、少报了哪些，新出的作废是不是两个模型说的同一处。

### Task 8: 文档

- [ ] 笔记本：第 5 格起默认就是对话格式；第三部分改成说明（对照实验做过了，要复现默认格式用 `TOPRewardJudge(chat=False)`、换一个缓存目录）。
- [ ] 设计文档：TOPReward 一节（两种格式的对照、冻结的选择、dev/train/test 成绩）；「整条判定在真数据上的样子」重跑；4.3 的现状。
- [ ] README：「怎么判」表格、成绩、「还在做的」、授权表里 TOPReward 那行。
- [ ] `.hackertrip/STATUS.md` 头部和日志。
