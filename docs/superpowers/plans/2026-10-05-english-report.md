# 英文报告、换档关键帧、小结建议：实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `jianjin check --lang en` 出英文报告；二次抓取的关键帧对准换档时刻；本批小结给采集员的建议只用我们写的句子；`jianjin summary-stats` 数小结的通过率。

**Architecture:** 设计见 `docs/superpowers/specs/2026-10-05-english-report-design.md`。双语文字用 `types.Bi`（`str` 的子类，`.en` 是英文），在产生文字的地方一次写好中英两份；名称表放 `i18n.py`；`report.render` 和命令行按 `lang` 取。

**Tech Stack:** Python 3.12、pytest、uv。

---

### Task 1：`Bi`、`en()`、`reason()`

**Files:** `src/jianjin/types.py`、`src/jianjin/checks/config.py`、`src/jianjin/checks/runner.py`、`src/jianjin/checks/batch.py`（`_compare`）、`src/jianjin/checks/video.py`、`tests/test_types.py`、`tests/helpers.py`

- [ ] 测试：`Bi("中", "en")` 等于 `"中"`、`en()` 取到 `"en"`；普通字符串 `en()` 原样返回；`copy.deepcopy`、`json.dumps` 后中文不变；`reason(NotApplicable(Bi(...)))` 保留 `.en`。
- [ ] `tests/helpers.py` 加 `assert_bilingual(text)`：英文非空、不含汉字、两份抽出的数字序列相同。
- [ ] 实现；`runner`、`_compare`、视频规则取原因时改用 `reason(exc)`。

### Task 2：名称表 `i18n.py`

**Files:** `src/jianjin/i18n.py`（新）、`src/jianjin/summary.py`、`tests/test_i18n.py`（新）

- [ ] 测试：每个 `Cause`、每个规则名、每个来源、每个 `Label` 都有英文名。
- [ ] `CAUSE_EN`、`CHECKS` 从 `summary.py` 搬到 `i18n.py`，加 `SOURCE_EN`（报告用）、`LABEL_EN`、`check_name(key, lang)`；`summary.py` 改为引用。

### Task 3：证据和没查原因的英文

**Files:** `src/jianjin/checks/{batch,integrity,motion,video}.py`、`src/jianjin/judges/cache.py`、`src/jianjin/scores/{frame,video}.py`、`src/jianjin/diagnose.py`（「视频评判」那条）；测试加在 `tests/test_{batch,integrity,motion,video_checks,diagnose}.py`

- [ ] 在已有测试里给每条规则的证据、每种没查原因加 `assert_bilingual(...)`，先看它们失败。
- [ ] 逐处改成 `Bi(中文, 英文)`，两份用同一组数，数字顺序一致。

### Task 4：判定理由的英文

**Files:** `src/jianjin/diagnose.py`、`tests/test_diagnose.py`

- [ ] 测试：硬规则、两个来源对上、同一件事、只有一个来源、对不上，五种理由都有英文，`assert_bilingual`。
- [ ] `label_of` 生成 `Bi`；`SAME_PROBLEM` 带英文。

### Task 5：换档关键帧

**Files:** `src/jianjin/types.py`（`Finding.moments`）、`src/jianjin/checks/batch.py`（`regrasp`）、`src/jianjin/keyframes.py`、`tests/test_keyframes.py`、`tests/test_batch.py`

- [ ] 测试：带 `moments=(2.9, 11.1, 12.8, 13.5, 20.9, 22.0)` 的整条发现取到 `[2.9, 12.8, 13.5, 22.0]`；二次抓取的发现带换档时刻。
- [ ] 实现。

### Task 6：英文报告

**Files:** `src/jianjin/report.py`、`tests/test_report.py`

- [ ] 测试：`render(..., lang="en")` 里除了中文小结那一段没有汉字；判定写 PASS、DOUBT、REJECT；`<html lang="en">`；页脚不再说「错杀......更贵」；中文版快照照旧通过（快照里的页脚和筛选按钮的内部值改了，更新快照）。
- [ ] 界面文字表；`lang` 参数一路传下去。

### Task 7：命令行 `--lang`

**Files:** `src/jianjin/cli.py`、`tests/test_cli.py`

- [ ] 测试：`check --lang en --out x.html` 写出英文报告，终端打印是英文。
- [ ] 实现。

### Task 8：小结建议

**Files:** `src/jianjin/summary.py`、`tests/test_summary.py`

- [ ] 测试：每个 `Cause` 都有中英建议、都不含数字；`batch_facts` 只给本批排在前面的原因加 `advice_i`；模板带 `{advice_1}`；提示第 5 条要求只用 `{advice_*}`；`PROMPT_VERSION == "4"`。
- [ ] 实现。

### Task 9：`jianjin summary-stats`

**Files:** `src/jianjin/summary.py`（`cache_stats`）、`src/jianjin/cli.py`、`tests/test_summary.py`、`tests/test_cli.py`

- [ ] 测试：造三条缓存（第一次就过、重试以后过、退回模板）和一条旧版本的，统计只算当前版本。
- [ ] 实现。

### Task 10：收尾

- [ ] 全部测试；叠方块、叠毛巾各出一份英文报告看一遍。
- [ ] README 中英、Devpost、视频脚本：试跑命令加 `--lang en`；写清建议来自哪里。
- [ ] 提交、合并、推送；更新 STATUS。
