# 本批小结实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `jianjin check --summary`：报告顶部多一节「本批小结」，Nemotron（Nebius Token Factory）写文字，数字由拣金填，中英各一份。

**Architecture:** 新模块 `src/jianjin/summary.py`：算事实、拼提示、调接口（可注入）、核对、填占位符、退回模板、缓存。`report.render` 多一个可选参数；`cli` 多两个参数。

**Tech Stack:** Python 3.12 标准库（urllib、hashlib、re）、pytest。

设计见 `docs/superpowers/specs/2026-10-05-nemotron-summary-design.md`。分支 `nemotron-summary`。

---

### Task 1：事实

- [ ] 测试先写：从判定和示范算出条数、原因前五（按示范计，带占比和最多三条示范编号）、没查的项目；有采集员编号时按人分；中英两套取值，原因名有英文。
- [ ] 实现 `batch_facts(verdicts, episodes, title)`；提交。

### Task 2：核对和填写

- [ ] 测试先写：只有占位符的文字通过；阿拉伯数字、「七条」、英文 seven、没给过的占位符都不通过；「一下」「同一处」放过；填写按语言取值。
- [ ] 实现 `check_text`、`fill`；提交。

### Task 3：调用、重试、回退、缓存

- [ ] 测试先写（假接口）：好回答直接用；第一次不合格、第二次合格用第二次；两次都不合格退回模板并记下问题；同样的事实第二次不调接口；没有密钥用模板；去掉 `<think>` 段。
- [ ] 实现 `summarize(...)` 和标准库的 `token_factory_chat`；提交。

### Task 4：报告和命令行

- [ ] 测试先写：给了小结，报告顶部有「本批小结」和出处；没给，报告不变（快照测试不动）；`jianjin check --summary` 没设密钥时用模板。
- [ ] 实现；提交。

### Task 5：真接口跑一次、文档

- [ ] 用户领额度、建密钥、在自己终端设 `NEBIUS_API_KEY`；跑一个公开数据集，看输出和核对结果。
- [ ] README（怎么跑、Nebius 一节）、设计文档第 7 节、STATUS；合并。

## 执行记录

（做的时候补）
