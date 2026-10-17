"""甄数在线演示：选一个公开的 LeRobot 数据集，或上传自己的数据，几十秒出质检报告。只跑关节规则（CPU）。"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi, snapshot_download

from assay_space import find_dataset_root, safe_extract, summarize, valid_repo_id

MAX_MB = 300
EXAMPLES = {
    "插头插入 · SO-101 · 72 条": "777im/so101_pick_plug_insert_v5_v6",
    "插座插头 · GIM 机械臂 · 376 条": "RDLwicked/gim_arm_socket_plug_teleop_3cam_joint",
    "打开开关 · SO-101 · 50 条": "RohanVasa/Switch_On_20260725_160650",
    "旋阀门 · SO-101 · 30 条": "plzsay/turn_valve_20260915_213218",
    "装电池 · ALOHA 双臂 · 49 条": "lerobot/aloha_static_battery",
    "工业装配 · Vega · 200 条": "rocochallenge2025/rocochallenge2026_Industrial_Assembly",
    "叠方块 · SO-100 · 56 条": "lerobot/svla_so100_stacking",
}
WORK = Path(tempfile.gettempdir()) / "assay_space"
# 浏览器是深色模式时 Gradio 也会变深色；这个演示固定用浅色
FORCE_LIGHT = """() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}"""

CSS = """
:root, .dark { --body-background-fill: #F4F7FB; }
.gradio-container { max-width: 1240px !important; margin: 0 auto; background: #F4F7FB; font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif; }
#hero { border-top: 4px solid #1D5DB0; background: #FFFFFF; padding: 28px 32px 22px; }
#hero h1 { font-size: 34px; margin: 0 0 6px; color: #13233A; letter-spacing: 1px; }
#hero h1 span { color: #1D5DB0; font-weight: 500; font-size: 22px; margin-left: 12px; }
#hero p { color: #4A5868; font-size: 16px; margin: 0; }
#hero .chips { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
#hero .chip { border: 1px solid #C9D6E6; padding: 4px 12px; font-size: 13px; color: #2B4A70; background: #F4F7FB; }
#hero .chip b { color: #C2531B; }
.panel { border-top: 2px solid #1D5DB0 !important; background: #FFFFFF !important; border-radius: 0 !important; }
#run_hub, #run_up { background: #1D5DB0 !important; color: #FFFFFF !important; border-radius: 0 !important; }
#summary { background: #FFFFFF; border-left: none !important; padding: 4px 8px; }
#report iframe { width: 100%; height: 900px; border: 1px solid #C9D6E6; background: #FFFFFF; color-scheme: light; }
footer { display: none !important; }
"""

HERO = """
<div id="hero">
  <h1>甄数 <span>Assay · 机器人遥操作示范数据的自动质检</span></h1>
  <p>每条示范判成合格、存疑、作废，写清原因、依据读数和出问题的时间段；质检员只复看存疑的。</p>
  <div class="chips">
    <div class="chip">北京人形质检标签：二次抓取 <b>99/162</b> 检出</div>
    <div class="chip">robomimic 真人示范：标出的 93 条里 <b>77</b> 条来自水平差的操作员</div>
    <div class="chip">30 个公开数据集 <b>17,646</b> 条，笔记本 CPU 上 8 分半钟</div>
    <div class="chip">代码开源 · Apache-2.0</div>
  </div>
</div>
"""

NOTE = """
**说明** 这个演示只跑关节规则（CPU），不跑视频评判；视频评判要显卡，在完整版里。上传的数据只在这次运行的临时目录里处理，出完报告就删掉，不留存。
代码：[github.com/hoongyu/assay](https://github.com/hoongyu/assay) ｜ 支持的格式：LeRobot v3、robomimic HDF5、RoboMIND HDF5
"""


def _repo_mb(repo: str) -> float:
    files = HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True)
    return sum(getattr(f, "size", 0) or 0 for f in files
               if f.path.startswith(("meta/", "data/"))) / 1e6


def _check(root: Path, out: Path, lang: str) -> tuple[str, str, list[str]]:
    report, detail = out / "report.html", out / "report.json"
    p = subprocess.run(["assay", "check", str(root), "--out", str(report), "--json", str(detail), "--no-frames",
                        "--lang", lang], capture_output=True, text=True, encoding="utf-8", timeout=600)
    if not report.exists():
        raise gr.Error(f"甄数没能出报告：{(p.stdout + p.stderr)[-400:]}")
    s = summarize(json.loads(detail.read_text(encoding="utf-8")))
    labels = s["labels"]
    lines = [f"### {s['n']:,} 条示范：合格 {labels.get('合格', 0):,} ｜ 存疑 {labels.get('存疑', 0):,} ｜ "
             f"作废 {labels.get('作废', 0):,}"]
    if s["causes"]:
        lines.append("问题分布（按条数）：" + "，".join(f"{c} {k}" for c, k in s["causes"][:6]))
    fails = [l.strip() for l in p.stdout.splitlines() if l.strip().startswith("读取失败 ")]
    if fails:
        lines.append(f"没检查的 {len(fails)} 项（数据本身的问题，甄数照实报出）：" + "；".join(fails[:3]))
    # 嵌进页面的这份固定用浅色（报告本身跟着系统深浅色走，下载的原文件不改）
    page = report.read_text(encoding="utf-8").replace("@media (prefers-color-scheme:dark)", "@media (max-width:0px)")
    frame = f'<iframe srcdoc="{html.escape(page, quote=True)}"></iframe>'
    return "\n\n".join(lines), frame, [str(report), str(detail)]


def _fresh(name: str) -> Path:
    d = WORK / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def run_hub(choice: str, repo: str, lang: str):
    repo = (repo or "").strip() or EXAMPLES.get(choice, "")
    if not valid_repo_id(repo):
        raise gr.Error("数据集名的格式是 作者/名字，比如 lerobot/aloha_static_battery")
    try:
        mb = _repo_mb(repo)
    except Exception as exc:
        raise gr.Error(f"Hugging Face 上找不到这个公开数据集：{repo}（{type(exc).__name__}）")
    if mb > MAX_MB:
        raise gr.Error(f"关节数据有 {mb:.0f} MB，演示限 {MAX_MB} MB；大数据集请在本地用命令行跑")
    work = _fresh("hub")
    data = work / repo.replace("/", "__")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=data, allow_patterns=["meta/*", "data/*"])
    out = work / "out"
    out.mkdir()
    try:
        return _check(data, out, lang)
    finally:
        shutil.rmtree(data, ignore_errors=True)


def run_upload(path: str | None, lang: str):
    if not path:
        raise gr.Error("先选一个文件：LeRobot v3 数据集打成的 zip，或一个 HDF5 文件")
    work = _fresh("upload")
    src = Path(path)
    try:
        if src.suffix.lower() == ".zip":
            root = find_dataset_root(safe_extract(src, work / "data", max_bytes=MAX_MB * 1_000_000))
        elif src.suffix.lower() in {".hdf5", ".h5"}:
            root = work / "data" / src.name
            root.parent.mkdir()
            shutil.copy(src, root)
        else:
            raise gr.Error("只收 .zip（LeRobot v3）或 .hdf5（robomimic、RoboMIND）")
        out = work / "out"
        out.mkdir()
        return _check(root, out, lang)
    except ValueError as exc:
        raise gr.Error(str(exc))
    finally:
        shutil.rmtree(work / "data", ignore_errors=True)
        src.unlink(missing_ok=True)


with gr.Blocks(title="甄数 Assay · 示范数据自动质检") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="panel"):
            lang = gr.Radio([("中文报告", "zh"), ("English report", "en")], value="zh", label="报告语言")
            with gr.Tab("公开数据集"):
                choice = gr.Dropdown(list(EXAMPLES), value=list(EXAMPLES)[0], label="选一个电气和工业操作的示例")
                repo = gr.Textbox(label="或填 Hugging Face 上任意 LeRobot v3 数据集", placeholder="作者/名字")
                run_h = gr.Button("开始体检", elem_id="run_hub")
            with gr.Tab("上传自己的数据"):
                up = gr.File(label=f"LeRobot v3 数据集的 zip（含 meta/ 和 data/），或 HDF5 文件，{MAX_MB} MB 以内",
                             file_types=[".zip", ".hdf5", ".h5"], type="filepath")
                run_u = gr.Button("开始体检", elem_id="run_up")
            gr.Markdown(NOTE)
        with gr.Column(scale=3):
            summary = gr.Markdown("选一个数据集，点“开始体检”。小数据集几十秒出结果。", elem_id="summary")
            files = gr.File(label="下载报告（HTML，断网也能打开）和逐条明细（JSON）", file_count="multiple")
    report = gr.HTML(elem_id="report")
    run_h.click(run_hub, [choice, repo, lang], [summary, report, files])
    run_u.click(run_upload, [up, lang], [summary, report, files])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base(primary_hue="blue", radius_size="none"), js=FORCE_LIGHT)
