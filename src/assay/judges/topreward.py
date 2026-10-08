"""TOPReward 适配器（LeRobot 0.6.1 的 TOPReward 移植，2026-10-03 读源码写成）。

TOPReward 没有自己的权重：拿现成的 Qwen3-VL，看一段视频和一句
「视频里的机器人完成了这个任务：<任务描述> …… 答案是：True」，读出最后那个「True」的对数概率。
要整条曲线，就对越来越长的前缀各打一次分（官方离线打标用 15 个锚点），
进度按官方做法在每条示范里做最小最大归一化；成功概率记成 exp(对数概率)，即「True」的概率，
它在示范之间可比，判断「做的是不是这个任务」靠它。

和 lerobot 自带加载方式的区别：自带的 TOPRewardModel 先把整个模型读进内存再搬到显卡，
Colab 免费档 13.6 GB 内存装不下 8B（2026-10-02 摸底时崩在这一步）。这里直接加载到显卡，
默认用 4B（半精度约 9 GB）。用到了 lerobot 的编码步骤和键名，所以锁 lerobot==0.6.1。

提示格式默认用对话格式（lerobot 的 add_chat_template=True：模型以助手身份回答，后面接「True」）。
lerobot 的默认格式把「The answer is: True」直接接在用户那段话后面，「True」的概率只有万分之几，
比出来的多半是任务描述本身的先验，不是视频。2026-10-04 两种格式都在 Colab 上跑了，在 dev、train 上比：
换描述 rel_success 检出 49/103 对 45/103（AUC 0.940 对 0.568），进度回落 14/60 对 2/60，
见设计文档 TOPReward 一节。chat=False 是 lerobot 默认格式，留着复现对照。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from assay.judges.base import Curve
from assay.judges.robometer import subsample

VLM = "Qwen/Qwen3-VL-4B-Instruct"
MAX_FRAMES = 32
N_ANCHORS = 15

ScoreFn = Callable[[np.ndarray, str, float], float]  # (前缀帧 T×H×W×3, 任务描述, 帧率) -> 对数概率


def anchors(n: int, k: int) -> np.ndarray:
    """前缀长度：从 2 帧到 n 帧均匀取不超过 k 个（只有 1 帧时就是 1）。"""
    return np.unique(np.linspace(min(2, n), n, k).round().astype(int))


def min_max(x) -> np.ndarray:
    """和官方 normalize_rewards 一样：全相等时都记 1。"""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return np.ones_like(x) if hi == lo else (x - lo) / (hi - lo)


class TOPRewardJudge:
    name = "topreward"

    def __init__(self, vlm_name: str = VLM, max_frames: int = MAX_FRAMES, n_anchors: int = N_ANCHORS,
                 chat: bool = True, score_fn: ScoreFn | None = None):
        self.vlm_name = vlm_name
        self.max_frames = max_frames
        self.n_anchors = n_anchors
        self.chat = chat
        self.version = (f"TOPReward/{vlm_name.split('/')[-1]}/lerobot-0.6.1/fp16/f{max_frames}a{n_anchors}"
                        + ("/chat" if chat else ""))
        self._score = score_fn

    def _load(self) -> None:
        if self._score is not None:
            return
        import torch
        from lerobot.lerobot_types import TransitionKey
        from lerobot.rewards.topreward.processor_topreward import (
            TOPREWARD_FEATURE_PREFIX,
            TOPREWARD_INPUT_KEYS,
            TOPRewardEncoderProcessorStep,
        )
        from torch.nn.functional import cross_entropy
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.vlm_name, dtype=torch.float16, device_map="cuda"  # T4 不支持 BF16
        ).eval()
        encoder = TOPRewardEncoderProcessorStep(vlm_name=self.vlm_name, max_frames=None,
                                                add_chat_template=self.chat)
        key = encoder.image_key

        def score(frames: np.ndarray, task: str, fps: float) -> float:
            encoder.fps = fps  # 给 Qwen 的视频时间戳用
            video = torch.from_numpy(np.ascontiguousarray(frames))[None]  # 1×T×H×W×3，uint8
            obs = encoder({TransitionKey.OBSERVATION: {key: video},
                           TransitionKey.COMPLEMENTARY_DATA: {"task": task}})[TransitionKey.OBSERVATION]
            inputs = {k: obs[f"{TOPREWARD_FEATURE_PREFIX}{k}"].to("cuda") for k in TOPREWARD_INPUT_KEYS}
            labels = inputs.pop("labels")
            with torch.no_grad():  # 照抄 TOPRewardModel.compute_reward
                logits = model(**inputs, logits_to_keep=2).logits
            return float(-cross_entropy(logits[:, -2, :].float(), labels[:, -1]))

        self._score = score

    def curve(self, frames: np.ndarray, times: np.ndarray, task: str) -> Curve:
        self._load()
        idx = subsample(len(frames), self.max_frames)
        f, t = frames[idx], np.asarray(times, dtype=np.float64)[idx]
        span = t[-1] - t[0]
        fps = (len(idx) - 1) / span if span > 0 else 2.0
        ends = anchors(len(idx), self.n_anchors)
        raw = [self._score(f[:k], task, fps) for k in ends]
        if not all(math.isfinite(r) for r in raw):
            raise RuntimeError(f"TOPReward 给了非有限的对数概率 {raw}（半精度溢出？）")
        return Curve(
            self.name, self.version,
            [float(t[k - 1]) for k in ends],
            [float(p) for p in min_max(raw)],
            [math.exp(r) for r in raw],
        )
