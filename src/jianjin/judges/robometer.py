"""Robometer-4B 适配器（LeRobot 0.6.1，2026-10-02 在 Colab T4 上摸底）。

摸底结论（notebooks/colab_probe.out.ipynb）：
- 配置里设 torch_dtype="float16" 就能按半精度加载，显存 9.0 GB；T4 不支持 BF16。
- compute_reward 只返回最后一帧；它内部先用 _compute_rbm_logits 和 decode_progress_outputs
  算出每一帧的进度和成功概率。这里直接调这两步，一次拿整条曲线。
- 配置里的 max_frames=8 没被编码器用上，帧数要自己抽；26 帧（长边 448）峰值 13.5 GB，43 帧爆显存。
用到了私有方法 _compute_rbm_logits，所以 judges 依赖组锁 lerobot==0.6.1。
"""

from __future__ import annotations

import numpy as np

from jianjin.judges.base import Curve

MAX_FRAMES = 24
INPUT_KEYS = (
    "input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw",
    "video_grid_thw", "second_per_grid_ts", "mm_token_type_ids",
    "prog_token_id", "vision_start_token_id", "vision_end_token_id", "video_merge_size",
)


def subsample(n: int, k: int) -> np.ndarray:
    """从 n 帧里均匀取不超过 k 帧，含首尾。"""
    return np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(int))


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


class RobometerJudge:
    name = "robometer"

    def __init__(self, max_frames: int = MAX_FRAMES, model=None, encoder=None, decode=None,
                 input_keys: tuple[str, ...] = INPUT_KEYS):
        self.max_frames = max_frames
        self.version = f"Robometer-4B/lerobot-0.6.1/fp16/f{max_frames}"
        self._model, self._encoder, self._decode = model, encoder, decode
        self._input_keys = input_keys
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
        from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
        from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

        cfg = RobometerConfig(pretrained_path="lerobot/Robometer-4B", device="cuda", reward_output="progress")
        if torch.cuda.get_device_capability()[0] < 8:
            cfg.torch_dtype = "float16"
        self._model = RobometerRewardModel.from_pretrained(cfg.pretrained_path, config=cfg)
        self._model.eval()
        self._encoder = RobometerEncoderProcessorStep(
            base_model_id=cfg.base_model_id,
            use_multi_image=cfg.use_multi_image,
            use_per_frame_progress_token=cfg.use_per_frame_progress_token,
            max_frames=cfg.max_frames,
        )
        self._decode = decode_progress_outputs
        self._device = "cuda"

    def curve(self, frames: np.ndarray, times: np.ndarray, task: str) -> Curve:
        self._load()
        idx = subsample(len(frames), self.max_frames)
        encoded = self._encoder.encode_samples([(frames[idx], task)])
        inputs = {k: v for k, v in encoded.items() if k in self._input_keys}
        if self._device:
            import torch

            inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                progress_logits, success_logits = self._model._compute_rbm_logits(inputs)
        else:
            progress_logits, success_logits = self._model._compute_rbm_logits(inputs)
        decoded = self._decode(progress_logits, success_logits,
                               is_discrete_mode=self._model.config.use_discrete_progress)
        progress = np.clip(_to_numpy(decoded["progress_pred"][0]), 0.0, 1.0)
        success = _to_numpy(decoded["success_probs"][0])
        if len(progress) != len(idx):
            raise RuntimeError(f"Robometer 给了 {len(progress)} 个进度，应该是 {len(idx)} 个")
        return Curve(
            self.name, self.version,
            [float(times[i]) for i in idx],
            [float(p) for p in progress],
            [float(s) for s in success] if len(success) == len(idx) else None,
        )
