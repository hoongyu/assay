"""规则名的中英对照，报告和本批小结共用。原因、来源、判定的英文名跟着 Cause、SOURCE_NAMES、Label 放。"""

from __future__ import annotations

from assay.types import source_name

CHECKS = {  # 规则名：（中文，英文）
    "missing_values": ("读数缺失", "missing readings"), "too_short": ("示范过短", "too short"),
    "timestamp_gaps": ("时间戳", "timestamp gaps"), "video_length": ("视频时长", "video length"),
    "long_pause": ("停顿过长", "long pause"), "extra_motion": ("多余动作", "extra motion"),
    "end_pose": ("没做完", "incomplete"), "empty_recording": ("空录", "empty recording"),
    "regrasp": ("二次抓取", "regrasp"), "jerky": ("动作不流畅", "unsmooth motion"),
    "wrong_task": ("指令不符", "wrong task"), "progress_drop": ("进度回落", "progress drop"),
    "video_judges": ("视频评判", "video judges"),
}


def check_name(key: str, lang: str = "zh") -> str:
    """规则名。视频规则在 not_checked 里的键前面带模型名，如 topreward.progress_drop。"""
    src, _, name = key.rpartition(".")
    zh, en = CHECKS.get(name, (name, name))
    if lang == "en":
        return f"{source_name(src, 'en')} {en}" if src else en
    return f"{source_name(src)}的「{zh}」" if src else zh
