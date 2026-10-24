"""视频复核：关节规则说什么时候、看哪一段，Nemotron 3.5 Super VL 回答画面里发生了什么。

只用标准库：取帧、编码视频、发请求都由调用方注入（命令行用 PyAV 和 urllib，网页用 <video>+canvas 和
Worker 里的 XHR），这里不 import av，也不联网。每条存疑或作废的示范只问一个由规则生成的固定问题，
模型只能回答列出的选项，选项以外的回答一律不收，模型自己写的数字不进报告。复核结论挂在体检单上，
不改判定：作废仍要两个独立来源。
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field, replace
from typing import Callable

from assay.diagnose import Label, Verdict
from assay.types import Episode, Finding

MODEL = "nvidia/Nemotron-3.5-Super-VL"
# 改了问题、选项或请求的写法就改它，缓存按它分开
PROMPT_VERSION = "1"
Chat = Callable[[list[dict], str], str]
YES_NO = ("yes", "no", "unclear")
# 一条示范只复核一项发现，按这个顺序挑：规则给出的时刻越具体越先问
ORDER = ("regrasp", "end_pose", "empty_recording", "progress_drop", "wrong_task", "video_length", "too_short")
PAD = 1.5  # 时刻前后各留多少秒
MAX_CLIP = 60.0  # 一段最长多少秒（模型卡：视频最长 2 分钟）
INTRO = "This is a robot teleoperation recording."


@dataclass(frozen=True)
class ReviewRequest:
    episode_id: str
    check: str
    t0: float  # 片段在示范里的起止（秒，从示范开头算）
    t1: float
    question: str
    choices: tuple[str, ...]
    confirms: tuple[str, ...]  # 这些回答支持规则的发现
    clears: tuple[str, ...]  # 这些回答和规则的发现相反


@dataclass(frozen=True)
class ReviewResult:
    episode_id: str
    check: str
    answer: str | None
    outcome: str  # confirms、clears 或 unclear
    model: str
    problems: list[str] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)


def _video_seconds(ep: Episode) -> float | None:
    known = [ref.to_s - ref.from_s for ref in ep.cameras.values() if ref.to_s is not None]
    return max(known) if known else None


def _request(v: Verdict, f: Finding, ep: Episode) -> ReviewRequest | None:
    task = ep.task.strip() or "the task"
    end = ep.duration
    if f.check == "regrasp" and len(f.moments) >= 2:
        ms = sorted(f.moments)
        a, b = min(zip(ms, ms[1:]), key=lambda p: p[1] - p[0])  # 隔得最近的两次换档：松开又抓
        t0, t1 = max(0.0, a - PAD), min(end, b + PAD)
        q = (f"{INTRO} Task: {task} Around {a - t0:.1f} s and {b - t0:.1f} s into this clip, does a robot gripper "
             "lose its grip on the object and grasp it again?")
        return ReviewRequest(v.episode_id, f.check, t0, t1, q, YES_NO, ("yes",), ("no",))
    if f.check == "end_pose":
        t0 = max(0.0, end - 3.0)
        q = (f"{INTRO} Task: {task} This clip is the last {end - t0:.1f} s of the recording. At the very end, is the "
             "robot arm still moving, or has it come to rest?")
        return ReviewRequest(v.episode_id, f.check, t0, end, q, ("still_moving", "at_rest", "unclear"),
                             ("still_moving",), ("at_rest",))
    if f.check == "empty_recording":
        q = f"{INTRO} Task: {task} Does the robot arm move at all during this clip?"
        return ReviewRequest(v.episode_id, f.check, 0.0, min(end, MAX_CLIP), q, YES_NO, ("no",), ("yes",))
    if f.check == "progress_drop":
        t0, t1 = max(0.0, f.t0 - PAD), min(end, f.t1 + PAD)
        q = (f"{INTRO} Task: {task} Between {f.t0 - t0:.1f} s and {f.t1 - t0:.1f} s into this clip, does the robot "
             "undo progress it had made, for example drop or knock over an object it had already placed?")
        return ReviewRequest(v.episode_id, f.check, t0, t1, q, YES_NO, ("yes",), ("no",))
    if f.check == "wrong_task":
        q = f"{INTRO} Task: {task} By the end of this clip, has the robot completed this task?"
        return ReviewRequest(v.episode_id, f.check, 0.0, min(end, MAX_CLIP), q, YES_NO, ("no",), ("yes",))
    if f.check == "video_length":
        video = _video_seconds(ep)
        if video is None or video <= end:  # 视频比关节短：缺的那段没有画面可看
            return None
        t0, t1 = max(0.0, end - 2.0), min(video, end + MAX_CLIP)
        q = (f"{INTRO} Task: {task} The robot's joint recording ends {end - t0:.1f} s into this clip; the rest of the "
             "video was recorded after that. What happens in the video after that point?")
        choices = ("robot_continues_task", "person_in_scene", "nothing_moves", "unclear")
        return ReviewRequest(v.episode_id, f.check, t0, t1, q, choices, choices[:3], ())
    if f.check == "too_short":
        q = f"{INTRO} Task: {task} Does this clip show the robot doing the task?"
        return ReviewRequest(v.episode_id, f.check, 0.0, end, q, YES_NO, ("no",), ("yes",))
    return None


def review_requests(verdicts: list[Verdict], episodes: list[Episode]) -> list[ReviewRequest]:
    """存疑和作废的示范各出一个复核请求（能问的话）；合格的不问。"""
    by_id = {ep.id: ep for ep in episodes}
    out = []
    for v in verdicts:
        if v.label == Label.PASS or v.episode_id not in by_id:
            continue
        for check in ORDER:
            f = next((f for f in v.causes if f.check == check), None)
            req = _request(v, f, by_id[v.episode_id]) if f else None
            if req is not None:
                out.append(req)
                break
    return out


def _format_line(choices: tuple[str, ...]) -> str:
    return 'Answer with JSON only, no other text: {"answer": ' + " | ".join(f'"{c}"' for c in choices) + "}"


def review_messages(req: ReviewRequest, frames: list[bytes] | None = None, frame_times: list[float] | None = None,
                    video: bytes | None = None) -> list[dict]:
    """OpenAI 格式的一条用户消息：视频（mp4）或几张关键帧（JPEG），后面跟问题。"""
    parts: list[dict] = []
    text = req.question
    if video is not None:
        parts.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video).decode("ascii")}})
    else:
        for img in frames or []:
            parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode("ascii")}})
        if frame_times:
            text = f"The images are frames of the clip at {', '.join(f'{t:.1f} s' for t in frame_times)}. " + text
    parts.append({"type": "text", "text": f"{text} {_format_line(req.choices)}"})
    return [{"role": "user", "content": parts}]


def parse_answer(text: str, choices: tuple[str, ...]) -> str | None:
    """只收一个 JSON 对象里 answer 等于某个选项的回答（不分大小写）；去掉推理段和代码块标记。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).rsplit("</think>", 1)[-1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    answer = data.get("answer") if isinstance(data, dict) else None
    if not isinstance(answer, str):
        return None
    answer = answer.strip().lower()
    return answer if answer in choices else None


def review(req: ReviewRequest, chat: Chat, model: str = MODEL, *, frames: list[bytes] | None = None,
           frame_times: list[float] | None = None, video: bytes | None = None) -> ReviewResult:
    """问一次；回答不在选项里就说明问题再问一次，还不行就记 unclear。网络错误交给调用方。"""
    messages = review_messages(req, frames, frame_times, video)
    problems, attempts = [], []
    for attempt in range(2):
        raw = chat(messages, model)
        attempts.append(raw)
        answer = parse_answer(raw, req.choices)
        if answer is not None:
            outcome = "confirms" if answer in req.confirms else "clears" if answer in req.clears else "unclear"
            return ReviewResult(req.episode_id, req.check, answer, outcome, model, problems, attempts)
        problems.append(f"attempt {attempt + 1}: the answer is not one of {', '.join(req.choices)}")
        messages = messages + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": "That cannot be used. " + _format_line(req.choices)}]
    return ReviewResult(req.episode_id, req.check, None, "unclear", model, problems, attempts)


def apply_reviews(verdicts: list[Verdict], results: list[ReviewResult]) -> list[Verdict]:
    """复核结论挂到对应示范的体检单上；判定不变。"""
    by_id: dict[str, list[ReviewResult]] = {}
    for r in results:
        by_id.setdefault(r.episode_id, []).append(r)
    return [replace(v, reviews=v.reviews + by_id[v.episode_id]) if v.episode_id in by_id else v for v in verdicts]
