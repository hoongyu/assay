"""本批小结：Nemotron（Nebius Token Factory）写给质检组长和采集员的一段话，数字由甄数填。

模型只看甄数算出的统计（条数、原因、示范编号），不看原始数据。它写的文字里，所有数量、编号、原因名都是占位符，
由代码按统计填进去；它自己写的数字一律不收（见 check_text），不合格重试一次，还不行退回模板。
见 docs/superpowers/specs/2026-10-05-nemotron-summary-design.md。
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from assay.diagnose import Label, Verdict
from assay.i18n import check_name
from assay.types import CAUSE_EN, Bi, Cause, Episode, en, spaced

# 给采集员的提醒：每类原因一句，中英各一句，由我们写。模型只能用 {advice_i} 占位符引用，不许自己编建议
# （2026-10-05 第一次真调用，模型自己写了「下次采集确保某项检查在所有示范上都能跑」，对采集员没用）。
ADVICE = {
    Cause.INCOMPLETE: ("做完整个任务、手臂回到起始姿态以后再停止录制。",
                       "Stop recording only after the task is done and the arm is back in its start pose."),
    Cause.EMPTY: ("开录前确认手臂真的在动；没有动作的录制直接删掉，不要提交。",
                  "Make sure the arm is actually moving before you record; delete recordings with no motion "
                  "instead of submitting them."),
    Cause.MISPLACED: ("放下物体前先对准目标位置，再松开夹爪。", "Line the object up with the target before opening the gripper."),
    Cause.GRASP_MISS: ("夹爪合紧、确认夹住物体以后再提起。", "Close the gripper fully and make sure the object is held before lifting."),
    Cause.REGRASP: ("抓取前先把夹爪对正物体，合上以后稍停再提起，避免松开重抓。",
                    "Square the gripper to the object before closing it and pause briefly before lifting, so it does "
                    "not have to let go and grasp again."),
    Cause.DROP: ("搬运时放慢速度，确认夹稳以后再移动。", "Carry the object more slowly and make sure the grip is secure before moving."),
    Cause.EXTRA_MOTION: ("按最直接的路线完成任务，少做来回调整。",
                         "Take the most direct path through the task and avoid back-and-forth adjustments."),
    Cause.LONG_PAUSE: ("中途卡住时停止录制、重新开始，不要在半空中长时间停住。",
                       "If you get stuck mid-task, stop and restart the recording instead of holding still for a long time."),
    Cause.JERKY: ("保持连贯的节奏完成动作，中途不要反复停顿和修正。",
                  "Keep a steady rhythm through the motion, without repeated stops and corrections."),
    Cause.WRONG_TASK: ("开录前核对任务描述和要做的动作是否一致。",
                       "Before recording, check that the task description matches what you are about to do."),
    Cause.DATA_ERROR: ("检查录制程序：视频和关节数据要同时开始、同时结束，时长过短的录制要删掉。",
                       "Check the recording setup: video and joint data should start and stop together, and recordings "
                       "that are too short should be deleted."),
}

TOP_CAUSES = 5
EXAMPLES = 3
TOP_OPERATORS = 5


@dataclass(frozen=True)
class Facts:
    """给模型看的事实。meaning[占位符名] 告诉模型它是什么；values[语言][占位符名] 是填进去的文字。"""

    meaning: dict[str, str]
    values: dict[str, dict[str, str]]


def _short(episode_id: str) -> str:
    return episode_id.rsplit("/", 1)[-1]


def batch_facts(verdicts: list[Verdict], episodes: list[Episode], title: str) -> Facts:
    """一批示范的统计：各判定条数，报得最多的原因（按示范计），没查的项目（没启用的不算），有采集员编号时按人分。"""
    meaning: dict[str, str] = {}
    zh: dict[str, str] = {}
    en: dict[str, str] = {}

    def put(key: str, what: str, z, e=None) -> None:
        meaning[key], zh[key], en[key] = what, str(z), str(z if e is None else e)

    order = list(Cause)
    n = len(verdicts)
    labels = Counter(v.label for v in verdicts)
    put("batch", "name of this batch of demonstrations", title)
    put("n_total", "number of episodes checked", n)
    put("n_pass", "episodes judged PASS", labels[Label.PASS])
    put("n_doubt", "episodes judged DOUBT (a human reviewer should look at them)", labels[Label.DOUBT])
    put("n_reject", "episodes judged REJECT", labels[Label.REJECT])

    by_cause: dict[Cause, list[str]] = defaultdict(list)
    for v in verdicts:
        for c in dict.fromkeys(f.cause for f in v.causes):
            by_cause[c].append(v.episode_id)
    ranked = sorted(by_cause.items(), key=lambda kv: (-len(kv[1]), order.index(kv[0])))[:TOP_CAUSES]
    for i, (cause, ids) in enumerate(ranked, 1):
        put(f"cause_{i}", f"name of the problem ranked #{i} by number of episodes", cause.value, CAUSE_EN[cause])
        put(f"cause_{i}_n", f"number of episodes with problem #{i}", len(ids))
        put(f"cause_{i}_pct", f"share of the batch with problem #{i}", f"{round(100 * len(ids) / n)}%")
        eg = [_short(x) for x in ids[:EXAMPLES]]
        put(f"cause_{i}_eg", f"example episode ids with problem #{i}", "、".join(eg), ", ".join(eg))
        put(f"advice_{i}", f"a fixed, pre-written tip for the operators about problem #{i}", *ADVICE[cause])

    unchecked = Counter(k for v in verdicts for k, why in v.not_checked.items() if "没启用" not in why)
    for i, (key, k) in enumerate(sorted(unchecked.items(), key=lambda kv: (-kv[1], kv[0]))[:3], 1):
        put(f"unchecked_{i}", f"name of a check that could not run on some episodes (#{i})", check_name(key),
            check_name(key, "en"))
        put(f"unchecked_{i}_n", f"number of episodes where check #{i} could not run", k)

    ops = {ep.id: ep.operator_id for ep in episodes}
    if any(ops.values()):
        per: dict[str, list[Verdict]] = defaultdict(list)
        for v in verdicts:
            per[ops.get(v.episode_id) or ""].append(v)
        ranked_ops = sorted(per.items(), key=lambda kv: (-sum(x.label != Label.PASS for x in kv[1]), kv[0]))
        for j, (op, vs) in enumerate(ranked_ops[:TOP_OPERATORS], 1):
            put(f"op_{j}", f"id of operator #{j} (ranked by flagged episodes)", op or "（未填）", op or "(unknown)")
            put(f"op_{j}_n", f"episodes recorded by operator #{j}", len(vs))
            put(f"op_{j}_flagged", f"episodes of operator #{j} judged DOUBT or REJECT", sum(x.label != Label.PASS for x in vs))
            cc = Counter(c for x in vs for c in dict.fromkeys(f.cause for f in x.causes))
            if cc:
                c, k = sorted(cc.items(), key=lambda kv: (-kv[1], order.index(kv[0])))[0]
                put(f"op_{j}_cause", f"most frequent problem of operator #{j}", c.value, CAUSE_EN[c])
                put(f"op_{j}_cause_n", f"episodes of operator #{j} with that problem", k)
    return Facts(meaning, {"zh": zh, "en": en})


PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_]+)\}")
DIGIT = re.compile(r"[0-9０-９]")
ZH_NUMBER = re.compile(r"[二两三四五六七八九十百千万][零〇一二两三四五六七八九十百千万]*\s*"
                       r"(?:条|个|次|秒|成|项|位|名|批|段|处|种|类|回|分钟|小时|倍)")
EN_NUMBER = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                       r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|hundreds?|"
                       r"thousands?|dozens?|half)\b", re.IGNORECASE)


def check_text(text: str, keys: set[str]) -> list[str]:
    """模型写的文字能不能收：占位符都得是给过的；去掉占位符以后，不许有它自己写的数字。返回问题，空表示能收。
    「一」太常见（一下、同一处），英文的 one 也是，都放过。"""
    problems = [Bi(f"用了没给过的占位符 {{{k}}}", f"used a placeholder that was not given: {{{k}}}")
                for k in dict.fromkeys(PLACEHOLDER.findall(text)) if k not in keys]
    rest = PLACEHOLDER.sub("", text)
    if "{" in rest or "}" in rest:
        problems.append(Bi("有写坏的占位符（花括号不成对）", "has a broken placeholder (unmatched braces)"))
    for name, name_en, pattern in (("阿拉伯数字", "digits", DIGIT), ("汉字数字", "Chinese number words", ZH_NUMBER),
                                   ("英文数字", "English number words", EN_NUMBER)):
        found = list(dict.fromkeys(m.group(0) for m in pattern.finditer(rest)))
        if found:
            problems.append(Bi(f"自己写了{name}：{'、'.join(found[:5])}", f"wrote {name_en} itself: {', '.join(found[:5])}"))
    return problems


def fill(text: str, facts: Facts, lang: str) -> str:
    values = facts.values[lang]
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


# ---- 调模型 ----

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"
# 2026-10-05 第一版在真接口上两种语言都没过核对：照抄了提示里给的值（含批次名里的 100），还自造了 {placeholder}、
# {x} 这样的占位符。第二版不给带数字的值，改成让它改写一份全是占位符的草稿，重试时逐个列出能用的占位符。
# 第二版存下的原文显示，真正的原因是推理过程混进了正文、写到长度上限被截断（见 token_factory_chat）；
# 第三版关掉推理。第四版（2026-10-05）给采集员的建议只许用 {advice_i}（见 ADVICE），重试时用英文说问题。
# 缓存按这个版本号分开，改了请求就要改它。
PROMPT_VERSION = "4"
SYSTEM = (
    "You rewrite a plain draft into a short, useful feedback note for the QA lead and the teleoperators of a robot "
    "data-collection site. The draft and the placeholder list are the ONLY facts. Rules: (1) Keep every placeholder "
    "in curly braces exactly as written, e.g. {n_doubt}; the software replaces them with the real numbers and names "
    "afterwards. (2) Never write digits or number words yourself, and never spell out a value that is only given as "
    "a placeholder. (3) Use only the listed placeholders; do not make up new ones. (4) Do not invent problems or "
    "facts. (5) You may reorder, merge or drop sentences, and tell the reviewers which episodes to look at first. "
    "Advice for the operators may only come from the {advice_N} placeholders: use the ones that fit and do not write "
    "tips of your own. (6) Output only the note: "
    "plain prose, no reasoning, no headings, no lists, no markdown."
)
LANG_SPEC = {
    "zh": "Write the note in Simplified Chinese, at most 200 Chinese characters.",
    "en": "Write the note in English, at most 120 words.",
}
Chat = Callable[[list[dict], str], str]


@dataclass(frozen=True)
class Summary:
    texts: dict[str, str]  # 语言 → 填好数字的文字
    source: dict[str, str]  # 语言 → model（模型写的）或 template（模板）
    model: str | None  # 用了哪个模型；全是模板时为 None
    problems: dict[str, list[str]]  # 语言 → 模型的输出没过核对、或者调用失败的原因


def _shown(value: str) -> str:
    """给模型看的取值：带数字的（数量、占比、编号、带数字的名字）不给，免得它照抄；零单独说；纯文字的照给。"""
    if value == "0":
        return "; value is zero"
    return "" if DIGIT.search(value) else f"; value: {value}"


def _messages(facts: Facts, lang: str) -> list[dict]:
    rows = "\n".join(f"{{{k}}}: {facts.meaning[k]}{_shown(facts.values[lang][k])}" for k in facts.meaning)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Placeholders you may use (meaning; value when it has no digits):\n{rows}\n\n"
                                        f"Plain draft:\n{template(facts, lang)}\n\n{LANG_SPEC[lang]} "
                                        "Output only the note."}]


def _strip_think(text: str) -> str:
    """去掉推理段：有的模型把推理写在 <think>…</think> 里，有的只留下结尾的 </think>。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.rsplit("</think>", 1)[-1].strip()


def template(facts: Facts, lang: str) -> str:
    """模板：同样的事实按固定句式写。没设密钥、模型的输出没过核对、调用失败时用它。"""
    v = facts.values[lang]
    zh = lang == "zh"
    parts = ["本批「{batch}」{n_total} 条示范：合格 {n_pass}，存疑 {n_doubt}，作废 {n_reject}。" if zh else
             "Batch {batch}: {n_total} episodes checked; {n_pass} passed, {n_doubt} need review, {n_reject} rejected."]
    if "cause_1" in v:
        parts.append("报得最多的是「{cause_1}」，{cause_1_n} 条（{cause_1_pct}），比如 {cause_1_eg}。" if zh else
                     " The most frequent problem is {cause_1}: {cause_1_n} episodes ({cause_1_pct}), e.g. {cause_1_eg}.")
    if "advice_1" in v:
        parts.append("给采集员的提醒：{advice_1}" if zh else " Tip for operators: {advice_1}")
    if "cause_2" in v:
        parts.append("其次是「{cause_2}」，{cause_2_n} 条。" if zh else " Next is {cause_2}: {cause_2_n} episodes.")
    if v["n_reject"] != "0":
        parts.append("先看作废的 {n_reject} 条，再看存疑的。" if zh else " Review the {n_reject} rejected episodes first.")
    elif v["n_doubt"] != "0":
        parts.append("先看存疑的 {n_doubt} 条。" if zh else " Review the {n_doubt} flagged episodes first.")
    if "unchecked_1" in v:
        parts.append("{unchecked_1}有 {unchecked_1_n} 条没查。" if zh else " {unchecked_1} could not run on {unchecked_1_n} episodes.")
    if "op_1_cause" in v:
        parts.append("按采集员看，{op_1} 的 {op_1_n} 条里 {op_1_flagged} 条被标出，多是「{op_1_cause}」。" if zh else
                     " By operator, {op_1} had {op_1_flagged} of {op_1_n} episodes flagged, mostly {op_1_cause}.")
    return "".join(parts)


def _tidy_zh(text: str) -> str:
    """中文段的空格：汉字和英文之间补一个；右引号、右括号后面紧跟汉字时去掉（模型常在占位符后面多打一个空格）。"""
    return re.sub(r"(?<=[」）】》])\s+(?=[一-鿿])", "", spaced(text))


def _ask(facts: Facts, lang: str, chat: Chat, model: str) -> tuple[str | None, list[str], list[dict]]:
    """问模型，不合格带着问题重试一次。返回能收的文字（带占位符）或 None、遇到的问题、每次尝试的原文。"""
    keys = set(facts.meaning)
    allowed = ", ".join(f"{{{k}}}" for k in facts.meaning)
    messages = _messages(facts, lang)
    problems: list[str] = []
    attempts: list[dict] = []
    for attempt in range(2):
        raw = chat(messages, model)
        text = _strip_think(raw)
        found = check_text(text, keys)
        attempts.append({"raw": raw, "problems": found})
        if not found:
            return text, problems, attempts
        problems += [Bi(f"第 {attempt + 1} 次：{p}", f"attempt {attempt + 1}: {en(p)}") for p in found]
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": "This note cannot be used: " + "; ".join(en(p) for p in found)
                                + f". Rewrite it from the plain draft. Allowed placeholders: {allowed}. "
                                "Do not write any digits or number words; output only the note."}]
    return None, problems, attempts


def summarize(facts: Facts, chat: Chat | None = None, model: str = DEFAULT_MODEL, cache_dir: str | Path | None = None,
              langs: tuple[str, ...] = ("zh", "en")) -> Summary:
    """每种语言写一段。模型的回答按「模型、提示版本、语言、事实」缓存；没有 chat（没设密钥）时用模板，不缓存。"""
    texts, source, problems = {}, {}, {}
    for lang in langs:
        key = hashlib.sha256(json.dumps({"model": model, "prompt": PROMPT_VERSION, "lang": lang,
                                         "meaning": facts.meaning, "values": facts.values[lang]},
                                        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        path = Path(cache_dir) / "summary" / f"{key}.json" if cache_dir else None
        if path is not None and path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))
            raw, source[lang], problems[lang] = hit["text"], hit["source"], hit["problems"]
        elif chat is None:
            raw, source[lang], problems[lang] = template(facts, lang), "template", []
        else:
            try:
                got, problems[lang], attempts = _ask(facts, lang, chat, model)
            except Exception as exc:  # 网络、额度、限流：这次用模板，不缓存，下次再试
                raw, source[lang], problems[lang] = template(facts, lang), "template", [Bi(f"调用失败：{exc}",
                                                                                           f"the call failed: {exc}")]
            else:
                raw, source[lang] = (got, "model") if got is not None else (template(facts, lang), "template")
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"model": model, "prompt": PROMPT_VERSION, "lang": lang, "text": raw,
                                                "source": source[lang], "problems": problems[lang],
                                                "attempts": attempts},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
        texts[lang] = _tidy_zh(fill(raw, facts, lang)) if lang == "zh" else fill(raw, facts, lang)
    used = model if "model" in source.values() or any(p for p in problems.values()) else None
    return Summary(texts, source, used, problems)


class Truncated(Exception):
    """回答写到长度上限被截断了，不拿去核对。"""


def token_factory_chat(api_key: str, base_url: str = BASE_URL, timeout: float = 120.0, thinking: bool = False) -> Chat:
    """Nebius Token Factory 的 OpenAI 兼容接口，用标准库发请求。

    推理默认关掉：Nemotron 3.5 Lightning 默认开推理（模型卡：chat_template_kwargs 的 enable_thinking），
    2026-10-05 在 Token Factory 上实测，推理过程以「Here's a thinking process」开头直接写进正文、没有 <think> 标签，
    每次六七千字符，写到 2,048 个词元的上限还没写出那段话。"""

    def chat(messages: list[dict], model: str) -> str:
        body = json.dumps({"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1024,
                           "chat_template_kwargs": {"enable_thinking": thinking}}).encode("utf-8")
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise Truncated("回答写到长度上限被截断了")
        return choice["message"].get("content") or ""

    return chat


def cache_stats(cache_dir: str | Path) -> dict[str, dict[str, int]]:
    """缓存里当前提示版本的小结，按语言数三种结果：第一次就过、重试以后过、退回模板。调用失败的不缓存，不在里面。"""
    out: dict[str, dict[str, int]] = {}
    for path in sorted((Path(cache_dir) / "summary").glob("*.json")):
        hit = json.loads(path.read_text(encoding="utf-8"))
        if hit.get("prompt") != PROMPT_VERSION:
            continue
        row = out.setdefault(hit["lang"], {"first_try": 0, "after_retry": 0, "template": 0})
        if hit["source"] != "model":
            row["template"] += 1
        else:
            row["first_try" if len(hit.get("attempts", [])) <= 1 else "after_retry"] += 1
    return out
