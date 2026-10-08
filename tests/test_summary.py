from dataclasses import replace

import numpy as np

from helpers import make_episode
from assay.diagnose import Label, Verdict
from assay.types import Cause, Finding


def f(cause, source="joints"):
    return Finding(check="x", cause=cause, t0=0.0, t1=1.0, score=1.0, evidence="e", source=source)


def batch(ops=None):
    """五条：两条合格、两条存疑（二次抓取；二次抓取加 Robometer 的指令不符）、一条作废（数据异常）。"""
    ops = ops or [None] * 5
    eps = [replace(make_episode(np.zeros((10, 2)), ep_id=f"ds/ep{i:06d}"), operator_id=o) for i, o in enumerate(ops)]
    off = "TOPReward 看最开头几帧时打分不稳，这条没启用"
    vs = [
        Verdict("ds/ep000000", Label.PASS, [], [], not_checked={"topreward.progress_drop": off}),
        Verdict("ds/ep000001", Label.DOUBT, ["r"], [f(Cause.REGRASP)], not_checked={"topreward.progress_drop": off}),
        Verdict("ds/ep000002", Label.DOUBT, ["r"], [f(Cause.REGRASP), f(Cause.WRONG_TASK, "robometer")]),
        Verdict("ds/ep000003", Label.REJECT, ["r"], [f(Cause.DATA_ERROR)], not_checked={"regrasp": "认不出夹爪通道"}),
        Verdict("ds/ep000004", Label.PASS, [], [], not_checked={"regrasp": "认不出夹爪通道"}),
    ]
    return vs, eps


def test_batch_facts_counts_causes_and_unchecked():
    from assay.summary import batch_facts

    vs, eps = batch()
    facts = batch_facts(vs, eps, title="ds")
    zh, en = facts.values["zh"], facts.values["en"]
    assert set(facts.meaning) == set(zh) == set(en)
    assert (zh["n_total"], zh["n_pass"], zh["n_doubt"], zh["n_reject"]) == ("5", "2", "2", "1")
    assert zh["batch"] == en["batch"] == "ds"
    assert (zh["cause_1"], en["cause_1"], zh["cause_1_n"], zh["cause_1_pct"]) == ("二次抓取", "regrasp", "2", "40%")
    assert zh["cause_1_eg"] == "ep000001、ep000002" and en["cause_1_eg"] == "ep000001, ep000002"
    assert {zh["cause_2"], zh["cause_3"]} == {"指令不符", "数据异常"} and "cause_4" not in zh
    assert (zh["unchecked_1"], en["unchecked_1"], zh["unchecked_1_n"]) == ("二次抓取", "regrasp", "2")
    assert "unchecked_2" not in zh  # 没启用的那条不算「没查」
    assert not any(k.startswith("op_") for k in zh)  # 没有采集员编号


def test_batch_facts_by_operator():
    from assay.summary import batch_facts

    vs, eps = batch(ops=["A", "A", "A", "B", "B"])
    zh = batch_facts(vs, eps, title="ds").values["zh"]
    assert (zh["op_1"], zh["op_1_n"], zh["op_1_flagged"], zh["op_1_cause"], zh["op_1_cause_n"]) == ("A", "3", "2", "二次抓取", "2")
    assert (zh["op_2"], zh["op_2_n"], zh["op_2_flagged"], zh["op_2_cause"], zh["op_2_cause_n"]) == ("B", "2", "1", "数据异常", "1")


def test_check_text_rejects_numbers_the_model_wrote_itself():
    from assay.summary import check_text

    keys = {"n_doubt", "cause_1", "cause_1_n"}
    assert check_text("本批有 {n_doubt} 条存疑，最多的是「{cause_1}」（{cause_1_n} 条），先看一下同一处问题。", keys) == []
    assert check_text("Check one example first; {n_doubt} episodes need review.", keys) == []  # one 太常见，放过
    assert check_text("有 7 条存疑", keys)
    assert check_text("有７条存疑", keys)  # 全角数字
    assert check_text("七条存疑，两回重抓", keys)
    assert check_text("seven episodes, a dozen more", keys)
    assert check_text("{n_reject} 条作废", keys)  # 没给过的占位符
    assert check_text("坏了的 {n_doubt", keys)  # 占位符写坏了


def test_fill_uses_values_of_the_language():
    from assay.summary import Facts, fill

    facts = Facts({"n": "条数", "c": "原因"}, {"zh": {"n": "3", "c": "二次抓取"}, "en": {"n": "3", "c": "regrasp"}})
    assert fill("{n} 条「{c}」", facts, "zh") == "3 条「二次抓取」"
    assert fill("{n} {c} episodes", facts, "en") == "3 regrasp episodes"


def fake_chat(*replies):
    """依次回答 replies；记下每次收到的消息。"""
    calls = []

    def chat(messages, model):
        calls.append(messages)
        return replies[min(len(calls), len(replies)) - 1]

    chat.calls = calls
    return chat


def facts5():
    from assay.summary import batch_facts

    vs, eps = batch()
    return batch_facts(vs, eps, title="ds")


def test_summarize_uses_model_text_with_numbers_filled(tmp_path):
    from assay.summary import summarize

    chat = fake_chat("<think>先算一下</think>本批 {n_total} 条，先看作废的 {n_reject} 条。")
    s = summarize(facts5(), chat=chat, cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "本批 5 条，先看作废的 1 条。" and s.source["zh"] == "model" and len(chat.calls) == 1
    prompt = chat.calls[0][-1]["content"]
    assert "{n_doubt}" in prompt and "合格 {n_pass}" in prompt  # 给它占位符的意思，和一份全是占位符的草稿
    assert "二次抓取" in prompt  # 纯文字的值照给
    assert "ep000001" not in prompt and "40%" not in prompt  # 带数字的值不给，免得它照抄


def test_prompt_hides_values_with_digits_and_says_which_counts_are_zero():
    from assay.summary import Facts, _messages

    keys = ["batch", "n_total", "n_pass", "n_doubt", "n_reject"]
    zh = dict(zip(keys, ["so100_stacking", "5", "5", "0", "0"]))
    facts = Facts({k: k for k in keys}, {"zh": zh, "en": zh})
    prompt = _messages(facts, "zh")[-1]["content"]
    assert "so100" not in prompt and "{batch}" in prompt
    row = next(line for line in prompt.splitlines() if line.startswith("{n_reject}"))
    assert "zero" in row


def test_summarize_retries_once_then_falls_back_to_template(tmp_path):
    from assay.summary import summarize

    chat = fake_chat("有 7 条存疑。", "本批 {n_total} 条。")
    s = summarize(facts5(), chat=chat, cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "本批 5 条。" and s.source["zh"] == "model" and len(chat.calls) == 2
    assert "wrote digits itself" in chat.calls[1][-1]["content"]  # 重试时用英文把问题告诉模型
    assert "{n_total}" in chat.calls[1][-1]["content"]  # 并且把能用的占位符逐个列出来
    import json

    (hit,) = (tmp_path / "summary").glob("*.json")
    attempts = json.loads(hit.read_text(encoding="utf-8"))["attempts"]
    assert [a["raw"] for a in attempts] == ["有 7 条存疑。", "本批 {n_total} 条。"]  # 每次尝试的原文都留着
    assert attempts[0]["problems"] and attempts[1]["problems"] == []
    bad = fake_chat("有 7 条存疑。", "还是 7 条。")
    t = summarize(facts5(), chat=bad, cache_dir=tmp_path / "b", langs=("zh",))
    assert t.source["zh"] == "template" and "5 条示范" in t.texts["zh"] and t.problems["zh"]


def test_summarize_caches_model_answers_and_uses_template_without_key(tmp_path):
    from assay.summary import summarize

    summarize(facts5(), chat=fake_chat("本批 {n_total} 条。"), cache_dir=tmp_path, langs=("zh",))
    again = fake_chat("不该被调用")
    s = summarize(facts5(), chat=again, cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "本批 5 条。" and again.calls == []
    t = summarize(facts5(), chat=None, cache_dir=tmp_path / "t", langs=("zh", "en"))
    assert t.source == {"zh": "template", "en": "template"} and t.model is None
    assert t.texts["en"].startswith("Batch ds: 5 episodes checked")
    assert not list((tmp_path / "t").rglob("*.json"))  # 模板不进缓存，有了密钥以后还会去调模型


def test_summarize_falls_back_when_the_call_fails(tmp_path):
    from assay.summary import summarize

    def broken(messages, model):
        raise OSError("网络断了")

    s = summarize(facts5(), chat=broken, cache_dir=tmp_path, langs=("zh",))
    assert s.source["zh"] == "template" and "网络断了" in s.problems["zh"][0]


def test_summarize_spaces_latin_words_in_chinese(tmp_path):
    from assay.summary import summarize

    s = summarize(facts5(), chat=fake_chat("{cause_1}和Robometer都要看。"), cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "二次抓取和 Robometer 都要看。"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        import json

        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_token_factory_chat_turns_thinking_off_and_rejects_truncated_answers(monkeypatch):
    import json

    import pytest

    from assay import summary

    sent = []

    def fake_urlopen(req, timeout):
        sent.append(req)
        return FakeResponse({"choices": [{"message": {"content": "好"}, "finish_reason": reply.pop(0)}]})

    reply = ["stop", "length"]
    monkeypatch.setattr(summary.urllib.request, "urlopen", fake_urlopen)
    chat = summary.token_factory_chat("k")
    assert chat([{"role": "user", "content": "hi"}], "nvidia/x") == "好"
    body = json.loads(sent[0].data.decode("utf-8"))
    assert body["chat_template_kwargs"] == {"enable_thinking": False}  # Nemotron 3.5 默认开推理，推理过程会混进正文
    assert sent[0].get_header("Authorization") == "Bearer k"
    with pytest.raises(summary.Truncated):
        chat([{"role": "user", "content": "hi"}], "nvidia/x")


def test_summarize_drops_space_between_closing_bracket_and_chinese(tmp_path):
    from assay.summary import summarize

    s = summarize(facts5(), chat=fake_chat("「{cause_1}」 有 {cause_1_n} 条。"), cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "「二次抓取」有 2 条。"


def test_every_cause_has_advice_without_numbers():
    from assay.summary import ADVICE, check_text

    assert set(ADVICE) == set(Cause)
    for zh, en in ADVICE.values():
        assert zh.endswith("。") and en.endswith(".")
        assert check_text(zh, set()) == [] and check_text(en, set()) == []


def test_advice_comes_from_placeholders():
    """给采集员的建议只能用我们写的句子（{advice_i}），模型不许自己编（2026-10-05 第一次真调用编出了没用的建议）。"""
    from assay.summary import ADVICE, PROMPT_VERSION, SYSTEM, batch_facts, template

    vs, eps = batch()
    facts = batch_facts(vs, eps, title="ds")
    assert facts.values["zh"]["advice_1"] == ADVICE[Cause.REGRASP][0]
    assert facts.values["en"]["advice_1"] == ADVICE[Cause.REGRASP][1]
    assert "advice_3" in facts.meaning and "advice_4" not in facts.meaning
    assert "{advice_1}" in template(facts, "zh") and "{advice_1}" in template(facts, "en")
    assert PROMPT_VERSION == "4" and "{advice_" in SYSTEM and "tips of your own" in SYSTEM


def test_check_text_problems_are_bilingual():
    from helpers import assert_bilingual
    from assay.summary import check_text

    for text in ("有 7 条存疑", "{n_reject} 条作废", "坏了的 {n_doubt"):
        assert_bilingual(check_text(text, {"n_doubt"})[0])


def write_cache_entry(cache, name, lang, source, tries, prompt=None):
    import json

    from assay.summary import PROMPT_VERSION

    d = cache / "summary"
    d.mkdir(parents=True, exist_ok=True)
    attempts = [{"raw": "x", "problems": ["p"]}] * (tries - 1) + [{"raw": "y", "problems": [] if source == "model" else ["p"]}]
    (d / f"{name}.json").write_text(json.dumps({"model": "m", "prompt": prompt or PROMPT_VERSION, "lang": lang, "text": "t",
                                                "source": source, "problems": [], "attempts": attempts}), encoding="utf-8")


def test_cache_stats_counts_the_current_prompt_version_only(tmp_path):
    from assay.summary import cache_stats

    write_cache_entry(tmp_path, "a", "zh", "model", 1)
    write_cache_entry(tmp_path, "b", "en", "model", 2)
    write_cache_entry(tmp_path, "c", "zh", "template", 2)
    write_cache_entry(tmp_path, "d", "zh", "model", 1, prompt="3")
    assert cache_stats(tmp_path) == {"zh": {"first_try": 1, "after_retry": 0, "template": 1},
                                     "en": {"first_try": 0, "after_retry": 1, "template": 0}}
