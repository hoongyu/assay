from dataclasses import replace

import numpy as np

from helpers import make_episode
from jianjin.diagnose import Label, Verdict
from jianjin.types import Cause, Finding


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
    from jianjin.summary import batch_facts

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
    from jianjin.summary import batch_facts

    vs, eps = batch(ops=["A", "A", "A", "B", "B"])
    zh = batch_facts(vs, eps, title="ds").values["zh"]
    assert (zh["op_1"], zh["op_1_n"], zh["op_1_flagged"], zh["op_1_cause"], zh["op_1_cause_n"]) == ("A", "3", "2", "二次抓取", "2")
    assert (zh["op_2"], zh["op_2_n"], zh["op_2_flagged"], zh["op_2_cause"], zh["op_2_cause_n"]) == ("B", "2", "1", "数据异常", "1")


def test_check_text_rejects_numbers_the_model_wrote_itself():
    from jianjin.summary import check_text

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
    from jianjin.summary import Facts, fill

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
    from jianjin.summary import batch_facts

    vs, eps = batch()
    return batch_facts(vs, eps, title="ds")


def test_summarize_uses_model_text_with_numbers_filled(tmp_path):
    from jianjin.summary import summarize

    chat = fake_chat("<think>先算一下</think>本批 {n_total} 条，先看作废的 {n_reject} 条。")
    s = summarize(facts5(), chat=chat, cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "本批 5 条，先看作废的 1 条。" and s.source["zh"] == "model" and len(chat.calls) == 1
    prompt = chat.calls[0][-1]["content"]
    assert "{n_doubt}" in prompt and "= 2" in prompt  # 模型看得到每个占位符的意思和值


def test_summarize_retries_once_then_falls_back_to_template(tmp_path):
    from jianjin.summary import summarize

    chat = fake_chat("有 7 条存疑。", "本批 {n_total} 条。")
    s = summarize(facts5(), chat=chat, cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "本批 5 条。" and s.source["zh"] == "model" and len(chat.calls) == 2
    assert "阿拉伯数字" in chat.calls[1][-1]["content"]  # 重试时把问题告诉模型
    bad = fake_chat("有 7 条存疑。", "还是 7 条。")
    t = summarize(facts5(), chat=bad, cache_dir=tmp_path / "b", langs=("zh",))
    assert t.source["zh"] == "template" and "5 条示范" in t.texts["zh"] and t.problems["zh"]


def test_summarize_caches_model_answers_and_uses_template_without_key(tmp_path):
    from jianjin.summary import summarize

    summarize(facts5(), chat=fake_chat("本批 {n_total} 条。"), cache_dir=tmp_path, langs=("zh",))
    again = fake_chat("不该被调用")
    s = summarize(facts5(), chat=again, cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "本批 5 条。" and again.calls == []
    t = summarize(facts5(), chat=None, cache_dir=tmp_path / "t", langs=("zh", "en"))
    assert t.source == {"zh": "template", "en": "template"} and t.model is None
    assert t.texts["en"].startswith("Batch ds: 5 episodes checked")
    assert not list((tmp_path / "t").rglob("*.json"))  # 模板不进缓存，有了密钥以后还会去调模型


def test_summarize_falls_back_when_the_call_fails(tmp_path):
    from jianjin.summary import summarize

    def broken(messages, model):
        raise OSError("网络断了")

    s = summarize(facts5(), chat=broken, cache_dir=tmp_path, langs=("zh",))
    assert s.source["zh"] == "template" and "网络断了" in s.problems["zh"][0]


def test_summarize_spaces_latin_words_in_chinese(tmp_path):
    from jianjin.summary import summarize

    s = summarize(facts5(), chat=fake_chat("{cause_1}和Robometer都要看。"), cache_dir=tmp_path, langs=("zh",))
    assert s.texts["zh"] == "二次抓取和 Robometer 都要看。"
