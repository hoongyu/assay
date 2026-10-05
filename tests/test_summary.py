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
