import base64
import json

import numpy as np
import pytest

from helpers import make_episode
from assay.diagnose import Label, Verdict
from assay.review import MODEL, apply_reviews, parse_answer, review, review_requests
from assay.summary import Truncated, token_factory_body, token_factory_text
from assay.types import Cause, Finding, VideoRef


def ep_of(seconds: float, ep_id: str = "ds/ep000002", task: str = "Fold the towel in half.", video_s: float | None = None):
    ep = make_episode(np.zeros((int(seconds * 30), 2)), ep_id=ep_id, task=task)
    if video_s is not None:
        ep.cameras = {"observation.images.top": VideoRef("v.mp4", 10.0, 10.0 + video_s)}
    return ep


def verdict(ep_id: str, *findings: Finding, label: Label = Label.DOUBT) -> Verdict:
    return Verdict(ep_id, label, ["reason"], list(findings))


def regrasp(moments=(2.9, 11.1, 12.833, 13.467, 20.9, 21.967), t1=31.0):
    return Finding("regrasp", Cause.REGRASP, 0.0, t1, 6.0, "six switches", moments=moments)


def wrong_task(t1=31.0, source="robometer"):
    return Finding("wrong_task", Cause.WRONG_TASK, 0.0, t1, 0.2, "low progress", source=source)


def test_regrasp_request_looks_at_the_two_closest_switches():
    (req,) = review_requests([verdict("ds/ep000002", regrasp())], [ep_of(31)])
    assert req.check == "regrasp" and req.episode_id == "ds/ep000002"
    assert req.t0 == pytest.approx(12.833 - 1.5) and req.t1 == pytest.approx(13.467 + 1.5)
    assert "1.5 s" in req.question and "2.1 s" in req.question
    assert req.choices == ("yes", "no", "unclear") and req.confirms == ("yes",) and req.clears == ("no",)


def test_video_length_request_covers_the_extra_footage():
    ep = ep_of(20, video_s=24.0)
    f = Finding("video_length", Cause.DATA_ERROR, 0.0, 20.0, 4.0, "video 4 s longer", source="meta")
    (req,) = review_requests([verdict(ep.id, f)], [ep])
    assert req.t0 == pytest.approx(18.0) and req.t1 == pytest.approx(24.0)
    assert "2.0 s" in req.question
    assert "person_in_scene" in req.choices and req.clears == ()


def test_wrong_task_request_watches_the_whole_episode_and_names_the_task():
    (req,) = review_requests([verdict("ds/ep000002", wrong_task())], [ep_of(31)])
    assert (req.t0, req.t1) == (0.0, pytest.approx(31.0))
    assert "Fold the towel in half." in req.question
    assert req.confirms == ("no",) and req.clears == ("yes",)


def test_one_request_per_episode_regrasp_before_wrong_task():
    (req,) = review_requests([verdict("ds/ep000002", wrong_task(), regrasp())], [ep_of(31)])
    assert req.check == "regrasp"


def test_passed_episodes_and_unreviewable_findings_get_no_request():
    pause = Finding("long_pause", Cause.LONG_PAUSE, 3.0, 8.0, 5.0, "pause")
    out = review_requests([verdict("ds/ep000002", regrasp(), label=Label.PASS), verdict("ds/ep000003", pause)],
                          [ep_of(31), ep_of(31, ep_id="ds/ep000003")])
    assert out == []


def test_parse_answer_accepts_only_the_listed_choices():
    choices = ("yes", "no", "unclear")
    assert parse_answer('{"answer": "no"}', choices) == "no"
    assert parse_answer('<think>hmm</think>```json\n{"answer": "YES"}\n```', choices) == "yes"
    assert parse_answer('{"answer": "maybe"}', choices) is None
    assert parse_answer("The gripper regrasps.", choices) is None


def capture(*answers):
    calls = []

    def chat(messages, model):
        calls.append((messages, model))
        return answers[len(calls) - 1]

    return chat, calls


def test_review_sends_the_clip_as_video_and_reads_the_answer():
    (req,) = review_requests([verdict("ds/ep000002", regrasp())], [ep_of(31)])
    chat, calls = capture('{"answer": "yes"}')
    res = review(req, chat, video=b"\x00mp4")
    assert (res.answer, res.outcome, res.problems) == ("yes", "confirms", [])
    (messages, model), = calls
    assert model == MODEL
    parts = messages[-1]["content"]
    assert parts[0] == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(b"\x00mp4").decode()}}
    assert parts[-1]["type"] == "text" and req.question in parts[-1]["text"]


def test_review_sends_keyframes_with_their_times():
    (req,) = review_requests([verdict("ds/ep000002", regrasp())], [ep_of(31)])
    chat, calls = capture('{"answer": "no"}')
    res = review(req, chat, frames=[b"a", b"b"], frame_times=[0.0, 1.5])
    assert (res.answer, res.outcome) == ("no", "clears")
    parts = calls[0][0][-1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "0.0 s, 1.5 s" in parts[-1]["text"]


def test_review_asks_again_once_when_the_answer_is_not_a_listed_choice():
    (req,) = review_requests([verdict("ds/ep000002", regrasp())], [ep_of(31)])
    chat, calls = capture('{"answer": "maybe"}', '{"answer": "no"}')
    res = review(req, chat, video=b"x")
    assert (res.answer, res.outcome, len(calls)) == ("no", "clears", 2)
    assert len(res.problems) == 1 and len(res.attempts) == 2


def test_review_is_unclear_after_two_unusable_answers():
    (req,) = review_requests([verdict("ds/ep000002", regrasp())], [ep_of(31)])
    chat, _ = capture("It looks like a regrasp.", "Yes.")
    res = review(req, chat, video=b"x")
    assert (res.answer, res.outcome, len(res.problems)) == (None, "unclear", 2)


def test_apply_reviews_attaches_results_without_changing_the_label():
    v = verdict("ds/ep000002", regrasp())
    (req,) = review_requests([v], [ep_of(31)])
    chat, _ = capture('{"answer": "yes"}')
    (out,) = apply_reviews([v], [review(req, chat, video=b"x")])
    assert out.label == Label.DOUBT and out.reviews[0].answer == "yes" and out.reviews[0].check == "regrasp"


def test_token_factory_body_has_the_summary_defaults_and_takes_overrides():
    msgs = [{"role": "user", "content": "hi"}]
    body = token_factory_body(msgs, "m")
    assert body == {"model": "m", "messages": msgs, "temperature": 0.2, "max_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False}}
    assert token_factory_body(msgs, "m", temperature=0.0, max_tokens=60)["max_tokens"] == 60


def test_token_factory_text_reads_the_answer_and_refuses_a_cut_one():
    ok = {"choices": [{"finish_reason": "stop", "message": {"content": "ready"}}]}
    assert token_factory_text(ok) == "ready"
    with pytest.raises(Truncated):
        token_factory_text({"choices": [{"finish_reason": "length", "message": {"content": "rea"}}]})
    assert json.dumps(token_factory_body([], "m"))  # plain JSON, no objects the browser cannot send
