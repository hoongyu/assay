import numpy as np

from helpers import make_episode
from assay.diagnose import Label, Verdict
from assay.report import _card
from assay.review import ReviewResult
from assay.review_run import run_reviews
from assay.types import Cause, Finding, VideoRef


def towel(ep_id="ds/ep000002", video=True, tmp=None):
    ep = make_episode(np.zeros((31 * 30, 2)), ep_id=ep_id, task="Fold the towel in half.")
    if video:
        path = tmp / f"{ep_id.replace('/', '_')}.mp4"
        path.write_bytes(b"fake")
        ep.cameras = {"observation.images.top": VideoRef(path, 0.0, 31.0)}
    return ep


def regrasp_verdict(ep_id="ds/ep000002"):
    f = Finding("regrasp", Cause.REGRASP, 0.0, 31.0, 6.0, "six switches", moments=(11.1, 12.833, 13.467, 20.9))
    return Verdict(ep_id, Label.DOUBT, ["r"], [f])


def test_run_reviews_asks_only_where_there_is_local_video(tmp_path):
    eps = [towel(tmp=tmp_path), towel("ds/ep000005", video=False, tmp=tmp_path)]
    verdicts = [regrasp_verdict(), regrasp_verdict("ds/ep000005")]
    cuts = []

    def media(ref, t0, t1):
        cuts.append((ref.path.name, round(t0, 2), round(t1, 2)))
        return b"mp4", [b"j"] * 8, [0.0] * 8

    out, skipped = run_reviews(verdicts, eps, lambda m, model: '{"answer": "yes"}', media=media)
    assert cuts == [("ds_ep000002.mp4", round(12.833 - 1.5, 2), round(13.467 + 1.5, 2))]
    assert out[0].reviews[0].answer == "yes" and out[0].reviews[0].question.startswith("This is a robot")
    assert out[1].reviews == [] and skipped == ["ds/ep000005"]


def test_run_reviews_with_keyframes_sends_images(tmp_path):
    sent = []

    def chat(messages, model):
        sent.append([p["type"] for p in messages[-1]["content"]])
        return '{"answer": "no"}'

    out, _ = run_reviews([regrasp_verdict()], [towel(tmp=tmp_path)], chat,
                         media=lambda ref, t0, t1: (b"mp4", [b"j"] * 8, [0.5 * i for i in range(8)]), mode="frames")
    assert sent == [["image_url"] * 8 + ["text"]] and out[0].reviews[0].outcome == "clears"


def test_card_shows_the_review_question_answer_and_outcome(tmp_path):
    v = regrasp_verdict()
    v.reviews = [ReviewResult("ds/ep000002", "regrasp", "yes", "confirms", "nvidia/Nemotron-3.5-Super-VL",
                              question="Does a robot gripper lose its grip and grasp again?")]
    ep = towel(tmp=tmp_path)
    en, zh = _card(v, ep, lang="en"), _card(v, ep, lang="zh")
    for html in (en, zh):
        assert "Nemotron 3.5 Super VL" in html and "Nebius Token Factory" in html
        assert "Does a robot gripper lose its grip and grasp again?" in html
    assert "agrees with the rule" in en and "支持规则的发现" in zh


def test_card_says_when_the_model_disagrees_or_gives_no_usable_answer(tmp_path):
    v = regrasp_verdict()
    ep = towel(tmp=tmp_path)
    v.reviews = [ReviewResult("ds/ep000002", "regrasp", "no", "clears", "m", question="q")]
    assert "disagrees with the rule" in _card(v, ep, lang="en")
    v.reviews = [ReviewResult("ds/ep000002", "regrasp", None, "unclear", "m", question="q")]
    assert "no usable answer" in _card(v, ep, lang="en") and "没给出能用的回答" in _card(v, ep, lang="zh")
