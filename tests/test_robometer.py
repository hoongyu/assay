import numpy as np

from assay.judges import JUDGES
from assay.judges.robometer import RobometerJudge, subsample


class FakeEncoder:
    def __init__(self):
        self.seen = None

    def encode_samples(self, samples):
        (frames, task), = samples
        self.seen = (frames, task)
        return {"input_ids": np.zeros((1, 5)), "n": len(frames), "unrelated": 1}


class FakeConfig:
    use_discrete_progress = True


class FakeModel:
    config = FakeConfig()

    def _compute_rbm_logits(self, inputs):
        n = inputs["n"]
        return np.linspace(0, 1.2, n)[None], np.linspace(1, 0, n)[None]


def fake_decode(progress_logits, success_logits, is_discrete_mode):
    assert is_discrete_mode is True
    return {"progress_pred": [progress_logits[0]], "success_probs": [success_logits[0]]}


def test_subsample():
    assert subsample(5, 24).tolist() == [0, 1, 2, 3, 4]
    idx = subsample(100, 24)
    assert len(idx) == 24 and idx[0] == 0 and idx[-1] == 99


def test_curve_uses_per_frame_outputs():
    enc = FakeEncoder()
    j = RobometerJudge(max_frames=4, model=FakeModel(), encoder=enc, decode=fake_decode, input_keys=("input_ids", "n"))
    frames = np.zeros((10, 8, 8, 3), np.uint8)
    times = np.arange(10) * 0.5
    c = j.curve(frames, times, "pick")
    assert enc.seen[1] == "pick" and len(enc.seen[0]) == 4
    assert c.times == [0.0, 1.5, 3.0, 4.5]
    assert len(c.progress) == 4 and c.progress[-1] == 1.0  # 截到 [0, 1]
    assert c.success[0] == 1.0 and c.success[-1] == 0.0
    assert c.model == "robometer" and "f4" in c.version


def test_registered():
    assert JUDGES["robometer"] is RobometerJudge
