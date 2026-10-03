import numpy as np

from helpers import make_episode, make_lerobot_v3, make_mp4
from jianjin.diagnose import Label, Verdict
from jianjin.ingest import load
from jianjin.keyframes import jpeg_data_uri, keyframe_times, shots_for
from jianjin.types import Cause, Finding


def v_with(*spans, label=Label.DOUBT):
    fs = [Finding("x", Cause.DROP, a, b, 1.0, "e") for a, b in spans]
    return Verdict("e", label, ["r"], fs)


def test_times_are_finding_edges():
    assert keyframe_times(v_with((1.0, 2.0)), 10.0) == [1.0, 2.0]


def test_whole_episode_finding_takes_start_middle_end():
    assert keyframe_times(v_with((0.0, 10.0)), 10.0) == [0.0, 5.0, 9.95]


def test_times_clamped_merged_and_capped():
    assert keyframe_times(v_with((9.0, 12.0)), 10.0) == [9.0, 9.95]
    assert keyframe_times(v_with((1.0, 1.2)), 10.0) == [1.0]
    assert keyframe_times(v_with((0.0, 1.0), (2.0, 3.0), (4.0, 5.0)), 10.0) == [0.0, 2.0, 3.0, 5.0]


def test_jpeg_data_uri():
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    assert jpeg_data_uri(img).startswith("data:image/jpeg;base64,/9j/")


def test_shots_only_for_doubt_and_reject_with_local_video(tmp_path):
    x = np.tile(np.linspace(0, 1, 90)[:, None], (1, 6))
    ds = make_lerobot_v3(tmp_path / "d", [("pick", x), ("pick", x)])
    eps = load(ds).episodes
    make_mp4(ds / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4", n=180)
    vs = [v_with((1.0, 2.0)), v_with(label=Label.PASS)]
    vs[0].episode_id, vs[1].episode_id = eps[0].id, eps[1].id
    shots, missing = shots_for(vs, eps)
    assert list(shots) == [eps[0].id] and missing == 0
    assert [t for t, _ in shots[eps[0].id]] == [1.0, 2.0]
    shots, missing = shots_for(vs, [make_episode(x, ep_id=eps[0].id), eps[1]])  # 没有视频
    assert shots == {} and missing == 1
