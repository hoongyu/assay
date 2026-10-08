import numpy as np

from helpers import make_episode, make_lerobot_v3, make_mp4
from assay.diagnose import Label, Verdict
from assay.ingest import load
from assay.keyframes import jpeg_data_uri, keyframe_times, shots_for
from assay.types import Cause, Finding


def v_with(*spans, label=Label.DOUBT):
    fs = [Finding("x", Cause.DROP, a, b, 1.0, "e") for a, b in spans]
    return Verdict("e", label, ["r"], fs)


def test_times_are_finding_edges():
    assert keyframe_times(v_with((1.0, 2.0)), 10.0) == [1.0, 2.0]


def test_whole_episode_finding_takes_start_middle_end():
    assert keyframe_times(v_with((0.0, 10.0)), 10.0) == [0.0, 5.0, 9.95]


def test_finding_with_moments_takes_those_moments():
    """二次抓取这类整条的发现带了换档时刻：按时刻取，超过 4 个时均匀取、保留首尾（叠毛巾 ep2 的六次换档）。"""
    f = Finding("regrasp", Cause.REGRASP, 0.0, 31.0, 2.0, "e", moments=(2.9, 11.1, 12.8, 13.5, 20.9, 22.0))
    assert keyframe_times(Verdict("e", Label.DOUBT, ["r"], [f]), 31.0) == [2.9, 12.8, 13.5, 22.0]


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


def test_video_past_joint_data_adds_one_frame_from_the_extra_part(tmp_path):
    from assay.types import VideoRef

    x = np.tile(np.linspace(0, 1, 90)[:, None], (1, 6))
    ds = make_lerobot_v3(tmp_path / "d", [("pick", x), ("pick", x)])
    eps = load(ds).episodes
    make_mp4(ds / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4", n=180)
    cam = "observation.images.front"
    ref = eps[0].cameras[cam]
    eps[0].cameras[cam] = VideoRef(ref.path, ref.from_s, ref.to_s + 2.0)  # 视频比关节长 2 秒
    longer = Finding("video_length", Cause.DATA_ERROR, 0.0, eps[0].duration, 2.0, "长", source="meta")
    v = Verdict(eps[0].id, Label.DOUBT, ["r"], [longer])
    shots, _ = shots_for([v], eps)
    assert [t for t, _ in shots[eps[0].id]] == [0.0, 1.5, 2.95, 4.0]  # 多出来那段的正中
