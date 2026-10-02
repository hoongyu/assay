import json

import numpy as np
import pytest

from helpers import make_lerobot_v3
from jianjin.ingest.lerobot import read_lerobot


def ramp6(n):
    return np.tile(np.linspace(0, 1, n)[:, None], (1, 6))


@pytest.fixture
def ds(tmp_path):
    return make_lerobot_v3(
        tmp_path / "so101_demo",
        [("pick the cube", ramp6(60)), ("place the cube", ramp6(90) * 2)],
    )


def test_reads_episodes(ds):
    res = read_lerobot(ds)
    assert res.failures == []
    assert [e.id for e in res.episodes] == ["so101_demo/ep000000", "so101_demo/ep000001"]
    e0, e1 = res.episodes
    assert e0.task == "pick the cube" and e1.task == "place the cube"
    assert e0.state.shape == (60, 6) and e1.state.shape == (90, 6)
    assert e0.action.shape == (60, 6)
    assert e0.fps == 30.0 and e0.source == "lerobot"
    assert e0.gripper_index == 5
    np.testing.assert_allclose(e1.timestamps[:2], [0, 1 / 30], atol=1e-6)


def test_video_refs(ds):
    e0, e1 = read_lerobot(ds).episodes
    ref = e1.cameras["observation.images.front"]
    assert ref.path == ds / "videos/observation.images.front/chunk-000/file-000.mp4"
    assert np.isclose(ref.from_s, 2.0) and np.isclose(ref.to_s, 5.0)


def test_rejects_old_version(ds):
    info = json.loads((ds / "meta/info.json").read_text(encoding="utf-8"))
    info["codebase_version"] = "v2.1"
    (ds / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="v2.1"):
        read_lerobot(ds)


def test_broken_data_file_is_recorded_not_raised(ds):
    (ds / "data/chunk-000/file-001.parquet").write_bytes(b"not a parquet file")
    res = read_lerobot(ds)
    assert len(res.episodes) == 2
    assert len(res.failures) == 1
    assert "file-001" in res.failures[0][0]
