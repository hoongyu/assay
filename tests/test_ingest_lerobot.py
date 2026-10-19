import json

import numpy as np
import pytest

from helpers import make_lerobot_v2, make_lerobot_v3
from assay.ingest.lerobot import read_lerobot


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


def test_rejects_unknown_version(ds):
    info = json.loads((ds / "meta/info.json").read_text(encoding="utf-8"))
    info["codebase_version"] = "v1.6"
    (ds / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="v1.6"):
        read_lerobot(ds)


@pytest.mark.parametrize("version", ["v2.0", "v2.1"])
def test_reads_v2_dataset(tmp_path, version):
    root = make_lerobot_v2(tmp_path / "v2demo", [("pick the cube", ramp6(60)), ("place the cube", ramp6(90) * 2)],
                           version=version)
    res = read_lerobot(root)
    assert res.failures == []
    assert [e.id for e in res.episodes] == ["v2demo/ep000000", "v2demo/ep000001"]
    e0, e1 = res.episodes
    assert e0.task == "pick the cube" and e1.task == "place the cube"
    assert e0.state.shape == (60, 6) and e1.state.shape == (90, 6)
    assert e0.action.shape == (60, 6)
    assert e0.fps == 30.0 and e0.source == "lerobot" and e0.gripper_index == 5
    assert e1.meta["episode_index"] == 1
    np.testing.assert_allclose(e1.timestamps[:2], [0, 1 / 30], atol=1e-6)
    ref = e1.cameras["observation.images.front"]
    assert ref.path == root / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    assert ref.from_s == 0.0 and ref.to_s is None


def test_v2_episode_shorter_than_metadata_is_a_failure(tmp_path):
    root = make_lerobot_v2(tmp_path / "short", [("pick", ramp6(60)), ("pick", ramp6(90))])
    lines = (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["length"] = 80
    lines[0] = json.dumps(row)
    (root / "meta/episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = read_lerobot(root)
    assert [e.id for e in res.episodes] == ["short/ep000001"]
    (name, why), = res.failures
    assert name == "short/ep000000" and "60/80" in why


def test_v2_broken_episode_file_is_recorded_not_raised(tmp_path):
    root = make_lerobot_v2(tmp_path / "broken", [("pick", ramp6(60)), ("pick", ramp6(90))])
    (root / "data/chunk-000/episode_000001.parquet").write_bytes(b"not a parquet file")
    res = read_lerobot(root)
    assert [e.id for e in res.episodes] == ["broken/ep000000"]
    assert len(res.failures) == 1 and "episode_000001" in res.failures[0][0]


def test_v2_repeated_episode_index_skips_only_that_episode(tmp_path):
    root = make_lerobot_v2(tmp_path / "rep", [("pick", ramp6(60)), ("pick", ramp6(90)), ("pick", ramp6(70))])
    with open(root / "meta/episodes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"episode_index": 1, "tasks": ["pick"], "length": 90}) + "\n")
    res = read_lerobot(root)
    assert [e.id for e in res.episodes] == ["rep/ep000000", "rep/ep000002"]
    (name, why), = res.failures
    assert name == "rep/ep000001" and "重复" in why


def _add_image_column(f):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(f)
    blob = pa.array([b"\x00" * 1000] * len(t), type=pa.binary())
    pq.write_table(t.append_column("observation.images.front", blob), f)


def test_only_needed_columns_are_read(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    root = make_lerobot_v3(tmp_path / "img", [("pick", ramp6(60)), ("pick", ramp6(90))])
    _add_image_column(root / "data/chunk-000/file-000.parquet")
    calls = []
    real = pq.read_table

    def spy(source, *args, columns=None, **kwargs):
        calls.append((str(source), columns))
        return real(source, *args, columns=columns, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    res = read_lerobot(root)
    assert res.failures == [] and [e.state.shape[0] for e in res.episodes] == [60, 90]
    data_calls = [cols for src, cols in calls if "data" in src.replace("\\", "/").split("/")]
    assert data_calls and all(cols is not None and "observation.images.front" not in cols for cols in data_calls)


def test_missing_optional_columns(tmp_path):
    import pyarrow.parquet as pq
    root = make_lerobot_v2(tmp_path / "bare", [("pick", ramp6(60))])
    f = root / "data/chunk-000/episode_000000.parquet"
    pq.write_table(pq.read_table(f).drop_columns(["action", "task_index"]), f)
    (e0,) = read_lerobot(root).episodes
    assert e0.action is None and e0.task == "" and e0.state.shape == (60, 6)


def test_broken_data_file_is_recorded_not_raised(ds):
    (ds / "data/chunk-000/file-001.parquet").write_bytes(b"not a parquet file")
    res = read_lerobot(ds)
    assert len(res.episodes) == 2
    assert len(res.failures) == 1
    assert "file-001" in res.failures[0][0]


def _split_data_file(root, cut):
    """把唯一的数据文件在第 cut 行切成两个文件，模拟一条示范跨文件。"""
    import pyarrow.parquet as pq
    f0 = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f0)
    pq.write_table(t.slice(0, cut), f0)
    pq.write_table(t.slice(cut), root / "data" / "chunk-000" / "file-001.parquet")


def test_episode_split_across_files_is_joined(tmp_path):
    root = make_lerobot_v3(tmp_path / "split", [("pick", ramp6(60)), ("pick", ramp6(90))])
    _split_data_file(root, 80)  # 第 2 条示范的前 20 帧在 file-000，后 70 帧在 file-001
    res = read_lerobot(root)
    assert res.failures == []
    assert [e.state.shape[0] for e in res.episodes] == [60, 90]
    np.testing.assert_allclose(res.episodes[1].state[:, 0], np.linspace(0, 1, 90), atol=1e-6)


def test_partly_downloaded_episode_is_a_failure(tmp_path):
    root = make_lerobot_v3(tmp_path / "part", [("pick", ramp6(60)), ("pick", ramp6(90))])
    _split_data_file(root, 80)
    (root / "data" / "chunk-000" / "file-001.parquet").unlink()  # 只下了第一个数据文件
    res = read_lerobot(root)
    assert [e.state.shape[0] for e in res.episodes] == [60]
    (name, why), = res.failures
    assert name == "part/ep000001" and "20/90" in why


def test_repeated_episode_index_in_metadata_is_a_dataset_failure(tmp_path):
    import pyarrow.parquet as pq
    root = make_lerobot_v3(tmp_path / "dup", [("pick", ramp6(60)), ("pick", ramp6(90))])
    meta = root / "meta" / "episodes" / "chunk-000"
    pq.write_table(pq.read_table(meta / "file-000.parquet"), meta / "file-001.parquet")  # 第二个文件又从 0 编号
    res = read_lerobot(root)
    assert res.episodes == []
    (name, why), = res.failures
    assert name == "meta/episodes" and "episode_index" in why and "重复" in why


def test_only_files_matching_data_path_template_are_read(tmp_path):
    import shutil
    root = make_lerobot_v3(tmp_path / "stale", [("pick", ramp6(60))])
    # 转换后没删的旧版 v2.1 文件，内容和新文件一样
    shutil.copy(root / "data" / "chunk-000" / "file-000.parquet", root / "data" / "chunk-000" / "episode_000000.parquet")
    res = read_lerobot(root)
    assert res.failures == []
    assert [e.state.shape[0] for e in res.episodes] == [60]


def test_a_few_repeated_indices_skip_only_those_episodes(tmp_path):
    import pyarrow.parquet as pq
    root = make_lerobot_v3(tmp_path / "few", [("pick", ramp6(60)), ("pick", ramp6(90)), ("pick", ramp6(70))])
    meta = root / "meta" / "episodes" / "chunk-000"
    pq.write_table(pq.read_table(meta / "file-000.parquet").slice(1, 1), meta / "file-001.parquet")  # 只有 1 号重复
    res = read_lerobot(root)
    assert [e.id for e in res.episodes] == ["few/ep000000", "few/ep000002"]
    (name, why), = res.failures
    assert name == "few/ep000001" and "重复" in why
