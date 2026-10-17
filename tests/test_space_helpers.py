import zipfile

import pytest

from assay_space import find_dataset_root, safe_extract, summarize, valid_repo_id


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def test_safe_extract_rejects_path_traversal(tmp_path):
    z = _zip(tmp_path / "a.zip", {"../evil.txt": "x"})
    with pytest.raises(ValueError, match="路径"):
        safe_extract(z, tmp_path / "out", max_bytes=10_000)


def test_safe_extract_rejects_too_big(tmp_path):
    z = _zip(tmp_path / "a.zip", {"ds/data.bin": "x" * 5_000})
    with pytest.raises(ValueError, match="MB"):
        safe_extract(z, tmp_path / "out", max_bytes=1_000)


def test_find_root_inside_extracted_archive(tmp_path):
    z = _zip(tmp_path / "a.zip", {"upload/my_ds/meta/info.json": "{}", "upload/my_ds/data/x.parquet": "x"})
    out = safe_extract(z, tmp_path / "out", max_bytes=10_000)
    assert find_dataset_root(out) == out / "upload" / "my_ds"


def test_find_root_accepts_hdf5_and_rejects_unknown(tmp_path):
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "low_dim.hdf5").write_bytes(b"")
    assert find_dataset_root(tmp_path / "h") == tmp_path / "h" / "low_dim.hdf5"
    (tmp_path / "n").mkdir()
    with pytest.raises(ValueError):
        find_dataset_root(tmp_path / "n")


def test_summarize_counts_labels_and_causes():
    rows = [
        {"label": "合格", "findings": []},
        {"label": "存疑", "findings": [{"cause": "二次抓取"}, {"cause": "二次抓取"}]},
        {"label": "作废", "findings": [{"cause": "数据异常"}, {"cause": "二次抓取"}]},
    ]
    s = summarize(rows)
    assert s["n"] == 3 and s["labels"] == {"合格": 1, "存疑": 1, "作废": 1}
    assert s["causes"][0] == ("二次抓取", 2)


def test_valid_repo_id():
    assert valid_repo_id("lerobot/aloha_static_battery")
    assert not valid_repo_id("lerobot")
    assert not valid_repo_id("../x/y")
    assert not valid_repo_id("a/b/c")
