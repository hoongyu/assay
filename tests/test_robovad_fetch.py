import zipfile

from jianjin.eval.robovad_fetch import SMALL_PREFIXES, extract


def make_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("RoboVAD/", "")
        z.writestr("RoboVAD/annotations.json", "{}")
        z.writestr("RoboVAD/evaluation/in_domain.csv", "episode_name,start_frame,end_frame_exclusive,label\n")
        z.writestr("RoboVAD/proprioception/h_pickandplace_ep1.parquet", b"x")
        z.writestr("RoboVAD/videos/h_pickandplace_ep1_left.mp4", b"v")
        z.writestr("RoboVAD/.DS_Store", b"")
        z.writestr("__MACOSX/RoboVAD/._annotations.json", b"")
    return path


def test_extract_small_parts_only(tmp_path):
    z = zipfile.ZipFile(make_zip(tmp_path / "r.zip"))
    n = extract(z, tmp_path / "out", SMALL_PREFIXES)
    assert n == 3
    assert (tmp_path / "out/annotations.json").exists()
    assert (tmp_path / "out/evaluation/in_domain.csv").exists()
    assert (tmp_path / "out/proprioception/h_pickandplace_ep1.parquet").exists()
    assert not (tmp_path / "out/videos").exists()
    assert not (tmp_path / "out/.DS_Store").exists()


def test_extract_skips_existing_same_size(tmp_path):
    z = zipfile.ZipFile(make_zip(tmp_path / "r.zip"))
    extract(z, tmp_path / "out", SMALL_PREFIXES)
    assert extract(z, tmp_path / "out", SMALL_PREFIXES) == 0
