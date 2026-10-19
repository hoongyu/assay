"""认出数据格式，分派给对应的读取器。"""

from __future__ import annotations

from pathlib import Path

from assay.ingest.lerobot import read_lerobot
from assay.ingest.robomimic import is_robomimic, read_robomimic
from assay.ingest.robomind import read_robomind
from assay.ingest.robovad import read_robovad
from assay.types import LoadResult


def load(path: str | Path) -> LoadResult:
    path = Path(path)
    if (path / "meta" / "info.json").exists():
        return read_lerobot(path)
    if (path / "annotations.json").exists() and (path / "proprioception").is_dir():
        return read_robovad(path)
    if path.is_file() and path.suffix in {".hdf5", ".h5"}:
        return read_robomimic(path) if is_robomimic(path) else read_robomind(path)
    if path.is_dir() and (any(path.rglob("*.hdf5")) or any(path.rglob("*.h5"))):
        return read_robomind(path)
    raise ValueError(
        f"认不出 {path} 的格式：既不是 LeRobot v2.x 或 v3（没有 meta/info.json）、"
        "RoboVAD（没有 annotations.json 和 proprioception/），也没有 HDF5 文件"
    )


__all__ = ["load"]
