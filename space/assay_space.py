"""在线演示用到的纯逻辑：安全解压、找数据集根目录、汇总判定。界面在 app.py。"""

from __future__ import annotations

import collections
import re
import zipfile
from pathlib import Path

REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def valid_repo_id(s: str) -> bool:
    return bool(REPO_ID.match(s.strip())) and ".." not in s


def safe_extract(zip_path: str | Path, dest: str | Path, max_bytes: int) -> Path:
    """解压访客上传的压缩包：拒绝跳出目标目录的路径，解压后总大小超过 max_bytes 就拒绝。"""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError(f"解压后超过 {max_bytes / 1e6:.0f} MB，演示里放不下")
        for m in members:
            target = (dest / m.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"压缩包里有跳出目录的路径：{m.filename}")
        z.extractall(dest)
    return dest


def find_dataset_root(folder: str | Path) -> Path:
    """LeRobot v3 认 meta/info.json 所在的目录；robomimic 或 RoboMIND 认第一个 HDF5 文件。"""
    folder = Path(folder)
    info = sorted(folder.rglob("meta/info.json"), key=lambda p: len(p.parts))
    if info:
        return info[0].parent.parent
    h5 = sorted([*folder.rglob("*.hdf5"), *folder.rglob("*.h5")], key=lambda p: len(p.parts))
    if h5:
        return h5[0]
    raise ValueError("没找到 LeRobot v3 数据（meta/info.json）或 HDF5 文件")


def summarize(rows: list[dict]) -> dict:
    labels = collections.Counter(r["label"] for r in rows)
    causes = collections.Counter()
    for r in rows:
        for c in {f.get("cause") for f in r["findings"]} - {None}:
            causes[c] += 1
    return {"n": len(rows), "labels": dict(labels), "causes": causes.most_common()}
